// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package imports

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"go/ast"
	"go/build"
	"go/parser"
	"go/token"
	"go/types"
	"io/fs"
	"io/ioutil"
	"os"
	"path"
	"path/filepath"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"sync"
	"unicode"
	"unicode/utf8"

	"golang.org/x/sync/errgroup"
	"golang.org/x/tools/go/ast/astutil"
	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/gocommand"
	"golang.org/x/tools/internal/gopathwalk"
	"golang.org/x/tools/internal/stdlib"
)

// importToGroup is a list of functions which map from an import path to
// a group number.
var importToGroup = []func(localPrefix, importPath string) (num int, ok bool){
	func(localPrefix, importPath string) (num int, ok bool) {
		if localPrefix == "" {
			return
		}
		for _, p := range strings.Split(localPrefix, ",") {
			if strings.HasPrefix(importPath, p) || strings.TrimSuffix(p, "/") == importPath {
				return 3, true
			}
		}
		return
	},
	func(_, importPath string) (num int, ok bool) {
		if strings.HasPrefix(importPath, "appengine") {
			return 2, true
		}
		return
	},
	func(_, importPath string) (num int, ok bool) {
		firstComponent := strings.Split(importPath, "/")[0]
		if strings.Contains(firstComponent, ".") {
			return 1, true
		}
		return
	},
}

func importGroup(localPrefix, importPath string) int {
	for _, fn := range importToGroup {
		if n, ok := fn(localPrefix, importPath); ok {
			return n
		}
	}
	return 0
}

type ImportFixType int

const (
	AddImport ImportFixType = iota
	DeleteImport
	SetImportName
)

type ImportFix struct {
	// StmtInfo represents the import statement this fix will add, remove, or change.
	StmtInfo ImportInfo
	// IdentName is the identifier that this fix will add or remove.
	IdentName string
	// FixType is the type of fix this is (AddImport, DeleteImport, SetImportName).
	FixType   ImportFixType
	Relevance float64 // see pkg
}

// An ImportInfo represents a single import statement.
type ImportInfo struct {
	ImportPath string // import path, e.g. "crypto/rand".
	Name       string // import name, e.g. "crand", or "" if none.
}

// A packageInfo represents what's known about a package.
type packageInfo struct {
	name    string          // real package name, if known.
	exports map[string]bool // known exports.
}

// parseOtherFiles parses all the Go files in srcDir except filename, including
// test files if filename looks like a test.
//
// It returns an error only if ctx is cancelled. Files with parse errors are
// ignored.
func parseOtherFiles(ctx context.Context, fset *token.FileSet, srcDir, filename string) ([]*ast.File, error) {
	// This could use go/packages but it doesn't buy much, and it fails
	// with https://golang.org/issue/26296 in LoadFiles mode in some cases.
	considerTests := strings.HasSuffix(filename, "_test.go")

	fileBase := filepath.Base(filename)
	packageFileInfos, err := os.ReadDir(srcDir)
	if err != nil {
		return nil, ctx.Err()
	}

	var files []*ast.File
	for _, fi := range packageFileInfos {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		if fi.Name() == fileBase || !strings.HasSuffix(fi.Name(), ".go") {
			continue
		}
		if !considerTests && strings.HasSuffix(fi.Name(), "_test.go") {
			continue
		}

		f, err := parser.ParseFile(fset, filepath.Join(srcDir, fi.Name()), nil, parser.SkipObjectResolution)
		if err != nil {
			continue
		}

		files = append(files, f)
	}

	return files, ctx.Err()
}

// addGlobals puts the names of package vars into the provided map.
func addGlobals(f *ast.File, globals map[string]bool) {
	for _, decl := range f.Decls {
		genDecl, ok := decl.(*ast.GenDecl)
		if !ok {
			continue
		}

		for _, spec := range genDecl.Specs {
			valueSpec, ok := spec.(*ast.ValueSpec)
			if !ok {
				continue
			}
			globals[valueSpec.Names[0].Name] = true
		}
	}
}

// collectReferences builds a map of selector expressions, from
// left hand side (X) to a set of right hand sides (Sel).
func collectReferences(f *ast.File) references {
	refs := references{}

	var visitor visitFn
	visitor = func(node ast.Node) ast.Visitor {
		if node == nil {
			return visitor
		}
		switch v := node.(type) {
		case *ast.SelectorExpr:
			xident, ok := v.X.(*ast.Ident)
			if !ok {
				break
			}
			if xident.Obj != nil {
				// If the parser can resolve it, it's not a package ref.
				break
			}
			if !ast.IsExported(v.Sel.Name) {
				// Whatever this is, it's not exported from a package.
				break
			}
			pkgName := xident.Name
			r := refs[pkgName]
			if r == nil {
				r = make(map[string]bool)
				refs[pkgName] = r
			}
			r[v.Sel.Name] = true
		}
		return visitor
	}
	ast.Walk(visitor, f)
	return refs
}

// collectImports returns all the imports in f.
// Unnamed imports (., _) and "C" are ignored.
func collectImports(f *ast.File) []*ImportInfo {
	var imports []*ImportInfo
	for _, imp := range f.Imports {
		var name string
		if imp.Name != nil {
			name = imp.Name.Name
		}
		if imp.Path.Value == `"C"` || name == "_" || name == "." {
			continue
		}
		path := strings.Trim(imp.Path.Value, `"`)
		imports = append(imports, &ImportInfo{
			Name:       name,
			ImportPath: path,
		})
	}
	return imports
}

// findMissingImport searches pass's candidates for an import that provides
// pkg, containing all of syms.
func (p *pass) findMissingImport(pkg string, syms map[string]bool) *ImportInfo {
	for _, candidate := range p.candidates {
		pkgInfo, ok := p.knownPackages[candidate.ImportPath]
		if !ok {
			continue
		}
		if p.importIdentifier(candidate) != pkg {
			continue
		}

		allFound := true
		for right := range syms {
			if !pkgInfo.exports[right] {
				allFound = false
				break
			}
		}

		if allFound {
			return candidate
		}
	}
	return nil
}

// references is set of references found in a Go file. The first map key is the
// left hand side of a selector expression, the second key is the right hand
// side, and the value should always be true.
type references map[string]map[string]bool

// A pass contains all the inputs and state necessary to fix a file's imports.
// It can be modified in some ways during use; see comments below.
type pass struct {
	// Inputs. These must be set before a call to load, and not modified after.
	fset                 *token.FileSet // fset used to parse f and its siblings.
	f                    *ast.File      // the file being fixed.
	srcDir               string         // the directory containing f.
	env                  *ProcessEnv    // the environment to use for go commands, etc.
	loadRealPackageNames bool           // if true, load package names from disk rather than guessing them.
	otherFiles           []*ast.File    // sibling files.

	// Intermediate state, generated by load.
	existingImports map[string][]*ImportInfo
	allRefs         references
	missingRefs     references

	// Inputs to fix. These can be augmented between successive fix calls.
	lastTry       bool                    // indicates that this is the last call and fix should clean up as best it can.
	candidates    []*ImportInfo           // candidate imports in priority order.
	knownPackages map[string]*packageInfo // information about all known packages.
}

// loadPackageNames saves the package names for everything referenced by imports.
func (p *pass) loadPackageNames(imports []*ImportInfo) error {
	if p.env.Logf != nil {
		p.env.Logf("loading package names for %v packages", len(imports))
		defer func() {
			p.env.Logf("done loading package names for %v packages", len(imports))
		}()
	}
	var unknown []string
	for _, imp := range imports {
		if _, ok := p.knownPackages[imp.ImportPath]; ok {
			continue
		}
		unknown = append(unknown, imp.ImportPath)
	}

	resolver, err := p.env.GetResolver()
	if err != nil {
		return err
	}

	names, err := resolver.loadPackageNames(unknown, p.srcDir)
	if err != nil {
		return err
	}

	for path, name := range names {
		p.knownPackages[path] = &packageInfo{
			name:    name,
			exports: map[string]bool{},
		}
	}
	return nil
}

// if there is a trailing major version, remove it
func withoutVersion(nm string) string {
	if v := path.Base(nm); len(v) > 0 && v[0] == 'v' {
		if _, err := strconv.Atoi(v[1:]); err == nil {
			// this is, for instance, called with rand/v2 and returns rand
			if len(v) < len(nm) {
				xnm := nm[:len(nm)-len(v)-1]
				return path.Base(xnm)
			}
		}
	}
	return nm
}

// importIdentifier returns the identifier that imp will introduce. It will
// guess if the package name has not been loaded, e.g. because the source
// is not available.
func (p *pass) importIdentifier(imp *ImportInfo) string {
	if imp.Name != "" {
		return imp.Name
	}
	known := p.knownPackages[imp.ImportPath]
	if known != nil && known.name != "" {
		return withoutVersion(known.name)
	}
	return ImportPathToAssumedName(imp.ImportPath)
}

// load reads in everything necessary to run a pass, and reports whether the
// file already has all the imports it needs. It fills in p.missingRefs with the
// file's missing symbols, if any, or removes unused imports if not.
func (p *pass) load() ([]*ImportFix, bool) {
	p.knownPackages = map[string]*packageInfo{}
	p.missingRefs = references{}
	p.existingImports = map[string][]*ImportInfo{}

	// Load basic information about the file in question.
	p.allRefs = collectReferences(p.f)

	// Load stuff from other files in the same package:
	// global variables so we know they don't need resolving, and imports
	// that we might want to mimic.
	globals := map[string]bool{}
	for _, otherFile := range p.otherFiles {
		// Don't load globals from files that are in the same directory
		// but a different package. Using them to suggest imports is OK.
		if p.f.Name.Name == otherFile.Name.Name {
			addGlobals(otherFile, globals)
		}
		p.candidates = append(p.candidates, collectImports(otherFile)...)
	}

	// Resolve all the import paths we've seen to package names, and store
	// f's imports by the identifier they introduce.
	imports := collectImports(p.f)
	if p.loadRealPackageNames {
		err := p.loadPackageNames(append(imports, p.candidates...))
		if err != nil {
			p.env.logf("loading package names: %v", err)
			return nil, false
		}
	}
	for _, imp := range imports {
		p.existingImports[p.importIdentifier(imp)] = append(p.existingImports[p.importIdentifier(imp)], imp)
	}

	// Find missing references.
	for left, rights := range p.allRefs {
		if globals[left] {
			continue
		}
		_, ok := p.existingImports[left]
		if !ok {
			p.missingRefs[left] = rights
			continue
		}
	}
	if len(p.missingRefs) != 0 {
		return nil, false
	}

	return p.fix()
}

// fix attempts to satisfy missing imports using p.candidates. If it finds
// everything, or if p.lastTry is true, it updates fixes to add the imports it found,
// delete anything unused, and update import names, and returns true.
func (p *pass) fix() ([]*ImportFix, bool) {
	// Find missing imports.
	var selected []*ImportInfo
	for left, rights := range p.missingRefs {
		if imp := p.findMissingImport(left, rights); imp != nil {
			selected = append(selected, imp)
		}
	}

	if !p.lastTry && len(selected) != len(p.missingRefs) {
		return nil, false
	}

	// Found everything, or giving up. Add the new imports and remove any unused.
	var fixes []*ImportFix
	for _, identifierImports := range p.existingImports {
		for _, imp := range identifierImports {
			// We deliberately ignore globals here, because we can't be sure
			// they're in the same package. People do things like put multiple
			// main packages in the same directory, and we don't want to
			// remove imports if they happen to have the same name as a var in
			// a different package.
			if _, ok := p.allRefs[p.importIdentifier(imp)]; !ok {
				fixes = append(fixes, &ImportFix{
					StmtInfo:  *imp,
					IdentName: p.importIdentifier(imp),
					FixType:   DeleteImport,
				})
				continue
			}

			// An existing import may need to update its import name to be correct.
			if name := p.importSpecName(imp); name != imp.Name {
				fixes = append(fixes, &ImportFix{
					StmtInfo: ImportInfo{
						Name:       name,
						ImportPath: imp.ImportPath,
					},
					IdentName: p.importIdentifier(imp),
					FixType:   SetImportName,
				})
			}
		}
	}
	// Collecting fixes involved map iteration, so sort for stability. See
	// golang/go#59976.
	sortFixes(fixes)

	// collect selected fixes in a separate slice, so that it can be sorted
	// separately. Note that these fixes must occur after fixes to existing
	// imports. TODO(rfindley): figure out why.
	var selectedFixes []*ImportFix
	for _, imp := range selected {
		selectedFixes = append(selectedFixes, &ImportFix{
			StmtInfo: ImportInfo{
				Name:       p.importSpecName(imp),
				ImportPath: imp.ImportPath,
			},
			IdentName: p.importIdentifier(imp),
			FixType:   AddImport,
		})
	}
	sortFixes(selectedFixes)

	return append(fixes, selectedFixes...), true
}

func sortFixes(fixes []*ImportFix) {
	sort.Slice(fixes, func(i, j int) bool {
		fi, fj := fixes[i], fixes[j]
		if fi.StmtInfo.ImportPath != fj.StmtInfo.ImportPath {
			return fi.StmtInfo.ImportPath < fj.StmtInfo.ImportPath
		}
		if fi.StmtInfo.Name != fj.StmtInfo.Name {
			return fi.StmtInfo.Name < fj.StmtInfo.Name
		}
		if fi.IdentName != fj.IdentName {
			return fi.IdentName < fj.IdentName
		}
		return fi.FixType < fj.FixType
	})
}

// importSpecName gets the import name of imp in the import spec.
//
// When the import identifier matches the assumed import name, the import name does
// not appear in the import spec.
func (p *pass) importSpecName(imp *ImportInfo) string {
	// If we did not load the real package names, or the name is already set,
	// we just return the existing name.
	if !p.loadRealPackageNames || imp.Name != "" {
		return imp.Name
	}

	ident := p.importIdentifier(imp)
	if ident == ImportPathToAssumedName(imp.ImportPath) {
		return "" // ident not needed since the assumed and real names are the same.
	}
	return ident
}

// apply will perform the fixes on f in order.
func apply(fset *token.FileSet, f *ast.File, fixes []*ImportFix) {
	for _, fix := range fixes {
		switch fix.FixType {
		case DeleteImport:
			astutil.DeleteNamedImport(fset, f, fix.StmtInfo.Name, fix.StmtInfo.ImportPath)
		case AddImport:
			astutil.AddNamedImport(fset, f, fix.StmtInfo.Name, fix.StmtInfo.ImportPath)
		case SetImportName:
			// Find the matching import path and change the name.
			for _, spec := range f.Imports {
				path := strings.Trim(spec.Path.Value, `"`)
				if path == fix.StmtInfo.ImportPath {
					spec.Name = &ast.Ident{
						Name:    fix.StmtInfo.Name,
						NamePos: spec.Pos(),
					}
				}
			}
		}
	}
}

// assumeSiblingImportsValid assumes that siblings' use of packages is valid,
// adding the exports they use.
func (p *pass) assumeSiblingImportsValid() {
	for _, f := range p.otherFiles {
		refs := collectReferences(f)
		imports := collectImports(f)
		importsByName := map[string]*ImportInfo{}
		for _, imp := range imports {
			importsByName[p.importIdentifier(imp)] = imp
		}
		for left, rights := range refs {
			if imp, ok := importsByName[left]; ok {
				if m, ok := stdlib.PackageSymbols[imp.ImportPath]; ok {
					// We have the stdlib in memory; no need to guess.
					rights = symbolNameSet(m)
				}
				p.addCandidate(imp, &packageInfo{
					// no name; we already know it.
					exports: rights,
				})
			}
		}
	}
}

// addCandidate adds a candidate import to p, and merges in the information
// in pkg.
func (p *pass) addCandidate(imp *ImportInfo, pkg *packageInfo) {
	p.candidates = append(p.candidates, imp)
	if existing, ok := p.knownPackages[imp.ImportPath]; ok {
		if existing.name == "" {
			existing.name = pkg.name
		}
		for export := range pkg.exports {
			existing.exports[export] = true
		}
	} else {
		p.knownPackages[imp.ImportPath] = pkg
	}
}

// fixImports adds and removes imports from f so that all its references are
// satisfied and there are no unused imports.
//
// This is declared as a variable rather than a function so goimports can
// easily be extended by adding a file with an init function.
//
// DO NOT REMOVE: used internally at Google.
var fixImports = fixImportsDefault

func fixImportsDefault(fset *token.FileSet, f *ast.File, filename string, env *ProcessEnv) error {
	fixes, err := getFixes(context.Background(), fset, f, filename, env)
	if err != nil {
		return err
	}
	apply(fset, f, fixes)
	return err
}

// getFixes gets the import fixes that need to be made to f in order to fix the imports.
// It does not modify the ast.
func getFixes(ctx context.Context, fset *token.FileSet, f *ast.File, filename string, env *ProcessEnv) ([]*ImportFix, error) {
	abs, err := filepath.Abs(filename)
	if err != nil {
		return nil, err
	}
	srcDir := filepath.Dir(abs)
	env.logf("fixImports(filename=%q), abs=%q, srcDir=%q ...", filename, abs, srcDir)

	// First pass: looking only at f, and using the naive algorithm to
	// derive package names from import paths, see if the file is already
	// complete. We can't add any imports yet, because we don't know
	// if missing references are actually package vars.
	p := &pass{fset: fset, f: f, srcDir: srcDir, env: env}
	if fixes, done := p.load(); done {
		return fixes, nil
	}

	otherFiles, err := parseOtherFiles(ctx, fset, srcDir, filename)
	if err != nil {
		return nil, err
	}

	// Second pass: add information from other files in the same package,
	// like their package vars and imports.
	p.otherFiles = otherFiles
	if fixes, done := p.load(); done {
		return fixes, nil
	}

	// Now we can try adding imports from the stdlib.
	p.assumeSiblingImportsValid()
	addStdlibCandidates(p, p.missingRefs)
	if fixes, done := p.fix(); done {
		return fixes, nil
	}

	// Third pass: get real package names where we had previously used
	// the naive algorithm.
	p = &pass{fset: fset, f: f, srcDir: srcDir, env: env}
	p.loadRealPackageNames = true
	p.otherFiles = otherFiles
	if fixes, done := p.load(); done {
		return fixes, nil
	}

	if err := addStdlibCandidates(p, p.missingRefs); err != nil {
		return nil, err
	}
	p.assumeSiblingImportsValid()
	if fixes, done := p.fix(); done {
		return fixes, nil
	}

	// Go look for candidates in $GOPATH, etc. We don't necessarily load
	// the real exports of sibling imports, so keep assuming their contents.
	if err := addExternalCandidates(ctx, p, p.missingRefs, filename); err != nil {
		return nil, err
	}

	p.lastTry = true
	fixes, _ := p.fix()
	return fixes, nil
}

// MaxRelevance is the highest relevance, used for the standard library.
// Chosen arbitrarily to match pre-existing gopls code.
const MaxRelevance = 7.0

// getCandidatePkgs works with the passed callback to find all acceptable packages.
// It deduplicates by import path, and uses a cached stdlib rather than reading
// from disk.
func getCandidatePkgs(ctx context.Context, wrappedCallback *scanCallback, filename, filePkg string, env *ProcessEnv) error {
	notSelf := func(p *pkg) bool {
		return p.packageName != filePkg || p.dir != filepath.Dir(filename)
	}
	goenv, err := env.goEnv()
	if err != nil {
		return err
	}

	var mu sync.Mutex // to guard asynchronous access to dupCheck
	dupCheck := map[string]struct{}{}

	// Start off with the standard library.
	for importPath, symbols := range stdlib.PackageSymbols {
		p := &pkg{
			dir:             filepath.Join(goenv["GOROOT"], "src", importPath),
			importPathShort: importPath,
			packageName:     path.Base(importPath),
			relevance:       MaxRelevance,
		}
		dupCheck[importPath] = struct{}{}
		if notSelf(p) && wrappedCallback.dirFound(p) && wrappedCallback.packageNameLoaded(p) {
			var exports []stdlib.Symbol
			for _, sym := range symbols {
				switch sym.Kind {
				case stdlib.Func, stdlib.Type, stdlib.Var, stdlib.Const:
					exports = append(exports, sym)
				}
			}
			wrappedCallback.exportsLoaded(p, exports)
		}
	}

	scanFilter := &scanCallback{
		rootFound: func(root gopathwalk.Root) bool {
			// Exclude goroot results -- getting them is relatively expensive, not cached,
			// and generally redundant with the in-memory version.
			return root.Type != gopathwalk.RootGOROOT && wrappedCallback.rootFound(root)
		},
		dirFound: wrappedCallback.dirFound,
		packageNameLoaded: func(pkg *pkg) bool {
			mu.Lock()
			defer mu.Unlock()
			if _, ok := dupCheck[pkg.importPathShort]; ok {
				return false
			}
			dupCheck[pkg.importPathShort] = struct{}{}
			return notSelf(pkg) && wrappedCallback.packageNameLoaded(pkg)
		},
		exportsLoaded: func(pkg *pkg, exports []stdlib.Symbol) {
			// If we're an x_test, load the package under test's test variant.
			if strings.HasSuffix(filePkg, "_test") && pkg.dir == filepath.Dir(filename) {
				var err error
				_, exports, err = loadExportsFromFiles(ctx, env, pkg.dir, true)
				if err != nil {
					return
				}
			}
			wrappedCallback.exportsLoaded(pkg, exports)
		},
	}
	resolver, err := env.GetResolver()
	if err != nil {
		return err
	}
	return resolver.scan(ctx, scanFilter)
}

func ScoreImportPaths(ctx context.Context, env *ProcessEnv, paths []string) (map[string]float64, error) {
	result := make(map[string]float64)
	resolver, err := env.GetResolver()
	if err != nil {
		return nil, err
	}
	for _, path := range paths {
		result[path] = resolver.scoreImportPath(ctx, path)
	}
	return result, nil
}

func PrimeCache(ctx context.Context, resolver Resolver) error {
	// Fully scan the disk for directories, but don't actually read any Go files.
	callback := &scanCallback{
		rootFound: func(root gopathwalk.Root) bool {
			// See getCandidatePkgs: walking GOROOT is apparently expensive and
			// unnecessary.
			return root.Type != gopathwalk.RootGOROOT
		},
		dirFound: func(pkg *pkg) bool {
			return false
		},
		// packageNameLoaded and exportsLoaded must never be called.
	}

	return resolver.scan(ctx, callback)
}

func candidateImportName(pkg *pkg) string {
	if ImportPathToAssumedName(pkg.importPathShort) != pkg.packageName {
		return pkg.packageName
	}
	return ""
}

// GetAllCandidates calls wrapped for each package whose name starts with
// searchPrefix, and can be imported from filename with the package name filePkg.
//
// Beware that the wrapped function may be called multiple times concurrently.
// TODO(adonovan): encapsulate the concurrency.
func GetAllCandidates(ctx context.Context, wrapped func(ImportFix), searchPrefix, filename, filePkg string, env *ProcessEnv) error {
	callback := &scanCallback{
		rootFound: func(gopathwalk.Root) bool {
			return true
		},
		dirFound: func(pkg *pkg) bool {
			if !canUse(filename, pkg.dir) {
				return false
			}
			// Try the assumed package name first, then a simpler path match
			// in case of packages named vN, which are not uncommon.
			return strings.HasPrefix(ImportPathToAssumedName(pkg.importPathShort), searchPrefix) ||
				strings.HasPrefix(path.Base(pkg.importPathShort), searchPrefix)
		},
		packageNameLoaded: func(pkg *pkg) bool {
			if !strings.HasPrefix(pkg.packageName, searchPrefix) {
				return false
			}
			wrapped(ImportFix{
				StmtInfo: ImportInfo{
					ImportPath: pkg.importPathShort,
					Name:       candidateImportName(pkg),
				},
				IdentName: pkg.packageName,
				FixType:   AddImport,
				Relevance: pkg.relevance,
			})
			return false
		},
	}
	return getCandidatePkgs(ctx, callback, filename, filePkg, env)
}

// GetImportPaths calls wrapped for each package whose import path starts with
// searchPrefix, and can be imported from filename with the package name filePkg.
func GetImportPaths(ctx context.Context, wrapped func(ImportFix), searchPrefix, filename, filePkg string, env *ProcessEnv) error {
	callback := &scanCallback{
		rootFound: func(gopathwalk.Root) bool {
			return true
		},
		dirFound: func(pkg *pkg) bool {
			if !canUse(filename, pkg.dir) {
				return false
			}
			return strings.HasPrefix(pkg.importPathShort, searchPrefix)
		},
		packageNameLoaded: func(pkg *pkg) bool {
			wrapped(ImportFix{
				StmtInfo: ImportInfo{
					ImportPath: pkg.importPathShort,
					Name:       candidateImportName(pkg),
				},
				IdentName: pkg.packageName,
				FixType:   AddImport,
				Relevance: pkg.relevance,
			})
			return false
		},
	}
	return getCandidatePkgs(ctx, callback, filename, filePkg, env)
}

// A PackageExport is a package and its exports.
type PackageExport struct {
	Fix     *ImportFix
	Exports []stdlib.Symbol
}

// GetPackageExports returns all known packages with name pkg and their exports.
func GetPackageExports(ctx context.Context, wrapped func(PackageExport), searchPkg, filename, filePkg string, env *ProcessEnv) error {
	callback := &scanCallback{
		rootFound: func(gopathwalk.Root) bool {
			return true
		},
		dirFound: func(pkg *pkg) bool {
			return pkgIsCandidate(filename, references{searchPkg: nil}, pkg)
		},
		packageNameLoaded: func(pkg *pkg) bool {
			return pkg.packageName == searchPkg
		},
		exportsLoaded: func(pkg *pkg, exports []stdlib.Symbol) {
			sortSymbols(exports)
			wrapped(PackageExport{
				Fix: &ImportFix{
					StmtInfo: ImportInfo{
						ImportPath: pkg.importPathShort,
						Name:       candidateImportName(pkg),
					},
					IdentName: pkg.packageName,
					FixType:   AddImport,
					Relevance: pkg.relevance,
				},
				Exports: exports,
			})
		},
	}
	return getCandidatePkgs(ctx, callback, filename, filePkg, env)
}

// TODO(rfindley): we should depend on GOOS and GOARCH, to provide accurate
// imports when doing cross-platform development.
var requiredGoEnvVars = []string{
	"GO111MODULE",
	"GOFLAGS",
	"GOINSECURE",
	"GOMOD",
	"GOMODCACHE",
	"GONOPROXY",
	"GONOSUMDB",
	"GOPATH",
	"GOPROXY",
	"GOROOT",
	"GOSUMDB",
	"GOWORK",
}

// ProcessEnv contains environment variables and settings that affect the use of
// the go command, the go/build package, etc.
//
// ...a ProcessEnv *also* overwrites its Env along with derived state in the
// form of the resolver. And because it is lazily initialized, an env may just
// be broken and unusable, but there is no way for the caller to detect that:
// all queries will just fail.
//
// TODO(rfindley): refactor this package so that this type (perhaps renamed to
// just Env or Config) is an immutable configuration struct, to be exchanged
// for an initialized object via a constructor that returns an error. Perhaps
// the signature should be `func NewResolver(*Env) (*Resolver, error)`, where
// resolver is a concrete type used for resolving imports. Via this
// refactoring, we can avoid the need to call ProcessEnv.init and
// ProcessEnv.GoEnv everywhere, and implicitly fix all the places where this
// these are misused. Also, we'd delegate the caller the decision of how to
// handle a broken environment.
type ProcessEnv struct {
	GocmdRunner *gocommand.Runner

	BuildFlags []string
	ModFlag    string

	// SkipPathInScan returns true if the path should be skipped from scans of
	// the RootCurrentModule root type. The function argument is a clean,
	// absolute path.
	SkipPathInScan func(string) bool

	// Env overrides the OS environment, and can be used to specify
	// GOPROXY, GO111MODULE, etc. PATH cannot be set here, because
	// exec.Command will not honor it.
	// Specifying all of requiredGoEnvVars avoids a call to `go env`.
	Env map[string]string

	WorkingDir string

	// If Logf is non-nil, debug logging is enabled through this function.
	Logf func(format string, args ...interface{})

	// If set, ModCache holds a shared cache of directory info to use across
	// multiple ProcessEnvs.
	ModCache *DirInfoCache

	initialized bool // see TODO above

	// resolver and resolverErr are lazily evaluated (see GetResolver).
	// This is unclean, but see the big TODO in the docstring for ProcessEnv
	// above: for now, we can't be sure that the ProcessEnv is fully initialized.
	resolver    Resolver
	resolverErr error
}

func (e *ProcessEnv) goEnv() (map[string]string, error) {
	if err := e.init(); err != nil {
		return nil, err
	}
	return e.Env, nil
}

func (e *ProcessEnv) matchFile(dir, name string) (bool, error) {
	bctx, err := e.buildContext()
	if err != nil {
		return false, err
	}
	return bctx.MatchFile(dir, name)
}

// CopyConfig copies the env's configuration into a new env.
func (e *ProcessEnv) CopyConfig() *ProcessEnv {
	copy := &ProcessEnv{
		GocmdRunner: e.GocmdRunner,
		initialized: e.initialized,
		BuildFlags:  e.BuildFlags,
		Logf:        e.Logf,
		WorkingDir:  e.WorkingDir,
		resolver:    nil,
		Env:         map[string]string{},
	}
	for k, v := range e.Env {
		copy.Env[k] = v
	}
	return copy
}

func (e *ProcessEnv) init() error {
	if e.initialized {
		return nil
	}

	foundAllRequired := true
	for _, k := range requiredGoEnvVars {
		if _, ok := e.Env[k]; !ok {
			foundAllRequired = false
			break
		}
	}
	if foundAllRequired {
		e.initialized = true
		return nil
	}

	if e.Env == nil {
		e.Env = map[string]string{}
	}

	goEnv := map[string]string{}
	stdout, err := e.invokeGo(context.TODO(), "env", append([]string{"-json"}, requiredGoEnvVars...)...)
	if err != nil {
		return err
	}
	if err := json.Unmarshal(stdout.Bytes(), &goEnv); err != nil {
		return err
	}
	for k, v := range goEnv {
		e.Env[k] = v
	}
	e.initialized = true
	return nil
}

func (e *ProcessEnv) env() []string {
	var env []string // the gocommand package will prepend os.Environ.
	for k, v := range e.Env {
		env = append(env, k+"="+v)
	}
	return env
}

func (e *ProcessEnv) GetResolver() (Resolver, error) {
	if err := e.init(); err != nil {
		return nil, err
	}

	if e.resolver == nil && e.resolverErr == nil {
		// TODO(rfindley): we should only use a gopathResolver here if the working
		// directory is actually *in* GOPATH. (I seem to recall an open gopls issue
		// for this behavior, but I can't find it).
		//
		// For gopls, we can optionally explicitly choose a resolver type, since we
		// already know the view type.
		if len(e.Env["GOMOD"]) == 0 && len(e.Env["GOWORK"]) == 0 {
			e.resolver = newGopathResolver(e)
			e.logf("created gopath resolver")
		} else if r, err := newModuleResolver(e, e.ModCache); err != nil {
			e.resolverErr = err
			e.logf("failed to create module resolver: %v", err)
		} else {
			e.resolver = Resolver(r)
			e.logf("created module resolver")
		}
	}

	return e.resolver, e.resolverErr
}

// logf logs if e.Logf is non-nil.
func (e *ProcessEnv) logf(format string, args ...any) {
	if e.Logf != nil {
		e.Logf(format, args...)
	}
}

// buildContext returns the build.Context to use for matching files.
//
// TODO(rfindley): support dynamic GOOS, GOARCH here, when doing cross-platform
// development.
func (e *ProcessEnv) buildContext() (*build.Context, error) {
	ctx := build.Default
	goenv, err := e.goEnv()
	if err != nil {
		return nil, err
	}
	ctx.GOROOT = goenv["GOROOT"]
	ctx.GOPATH = goenv["GOPATH"]

	// As of Go 1.14, build.Context has a Dir field
	// (see golang.org/issue/34860).
	// Populate it only if present.
	rc := reflect.ValueOf(&ctx).Elem()
	dir := rc.FieldByName("Dir")
	if dir.IsValid() && dir.Kind() == reflect.String {
		dir.SetString(e.WorkingDir)
	}

	// Since Go 1.11, go/build.Context.Import may invoke 'go list' depending on
	// the value in GO111MODULE in the process's environment. We always want to
	// run in GOPATH mode when calling Import, so we need to prevent this from
	// happening. In Go 1.16, GO111MODULE defaults to "on", so this problem comes
	// up more frequently.
	//
	// HACK: setting any of the Context I/O hooks prevents Import from invoking
	// 'go list', regardless of GO111MODULE. This is undocumented, but it's
	// unlikely to change before GOPATH support is removed.
	ctx.ReadDir = ioutil.ReadDir

	return &ctx, nil
}

func (e *ProcessEnv) invokeGo(ctx context.Context, verb string, args ...string) (*bytes.Buffer, error) {
	inv := gocommand.Invocation{
		Verb:       verb,
		Args:       args,
		BuildFlags: e.BuildFlags,
		Env:        e.env(),
		Logf:       e.Logf,
		WorkingDir: e.WorkingDir,
	}
	return e.GocmdRunner.Run(ctx, inv)
}

func addStdlibCandidates(pass *pass, refs references) error {
	goenv, err := pass.env.goEnv()
	if err != nil {
		return err
	}
	localbase := func(nm string) string {
		ans := path.Base(nm)
		if ans[0] == 'v' {
			// this is called, for instance, with math/rand/v2 and returns rand/v2
			if _, err := strconv.Atoi(ans[1:]); err == nil {
				ix := strings.LastIndex(nm, ans)
				more := path.Base(nm[:ix])
				ans = path.Join(more, ans)
			}
		}
		return ans
	}
	add := func(pkg string) {
		// Prevent self-imports.
		if path.Base(pkg) == pass.f.Name.Name && filepath.Join(goenv["GOROOT"], "src", pkg) == pass.srcDir {
			return
		}
		exports := symbolNameSet(stdlib.PackageSymbols[pkg])
		pass.addCandidate(
			&ImportInfo{ImportPath: pkg},
			&packageInfo{name: localbase(pkg), exports: exports})
	}
	for left := range refs {
		if left == "rand" {
			// Make sure we try crypto/rand before any version of math/rand as both have Int()
			// and our policy is to recommend crypto
			add("crypto/rand")
			// if the user's no later than go1.21, this should be "math/rand"
			// but we have no way of figuring out what the user is using
			// TODO: investigate using the toolchain version to disambiguate in the stdlib
			add("math/rand/v2")
			continue
		}
		for importPath := range stdlib.PackageSymbols {
			if path.Base(importPath) == left {
				add(importPath)
			}
		}
	}
	return nil
}

// A Resolver does the build-system-specific parts of goimports.
type Resolver interface {
	// loadPackageNames loads the package names in importPaths.
	loadPackageNames(importPaths []string, srcDir string) (map[string]string, error)

	// scan works with callback to search for packages. See scanCallback for details.
	scan(ctx context.Context, callback *scanCallback) error

	// loadExports returns the package name and set of exported symbols in the
	// package at dir. loadExports may be called concurrently.
	loadExports(ctx context.Context, pkg *pkg, includeTest bool) (string, []stdlib.Symbol, error)

	// scoreImportPath returns the relevance for an import path.
	scoreImportPath(ctx context.Context, path string) float64

	// ClearForNewScan returns a new Resolver based on the receiver that has
	// cleared its internal caches of directory contents.
	//
	// The new resolver should be primed and then set via
	// [ProcessEnv.UpdateResolver].
	ClearForNewScan() Resolver
}

// A scanCallback controls a call to scan and receives its results.
// In general, minor errors will be silently discarded; a user should not
// expect to receive a full series of calls for everything.
type scanCallback struct {
	// rootFound is called before scanning a new root dir. If it returns true,
	// the root will be scanned. Returning false will not necessarily prevent
	// directories from that root making it to dirFound.
	rootFound func(gopathwalk.Root) bool
	// dirFound is called when a directory is found that is possibly a Go package.
	// pkg will be populated with everything except packageName.
	// If it returns true, the package's name will be loaded.
	dirFound func(pkg *pkg) bool
	// packageNameLoaded is called when a package is found and its name is loaded.
	// If it returns true, the package's exports will be loaded.
	packageNameLoaded func(pkg *pkg) bool
	// exportsLoaded is called when a package's exports have been loaded.
	exportsLoaded func(pkg *pkg, exports []stdlib.Symbol)
}

func addExternalCandidates(ctx context.Context, pass *pass, refs references, filename string) error {
	ctx, done := event.Start(ctx, "imports.addExternalCandidates")
	defer done()

	var mu sync.Mutex
	found := make(map[string][]pkgDistance)
	callback := &scanCallback{
		rootFound: func(gopathwalk.Root) bool {
			return true // We want everything.
		},
		dirFound: func(pkg *pkg) bool {
			return pkgIsCandidate(filename, refs, pkg)
		},
		packageNameLoaded: func(pkg *pkg) bool {
			if _, want := refs[pkg.packageName]; !want {
				return false
			}
			if pkg.dir == pass.srcDir && pass.f.Name.Name == pkg.packageName {
				// The candidate is in the same directory and has the
				// same package name. Don't try to import ourselves.
				return false
			}
			if !canUse(filename, pkg.dir) {
				return false
			}
			mu.Lock()
			defer mu.Unlock()
			found[pkg.packageName] = append(found[pkg.packageName], pkgDistance{pkg, distance(pass.srcDir, pkg.dir)})
			return false // We'll do our own loading after we sort.
		},
	}
	resolver, err := pass.env.GetResolver()
	if err != nil {
		return err
	}
	if err = resolver.scan(ctx, callback); err != nil {
		return err
	}

	// Search for imports matching potential package references.
	type result struct {
		imp *ImportInfo
		pkg *packageInfo
	}
	results := make([]*result, len(refs))

	g, ctx := errgroup.WithContext(ctx)

	searcher := symbolSearcher{
		logf:        pass.env.logf,
		srcDir:      pass.srcDir,
		xtest:       strings.HasSuffix(pass.f.Name.Name, "_test"),
		loadExports: resolver.loadExports,
	}

	i := 0
	for pkgName, symbols := range refs {
		index := i // claim an index in results
		i++
		pkgName := pkgName
		symbols := symbols

		g.Go(func() error {
			found, err := searcher.search(ctx, found[pkgName], pkgName, symbols)
			if err != nil {
				return err
			}
			if found == nil {
				return nil // No matching package.
			}

			imp := &ImportInfo{
				ImportPath: found.importPathShort,
			}
			pkg := &packageInfo{
				name:    pkgName,
				exports: symbols,
			}
			results[index] = &result{imp, pkg}
			return nil
		})
	}
	if err := g.Wait(); err != nil {
		return err
	}

	for _, result := range results {
		if result == nil {
			continue
		}
		// Don't offer completions that would shadow predeclared
		// names, such as github.com/coreos/etcd/error.
		if types.Universe.Lookup(result.pkg.name) != nil { // predeclared
			// Ideally we would skip this candidate only
			// if the predeclared name is actually
			// referenced by the file, but that's a lot
			// trickier to compute and would still create
			// an import that is likely to surprise the
			// user before long.
			continue
		}
		pass.addCandidate(result.imp, result.pkg)
	}
	return nil
}

// notIdentifier reports whether ch is an invalid identifier character.
func notIdentifier(ch rune) bool {
	return !('a' <= ch && ch <= 'z' || 'A' <= ch && ch <= 'Z' ||
		'0' <= ch && ch <= '9' ||
		ch == '_' ||
		ch >= utf8.RuneSelf && (unicode.IsLetter(ch) || unicode.IsDigit(ch)))
}

// ImportPathToAssumedName returns the assumed package name of an import path.
// It does this using only string parsing of the import path.
// It picks the last element of the path that does not look like a major
// version, and then picks the valid identifier off the start of that element.
// It is used to determine if a local rename should be added to an import for
// clarity.
// This function could be moved to a standard package and exported if we want
// for use in other tools.
func ImportPathToAssumedName(importPath string) string {
	base := path.Base(importPath)
	if strings.HasPrefix(base, "v") {
		if _, err := strconv.Atoi(base[1:]); err == nil {
			dir := path.Dir(importPath)
			if dir != "." {
				base = path.Base(dir)
			}
		}
	}
	base = strings.TrimPrefix(base, "go-")
	if i := strings.IndexFunc(base, notIdentifier); i >= 0 {
		base = base[:i]
	}
	return base
}

// gopathResolver implements resolver for GOPATH workspaces.
type gopathResolver struct {
	env      *ProcessEnv
	walked   bool
	cache    *DirInfoCache
	scanSema chan struct{} // scanSema prevents concurrent scans.
}

func newGopathResolver(env *ProcessEnv) *gopathResolver {
	r := &gopathResolver{
		env:      env,
		cache:    NewDirInfoCache(),
		scanSema: make(chan struct{}, 1),
	}
	r.scanSema <- struct{}{}
	return r
}

func (r *gopathResolver) ClearForNewScan() Resolver {
	return newGopathResolver(r.env)
}

func (r *gopathResolver) loadPackageNames(importPaths []string, srcDir string) (map[string]string, error) {
	names := map[string]string{}
	bctx, err := r.env.buildContext()
	if err != nil {
		return nil, err
	}
	for _, path := range importPaths {
		names[path] = importPathToName(bctx, path, srcDir)
	}
	return names, nil
}

// importPathToName finds out the actual package name, as declared in its .go files.
func importPathToName(bctx *build.Context, importPath, srcDir string) string {
	// Fast path for standard library without going to disk.
	if stdlib.HasPackage(importPath) {
		return path.Base(importPath) // stdlib packages always match their paths.
	}

	buildPkg, err := bctx.Import(importPath, srcDir, build.FindOnly)
	if err != nil {
		return ""
	}
	pkgName, err := packageDirToName(buildPkg.Dir)
	if err != nil {
		return ""
	}
	return pkgName
}

// packageDirToName is a faster version of build.Import if
// the only thing desired is the package name. Given a directory,
// packageDirToName then only parses one file in the package,
// trusting that the files in the directory are consistent.
func packageDirToName(dir string) (packageName string, err error) {
	d, err := os.Open(dir)
	if err != nil {
		return "", err
	}
	names, err := d.Readdirnames(-1)
	d.Close()
	if err != nil {
		return "", err
	}
	sort.Strings(names) // to have predictable behavior
	var lastErr error
	var nfile int
	for _, name := range names {
		if !strings.HasSuffix(name, ".go") {
			continue
		}
		if strings.HasSuffix(name, "_test.go") {
			continue
		}
		nfile++
		fullFile := filepath.Join(dir, name)

		fset := token.NewFileSet()
		f, err := parser.ParseFile(fset, fullFile, nil, parser.PackageClauseOnly)
		if err != nil {
			lastErr = err
			continue
		}
		pkgName := f.Name.Name
		if pkgName == "documentation" {
			// Special case from go/build.ImportDir, not
			// handled by ctx.MatchFile.
			continue
		}
		if pkgName == "main" {
			// Also skip package main, assuming it's a +build ignore generator or example.
			// Since you can't import a package main anyway, there's no harm here.
			continue
		}
		return pkgName, nil
	}
	if lastErr != nil {
		return "", lastErr
	}
	return "", fmt.Errorf("no importable package found in %d Go files", nfile)
}

type pkg struct {
	dir             string  // absolute file path to pkg directory ("/usr/lib/go/src/net/http")
	importPathShort string  // vendorless import path ("net/http", "a/b")
	packageName     string  // package name loaded from source if requested
	relevance       float64 // a weakly-defined score of how relevant a package is. 0 is most relevant.
}

type pkgDistance struct {
	pkg      *pkg
	distance int // relative distance to target
}

// byDistanceOrImportPathShortLength sorts by relative distance breaking ties
// on the short import path length and then the import string itself.
type byDistanceOrImportPathShortLength []pkgDistance

func (s byDistanceOrImportPathShortLength) Len() int { return len(s) }
func (s byDistanceOrImportPathShortLength) Less(i, j int) bool {
	di, dj := s[i].distance, s[j].distance
	if di == -1 {
		return false
	}
	if dj == -1 {
		return true
	}
	if di != dj {
		return di < dj
	}

	vi, vj := s[i].pkg.importPathShort, s[j].pkg.importPathShort
	if len(vi) != len(vj) {
		return len(vi) < len(vj)
	}
	return vi < vj
}
func (s byDistanceOrImportPathShortLength) Swap(i, j int) { s[i], s[j] = s[j], s[i] }

func distance(basepath, targetpath string) int {
	p, err := filepath.Rel(basepath, targetpath)
	if err != nil {
		return -1
	}
	if p == "." {
		return 0
	}
	return strings.Count(p, string(filepath.Separator)) + 1
}

func (r *gopathResolver) scan(ctx context.Context, callback *scanCallback) error {
	add := func(root gopathwalk.Root, dir string) {
		// We assume cached directories have not changed. We can skip them and their
		// children.
		if _, ok := r.cache.Load(dir); ok {
			return
		}

		importpath := filepath.ToSlash(dir[len(root.Path)+len("/"):])
		info := directoryPackageInfo{
			status:                 directoryScanned,
			dir:                    dir,
			rootType:               root.Type,
			nonCanonicalImportPath: VendorlessPath(importpath),
		}
		r.cache.Store(dir, info)
	}
	processDir := func(info directoryPackageInfo) {
		// Skip this directory if we were not able to get the package information successfully.
		if scanned, err := info.reachedStatus(directoryScanned); !scanned || err != nil {
			return
		}

		p := &pkg{
			importPathShort: info.nonCanonicalImportPath,
			dir:             info.dir,
			relevance:       MaxRelevance - 1,
		}
		if info.rootType == gopathwalk.RootGOROOT {
			p.relevance = MaxRelevance
		}

		if !callback.dirFound(p) {
			return
		}
		var err error
		p.packageName, err = r.cache.CachePackageName(info)
		if err != nil {
			return
		}

		if !callback.packageNameLoaded(p) {
			return
		}
		if _, exports, err := r.loadExports(ctx, p, false); err == nil {
			callback.exportsLoaded(p, exports)
		}
	}
	stop := r.cache.ScanAndListen(ctx, processDir)
	defer stop()

	goenv, err := r.env.goEnv()
	if err != nil {
		return err
	}
	var roots []gopathwalk.Root
	roots = append(roots, gopathwalk.Root{Path: filepath.Join(goenv["GOROOT"], "src"), Type: gopathwalk.RootGOROOT})
	for _, p := range filepath.SplitList(goenv["GOPATH"]) {
		roots = append(roots, gopathwalk.Root{Path: filepath.Join(p, "src"), Type: gopathwalk.RootGOPATH})
	}
	// The callback is not necessarily safe to use in the goroutine below. Process roots eagerly.
	roots = filterRoots(roots, callback.rootFound)
	// We can't cancel walks, because we need them to finish to have a usable
	// cache. Instead, run them in a separate goroutine and detach.
	scanDone := make(chan struct{})
	go func() {
		select {
		case <-ctx.Done():
			return
		case <-r.scanSema:
		}
		defer func() { r.scanSema <- struct{}{} }()
		gopathwalk.Walk(roots, add, gopathwalk.Options{Logf: r.env.Logf, ModulesEnabled: false})
		close(scanDone)
	}()
	select {
	case <-ctx.Done():
	case <-scanDone:
	}
	return nil
}

func (r *gopathResolver) scoreImportPath(ctx context.Context, path string) float64 {
	if stdlib.HasPackage(path) {
		return MaxRelevance
	}
	return MaxRelevance - 1
}

func filterRoots(roots []gopathwalk.Root, include func(gopathwalk.Root) bool) []gopathwalk.Root {
	var result []gopathwalk.Root
	for _, root := range roots {
		if !include(root) {
			continue
		}
		result = append(result, root)
	}
	return result
}

func (r *gopathResolver) loadExports(ctx context.Context, pkg *pkg, includeTest bool) (string, []stdlib.Symbol, error) {
	if info, ok := r.cache.Load(pkg.dir); ok && !includeTest {
		return r.cache.CacheExports(ctx, r.env, info)
	}
	return loadExportsFromFiles(ctx, r.env, pkg.dir, includeTest)
}

// VendorlessPath returns the devendorized version of the import path ipath.
// For example, VendorlessPath("foo/bar/vendor/a/b") returns "a/b".
func VendorlessPath(ipath string) string {
	// Devendorize for use in import statement.
	if i := strings.LastIndex(ipath, "/vendor/"); i >= 0 {
		return ipath[i+len("/vendor/"):]
	}
	if strings.HasPrefix(ipath, "vendor/") {
		return ipath[len("vendor/"):]
	}
	return ipath
}

func loadExportsFromFiles(ctx context.Context, env *ProcessEnv, dir string, includeTest bool) (string, []stdlib.Symbol, error) {
	// Look for non-test, buildable .go files which could provide exports.
	all, err := os.ReadDir(dir)
	if err != nil {
		return "", nil, err
	}
	var files []fs.DirEntry
	for _, fi := range all {
		name := fi.Name()
		if !strings.HasSuffix(name, ".go") || (!includeTest && strings.HasSuffix(name, "_test.go")) {
			continue
		}
		match, err := env.matchFile(dir, fi.Name())
		if err != nil || !match {
			continue
		}
		files = append(files, fi)
	}

	if len(files) == 0 {
		return "", nil, fmt.Errorf("dir %v contains no buildable, non-test .go files", dir)
	}

	var pkgName string
	var exports []stdlib.Symbol
	fset := token.NewFileSet()
	for _, fi := range files {
		select {
		case <-ctx.Done():
			return "", nil, ctx.Err()
		default:
		}

		fullFile := filepath.Join(dir, fi.Name())
		// Legacy ast.Object resolution is needed here.
		f, err := parser.ParseFile(fset, fullFile, nil, 0)
		if err != nil {
			env.logf("error parsing %v: %v", fullFile, err)
			continue
		}
		if f.Name.Name == "documentation" {
			// Special case from go/build.ImportDir, not
			// handled by MatchFile above.
			continue
		}
		if includeTest && strings.HasSuffix(f.Name.Name, "_test") {
			// x_test package. We want internal test files only.
			continue
		}
		pkgName = f.Name.Name
		for name, obj := range f.Scope.Objects {
			if ast.IsExported(name) {
				var kind stdlib.Kind
				switch obj.Kind {
				case ast.Con:
					kind = stdlib.Const
				case ast.Typ:
					kind = stdlib.Type
				case ast.Var:
					kind = stdlib.Var
				case ast.Fun:
					kind = stdlib.Func
				}
				exports = append(exports, stdlib.Symbol{
					Name:    name,
					Kind:    kind,
					Version: 0, // unknown; be permissive
				})
			}
		}
	}
	sortSymbols(exports)

	env.logf("loaded exports in dir %v (package %v): %v", dir, pkgName, exports)
	return pkgName, exports, nil
}

func sortSymbols(syms []stdlib.Symbol) {
	sort.Slice(syms, func(i, j int) bool {
		return syms[i].Name < syms[j].Name
	})
}

// A symbolSearcher searches for a package with a set of symbols, among a set
// of candidates. See [symbolSearcher.search].
//
// The search occurs within the scope of a single file, with context captured
// in srcDir and xtest.
type symbolSearcher struct {
	logf        func(string, ...any)
	srcDir      string // directory containing the file
	xtest       bool   // if set, the file containing is an x_test file
	loadExports func(ctx context.Context, pkg *pkg, includeTest bool) (string, []stdlib.Symbol, error)
}

// search searches the provided candidates for a package containing all
// exported symbols.
//
// If successful, returns the resulting package.
func (s *symbolSearcher) search(ctx context.Context, candidates []pkgDistance, pkgName string, symbols map[string]bool) (*pkg, error) {
	// Sort the candidates by their import package length,
	// assuming that shorter package names are better than long
	// ones.  Note that this sorts by the de-vendored name, so
	// there's no "penalty" for vendoring.
	sort.Sort(byDistanceOrImportPathShortLength(candidates))
	if s.logf != nil {
		for i, c := range candidates {
			s.logf("%s candidate %d/%d: %v in %v", pkgName, i+1, len(candidates), c.pkg.importPathShort, c.pkg.dir)
		}
	}

	// Arrange rescv so that we can we can await results in order of relevance
	// and exit as soon as we find the first match.
	//
	// Search with bounded concurrency, returning as soon as the first result
	// among rescv is non-nil.
	rescv := make([]chan *pkg, len(candidates))
	for i := range candidates {
		rescv[i] = make(chan *pkg, 1)
	}
	const maxConcurrentPackageImport = 4
	loadExportsSem := make(chan struct{}, maxConcurrentPackageImport)

	// Ensure that all work is completed at exit.
	ctx, cancel := context.WithCancel(ctx)
	var wg sync.WaitGroup
	defer func() {
		cancel()
		wg.Wait()
	}()

	// Start the search.
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i, c := range candidates {
			select {
			case loadExportsSem <- struct{}{}:
			case <-ctx.Done():
				return
			}

			i := i
			c := c
			wg.Add(1)
			go func() {
				defer func() {
					<-loadExportsSem
					wg.Done()
				}()
				if s.logf != nil {
					s.logf("loading exports in dir %s (seeking package %s)", c.pkg.dir, pkgName)
				}
				pkg, err := s.searchOne(ctx, c, symbols)
				if err != nil {
					if s.logf != nil && ctx.Err() == nil {
						s.logf("loading exports in dir %s (seeking package %s): %v", c.pkg.dir, pkgName, err)
					}
					pkg = nil
				}
				rescv[i] <- pkg // may be nil
			}()
		}
	}()

	// Await the first (best) result.
	for _, resc := range rescv {
		select {
		case r := <-resc:
			if r != nil {
				return r, nil
			}
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	return nil, nil
}

func (s *symbolSearcher) searchOne(ctx context.Context, c pkgDistance, symbols map[string]bool) (*pkg, error) {
	if ctx.Err() != nil {
		return nil, ctx.Err()
	}
	// If we're considering the package under test from an x_test, load the
	// test variant.
	includeTest := s.xtest && c.pkg.dir == s.srcDir
	_, exports, err := s.loadExports(ctx, c.pkg, includeTest)
	if err != nil {
		return nil, err
	}

	exportsMap := make(map[string]bool, len(exports))
	for _, sym := range exports {
		exportsMap[sym.Name] = true
	}
	for symbol := range symbols {
		if !exportsMap[symbol] {
			return nil, nil // no match
		}
	}
	return c.pkg, nil
}

// pkgIsCandidate reports whether pkg is a candidate for satisfying the
// finding which package pkgIdent in the file named by filename is trying
// to refer to.
//
// This check is purely lexical and is meant to be as fast as possible
// because it's run over all $GOPATH directories to filter out poor
// candidates in order to limit the CPU and I/O later parsing the
// exports in candidate packages.
//
// filename is the file being formatted.
// pkgIdent is the package being searched for, like "client" (if
// searching for "client.New")
func pkgIsCandidate(filename string, refs references, pkg *pkg) bool {
	// Check "internal" and "vendor" visibility:
	if !canUse(filename, pkg.dir) {
		return false
	}

	// Speed optimization to minimize disk I/O:
	//
	// Use the matchesPath heuristic to filter to package paths that could
	// reasonably match a dangling reference.
	//
	// This permits mismatch naming like directory "go-foo" being package "foo",
	// or "pkg.v3" being "pkg", or directory
	// "google.golang.org/api/cloudbilling/v1" being package "cloudbilling", but
	// doesn't permit a directory "foo" to be package "bar", which is strongly
	// discouraged anyway. There's no reason goimports needs to be slow just to
	// accommodate that.
	for pkgIdent := range refs {
		if matchesPath(pkgIdent, pkg.importPathShort) {
			return true
		}
	}
	return false
}

// canUse reports whether the package in dir is usable from filename,
// respecting the Go "internal" and "vendor" visibility rules.
func canUse(filename, dir string) bool {
	// Fast path check, before any allocations. If it doesn't contain vendor
	// or internal, it's not tricky:
	// Note that this can false-negative on directories like "notinternal",
	// but we check it correctly below. This is just a fast path.
	if !strings.Contains(dir, "vendor") && !strings.Contains(dir, "internal") {
		return true
	}

	dirSlash := filepath.ToSlash(dir)
	if !strings.Contains(dirSlash, "/vendor/") && !strings.Contains(dirSlash, "/internal/") && !strings.HasSuffix(dirSlash, "/internal") {
		return true
	}
	// Vendor or internal directory only visible from children of parent.
	// That means the path from the current directory to the target directory
	// can contain ../vendor or ../internal but not ../foo/vendor or ../foo/internal
	// or bar/vendor or bar/internal.
	// After stripping all the leading ../, the only okay place to see vendor or internal
	// is at the very beginning of the path.
	absfile, err := filepath.Abs(filename)
	if err != nil {
		return false
	}
	absdir, err := filepath.Abs(dir)
	if err != nil {
		return false
	}
	rel, err := filepath.Rel(absfile, absdir)
	if err != nil {
		return false
	}
	relSlash := filepath.ToSlash(rel)
	if i := strings.LastIndex(relSlash, "../"); i >= 0 {
		relSlash = relSlash[i+len("../"):]
	}
	return !strings.Contains(relSlash, "/vendor/") && !strings.Contains(relSlash, "/internal/") && !strings.HasSuffix(relSlash, "/internal")
}

// matchesPath reports whether ident may match a potential package name
// referred to by path, using heuristics to filter out unidiomatic package
// names.
//
// Specifically, it checks whether either of the last two '/'- or '\'-delimited
// path segments matches the identifier. The segment-matching heuristic must
// allow for various conventions around segment naming, including go-foo,
// foo-go, and foo.v3. To handle all of these, matching considers both (1) the
// entire segment, ignoring '-' and '.', as well as (2) the last subsegment
// separated by '-' or '.'. So the segment foo-go matches all of the following
// identifiers: foo, go, and foogo. All matches are case insensitive (for ASCII
// identifiers).
//
// See the docstring for [pkgIsCandidate] for an explanation of how this
// heuristic filters potential candidate packages.
func matchesPath(ident, path string) bool {
	// Ignore case, for ASCII.
	lowerIfASCII := func(b byte) byte {
		if 'A' <= b && b <= 'Z' {
			return b + ('a' - 'A')
		}
		return b
	}

	// match reports whether path[start:end] matches ident, ignoring [.-].
	match := func(start, end int) bool {
		ii := len(ident) - 1 // current byte in ident
		pi := end - 1        // current byte in path
		for ; pi >= start && ii >= 0; pi-- {
			pb := path[pi]
			if pb == '-' || pb == '.' {
				continue
			}
			pb = lowerIfASCII(pb)
			ib := lowerIfASCII(ident[ii])
			if pb != ib {
				return false
			}
			ii--
		}
		return ii < 0 && pi < start // all bytes matched
	}

	// segmentEnd and subsegmentEnd hold the end points of the current segment
	// and subsegment intervals.
	segmentEnd := len(path)
	subsegmentEnd := len(path)

	// Count slashes; we only care about the last two segments.
	nslash := 0

	for i := len(path) - 1; i >= 0; i-- {
		switch b := path[i]; b {
		// TODO(rfindley): we handle backlashes here only because the previous
		// heuristic handled backslashes. This is perhaps overly defensive, but is
		// the result of many lessons regarding Chesterton's fence and the
		// goimports codebase.
		//
		// However, this function is only ever called with something called an
		// 'importPath'. Is it possible that this is a real import path, and
		// therefore we need only consider forward slashes?
		case '/', '\\':
			if match(i+1, segmentEnd) || match(i+1, subsegmentEnd) {
				return true
			}
			nslash++
			if nslash == 2 {
				return false // did not match above
			}
			segmentEnd, subsegmentEnd = i, i // reset
		case '-', '.':
			if match(i+1, subsegmentEnd) {
				return true
			}
			subsegmentEnd = i
		}
	}
	return match(0, segmentEnd) || match(0, subsegmentEnd)
}

type visitFn func(node ast.Node) ast.Visitor

func (fn visitFn) Visit(node ast.Node) ast.Visitor {
	return fn(node)
}

func symbolNameSet(symbols []stdlib.Symbol) map[string]bool {
	names := make(map[string]bool)
	for _, sym := range symbols {
		switch sym.Kind {
		case stdlib.Const, stdlib.Var, stdlib.Type, stdlib.Func:
			names[sym.Name] = true
		}
	}
	return names
}
