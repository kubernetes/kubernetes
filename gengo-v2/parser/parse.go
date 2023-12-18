/*
Copyright 2015 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package parser

import (
	"fmt"
	"go/ast"
	"go/build"
	"go/constant"
	"go/parser"
	"go/token"
	tc "go/types"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"sync"

	"golang.org/x/tools/go/packages"
	"k8s.io/gengo/v2/types"
	"k8s.io/klog/v2"
)

// This clarifies when a pkg path has been canonicalized.
type importPathString string

// Builder lets you add all the go files in all the packages that you care
// about, then constructs the type source data.
type Builder struct {
	// If true, include *_test.go
	IncludeTestFiles bool

	// Map of package ID to definitions.  These keys should be canonical
	// Go pkg names (example.com/foo/bar) and not local paths (./foo/bar).
	// FIXME: evaluate: previously said "This must only be accessed via getLoadedBuildPackage and setLoadedBuildPackage"
	pkgMap map[importPathString]*packages.Package

	// Map of package ID to whether the user requested it or it was from
	// an import.
	userRequested map[importPathString]bool

	// Build tags to set when loading packages.
	buildTags []string

	// Tracks accumulated parsed files, so we don't have to re-parse.
	fset *token.FileSet

	// Provides mutual-exclusion while parsing.
	parseLock sync.Mutex

	////////////////////////
	// everything below this seems to not be needed any more
	////////////////////////

	// All comments from everywhere in every parsed file.
	endLineToCommentGroup map[fileLine]*ast.CommentGroup

	// map of package to list of packages it imports.
	importGraph map[importPathString]map[string]struct{}
}

// parsedFile is for tracking files with name
type parsedFile struct {
	name string
	file *ast.File
}

// key type for finding comments.
type fileLine struct {
	file string
	line int
}

// New constructs a new builder.
func New() *Builder {
	c := build.Default
	if c.GOROOT == "" {
		if p, err := exec.Command("which", "go").CombinedOutput(); err == nil {
			// The returned string will have some/path/bin/go, so remove the last two elements.
			c.GOROOT = filepath.Dir(filepath.Dir(strings.Trim(string(p), "\n")))
		} else {
			klog.Warningf("Warning: $GOROOT not set, and unable to run `which go` to find it: %v\n", err)
		}
	}
	// Force this to off, since we don't properly parse CGo.  All symbols must
	// have non-CGo equivalents.
	c.CgoEnabled = false
	return &Builder{
		pkgMap:        map[importPathString]*packages.Package{},
		userRequested: map[importPathString]bool{},
		fset:          token.NewFileSet(),
		//// Everything else may not be needed
		endLineToCommentGroup: map[fileLine]*ast.CommentGroup{},
		importGraph:           map[importPathString]map[string]struct{}{},
	}
}

// AddBuildTags adds the specified build tags for subsequent package loads.
func (b *Builder) AddBuildTags(tags ...string) {
	b.buildTags = append(b.buildTags, tags...)
}

/*
// FIXME: drop
func (b *Builder) getLoadedBuildPackage(importPath string) (*build.Package, bool) {
	canonicalized := canonicalizeImportPath(importPath)
	if string(canonicalized) != importPath {
		klog.V(5).Infof("getLoadedBuildPackage: %s normalized to %s", importPath, canonicalized)
	}
	buildPkg, ok := b.buildPackages[canonicalized]
	return buildPkg, ok
}

// FIXME: drop
func (b *Builder) setLoadedBuildPackage(importPath string, buildPkg *build.Package) {
	canonicalizedImportPath := canonicalizeImportPath(importPath)
	if string(canonicalizedImportPath) != importPath {
		klog.V(5).Infof("setLoadedBuildPackage: importPath %s normalized to %s", importPath, canonicalizedImportPath)
	}

	canonicalizedBuildPkgImportPath := canonicalizeImportPath(buildPkg.ImportPath)
	if string(canonicalizedBuildPkgImportPath) != buildPkg.ImportPath {
		klog.V(5).Infof("setLoadedBuildPackage: buildPkg.ImportPath %s normalized to %s", buildPkg.ImportPath, canonicalizedBuildPkgImportPath)
	}

	if canonicalizedImportPath != canonicalizedBuildPkgImportPath {
		klog.V(5).Infof("setLoadedBuildPackage: normalized importPath (%s) differs from buildPkg.ImportPath (%s)", canonicalizedImportPath, canonicalizedBuildPkgImportPath)
	}
	b.buildPackages[canonicalizedImportPath] = buildPkg
	b.buildPackages[canonicalizedBuildPkgImportPath] = buildPkg
}
*/

/* FIXME
// AddFileForTest adds a file to the set, without verifying that the provided
// pkg actually exists on disk. The pkg must be of the form "canonical/pkg/path"
// and the path must be the absolute path to the file.  Because this bypasses
// the normal recursive finding of package dependencies (on disk), test should
// sort their test files topologically first, so all deps are resolved by the
// time we need them.
func (b *Builder) AddFileForTest(pkg string, path string, src []byte) error {
	if err := b.addFile(importPathString(pkg), path, src, true); err != nil {
		return err
	}
	if _, err := b.typeCheckPackage(importPathString(pkg), true); err != nil {
		return err
	}
	return nil
}

// addFile adds a file to the set. The pkgPath must be of the form
// "canonical/pkg/path" and the path must be the absolute path to the file. A
// flag indicates whether this file was user-requested or just from following
// the import graph.
func (b *Builder) addFile(pkgPath importPathString, path string, src []byte, userRequested bool) error {
	for _, p := range b.parsed[pkgPath] {
		if path == p.name {
			klog.V(5).Infof("addFile %s %s already parsed, skipping", pkgPath, path)
			return nil
		}
	}
	klog.V(6).Infof("addFile %s %s", pkgPath, path)
	p, err := parser.ParseFile(b.fset, path, src, parser.DeclarationErrors|parser.ParseComments)
	if err != nil {
		return err
	}

	// This is redundant with addDir, but some tests call AddFileForTest, which
	// call into here without calling addDir.
	b.userRequested[pkgPath] = userRequested || b.userRequested[pkgPath]

	b.parsed[pkgPath] = append(b.parsed[pkgPath], parsedFile{path, p})
	for _, c := range p.Comments {
		position := b.fset.Position(c.End())
		b.endLineToCommentGroup[fileLine{position.Filename, position.Line}] = c
	}

	// We have to get the packages from this specific file, in case the
	// user added individual files instead of entire directories.
	if b.importGraph[pkgPath] == nil {
		b.importGraph[pkgPath] = map[string]struct{}{}
	}
	for _, im := range p.Imports {
		importedPath := strings.Trim(im.Path.Value, `"`)
		b.importGraph[pkgPath][importedPath] = struct{}{}
	}
	return nil
}
*/

func (b *Builder) LoadPackages(patterns ...string) error {
	_, err := b.loadPackages(patterns...)
	return err
}

func (b *Builder) loadPackages(patterns ...string) ([]*packages.Package, error) {
	netNewPkgs := make([]string, 0, len(patterns))
	existingPkgs := make([]*packages.Package, 0, len(patterns))
	for _, d := range patterns {
		// This does not properly handle "pkg/..." patterns but there's not
		// much we can do without actually running the load on them.
		// FIXME: save ... results and return them?
		ip := importPathString(d)
		if b.pkgMap[ip] == nil {
			netNewPkgs = append(netNewPkgs, d)
		} else {
			//FIXME: streamline wrt processPkg() - settiing in 2 places is ick
			b.userRequested[ip] = true
			existingPkgs = append(existingPkgs, b.pkgMap[ip])
		}
	}
	if len(netNewPkgs) == 0 {
		return existingPkgs, nil
	}

	//FIXME: are all these "need" right?
	//FIXME: does -i take directories, pkgs, or both?  e.g. sahould I add leading ./ ?
	cfg := packages.Config{
		Mode:       packages.NeedName | packages.NeedFiles | packages.NeedImports | packages.NeedDeps | packages.NeedModule | packages.NeedTypes | packages.NeedTypesSizes | packages.NeedTypesInfo | packages.NeedSyntax,
		Tests:      b.IncludeTestFiles,
		BuildFlags: []string{"-tags", strings.Join(b.buildTags, ",")},
		Fset:       b.fset,
		// ParseFile is called to read and parse each file when preparing a
		// package's type-checked syntax tree.  It must be safe to call
		// ParseFile simultaneously from multiple goroutines.
		ParseFile: func(fset *token.FileSet, filename string, src []byte) (*ast.File, error) {
			const mode = parser.DeclarationErrors | parser.ParseComments
			//FIXME: do I need the parseLock? Check default impl
			p, err := parser.ParseFile(b.fset, filename, src, mode)
			if err != nil {
				return nil, err
			}
			//FIXME: is this the best way?
			for _, c := range p.Comments {
				//FIXME: inside lock?
				position := b.fset.Position(c.End())
				b.parseLock.Lock()
				//FIXME: only do this if it is user-requested
				b.endLineToCommentGroup[fileLine{position.Filename, position.Line}] = c
				b.parseLock.Unlock()
			}
			return p, nil
		},
	}
	pkgs, err := packages.Load(&cfg, netNewPkgs...)
	if err != nil {
		return nil, fmt.Errorf("error loading packages: %w\n", err)
	}

	for _, p := range pkgs {
		b.processPkg(p, true)
		//FIXME: err?
	}
	return pkgs, nil
}

// FIXME: comment
func (b *Builder) processPkg(pkg *packages.Package, userRequested bool) {
	klog.V(5).Infof("processPkg %q", pkg.ID)

	pkgPath := importPathString(pkg.ID)
	b.pkgMap[pkgPath] = pkg
	if userRequested {
		b.userRequested[pkgPath] = true
	}

	if len(pkg.Errors) != 0 {
		if pkg.Errors[0].Kind == packages.ListError {
			//FIXME: do this in load (if at all)
			fmt.Printf("Error(s) in %q:\n", pkg.ID)
			for _, e := range pkg.Errors {
				fmt.Printf("  %s\n", e)
			}
			os.Exit(1)
		}
		//FIXME: return error
		//FIXME: this can flag things like "missing method DeepCopyObject"
		//  which happens if the deepcopy file is removed, but doesn't prevent
		//  codegen from happening to fix the situation.  chicken-egg.  Can I
		//  load without parsing code (or per-generator choose to ignore some
		//  errors)?
	} else {
		/*
		   c := spew.ConfigState{
		           DisableMethods: true,
		           MaxDepth:       2,
		           Indent:         "  ",
		   }
		   c.Dump(*pkg)
		*/
	}
	//FIXME: walk imports and add to pkgMap?

	/*
	   var pkgPath = importPathString(dir)

	   // Get the canonical path if we can.
	   if buildPkg, _ := b.getLoadedBuildPackage(dir); buildPkg != nil {
	           canonicalPackage := canonicalizeImportPath(buildPkg.ImportPath)
	           klog.V(5).Infof("importPackage %s, canonical path is %s", dir, canonicalPackage)
	           pkgPath = canonicalPackage
	   }

	   // If we have not seen this before, process it now.
	   ignoreError := false
	   if _, found := b.parsed[pkgPath]; !found {
	           // Ignore errors in paths that we're importing solely because
	           // they're referenced by other packages.
	           ignoreError = true

	           // Add it.
	           if err := b.addDir(dir, userRequested); err != nil {
	                   if isErrPackageNotFound(err) {
	                           klog.V(6).Info(err)
	                           return nil, nil
	                   }

	                   return nil, err
	           }

	           // Get the canonical path now that it has been added.
	           if buildPkg, _ := b.getLoadedBuildPackage(dir); buildPkg != nil {
	                   canonicalPackage := canonicalizeImportPath(buildPkg.ImportPath)
	                   klog.V(5).Infof("importPackage %s, canonical path is %s", dir, canonicalPackage)
	                   pkgPath = canonicalPackage
	           }
	   }

	   // If it was previously known, just check that the user-requestedness hasn't
	   // changed.
	   b.userRequested[pkgPath] = userRequested || b.userRequested[pkgPath]

	   // Run the type checker.  We may end up doing this to pkgs that are already
	   // done, or are in the queue to be done later, but it will short-circuit,
	   // and we can't miss pkgs that are only depended on.
	   pkg, err := b.typeCheckPackage(pkgPath, !ignoreError)
	   if err != nil {
	           switch {
	           case ignoreError && pkg != nil:
	                   klog.V(4).Infof("type checking encountered some issues in %q, but ignoring.\n", pkgPath)
	           case !ignoreError && pkg != nil:
	                   klog.V(3).Infof("type checking encountered some errors in %q\n", pkgPath)
	                   return nil, err
	           default:
	                   return nil, err
	           }
	   }

	   return pkg, nil
	*/
}

func (b *Builder) LoadPackagesTo(u *types.Universe, patterns ...string) ([]*types.Package, error) {
	pkgs, err := b.loadPackages(patterns...)
	if err != nil {
		return nil, err
	}
	// Collect the results as gengo types.Packages.
	ret := make([]*types.Package, 0, len(pkgs))
	//FIXME: should we just do this in loadPackages?
	for _, pkg := range pkgs {
		// FIXME: name or ID?
		pkgPath := importPathString(pkg.ID)
		if b.userRequested[pkgPath] {
			if err := b.findTypesIn(pkgPath, u); err != nil {
				//FIXME: what to do?
				return nil, err
			}
			ret = append(ret, u.Package(pkg.ID))
		}
	}
	return ret, nil
}

///////////////////////////////

// FindPackages fetches a list of the user-imported packages.
// Note that you need to call b.FindTypes() first.
func (b *Builder) FindPackages() []string {
	//FIXME: just return b.userRequested, sorted?
	// Iterate packages in a predictable order.
	pkgPaths := []string{}
	for k := range b.pkgMap {
		pkgPaths = append(pkgPaths, string(k))
	}
	//FIXME: save this index?
	sort.Strings(pkgPaths)

	result := []string{}
	for _, pkgPath := range pkgPaths {
		if b.userRequested[importPathString(pkgPath)] {
			// Since walkType is recursive, all types that are in packages that
			// were directly mentioned will be included.  We don't need to
			// include all types in all transitive packages, though.
			result = append(result, pkgPath)
		}
	}
	return result
}

// FindTypes finalizes the package imports, and searches through all the
// packages for types.
func (b *Builder) FindTypes() (types.Universe, error) {
	// Take a snapshot of pkgs to iterate, since this will recursively mutate
	// b.parsed. Iterate in a predictable order.
	pkgPaths := []string{}
	for pkgPath := range b.pkgMap {
		pkgPaths = append(pkgPaths, string(pkgPath))
	}
	sort.Strings(pkgPaths)

	u := types.Universe{}
	for _, pkgPath := range pkgPaths {
		if err := b.findTypesIn(importPathString(pkgPath), &u); err != nil {
			return nil, err
		}
	}
	return u, nil
}

// addCommentsToType takes any accumulated comment lines prior to obj and
// attaches them to the type t.
func (b *Builder) addCommentsToType(obj tc.Object, t *types.Type) {
	c1 := b.priorCommentLines(obj.Pos(), 1)
	// c1.Text() is safe if c1 is nil
	t.CommentLines = splitLines(c1.Text())
	if c1 == nil {
		t.SecondClosestCommentLines = splitLines(b.priorCommentLines(obj.Pos(), 2).Text())
	} else {
		t.SecondClosestCommentLines = splitLines(b.priorCommentLines(c1.List[0].Slash, 2).Text())
	}
}

// FIXME: hack - store in Builder
var typesFound = map[importPathString]bool{}

// findTypesIn finalizes the package import and searches through the package
// for types.
func (b *Builder) findTypesIn(pkgPath importPathString, u *types.Universe) error {
	if typesFound[pkgPath] {
		return nil
	}
	//FIXME: wromng om face of errors
	typesFound[pkgPath] = true

	klog.V(5).Infof("findTypesIn %s", pkgPath)
	pkg := b.pkgMap[pkgPath]
	if pkg == nil {
		//FIXME: can this happen?
		return fmt.Errorf("findTypesIn(%s): package is not known", pkgPath)
	}
	if !b.userRequested[pkgPath] {
		// Since walkType is recursive, all types that the
		// packages they asked for depend on will be included.
		// But we don't need to include all types in all
		// *packages* they depend on.
		klog.V(5).Infof("findTypesIn %s: package is not user requested", pkgPath)
		return nil
	}
	if len(pkg.GoFiles) == 0 {
		klog.V(5).Infof("findTypesIn %s: package has no Go files", pkgPath)
		return nil
	}
	absPath, _ := filepath.Split(pkg.GoFiles[0]) //FIXME: no better way?

	// We're keeping this package.  This call will create the record.
	//FIXME: store the pkg instead of these fields
	u.Package(string(pkgPath)).Name = pkg.Name
	u.Package(string(pkgPath)).Path = pkg.ID // FIXME or pkgPath?  see go core vendor weirdness (handled in go2make)
	u.Package(string(pkgPath)).SourcePath = absPath

	for i, f := range pkg.Syntax {
		/*
		   c := spew.ConfigState{
		           DisableMethods: true,
		           MaxDepth:       2,
		           Indent:         "  ",
		   }
		   c.Dump(*f)
		*/

		//FIXME: this is hacky and not obvious how to support - do we need it?
		//FIXME: something like this is safer?  Or index from CompiledGoFiles?
		//   s := pkg.Syntax[0]
		//   pos := s.Package
		//   file := pkg.Fset.File(pos).Name()
		if _, fileName := filepath.Split(pkg.GoFiles[i]); fileName == "doc.go" {
			tp := u.Package(string(pkgPath))
			// findTypesIn might be called multiple times. Clean up tp.Comments
			// to avoid repeatedly fill same comments to it.
			tp.Comments = []string{}
			for i := range f.Comments {
				tp.Comments = append(tp.Comments, splitLines(f.Comments[i].Text())...)
			}
			if f.Doc != nil {
				tp.DocComments = splitLines(f.Doc.Text())
			}
		}
	}

	s := pkg.Types.Scope()
	for _, n := range s.Names() {
		obj := s.Lookup(n)
		tn, ok := obj.(*tc.TypeName)
		if ok {
			t := b.walkType(*u, nil, tn.Type())
			b.addCommentsToType(obj, t)
		}
		tf, ok := obj.(*tc.Func)
		// We only care about functions, not concrete/abstract methods.
		if ok && tf.Type() != nil && tf.Type().(*tc.Signature).Recv() == nil {
			t := b.addFunction(*u, nil, tf)
			b.addCommentsToType(obj, t)
		}
		tv, ok := obj.(*tc.Var)
		if ok && !tv.IsField() {
			t := b.addVariable(*u, nil, tv)
			b.addCommentsToType(obj, t)
		}
		tconst, ok := obj.(*tc.Const)
		if ok {
			t := b.addConstant(*u, nil, tconst)
			b.addCommentsToType(obj, t)
		}
	}

	/* FIXME - need this?
	importedPkgs := []string{}
	for k := range b.importGraph[pkgPath] {
		importedPkgs = append(importedPkgs, string(k))
	}
	sort.Strings(importedPkgs)
	for _, p := range importedPkgs {
		u.AddImports(string(pkgPath), p)
	}
	*/
	return nil
}

// if there's a comment on the line `lines` before pos, return its text, otherwise "".
func (b *Builder) priorCommentLines(pos token.Pos, lines int) *ast.CommentGroup {
	position := b.fset.Position(pos)
	key := fileLine{position.Filename, position.Line - lines}
	return b.endLineToCommentGroup[key]
}

func splitLines(str string) []string {
	return strings.Split(strings.TrimRight(str, "\n"), "\n")
}

func tcFuncNameToName(in string) types.Name {
	name := strings.TrimPrefix(in, "func ")
	nameParts := strings.Split(name, "(")
	return tcNameToName(nameParts[0])
}

func tcVarNameToName(in string) types.Name {
	nameParts := strings.Split(in, " ")
	// nameParts[0] is "var".
	// nameParts[2:] is the type of the variable, we ignore it for now.
	return tcNameToName(nameParts[1])
}

func tcNameToName(in string) types.Name {
	// Detect anonymous type names. (These may have '.' characters because
	// embedded types may have packages, so we detect them specially.)
	if strings.HasPrefix(in, "struct{") ||
		strings.HasPrefix(in, "<-chan") ||
		strings.HasPrefix(in, "chan<-") ||
		strings.HasPrefix(in, "chan ") ||
		strings.HasPrefix(in, "func(") ||
		strings.HasPrefix(in, "func (") ||
		strings.HasPrefix(in, "*") ||
		strings.HasPrefix(in, "map[") ||
		strings.HasPrefix(in, "[") {
		return types.Name{Name: in}
	}

	// Otherwise, if there are '.' characters present, the name has a
	// package path in front.
	nameParts := strings.Split(in, ".")
	name := types.Name{Name: in}
	if n := len(nameParts); n >= 2 {
		// The final "." is the name of the type--previous ones must
		// have been in the package path.
		name.Package, name.Name = strings.Join(nameParts[:n-1], "."), nameParts[n-1]
	}
	return name
}

func (b *Builder) convertSignature(u types.Universe, t *tc.Signature) *types.Signature {
	signature := &types.Signature{}
	for i := 0; i < t.Params().Len(); i++ {
		signature.Parameters = append(signature.Parameters, b.walkType(u, nil, t.Params().At(i).Type()))
		signature.ParameterNames = append(signature.ParameterNames, t.Params().At(i).Name())
	}
	for i := 0; i < t.Results().Len(); i++ {
		signature.Results = append(signature.Results, b.walkType(u, nil, t.Results().At(i).Type()))
		signature.ResultNames = append(signature.ResultNames, t.Results().At(i).Name())
	}
	if r := t.Recv(); r != nil {
		signature.Receiver = b.walkType(u, nil, r.Type())
	}
	signature.Variadic = t.Variadic()
	return signature
}

// walkType adds the type, and any necessary child types.
func (b *Builder) walkType(u types.Universe, useName *types.Name, in tc.Type) *types.Type {
	// Most of the cases are underlying types of the named type.
	name := tcNameToName(in.String())
	if useName != nil {
		name = *useName
	}

	switch t := in.(type) {
	case *tc.Struct:
		out := u.Type(name)
		if out.Kind != types.Unknown {
			return out
		}
		out.Kind = types.Struct
		for i := 0; i < t.NumFields(); i++ {
			f := t.Field(i)
			m := types.Member{
				Name:         f.Name(),
				Embedded:     f.Anonymous(),
				Tags:         t.Tag(i),
				Type:         b.walkType(u, nil, f.Type()),
				CommentLines: splitLines(b.priorCommentLines(f.Pos(), 1).Text()),
			}
			out.Members = append(out.Members, m)
		}
		return out
	case *tc.Map:
		out := u.Type(name)
		if out.Kind != types.Unknown {
			return out
		}
		out.Kind = types.Map
		out.Elem = b.walkType(u, nil, t.Elem())
		out.Key = b.walkType(u, nil, t.Key())
		return out
	case *tc.Pointer:
		out := u.Type(name)
		if out.Kind != types.Unknown {
			return out
		}
		out.Kind = types.Pointer
		out.Elem = b.walkType(u, nil, t.Elem())
		return out
	case *tc.Slice:
		out := u.Type(name)
		if out.Kind != types.Unknown {
			return out
		}
		out.Kind = types.Slice
		out.Elem = b.walkType(u, nil, t.Elem())
		return out
	case *tc.Array:
		out := u.Type(name)
		if out.Kind != types.Unknown {
			return out
		}
		out.Kind = types.Array
		out.Elem = b.walkType(u, nil, t.Elem())
		out.Len = in.(*tc.Array).Len()
		return out
	case *tc.Chan:
		out := u.Type(name)
		if out.Kind != types.Unknown {
			return out
		}
		out.Kind = types.Chan
		out.Elem = b.walkType(u, nil, t.Elem())
		// TODO: need to store direction, otherwise raw type name
		// cannot be properly written.
		return out
	case *tc.Basic:
		out := u.Type(types.Name{
			Package: "",
			Name:    t.Name(),
		})
		if out.Kind != types.Unknown {
			return out
		}
		out.Kind = types.Unsupported
		return out
	case *tc.Signature:
		out := u.Type(name)
		if out.Kind != types.Unknown {
			return out
		}
		out.Kind = types.Func
		out.Signature = b.convertSignature(u, t)
		return out
	case *tc.Interface:
		out := u.Type(name)
		if out.Kind != types.Unknown {
			return out
		}
		out.Kind = types.Interface
		t.Complete()
		for i := 0; i < t.NumMethods(); i++ {
			if out.Methods == nil {
				out.Methods = map[string]*types.Type{}
			}
			method := t.Method(i)
			name := tcNameToName(method.String())
			mt := b.walkType(u, &name, method.Type())
			mt.CommentLines = splitLines(b.priorCommentLines(method.Pos(), 1).Text())
			out.Methods[method.Name()] = mt
		}
		return out
	case *tc.Named:
		var out *types.Type
		switch t.Underlying().(type) {
		case *tc.Named, *tc.Basic, *tc.Map, *tc.Slice:
			name := tcNameToName(t.String())
			out = u.Type(name)
			if out.Kind != types.Unknown {
				return out
			}
			out.Kind = types.Alias
			out.Underlying = b.walkType(u, nil, t.Underlying())
		default:
			// tc package makes everything "named" with an
			// underlying anonymous type--we remove that annoying
			// "feature" for users. This flattens those types
			// together.
			name := tcNameToName(t.String())
			if out := u.Type(name); out.Kind != types.Unknown {
				return out // short circuit if we've already made this.
			}
			out = b.walkType(u, &name, t.Underlying())
		}
		// If the underlying type didn't already add methods, add them.
		// (Interface types will have already added methods.)
		if len(out.Methods) == 0 {
			for i := 0; i < t.NumMethods(); i++ {
				if out.Methods == nil {
					out.Methods = map[string]*types.Type{}
				}
				method := t.Method(i)
				name := tcNameToName(method.String())
				mt := b.walkType(u, &name, method.Type())
				mt.CommentLines = splitLines(b.priorCommentLines(method.Pos(), 1).Text())
				out.Methods[method.Name()] = mt
			}
		}
		return out
	default:
		out := u.Type(name)
		if out.Kind != types.Unknown {
			return out
		}
		out.Kind = types.Unsupported
		klog.Warningf("Making unsupported type entry %q for: %#v\n", out, t)
		return out
	}
}

func (b *Builder) addFunction(u types.Universe, useName *types.Name, in *tc.Func) *types.Type {
	name := tcFuncNameToName(in.String())
	if useName != nil {
		name = *useName
	}
	out := u.Function(name)
	out.Kind = types.DeclarationOf
	out.Underlying = b.walkType(u, nil, in.Type())
	return out
}

func (b *Builder) addVariable(u types.Universe, useName *types.Name, in *tc.Var) *types.Type {
	name := tcVarNameToName(in.String())
	if useName != nil {
		name = *useName
	}
	out := u.Variable(name)
	out.Kind = types.DeclarationOf
	out.Underlying = b.walkType(u, nil, in.Type())
	return out
}

func (b *Builder) addConstant(u types.Universe, useName *types.Name, in *tc.Const) *types.Type {
	name := tcVarNameToName(in.String())
	if useName != nil {
		name = *useName
	}
	out := u.Constant(name)
	out.Kind = types.DeclarationOf
	out.Underlying = b.walkType(u, nil, in.Type())

	var constval string

	// For strings, we use `StringVal()` to get the un-truncated,
	// un-quoted string. For other values, `.String()` is preferable to
	// get something relatively human readable (especially since for
	// floating point types, `ExactString()` will generate numeric
	// expressions using `big.(*Float).Text()`.
	switch in.Val().Kind() {
	case constant.String:
		constval = constant.StringVal(in.Val())
	default:
		constval = in.Val().String()
	}

	out.ConstValue = &constval
	return out
}

// FIXME: drop
// canonicalizeImportPath takes an import path and returns the actual package.
// It doesn't support nested vendoring.
func canonicalizeImportPath(importPath string) importPathString {
	if !strings.Contains(importPath, "/vendor/") {
		return importPathString(importPath)
	}

	return importPathString(importPath[strings.Index(importPath, "/vendor/")+len("/vendor/"):])
}
