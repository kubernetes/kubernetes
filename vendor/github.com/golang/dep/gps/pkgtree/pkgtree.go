// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pkgtree

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/build"
	"go/parser"
	gscan "go/scanner"
	"go/token"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"unicode"
)

// Package represents a Go package. It contains a subset of the information
// go/build.Package does.
type Package struct {
	Name        string   // Package name, as declared in the package statement
	ImportPath  string   // Full import path, including the prefix provided to ListPackages()
	CommentPath string   // Import path given in the comment on the package statement
	Imports     []string // Imports from all go and cgo files
	TestImports []string // Imports from all go test files (in go/build parlance: both TestImports and XTestImports)
}

// vcsRoots is a set of directories we should not descend into in ListPackages when
// searching for Go packages
var vcsRoots = map[string]struct{}{
	".git": {},
	".bzr": {},
	".svn": {},
	".hg":  {},
}

// Strategy describes a build-tag strategy.
type Strategy struct {
	uninterestingTags []string
}

var defaultStrategySingleton = Strategy{
	uninterestingTags: []string{"ignore"},
}

// DefaultStrategy returns the default build-tag strategy.
func DefaultStrategy() *Strategy {
	return &defaultStrategySingleton
}

// AddUninterestingTag teaches a strategy to disregard a particular tag.
func (s *Strategy) AddUninterestingTag(tag string) {
	s.uninterestingTags = append(s.uninterestingTags, tag)
}

func (s *Strategy) isInterestingTag(tag string) bool {
	for _, t := range s.uninterestingTags {
		if t == tag {
			return false
		}
	}
	return true
}

// ListPackages reports Go package information about all directories in the tree
// at or below the provided fileRoot.
//
// The importRoot parameter is prepended to the relative path when determining
// the import path for each package. The obvious case is for something typical,
// like:
//
//  fileRoot = "/home/user/go/src/github.com/foo/bar"
//  importRoot = "github.com/foo/bar"
//
// where the fileRoot and importRoot align. However, if you provide:
//
//  fileRoot = "/home/user/workspace/path/to/repo"
//  importRoot = "github.com/foo/bar"
//
// then the root package at path/to/repo will be ascribed import path
// "github.com/foo/bar", and the package at
// "/home/user/workspace/path/to/repo/baz" will be "github.com/foo/bar/baz".
//
// A PackageTree is returned, which contains the ImportRoot and map of import path
// to PackageOrErr - each path under the root that exists will have either a
// Package, or an error describing why the directory is not a valid package.
func ListPackages(fileRoot, importRoot string) (PackageTree, error) {
	ptree := PackageTree{
		ImportRoot: importRoot,
		Packages:   make(map[string]PackageOrErr),
	}

	var err error
	fileRoot, err = filepath.Abs(fileRoot)
	if err != nil {
		return PackageTree{}, err
	}

	err = filepath.Walk(fileRoot, func(wp string, fi os.FileInfo, err error) error {
		if err != nil && err != filepath.SkipDir {
			if os.IsPermission(err) {
				return filepath.SkipDir
			}
			return err
		}
		if !fi.IsDir() {
			return nil
		}

		// Skip dirs that are known to hold non-local/dependency code.
		//
		// We don't skip _*, or testdata dirs because, while it may be poor
		// form, importing them is not a compilation error.
		switch fi.Name() {
		case "vendor", "Godeps":
			return filepath.SkipDir
		}

		// Skip dirs that are known to be VCS roots.
		//
		// Note that there are some pathological edge cases this doesn't cover,
		// such as a user using Git for version control, but having a package
		// named "svn" in a directory named ".svn".
		if _, ok := vcsRoots[fi.Name()]; ok {
			return filepath.SkipDir
		}

		{
			// For Go 1.9 and earlier:
			//
			// The entry error is nil when visiting a directory that itself is
			// untraversable, as it's still governed by the parent directory's
			// perms. We have to check readability of the dir here, because
			// otherwise we'll have an empty package entry when we fail to read any
			// of the dir's contents.
			//
			// If we didn't check here, then the next time this closure is called it
			// would have an err with the same path as is called this time, as only
			// then will filepath.Walk have attempted to descend into the directory
			// and encountered an error.
			var f *os.File
			f, err = os.Open(wp)
			if err != nil {
				if os.IsPermission(err) {
					return filepath.SkipDir
				}
				return err
			}
			f.Close()
		}

		// Compute the import path. Run the result through ToSlash(), so that
		// windows file paths are normalized to slashes, as is expected of
		// import paths.
		ip := filepath.ToSlash(filepath.Join(importRoot, strings.TrimPrefix(wp, fileRoot)))

		// Find all the imports, across all os/arch combos
		p := &build.Package{
			Dir:        wp,
			ImportPath: ip,
		}
		err = DefaultStrategy().fillPackage(p)

		if err != nil {
			switch err.(type) {
			case gscan.ErrorList, *gscan.Error, *build.NoGoError, *ConflictingImportComments:
				// Assorted cases in which we've encounter malformed or
				// nonexistent Go source code.
				ptree.Packages[ip] = PackageOrErr{
					Err: err,
				}
				return nil
			default:
				return err
			}
		}

		pkg := Package{
			ImportPath:  ip,
			CommentPath: p.ImportComment,
			Name:        p.Name,
			Imports:     p.Imports,
			TestImports: dedupeStrings(p.TestImports, p.XTestImports),
		}

		if pkg.CommentPath != "" && !strings.HasPrefix(pkg.CommentPath, importRoot) {
			ptree.Packages[ip] = PackageOrErr{
				Err: &NonCanonicalImportRoot{
					ImportRoot: importRoot,
					Canonical:  pkg.CommentPath,
				},
			}
			return nil
		}

		// This area has some...fuzzy rules, but check all the imports for
		// local/relative/dot-ness, and record an error for the package if we
		// see any.
		var lim []string
		for _, imp := range append(pkg.Imports, pkg.TestImports...) {
			if build.IsLocalImport(imp) {
				// Do allow the single-dot, at least for now
				if imp == "." {
					continue
				}
				lim = append(lim, imp)
			}
		}

		if len(lim) > 0 {
			ptree.Packages[ip] = PackageOrErr{
				Err: &LocalImportsError{
					Dir:          wp,
					ImportPath:   ip,
					LocalImports: lim,
				},
			}
		} else {
			ptree.Packages[ip] = PackageOrErr{
				P: pkg,
			}
		}

		return nil
	})

	if err != nil {
		return PackageTree{}, err
	}

	return ptree, nil
}

// fillPackage full of info. Assumes p.Dir is set at a minimum
func (s *Strategy) fillPackage(p *build.Package) error {
	var buildPrefix = "// +build "
	var buildFieldSplit = func(r rune) bool {
		return unicode.IsSpace(r) || r == ','
	}

	gofiles, err := filepath.Glob(filepath.Join(p.Dir, "*.go"))
	if err != nil {
		return err
	}

	if len(gofiles) == 0 {
		return &build.NoGoError{Dir: p.Dir}
	}

	var testImports []string
	var imports []string
	var importComments []string
	for _, file := range gofiles {
		// Skip underscore-led or dot-led files, in keeping with the rest of the toolchain.
		bPrefix := filepath.Base(file)[0]
		if bPrefix == '_' || bPrefix == '.' {
			continue
		}

		// Skip any directories that happened to get caught by glob
		if stat, err := os.Stat(file); err == nil && stat.IsDir() {
			continue
		}

		pf, err := parser.ParseFile(token.NewFileSet(), file, nil, parser.ImportsOnly|parser.ParseComments)
		if err != nil {
			if os.IsPermission(err) {
				continue
			}
			return err
		}
		testFile := strings.HasSuffix(file, "_test.go")
		fname := filepath.Base(file)

		allIgnored := true
		hasExplicitTags := false
		for _, c := range pf.Comments {
			ic := findImportComment(pf.Name, c)
			if ic != "" {
				importComments = append(importComments, ic)
			}
			if c.Pos() > pf.Package { // +build comment must come before package
				continue
			}

			var ct string
			for _, cl := range c.List {
				if strings.HasPrefix(cl.Text, buildPrefix) {
					ct = cl.Text
					break
				}
			}
			if ct == "" {
				continue
			}

			tags := strings.FieldsFunc(ct[len(buildPrefix):], buildFieldSplit)
			for _, t := range tags {
				hasExplicitTags = true
				if s.isInterestingTag(t) {
					allIgnored = false
					break
				}
			}
		}

		important := !hasExplicitTags || !allIgnored

		if testFile {
			p.TestGoFiles = append(p.TestGoFiles, fname)
			if p.Name == "" && important {
				p.Name = strings.TrimSuffix(pf.Name.Name, "_test")
			}
		} else {
			if p.Name == "" && important {
				p.Name = pf.Name.Name
			}
			p.GoFiles = append(p.GoFiles, fname)
		}

		if !important {
			continue
		}

		for _, is := range pf.Imports {
			name, err := strconv.Unquote(is.Path.Value)
			if err != nil {
				return err // can't happen?
			}
			if testFile {
				testImports = append(testImports, name)
			} else {
				imports = append(imports, name)
			}
		}
	}
	importComments = uniq(importComments)
	if len(importComments) > 1 {
		return &ConflictingImportComments{
			ImportPath:                p.ImportPath,
			ConflictingImportComments: importComments,
		}
	}
	if len(importComments) > 0 {
		p.ImportComment = importComments[0]
	}
	imports = uniq(imports)
	testImports = uniq(testImports)
	p.Imports = imports
	p.TestImports = testImports
	return nil
}

var (
	slashSlash = []byte("//")
	slashStar  = []byte("/*")
	starSlash  = []byte("*/")
	importKwd  = []byte("import ")
)

func findImportComment(pkgName *ast.Ident, c *ast.CommentGroup) string {
	afterPkg := pkgName.NamePos + token.Pos(len(pkgName.Name)) + 1
	commentSlash := c.List[0].Slash
	if afterPkg != commentSlash {
		return ""
	}
	text := []byte(c.List[0].Text)
	switch {
	case bytes.HasPrefix(text, slashSlash):
		eol := bytes.IndexByte(text, '\n')
		if eol < 0 {
			eol = len(text)
		}
		text = text[2:eol]
	case bytes.HasPrefix(text, slashStar):
		text = text[2:]
		end := bytes.Index(text, starSlash)
		if end < 0 {
			// malformed comment
			return ""
		}
		text = text[:end]
		if bytes.IndexByte(text, '\n') > 0 {
			// multiline comment, can't be an import comment
			return ""
		}
	}
	text = bytes.TrimSpace(text)
	if !bytes.HasPrefix(text, importKwd) {
		return ""
	}
	quotedPath := bytes.TrimSpace(text[len(importKwd):])
	return string(bytes.Trim(quotedPath, `"`))
}

// ConflictingImportComments indicates that the package declares more than one
// different canonical path.
type ConflictingImportComments struct {
	ImportPath                string   // An import path referring to this package
	ConflictingImportComments []string // All distinct "canonical" paths encountered in the package files
}

func (e *ConflictingImportComments) Error() string {
	return fmt.Sprintf("import path %s had conflicting import comments: %s",
		e.ImportPath, quotedPaths(e.ConflictingImportComments))
}

// NonCanonicalImportRoot reports the situation when the dependee imports a
// package via something other than the package's declared canonical path.
type NonCanonicalImportRoot struct {
	ImportRoot string // A root path that is being used to import a package
	Canonical  string // A canonical path declared by the package being imported
}

func (e *NonCanonicalImportRoot) Error() string {
	return fmt.Sprintf("import root %q is not a prefix for the package's declared canonical path %q",
		e.ImportRoot, e.Canonical)
}

func quotedPaths(ps []string) string {
	quoted := make([]string, 0, len(ps))
	for _, p := range ps {
		quoted = append(quoted, fmt.Sprintf("%q", p))
	}
	return strings.Join(quoted, ", ")
}

// LocalImportsError indicates that a package contains at least one relative
// import that will prevent it from compiling.
//
// TODO(sdboyer) add a Files property once we're doing our own per-file parsing
type LocalImportsError struct {
	ImportPath   string
	Dir          string
	LocalImports []string
}

func (e *LocalImportsError) Error() string {
	switch len(e.LocalImports) {
	case 0:
		// shouldn't be possible, but just cover the case
		return fmt.Sprintf("import path %s had bad local imports", e.ImportPath)
	case 1:
		return fmt.Sprintf("import path %s had a local import: %q", e.ImportPath, e.LocalImports[0])
	default:
		return fmt.Sprintf("import path %s had local imports: %s", e.ImportPath, quotedPaths(e.LocalImports))
	}
}

type wm struct {
	err error
	ex  map[string]bool
	in  map[string]bool
}

// PackageOrErr stores the results of attempting to parse a single directory for
// Go source code.
type PackageOrErr struct {
	P   Package
	Err error
}

// ProblemImportError describes the reason that a particular import path is
// not safely importable.
type ProblemImportError struct {
	// The import path of the package with some problem rendering it
	// unimportable.
	ImportPath string
	// The path to the internal package the problem package imports that is the
	// original cause of this issue. If empty, the package itself is the
	// problem.
	Cause []string
	// The actual error from ListPackages that is undermining importability for
	// this package.
	Err error
}

// Error formats the ProblemImportError as a string, reflecting whether the
// error represents a direct or transitive problem.
func (e *ProblemImportError) Error() string {
	switch len(e.Cause) {
	case 0:
		return fmt.Sprintf("%q contains malformed code: %s", e.ImportPath, e.Err.Error())
	case 1:
		return fmt.Sprintf("%q imports %q, which contains malformed code: %s", e.ImportPath, e.Cause[0], e.Err.Error())
	default:
		return fmt.Sprintf("%q transitively (through %v packages) imports %q, which contains malformed code: %s", e.ImportPath, len(e.Cause)-1, e.Cause[len(e.Cause)-1], e.Err.Error())
	}
}

// Helper func to create an error when a package is missing.
func missingPkgErr(pkg string) error {
	return fmt.Errorf("no package exists at %q", pkg)
}

// A PackageTree represents the results of recursively parsing a tree of
// packages, starting at the ImportRoot. The results of parsing the files in the
// directory identified by each import path - a Package or an error - are stored
// in the Packages map, keyed by that import path.
type PackageTree struct {
	ImportRoot string
	Packages   map[string]PackageOrErr
}

// ToReachMap looks through a PackageTree and computes the list of external
// import statements (that is, import statements pointing to packages that are
// not logical children of PackageTree.ImportRoot) that are transitively
// imported by the internal packages in the tree.
//
// main indicates whether (true) or not (false) to include main packages in the
// analysis. When utilized by gps' solver, main packages are generally excluded
// from analyzing anything other than the root project, as they necessarily can't
// be imported.
//
// tests indicates whether (true) or not (false) to include imports from test
// files in packages when computing the reach map.
//
// backprop indicates whether errors (an actual PackageOrErr.Err, or an import
// to a nonexistent internal package) should be backpropagated, transitively
// "poisoning" all corresponding importers to all importers.
//
// ignore is a map of import paths that, if encountered, should be excluded from
// analysis. This exclusion applies to both internal and external packages. If
// an external import path is ignored, it is simply omitted from the results.
//
// If an internal path is ignored, then it not only does not appear in the final
// map, but it is also excluded from the transitive calculations of other
// internal packages.  That is, if you ignore A/foo, then the external package
// list for all internal packages that import A/foo will not include external
// packages that are only reachable through A/foo.
//
// Visually, this means that, given a PackageTree with root A and packages at A,
// A/foo, and A/bar, and the following import chain:
//
//  A -> A/foo -> A/bar -> B/baz
//
// In this configuration, all of A's packages transitively import B/baz, so the
// returned map would be:
//
//  map[string][]string{
// 	"A": []string{"B/baz"},
// 	"A/foo": []string{"B/baz"}
// 	"A/bar": []string{"B/baz"},
//  }
//
// However, if you ignore A/foo, then A's path to B/baz is broken, and A/foo is
// omitted entirely. Thus, the returned map would be:
//
//  map[string][]string{
// 	"A": []string{},
// 	"A/bar": []string{"B/baz"},
//  }
//
// If there are no packages to ignore, it is safe to pass a nil map.
//
// Finally, if an internal PackageOrErr contains an error, it is always omitted
// from the result set. If backprop is true, then the error from that internal
// package will be transitively propagated back to any other internal
// PackageOrErrs that import it, causing them to also be omitted. So, with the
// same import chain:
//
//  A -> A/foo -> A/bar -> B/baz
//
// If A/foo has an error, then it would backpropagate to A, causing both to be
// omitted, and the returned map to contain only A/bar:
//
//  map[string][]string{
// 	"A/bar": []string{"B/baz"},
//  }
//
// If backprop is false, then errors will not backpropagate to internal
// importers. So, with an error in A/foo, this would be the result map:
//
//  map[string][]string{
// 	"A": []string{},
// 	"A/bar": []string{"B/baz"},
//  }
func (t PackageTree) ToReachMap(main, tests, backprop bool, ignore *IgnoredRuleset) (ReachMap, map[string]*ProblemImportError) {
	// world's simplest adjacency list
	workmap := make(map[string]wm)

	var imps []string
	for ip, perr := range t.Packages {
		if perr.Err != nil {
			workmap[ip] = wm{
				err: perr.Err,
			}
			continue
		}
		p := perr.P

		// Skip main packages, unless param says otherwise
		if p.Name == "main" && !main {
			continue
		}
		// Skip ignored packages
		if ignore.IsIgnored(ip) {
			continue
		}

		// TODO (kris-nova) Disable to get staticcheck passing
		//imps = imps[:0]

		if tests {
			imps = dedupeStrings(p.Imports, p.TestImports)
		} else {
			imps = p.Imports
		}

		w := wm{
			ex: make(map[string]bool),
			in: make(map[string]bool),
		}

		// For each import, decide whether it should be ignored, or if it
		// belongs in the external or internal imports list.
		for _, imp := range imps {
			if ignore.IsIgnored(imp) || imp == "." {
				continue
			}

			if !eqOrSlashedPrefix(imp, t.ImportRoot) {
				w.ex[imp] = true
			} else {
				w.in[imp] = true
			}
		}

		workmap[ip] = w
	}

	return wmToReach(workmap, backprop)
}

// Copy copies the PackageTree.
//
// This is really only useful as a defensive measure to prevent external state
// mutations.
func (t PackageTree) Copy() PackageTree {
	t2 := PackageTree{
		ImportRoot: t.ImportRoot,
		Packages:   make(map[string]PackageOrErr, len(t.Packages)),
	}

	// Walk through and count up the total number of string slice elements we'll
	// need, then allocate them all at once.
	strcount := 0
	for _, poe := range t.Packages {
		strcount = strcount + len(poe.P.Imports) + len(poe.P.TestImports)
	}
	pool := make([]string, strcount)

	for path, poe := range t.Packages {
		var poe2 PackageOrErr

		if poe.Err != nil {
			refl := reflect.ValueOf(poe.Err)
			switch refl.Kind() {
			case reflect.Ptr:
				poe2.Err = reflect.New(refl.Elem().Type()).Interface().(error)
			case reflect.Slice:
				err2 := reflect.MakeSlice(refl.Type(), refl.Len(), refl.Len())
				reflect.Copy(err2, refl)
				poe2.Err = err2.Interface().(error)
			default:
				// This shouldn't be too onerous to maintain - the set of errors
				// we can get here is restricted by what ListPackages() allows.
				// So just panic if one is outside the expected kinds of ptr or
				// slice, as that would mean we've missed something notable.
				panic(fmt.Sprintf("unrecognized PackgeOrErr error type, %T", poe.Err))
			}
		} else {
			poe2.P = poe.P
			il, til := len(poe.P.Imports), len(poe.P.TestImports)
			if il > 0 {
				poe2.P.Imports, pool = pool[:il], pool[il:]
				copy(poe2.P.Imports, poe.P.Imports)
			}
			if til > 0 {
				poe2.P.TestImports, pool = pool[:til], pool[til:]
				copy(poe2.P.TestImports, poe.P.TestImports)
			}
		}

		t2.Packages[path] = poe2
	}

	return t2
}

// TrimHiddenPackages returns a new PackageTree where packages that are ignored,
// or both hidden and unreachable, have been removed.
//
// The package list is partitioned into two sets: visible, and hidden, where
// packages are considered hidden if they are within or beneath directories
// with:
//
//  * leading dots
//  * leading underscores
//  * the exact name "testdata"
//
// Packages in the hidden set are dropped from the returned PackageTree, unless
// they are transitively reachable from imports in the visible set.
//
// The "main", "tests" and "ignored" parameters have the same behavior as with
// PackageTree.ToReachMap(): the first two determine, respectively, whether
// imports from main packages, and imports from tests, should be considered for
// reachability checks. Setting 'main' to true will additionally result in main
// packages being trimmed.
//
// "ignored" designates import paths, or patterns of import paths, where the
// corresponding packages should be excluded from reachability checks, if
// encountered. Ignored packages are also removed from the final set.
//
// Note that it is not recommended to call this method if the goal is to obtain
// a set of tree-external imports; calling ToReachMap and FlattenFn will achieve
// the same effect.
func (t PackageTree) TrimHiddenPackages(main, tests bool, ignore *IgnoredRuleset) PackageTree {
	rm, pie := t.ToReachMap(main, tests, false, ignore)
	t2 := t.Copy()
	preserve := make(map[string]bool)

	for pkg, ie := range rm {
		if pkgFilter(pkg) && !ignore.IsIgnored(pkg) {
			preserve[pkg] = true
			for _, in := range ie.Internal {
				preserve[in] = true
			}
		}
	}

	// Also process the problem map, as packages in the visible set with errors
	// need to be included in the return values.
	for pkg := range pie {
		if pkgFilter(pkg) && !ignore.IsIgnored(pkg) {
			preserve[pkg] = true
		}
	}

	for ip := range t.Packages {
		if !preserve[ip] {
			delete(t2.Packages, ip)
		}
	}

	return t2
}

// wmToReach takes an internal "workmap" constructed by
// PackageTree.ExternalReach(), transitively walks (via depth-first traversal)
// all internal imports until they reach an external path or terminate, then
// translates the results into a slice of external imports for each internal
// pkg.
//
// It drops any packages with errors, and - if backprop is true - backpropagates
// those errors, causing internal packages that (transitively) import other
// internal packages having errors to also be dropped.
func wmToReach(workmap map[string]wm, backprop bool) (ReachMap, map[string]*ProblemImportError) {
	// Uses depth-first exploration to compute reachability into external
	// packages, dropping any internal packages on "poisoned paths" - a path
	// containing a package with an error, or with a dep on an internal package
	// that's missing.

	const (
		white uint8 = iota
		grey
		black
	)

	colors := make(map[string]uint8)
	exrsets := make(map[string]map[string]struct{})
	inrsets := make(map[string]map[string]struct{})
	errmap := make(map[string]*ProblemImportError)

	// poison is a helper func to eliminate specific reachsets from exrsets and
	// inrsets, and populate error information along the way.
	poison := func(path []string, err *ProblemImportError) {
		for k, ppkg := range path {
			delete(exrsets, ppkg)
			delete(inrsets, ppkg)

			// Duplicate the err for this package
			kerr := &ProblemImportError{
				ImportPath: ppkg,
				Err:        err.Err,
			}

			// Shift the slice bounds on the incoming err.Cause.
			//
			// This check will only be false on the final path element when
			// entering via poisonWhite, where the last pkg is the underlying
			// cause of the problem, and is thus expected to have an empty Cause
			// slice.
			if k+1 < len(err.Cause) {
				// reuse the slice
				kerr.Cause = err.Cause[k+1:]
			}

			// Both black and white cases can have the final element be a
			// package that doesn't exist. If that's the case, don't write it
			// directly to the errmap, as presence in the errmap indicates the
			// package was present in the input PackageTree.
			if k == len(path)-1 {
				if _, exists := workmap[path[len(path)-1]]; !exists {
					continue
				}
			}

			// Direct writing to the errmap means that if multiple errors affect
			// a given package, only the last error visited will be reported.
			// But that should be sufficient; presumably, the user can
			// iteratively resolve the errors.
			errmap[ppkg] = kerr
		}
	}

	// poisonWhite wraps poison for error recording in the white-poisoning case,
	// where we're constructing a new poison path.
	poisonWhite := func(path []string) {
		err := &ProblemImportError{
			Cause: make([]string, len(path)),
		}
		copy(err.Cause, path)

		// find the tail err
		tail := path[len(path)-1]
		if w, exists := workmap[tail]; exists {
			// If we make it to here, the dfe guarantees that the workmap
			// will contain an error for this pkg.
			err.Err = w.err
		} else {
			err.Err = missingPkgErr(tail)
		}

		poison(path, err)
	}
	// poisonBlack wraps poison for error recording in the black-poisoning case,
	// where we're connecting to an existing poison path.
	poisonBlack := func(path []string, from string) {
		// Because the outer dfe loop ensures we never directly re-visit a pkg
		// that was already completed (black), we don't have to defend against
		// an empty path here.

		fromErr, exists := errmap[from]
		// FIXME: It should not be possible for fromErr to not exist,
		// See issue https://github.com/golang/dep/issues/351
		// This is a temporary solution to avoid a panic.
		if !exists {
			fromErr = &ProblemImportError{
				Err: fmt.Errorf("unknown error for %q, if you get this error see https://github.com/golang/dep/issues/351", from),
			}
		}
		err := &ProblemImportError{
			Err:   fromErr.Err,
			Cause: make([]string, 0, len(path)+len(fromErr.Cause)+1),
		}
		err.Cause = append(err.Cause, path...)
		err.Cause = append(err.Cause, from)
		err.Cause = append(err.Cause, fromErr.Cause...)

		poison(path, err)
	}

	var dfe func(string, []string) bool

	// dfe is the depth-first-explorer that computes a safe, error-free external
	// reach map.
	//
	// pkg is the import path of the pkg currently being visited; path is the
	// stack of parent packages we've visited to get to pkg. The return value
	// indicates whether the level completed successfully (true) or if it was
	// poisoned (false).
	dfe = func(pkg string, path []string) bool {
		// white is the zero value of uint8, which is what we want if the pkg
		// isn't in the colors map, so this works fine
		switch colors[pkg] {
		case white:
			// first visit to this pkg; mark it as in-process (grey)
			colors[pkg] = grey

			// make sure it's present and w/out errs
			w, exists := workmap[pkg]

			// Push current visitee onto the path slice. Passing path through
			// recursion levels as a value has the effect of auto-popping the
			// slice, while also giving us safe memory reuse.
			path = append(path, pkg)

			if !exists || w.err != nil {
				if backprop {
					// Does not exist or has an err; poison self and all parents
					poisonWhite(path)
				} else if exists {
					// Only record something in the errmap if there's actually a
					// package there, per the semantics of the errmap
					errmap[pkg] = &ProblemImportError{
						ImportPath: pkg,
						Err:        w.err,
					}
				}

				// we know we're done here, so mark it black
				colors[pkg] = black
				return false
			}
			// pkg exists with no errs; start internal and external reachsets for it.
			rs := make(map[string]struct{})
			irs := make(map[string]struct{})

			// Dump this package's external pkgs into its own reachset. Separate
			// loop from the parent dump to avoid nested map loop lookups.
			for ex := range w.ex {
				rs[ex] = struct{}{}
			}
			exrsets[pkg] = rs
			// Same deal for internal imports
			for in := range w.in {
				irs[in] = struct{}{}
			}
			inrsets[pkg] = irs

			// Push this pkg's imports into all parent reachsets. Not all
			// parents will necessarily have a reachset; none, some, or all
			// could have been poisoned by a different path than what we're on
			// right now.
			for _, ppkg := range path {
				if prs, exists := exrsets[ppkg]; exists {
					for ex := range w.ex {
						prs[ex] = struct{}{}
					}
				}

				if prs, exists := inrsets[ppkg]; exists {
					for in := range w.in {
						prs[in] = struct{}{}
					}
				}
			}

			// Now, recurse until done, or a false bubbles up, indicating the
			// path is poisoned.
			for in := range w.in {
				// It's possible, albeit weird, for a package to import itself.
				// If we try to visit self, though, then it erroneously poisons
				// the path, as it would be interpreted as grey. In practice,
				// self-imports are a no-op, so we can just skip it.
				if in == pkg {
					continue
				}

				clean := dfe(in, path)
				if !clean && backprop {
					// Path is poisoned. If we're backpropagating errors, then
					// the  reachmap for the visitee was already deleted by the
					// path we're returning from; mark the visitee black, then
					// return false to bubble up the poison. This is OK to do
					// early, before exploring all internal imports, because the
					// outer loop visits all internal packages anyway.
					//
					// In fact, stopping early is preferable - white subpackages
					// won't have to iterate pointlessly through a parent path
					// with no reachset.
					colors[pkg] = black
					return false
				}
			}

			// Fully done with this pkg; no transitive problems.
			colors[pkg] = black
			return true

		case grey:
			// Import cycles can arise in healthy situations through xtests, so
			// allow them for now.
			//
			// FIXME(sdboyer) we need an improved model that allows us to
			// accurately reject real import cycles.
			return true
			// grey means an import cycle; guaranteed badness right here. You'd
			// hope we never encounter it in a dependency (really? you published
			// that code?), but we have to defend against it.
			//colors[pkg] = black
			//poison(append(path, pkg)) // poison self and parents

		case black:
			// black means we're revisiting a package that was already
			// completely explored. If it has an entry in exrsets, it completed
			// successfully. If not, it was poisoned, and we need to bubble the
			// poison back up.
			rs, exists := exrsets[pkg]
			if !exists {
				if backprop {
					// just poison parents; self was necessarily already poisoned
					poisonBlack(path, pkg)
				}
				return false
			}
			// If external reachset existed, internal must (even if empty)
			irs := inrsets[pkg]

			// It's good; pull over the imports from its reachset into all
			// non-poisoned parent reachsets
			for _, ppkg := range path {
				if prs, exists := exrsets[ppkg]; exists {
					for ex := range rs {
						prs[ex] = struct{}{}
					}
				}

				if prs, exists := inrsets[ppkg]; exists {
					for in := range irs {
						prs[in] = struct{}{}
					}
				}
			}
			return true

		default:
			panic(fmt.Sprintf("invalid color marker %v for %s", colors[pkg], pkg))
		}
	}

	// Run the depth-first exploration.
	//
	// Don't bother computing graph sources, this straightforward loop works
	// comparably well, and fits nicely with an escape hatch in the dfe.
	var path []string
	for pkg := range workmap {
		// However, at least check that the package isn't already fully visited;
		// this saves a bit of time and implementation complexity inside the
		// closures.
		if colors[pkg] != black {
			dfe(pkg, path)
		}
	}

	type ie struct {
		Internal, External []string
	}

	// Flatten exrsets into reachmap
	rm := make(ReachMap)
	for pkg, rs := range exrsets {
		rlen := len(rs)
		if rlen == 0 {
			rm[pkg] = ie{}
			continue
		}

		edeps := make([]string, 0, rlen)
		for opkg := range rs {
			edeps = append(edeps, opkg)
		}

		sort.Strings(edeps)

		sets := rm[pkg]
		sets.External = edeps
		rm[pkg] = sets
	}

	// Flatten inrsets into reachmap
	for pkg, rs := range inrsets {
		rlen := len(rs)
		if rlen == 0 {
			continue
		}

		ideps := make([]string, 0, rlen)
		for opkg := range rs {
			ideps = append(ideps, opkg)
		}

		sort.Strings(ideps)

		sets := rm[pkg]
		sets.Internal = ideps
		rm[pkg] = sets
	}

	return rm, errmap
}

// eqOrSlashedPrefix checks to see if the prefix is either equal to the string,
// or that it is a prefix and the next char in the string is "/".
func eqOrSlashedPrefix(s, prefix string) bool {
	if !strings.HasPrefix(s, prefix) {
		return false
	}

	prflen, pathlen := len(prefix), len(s)
	return prflen == pathlen || strings.Index(s[prflen:], "/") == 0
}

// helper func to merge, dedupe, and sort strings
func dedupeStrings(s1, s2 []string) (r []string) {
	dedupe := make(map[string]bool)

	if len(s1) > 0 && len(s2) > 0 {
		for _, i := range s1 {
			dedupe[i] = true
		}
		for _, i := range s2 {
			dedupe[i] = true
		}

		for i := range dedupe {
			r = append(r, i)
		}
		// And then re-sort them
		sort.Strings(r)
	} else if len(s1) > 0 {
		r = s1
	} else if len(s2) > 0 {
		r = s2
	}

	return
}

func uniq(a []string) []string {
	if a == nil {
		return make([]string, 0)
	}
	var s string
	var i int
	if !sort.StringsAreSorted(a) {
		sort.Strings(a)
	}
	for _, t := range a {
		if t != s {
			a[i] = t
			i++
			s = t
		}
	}
	return a[:i]
}
