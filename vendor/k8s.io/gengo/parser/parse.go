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
	"io/ioutil"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"k8s.io/gengo/types"
	"k8s.io/klog/v2"
)

// This clarifies when a pkg path has been canonicalized.
type importPathString string

// Builder lets you add all the go files in all the packages that you care
// about, then constructs the type source data.
type Builder struct {
	context *build.Context

	// If true, include *_test.go
	IncludeTestFiles bool

	// Map of package names to more canonical information about the package.
	// This might hold the same value for multiple names, e.g. if someone
	// referenced ./pkg/name or in the case of vendoring, which canonicalizes
	// differently that what humans would type.
	buildPackages map[string]*build.Package

	fset *token.FileSet
	// map of package path to list of parsed files
	parsed map[importPathString][]parsedFile
	// map of package path to absolute path (to prevent overlap)
	absPaths map[importPathString]string

	// Set by typeCheckPackage(), used by importPackage() and friends.
	typeCheckedPackages map[importPathString]*tc.Package

	// Map of package path to whether the user requested it or it was from
	// an import.
	userRequested map[importPathString]bool

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
		context:               &c,
		buildPackages:         map[string]*build.Package{},
		typeCheckedPackages:   map[importPathString]*tc.Package{},
		fset:                  token.NewFileSet(),
		parsed:                map[importPathString][]parsedFile{},
		absPaths:              map[importPathString]string{},
		userRequested:         map[importPathString]bool{},
		endLineToCommentGroup: map[fileLine]*ast.CommentGroup{},
		importGraph:           map[importPathString]map[string]struct{}{},
	}
}

// AddBuildTags adds the specified build tags to the parse context.
func (b *Builder) AddBuildTags(tags ...string) {
	b.context.BuildTags = append(b.context.BuildTags, tags...)
}

// Get package information from the go/build package. Automatically excludes
// e.g. test files and files for other platforms-- there is quite a bit of
// logic of that nature in the build package.
func (b *Builder) importBuildPackage(dir string) (*build.Package, error) {
	if buildPkg, ok := b.buildPackages[dir]; ok {
		return buildPkg, nil
	}
	// This validates the `package foo // github.com/bar/foo` comments.
	buildPkg, err := b.importWithMode(dir, build.ImportComment)
	if err != nil {
		if _, ok := err.(*build.NoGoError); !ok {
			return nil, fmt.Errorf("unable to import %q: %v", dir, err)
		}
	}
	if buildPkg == nil {
		// Might be an empty directory. Try to just find the dir.
		buildPkg, err = b.importWithMode(dir, build.FindOnly)
		if err != nil {
			return nil, err
		}
	}

	// Remember it under the user-provided name.
	klog.V(5).Infof("saving buildPackage %s", dir)
	b.buildPackages[dir] = buildPkg
	canonicalPackage := canonicalizeImportPath(buildPkg.ImportPath)
	if dir != string(canonicalPackage) {
		// Since `dir` is not the canonical name, see if we knew it under another name.
		if buildPkg, ok := b.buildPackages[string(canonicalPackage)]; ok {
			return buildPkg, nil
		}
		// Must be new, save it under the canonical name, too.
		klog.V(5).Infof("saving buildPackage %s", canonicalPackage)
		b.buildPackages[string(canonicalPackage)] = buildPkg
	}

	return buildPkg, nil
}

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
	if _, err := b.typeCheckPackage(importPathString(pkg)); err != nil {
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

// AddDir adds an entire directory, scanning it for go files. 'dir' should have
// a single go package in it. GOPATH, GOROOT, and the location of your go
// binary (`which go`) will all be searched if dir doesn't literally resolve.
func (b *Builder) AddDir(dir string) error {
	_, err := b.importPackage(dir, true)
	return err
}

// AddDirRecursive is just like AddDir, but it also recursively adds
// subdirectories; it returns an error only if the path couldn't be resolved;
// any directories recursed into without go source are ignored.
func (b *Builder) AddDirRecursive(dir string) error {
	// Add the root.
	if _, err := b.importPackage(dir, true); err != nil {
		klog.Warningf("Ignoring directory %v: %v", dir, err)
	}

	// filepath.Walk does not follow symlinks. We therefore evaluate symlinks and use that with
	// filepath.Walk.
	realPath, err := filepath.EvalSymlinks(b.buildPackages[dir].Dir)
	if err != nil {
		return err
	}

	fn := func(filePath string, info os.FileInfo, err error) error {
		if info != nil && info.IsDir() {
			rel := filepath.ToSlash(strings.TrimPrefix(filePath, realPath))
			if rel != "" {
				// Make a pkg path.
				pkg := path.Join(string(canonicalizeImportPath(b.buildPackages[dir].ImportPath)), rel)

				// Add it.
				if _, err := b.importPackage(pkg, true); err != nil {
					klog.Warningf("Ignoring child directory %v: %v", pkg, err)
				}
			}
		}
		return nil
	}
	if err := filepath.Walk(realPath, fn); err != nil {
		return err
	}
	return nil
}

// AddDirTo adds an entire directory to a given Universe. Unlike AddDir, this
// processes the package immediately, which makes it safe to use from within a
// generator (rather than just at init time. 'dir' must be a single go package.
// GOPATH, GOROOT, and the location of your go binary (`which go`) will all be
// searched if dir doesn't literally resolve.
// Deprecated. Please use AddDirectoryTo.
func (b *Builder) AddDirTo(dir string, u *types.Universe) error {
	// We want all types from this package, as if they were directly added
	// by the user.  They WERE added by the user, in effect.
	if _, err := b.importPackage(dir, true); err != nil {
		return err
	}
	return b.findTypesIn(canonicalizeImportPath(b.buildPackages[dir].ImportPath), u)
}

// AddDirectoryTo adds an entire directory to a given Universe. Unlike AddDir,
// this processes the package immediately, which makes it safe to use from
// within a generator (rather than just at init time. 'dir' must be a single go
// package. GOPATH, GOROOT, and the location of your go binary (`which go`)
// will all be searched if dir doesn't literally resolve.
func (b *Builder) AddDirectoryTo(dir string, u *types.Universe) (*types.Package, error) {
	// We want all types from this package, as if they were directly added
	// by the user.  They WERE added by the user, in effect.
	if _, err := b.importPackage(dir, true); err != nil {
		return nil, err
	}
	path := canonicalizeImportPath(b.buildPackages[dir].ImportPath)
	if err := b.findTypesIn(path, u); err != nil {
		return nil, err
	}
	return u.Package(string(path)), nil
}

// The implementation of AddDir. A flag indicates whether this directory was
// user-requested or just from following the import graph.
func (b *Builder) addDir(dir string, userRequested bool) error {
	klog.V(5).Infof("addDir %s", dir)
	buildPkg, err := b.importBuildPackage(dir)
	if err != nil {
		return err
	}
	canonicalPackage := canonicalizeImportPath(buildPkg.ImportPath)
	pkgPath := canonicalPackage
	if dir != string(canonicalPackage) {
		klog.V(5).Infof("addDir %s, canonical path is %s", dir, pkgPath)
	}

	// Sanity check the pkg dir has not changed.
	if prev, found := b.absPaths[pkgPath]; found {
		if buildPkg.Dir != prev {
			return fmt.Errorf("package %q (%s) previously resolved to %s", pkgPath, buildPkg.Dir, prev)
		}
	} else {
		b.absPaths[pkgPath] = buildPkg.Dir
	}

	files := []string{}
	files = append(files, buildPkg.GoFiles...)
	if b.IncludeTestFiles {
		files = append(files, buildPkg.TestGoFiles...)
	}

	for _, file := range files {
		if !strings.HasSuffix(file, ".go") {
			continue
		}
		absPath := filepath.Join(buildPkg.Dir, file)
		data, err := ioutil.ReadFile(absPath)
		if err != nil {
			return fmt.Errorf("while loading %q: %v", absPath, err)
		}
		err = b.addFile(pkgPath, absPath, data, userRequested)
		if err != nil {
			return fmt.Errorf("while parsing %q: %v", absPath, err)
		}
	}
	return nil
}

// regexErrPackageNotFound helps test the expected error for not finding a package.
var regexErrPackageNotFound = regexp.MustCompile(`^unable to import ".*?":.*`)

func isErrPackageNotFound(err error) bool {
	return regexErrPackageNotFound.MatchString(err.Error())
}

// importPackage is a function that will be called by the type check package when it
// needs to import a go package. 'path' is the import path.
func (b *Builder) importPackage(dir string, userRequested bool) (*tc.Package, error) {
	klog.V(5).Infof("importPackage %s", dir)
	var pkgPath = importPathString(dir)

	// Get the canonical path if we can.
	if buildPkg := b.buildPackages[dir]; buildPkg != nil {
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
		if buildPkg := b.buildPackages[dir]; buildPkg != nil {
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
	pkg, err := b.typeCheckPackage(pkgPath)
	if err != nil {
		switch {
		case ignoreError && pkg != nil:
			klog.V(2).Infof("type checking encountered some issues in %q, but ignoring.\n", pkgPath)
		case !ignoreError && pkg != nil:
			klog.V(2).Infof("type checking encountered some errors in %q\n", pkgPath)
			return nil, err
		default:
			return nil, err
		}
	}

	return pkg, nil
}

type importAdapter struct {
	b *Builder
}

func (a importAdapter) Import(path string) (*tc.Package, error) {
	return a.b.importPackage(path, false)
}

// typeCheckPackage will attempt to return the package even if there are some
// errors, so you may check whether the package is nil or not even if you get
// an error.
func (b *Builder) typeCheckPackage(pkgPath importPathString) (*tc.Package, error) {
	klog.V(5).Infof("typeCheckPackage %s", pkgPath)
	if pkg, ok := b.typeCheckedPackages[pkgPath]; ok {
		if pkg != nil {
			klog.V(6).Infof("typeCheckPackage %s already done", pkgPath)
			return pkg, nil
		}
		// We store a nil right before starting work on a package. So
		// if we get here and it's present and nil, that means there's
		// another invocation of this function on the call stack
		// already processing this package.
		return nil, fmt.Errorf("circular dependency for %q", pkgPath)
	}
	parsedFiles, ok := b.parsed[pkgPath]
	if !ok {
		return nil, fmt.Errorf("No files for pkg %q", pkgPath)
	}
	files := make([]*ast.File, len(parsedFiles))
	for i := range parsedFiles {
		files[i] = parsedFiles[i].file
	}
	b.typeCheckedPackages[pkgPath] = nil
	c := tc.Config{
		IgnoreFuncBodies: true,
		// Note that importAdapter can call b.importPackage which calls this
		// method. So there can't be cycles in the import graph.
		Importer: importAdapter{b},
		Error: func(err error) {
			klog.V(2).Infof("type checker: %v\n", err)
		},
	}
	pkg, err := c.Check(string(pkgPath), b.fset, files, nil)
	b.typeCheckedPackages[pkgPath] = pkg // record the result whether or not there was an error
	return pkg, err
}

// FindPackages fetches a list of the user-imported packages.
// Note that you need to call b.FindTypes() first.
func (b *Builder) FindPackages() []string {
	// Iterate packages in a predictable order.
	pkgPaths := []string{}
	for k := range b.typeCheckedPackages {
		pkgPaths = append(pkgPaths, string(k))
	}
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
	for pkgPath := range b.parsed {
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

// findTypesIn finalizes the package import and searches through the package
// for types.
func (b *Builder) findTypesIn(pkgPath importPathString, u *types.Universe) error {
	klog.V(5).Infof("findTypesIn %s", pkgPath)
	pkg := b.typeCheckedPackages[pkgPath]
	if pkg == nil {
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

	// We're keeping this package.  This call will create the record.
	u.Package(string(pkgPath)).Name = pkg.Name()
	u.Package(string(pkgPath)).Path = pkg.Path()
	u.Package(string(pkgPath)).SourcePath = b.absPaths[pkgPath]

	for _, f := range b.parsed[pkgPath] {
		if _, fileName := filepath.Split(f.name); fileName == "doc.go" {
			tp := u.Package(string(pkgPath))
			// findTypesIn might be called multiple times. Clean up tp.Comments
			// to avoid repeatedly fill same comments to it.
			tp.Comments = []string{}
			for i := range f.file.Comments {
				tp.Comments = append(tp.Comments, splitLines(f.file.Comments[i].Text())...)
			}
			if f.file.Doc != nil {
				tp.DocComments = splitLines(f.file.Doc.Text())
			}
		}
	}

	s := pkg.Scope()
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

	importedPkgs := []string{}
	for k := range b.importGraph[pkgPath] {
		importedPkgs = append(importedPkgs, string(k))
	}
	sort.Strings(importedPkgs)
	for _, p := range importedPkgs {
		u.AddImports(string(pkgPath), p)
	}
	return nil
}

func (b *Builder) importWithMode(dir string, mode build.ImportMode) (*build.Package, error) {
	// This is a bit of a hack.  The srcDir argument to Import() should
	// properly be the dir of the file which depends on the package to be
	// imported, so that vendoring can work properly and local paths can
	// resolve.  We assume that there is only one level of vendoring, and that
	// the CWD is inside the GOPATH, so this should be safe. Nobody should be
	// using local (relative) paths except on the CLI, so CWD is also
	// sufficient.
	cwd, err := os.Getwd()
	if err != nil {
		return nil, fmt.Errorf("unable to get current directory: %v", err)
	}
	buildPkg, err := b.context.Import(filepath.ToSlash(dir), cwd, mode)
	if err != nil {
		return nil, err
	}
	return buildPkg, nil
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
	}
	for i := 0; i < t.Results().Len(); i++ {
		signature.Results = append(signature.Results, b.walkType(u, nil, t.Results().At(i).Type()))
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
		// TODO: need to store array length, otherwise raw type name
		// cannot be properly written.
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

// canonicalizeImportPath takes an import path and returns the actual package.
// It doesn't support nested vendoring.
func canonicalizeImportPath(importPath string) importPathString {
	if !strings.Contains(importPath, "/vendor/") {
		return importPathString(importPath)
	}

	return importPathString(importPath[strings.Index(importPath, "/vendor/")+len("/vendor/"):])
}
