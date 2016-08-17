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
	"go/parser"
	"go/token"
	tc "go/types"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/golang/glog"
	"k8s.io/kubernetes/cmd/libs/go2idl/types"
)

// Builder lets you add all the go files in all the packages that you care
// about, then constructs the type source data.
type Builder struct {
	context   *build.Context
	buildInfo map[string]*build.Package

	fset *token.FileSet
	// map of package id to list of parsed files
	parsed map[string][]parsedFile
	// map of package id to absolute path (to prevent overlap)
	absPaths map[string]string

	// Set by makePackage(), used by importer() and friends.
	pkgs map[string]*tc.Package

	// Map of package path to whether the user requested it or it was from
	// an import.
	userRequested map[string]bool

	// All comments from everywhere in every parsed file.
	endLineToCommentGroup map[fileLine]*ast.CommentGroup

	// map of package to list of packages it imports.
	importGraph map[string]map[string]struct{}
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
			glog.Warningf("Warning: $GOROOT not set, and unable to run `which go` to find it: %v\n", err)
		}
	}
	// Force this to off, since we don't properly parse CGo.  All symbols must
	// have non-CGo equivalents.
	c.CgoEnabled = false
	return &Builder{
		context:               &c,
		buildInfo:             map[string]*build.Package{},
		fset:                  token.NewFileSet(),
		parsed:                map[string][]parsedFile{},
		absPaths:              map[string]string{},
		userRequested:         map[string]bool{},
		endLineToCommentGroup: map[fileLine]*ast.CommentGroup{},
		importGraph:           map[string]map[string]struct{}{},
	}
}

// AddBuildTags adds the specified build tags to the parse context.
func (b *Builder) AddBuildTags(tags ...string) {
	b.context.BuildTags = append(b.context.BuildTags, tags...)
}

// Get package information from the go/build package. Automatically excludes
// e.g. test files and files for other platforms-- there is quite a bit of
// logic of that nature in the build package.
func (b *Builder) buildPackage(pkgPath string) (*build.Package, error) {
	// This is a bit of a hack.  The srcDir argument to Import() should
	// properly be the dir of the file which depends on the package to be
	// imported, so that vendoring can work properly.  We assume that there is
	// only one level of vendoring, and that the CWD is inside the GOPATH, so
	// this should be safe.
	cwd, err := os.Getwd()
	if err != nil {
		return nil, fmt.Errorf("unable to get current directory: %v", err)
	}

	// First, find it, so we know what path to use.
	pkg, err := b.context.Import(pkgPath, cwd, build.FindOnly)
	if err != nil {
		return nil, fmt.Errorf("unable to *find* %q: %v", pkgPath, err)
	}

	pkgPath = pkg.ImportPath

	if pkg, ok := b.buildInfo[pkgPath]; ok {
		return pkg, nil
	}
	pkg, err = b.context.Import(pkgPath, cwd, build.ImportComment)
	if err != nil {
		if _, ok := err.(*build.NoGoError); !ok {
			return nil, fmt.Errorf("unable to import %q: %v", pkgPath, err)
		}
	}
	b.buildInfo[pkgPath] = pkg

	if b.importGraph[pkgPath] == nil {
		b.importGraph[pkgPath] = map[string]struct{}{}
	}
	for _, p := range pkg.Imports {
		b.importGraph[pkgPath][p] = struct{}{}
	}
	return pkg, nil
}

// AddFile adds a file to the set. The pkg must be of the form
// "canonical/pkg/path" and the path must be the absolute path to the file.
func (b *Builder) AddFile(pkg string, path string, src []byte) error {
	return b.addFile(pkg, path, src, true)
}

// addFile adds a file to the set. The pkg must be of the form
// "canonical/pkg/path" and the path must be the absolute path to the file. A
// flag indicates whether this file was user-requested or just from following
// the import graph.
func (b *Builder) addFile(pkg string, path string, src []byte, userRequested bool) error {
	p, err := parser.ParseFile(b.fset, path, src, parser.DeclarationErrors|parser.ParseComments)
	if err != nil {
		return err
	}
	dirPath := filepath.Dir(path)
	if prev, found := b.absPaths[pkg]; found {
		if dirPath != prev {
			return fmt.Errorf("package %q (%s) previously resolved to %s", pkg, dirPath, prev)
		}
	} else {
		b.absPaths[pkg] = dirPath
	}

	b.parsed[pkg] = append(b.parsed[pkg], parsedFile{path, p})
	b.userRequested[pkg] = userRequested
	for _, c := range p.Comments {
		position := b.fset.Position(c.End())
		b.endLineToCommentGroup[fileLine{position.Filename, position.Line}] = c
	}

	// We have to get the packages from this specific file, in case the
	// user added individual files instead of entire directories.
	if b.importGraph[pkg] == nil {
		b.importGraph[pkg] = map[string]struct{}{}
	}
	for _, im := range p.Imports {
		importedPath := strings.Trim(im.Path.Value, `"`)
		b.importGraph[pkg][importedPath] = struct{}{}
	}
	return nil
}

// AddDir adds an entire directory, scanning it for go files. 'dir' should have
// a single go package in it. GOPATH, GOROOT, and the location of your go
// binary (`which go`) will all be searched if dir doesn't literally resolve.
func (b *Builder) AddDir(dir string) error {
	return b.addDir(dir, true)
}

// AddDirRecursive is just like AddDir, but it also recursively adds
// subdirectories; it returns an error only if the path couldn't be resolved;
// any directories recursed into without go source are ignored.
func (b *Builder) AddDirRecursive(dir string) error {
	// This is a bit of a hack.  The srcDir argument to Import() should
	// properly be the dir of the file which depends on the package to be
	// imported, so that vendoring can work properly.  We assume that there is
	// only one level of vendoring, and that the CWD is inside the GOPATH, so
	// this should be safe.
	cwd, err := os.Getwd()
	if err != nil {
		return fmt.Errorf("unable to get current directory: %v", err)
	}

	// First, find it, so we know what path to use.
	pkg, err := b.context.Import(dir, cwd, build.FindOnly)
	if err != nil {
		return fmt.Errorf("unable to *find* %q: %v", dir, err)
	}

	if err := b.addDir(dir, true); err != nil {
		glog.Warningf("Ignoring directory %v: %v", dir, err)
	}

	prefix := strings.TrimSuffix(pkg.Dir, strings.TrimSuffix(dir, "/"))
	filepath.Walk(pkg.Dir, func(path string, info os.FileInfo, err error) error {
		if info != nil && info.IsDir() {
			trimmed := strings.TrimPrefix(path, prefix)
			if trimmed != "" {
				if err := b.addDir(trimmed, true); err != nil {
					glog.Warningf("Ignoring child directory %v: %v", trimmed, err)
				}
			}
		}
		return nil
	})
	return nil
}

// AddDirTo adds an entire directory to a given Universe. Unlike AddDir, this
// processes the package immediately, which makes it safe to use from within a
// generator (rather than just at init time. 'dir' must be a single go package.
// GOPATH, GOROOT, and the location of your go binary (`which go`) will all be
// searched if dir doesn't literally resolve.
func (b *Builder) AddDirTo(dir string, u *types.Universe) error {
	if _, found := b.parsed[dir]; !found {
		// We want all types from this package, as if they were directly added
		// by the user.  They WERE added by the user, in effect.
		if err := b.addDir(dir, true); err != nil {
			return err
		}
	} else {
		// We already had this package, but we want it to be considered as if
		// the user addid it directly.
		b.userRequested[dir] = true
	}
	return b.findTypesIn(dir, u)
}

// The implementation of AddDir. A flag indicates whether this directory was
// user-requested or just from following the import graph.
func (b *Builder) addDir(dir string, userRequested bool) error {
	pkg, err := b.buildPackage(dir)
	if err != nil {
		return err
	}
	// Check in case this package was added (maybe dir was not canonical)
	if wasRequested, wasAdded := b.userRequested[dir]; wasAdded {
		if !userRequested || userRequested == wasRequested {
			return nil
		}
	}

	for _, n := range pkg.GoFiles {
		if !strings.HasSuffix(n, ".go") {
			continue
		}
		absPath := filepath.Join(pkg.Dir, n)
		data, err := ioutil.ReadFile(absPath)
		if err != nil {
			return fmt.Errorf("while loading %q: %v", absPath, err)
		}
		err = b.addFile(dir, absPath, data, userRequested)
		if err != nil {
			return fmt.Errorf("while parsing %q: %v", absPath, err)
		}
	}
	return nil
}

// importer is a function that will be called by the type check package when it
// needs to import a go package. 'path' is the import path. go1.5 changes the
// interface, and importAdapter below implements the new interface in terms of
// the old one.
func (b *Builder) importer(imports map[string]*tc.Package, path string) (*tc.Package, error) {
	if pkg, ok := imports[path]; ok {
		return pkg, nil
	}
	ignoreError := false
	if _, ours := b.parsed[path]; !ours {
		// Ignore errors in paths that we're importing solely because
		// they're referenced by other packages.
		ignoreError = true
		if err := b.addDir(path, false); err != nil {
			return nil, err
		}
	}
	pkg, err := b.typeCheckPackage(path)
	if err != nil {
		if ignoreError && pkg != nil {
			glog.V(2).Infof("type checking encountered some errors in %q, but ignoring.\n", path)
		} else {
			return nil, err
		}
	}
	imports[path] = pkg
	return pkg, nil
}

type importAdapter struct {
	b *Builder
}

func (a importAdapter) Import(path string) (*tc.Package, error) {
	return a.b.importer(a.b.pkgs, path)
}

// typeCheckPackage will attempt to return the package even if there are some
// errors, so you may check whether the package is nil or not even if you get
// an error.
func (b *Builder) typeCheckPackage(id string) (*tc.Package, error) {
	if pkg, ok := b.pkgs[id]; ok {
		if pkg != nil {
			return pkg, nil
		}
		// We store a nil right before starting work on a package. So
		// if we get here and it's present and nil, that means there's
		// another invocation of this function on the call stack
		// already processing this package.
		return nil, fmt.Errorf("circular dependency for %q", id)
	}
	parsedFiles, ok := b.parsed[id]
	if !ok {
		return nil, fmt.Errorf("No files for pkg %q: %#v", id, b.parsed)
	}
	files := make([]*ast.File, len(parsedFiles))
	for i := range parsedFiles {
		files[i] = parsedFiles[i].file
	}
	b.pkgs[id] = nil
	c := tc.Config{
		IgnoreFuncBodies: true,
		// Note that importAdater can call b.import which calls this
		// method. So there can't be cycles in the import graph.
		Importer: importAdapter{b},
		Error: func(err error) {
			glog.V(2).Infof("type checker error: %v\n", err)
		},
	}
	pkg, err := c.Check(id, b.fset, files, nil)
	b.pkgs[id] = pkg // record the result whether or not there was an error
	return pkg, err
}

func (b *Builder) makeAllPackages() error {
	// Take a snapshot to iterate, since this will recursively mutate b.parsed.
	keys := []string{}
	for id := range b.parsed {
		keys = append(keys, id)
	}
	for _, id := range keys {
		if _, err := b.makePackage(id); err != nil {
			return err
		}
	}
	return nil
}

func (b *Builder) makePackage(id string) (*tc.Package, error) {
	if b.pkgs == nil {
		b.pkgs = map[string]*tc.Package{}
	}

	// We have to check here even though we made a new one above,
	// because typeCheckPackage follows the import graph, which may
	// cause a package to be filled before we get to it in this
	// loop.
	if pkg, done := b.pkgs[id]; done {
		return pkg, nil
	}
	return b.typeCheckPackage(id)
}

// FindPackages fetches a list of the user-imported packages.
func (b *Builder) FindPackages() []string {
	result := []string{}
	for pkgPath := range b.pkgs {
		if b.userRequested[pkgPath] {
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
	if err := b.makeAllPackages(); err != nil {
		return nil, err
	}

	u := types.Universe{}

	for pkgPath := range b.parsed {
		if err := b.findTypesIn(pkgPath, &u); err != nil {
			return nil, err
		}
	}
	return u, nil
}

// findTypesIn finalizes the package import and searches through the package
// for types.
func (b *Builder) findTypesIn(pkgPath string, u *types.Universe) error {
	pkg, err := b.makePackage(pkgPath)
	if err != nil {
		return err
	}
	if !b.userRequested[pkgPath] {
		// Since walkType is recursive, all types that the
		// packages they asked for depend on will be included.
		// But we don't need to include all types in all
		// *packages* they depend on.
		return nil
	}

	for _, f := range b.parsed[pkgPath] {
		if strings.HasSuffix(f.name, "/doc.go") {
			tp := u.Package(pkgPath)
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
			c1 := b.priorCommentLines(obj.Pos(), 1)
			t.CommentLines = splitLines(c1.Text())
			if c1 == nil {
				t.SecondClosestCommentLines = splitLines(b.priorCommentLines(obj.Pos(), 2).Text())
			} else {
				t.SecondClosestCommentLines = splitLines(b.priorCommentLines(c1.List[0].Slash, 2).Text())
			}
		}
		tf, ok := obj.(*tc.Func)
		// We only care about functions, not concrete/abstract methods.
		if ok && tf.Type() != nil && tf.Type().(*tc.Signature).Recv() == nil {
			b.addFunction(*u, nil, tf)
		}
		tv, ok := obj.(*tc.Var)
		if ok && !tv.IsField() {
			b.addVariable(*u, nil, tv)
		}
	}
	for p := range b.importGraph[pkgPath] {
		u.AddImports(pkgPath, p)
	}
	u.Package(pkgPath).Name = pkg.Name()
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
	name := strings.TrimLeft(in, "func ")
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
			out.Methods[t.Method(i).Name()] = b.walkType(u, nil, t.Method(i).Type())
		}
		return out
	case *tc.Named:
		switch t.Underlying().(type) {
		case *tc.Named, *tc.Basic, *tc.Map, *tc.Slice:
			name := tcNameToName(t.String())
			out := u.Type(name)
			if out.Kind != types.Unknown {
				return out
			}
			out.Kind = types.Alias
			out.Underlying = b.walkType(u, nil, t.Underlying())
			return out
		default:
			// tc package makes everything "named" with an
			// underlying anonymous type--we remove that annoying
			// "feature" for users. This flattens those types
			// together.
			name := tcNameToName(t.String())
			if out := u.Type(name); out.Kind != types.Unknown {
				return out // short circuit if we've already made this.
			}
			out := b.walkType(u, &name, t.Underlying())
			if len(out.Methods) == 0 {
				// If the underlying type didn't already add
				// methods, add them. (Interface types will
				// have already added methods.)
				for i := 0; i < t.NumMethods(); i++ {
					if out.Methods == nil {
						out.Methods = map[string]*types.Type{}
					}
					out.Methods[t.Method(i).Name()] = b.walkType(u, nil, t.Method(i).Type())
				}
			}
			return out
		}
	default:
		out := u.Type(name)
		if out.Kind != types.Unknown {
			return out
		}
		out.Kind = types.Unsupported
		glog.Warningf("Making unsupported type entry %q for: %#v\n", out, t)
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
