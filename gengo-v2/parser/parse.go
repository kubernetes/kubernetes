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
	"errors"
	"fmt"
	"go/ast"
	"go/constant"
	"go/token"
	tc "go/types"
	"path/filepath"
	"sort"
	"strings"

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

	// Map of package paths to definitions.  These keys should be canonical
	// Go import paths (example.com/foo/bar) and not local paths (./foo/bar).
	goPkgs map[importPathString]*packages.Package

	// Keep track of which packages were directly requested (as opposed to
	// those which are transitively loaded).
	userRequested map[importPathString]bool

	// Keep track of which packages have already been scanned for types.
	fullyProcessed map[importPathString]bool

	// Build tags to set when loading packages.
	buildTags []string

	// Tracks accumulated parsed files, so we can do position lookups later.
	fset *token.FileSet

	// All comments from everywhere in every parsed file.  This map is keyed by
	// the file-line on which the comment block ends, which makes it easy to
	// look up comments which immediately precede a given obect (e.g. a type or
	// function definition), which is what we almost always want.  We need this
	// because Go's own ast package does a very poor job of handling comments.
	endLineToCommentGroup map[fileLine]*ast.CommentGroup
}

// key type for finding comments.
type fileLine struct {
	file string
	line int
}

// New constructs a new builder.
func New() *Builder {
	return &Builder{
		goPkgs:                map[importPathString]*packages.Package{},
		userRequested:         map[importPathString]bool{},
		fullyProcessed:        map[importPathString]bool{},
		fset:                  token.NewFileSet(),
		endLineToCommentGroup: map[fileLine]*ast.CommentGroup{},
	}
}

// AddBuildTags adds the specified build tags for subsequent package loads.
func (b *Builder) AddBuildTags(tags ...string) {
	b.buildTags = append(b.buildTags, tags...)
}

// FindPackages expands the provided patterns into a list of Go import-paths,
// much like `go list -find`.
func (b *Builder) FindPackages(patterns ...string) ([]string, error) {
	toFind := make([]string, 0, len(patterns))
	results := make([]string, 0, len(patterns))
	for _, pat := range patterns {
		ip := importPathString(pat)
		if pkg := b.goPkgs[ip]; pkg != nil {
			results = append(results, pkg.PkgPath)
		} else {
			toFind = append(toFind, pat)
		}
	}
	if len(toFind) == 0 {
		return results, nil
	}

	cfg := packages.Config{
		Mode:       packages.NeedName | packages.NeedFiles,
		BuildFlags: []string{"-tags", strings.Join(b.buildTags, ",")},
	}
	pkgs, err := packages.Load(&cfg, toFind...)
	if err != nil {
		return nil, fmt.Errorf("error loading packages: %w", err)
	}
	var allErrs []error
	for _, pkg := range pkgs {
		results = append(results, pkg.PkgPath)

		// pkg.Errors is not a slice of `error`, but concrete types.  We have
		// to iteratively convert each one into `error`.
		var errs []error
		for _, e := range pkg.Errors {
			errs = append(errs, e)
		}
		if len(errs) > 0 {
			allErrs = append(allErrs, fmt.Errorf("error(s) in %q:\n%w", pkg.PkgPath, errors.Join(errs...)))
		}
	}
	if len(allErrs) != 0 {
		return nil, errors.Join(allErrs...)
	}
	return results, nil
}

// LoadPackages loads and parses the specified Go packages.  Specifically
// named packages (without a trailing "/...") which do not exist or have no Go
// files are an error.
func (b *Builder) LoadPackages(patterns ...string) error {
	_, err := b.loadPackages(patterns...)
	return err
}

// LoadPackagesTo loads and parses the specified Go packages, and inserts them
// into the specified Universe. It returns the packages which match the
// patterns, but loads all packages and their imports, recursively, into the
// universe.  See NewUniverse for more.
func (b *Builder) LoadPackagesTo(u *types.Universe, patterns ...string) ([]*types.Package, error) {
	// Load Packages.
	pkgs, err := b.loadPackages(patterns...)
	if err != nil {
		return nil, err
	}

	// Load types in all packages (it will internally filter).
	if err := b.addPkgsToUniverse(pkgs, u); err != nil {
		return nil, err
	}

	// Return the results as gengo types.Packages.
	ret := make([]*types.Package, 0, len(pkgs))
	for _, pkg := range pkgs {
		ret = append(ret, u.Package(pkg.PkgPath))
	}

	return ret, nil
}

func (b *Builder) loadPackages(patterns ...string) ([]*packages.Package, error) {
	// Loading packages is slow - only do ones we know we have not already done
	// (e.g. if a tool calls LoadPackages itself).
	existingPkgs, netNewPkgs, err := b.alreadyLoaded(patterns...)
	if err != nil {
		return nil, err
	}

	// If these were not user-requested before, they are now.
	for _, pkg := range existingPkgs {
		ip := importPathString(pkg.PkgPath)
		if !b.userRequested[ip] {
			b.userRequested[ip] = true
		}
	}
	for _, pkgPath := range netNewPkgs {
		ip := importPathString(pkgPath)
		if !b.userRequested[ip] {
			b.userRequested[ip] = true
		}
	}

	if len(netNewPkgs) == 0 {
		return existingPkgs, nil
	}

	cfg := packages.Config{
		Mode: packages.NeedName |
			packages.NeedFiles | packages.NeedImports | packages.NeedDeps |
			packages.NeedModule | packages.NeedTypes | packages.NeedSyntax,
		Tests:      b.IncludeTestFiles,
		BuildFlags: []string{"-tags", strings.Join(b.buildTags, ",")},
		Fset:       b.fset,
	}

	pkgs, err := packages.Load(&cfg, netNewPkgs...)
	if err != nil {
		return nil, fmt.Errorf("error loading packages: %w", err)
	}

	// Handle any errors.
	collectErrors := func(pkg *packages.Package) error {
		var errs []error
		for _, e := range pkg.Errors {
			if e.Kind == packages.ListError || e.Kind == packages.ParseError {
				errs = append(errs, e)
			}
		}
		if len(errs) > 0 {
			return fmt.Errorf("error(s) in %q:\n%w", pkg.PkgPath, errors.Join(errs...))
		}
		return nil
	}
	if err := forEachPackageRecursive(pkgs, collectErrors); err != nil {
		return nil, err
	}

	// Finish integrating packages into our state.
	absorbPkg := func(pkg *packages.Package) error {
		pkgPath := importPathString(pkg.PkgPath)
		b.goPkgs[pkgPath] = pkg

		for _, f := range pkg.Syntax {
			for _, c := range f.Comments {
				// We need to do this on _every_ pkg, not just user-requested
				// ones, because some generators look at tags in other
				// packages.
				//
				// TODO: It would be nice if we only did this on user-requested
				// packages.  The problem is that we don't always know which
				// other packages will need this information, and even when we
				// do we may have already loaded the package (as a transitive
				// dep) and might have stored pointers into it.  Doing a
				// thorough "reload" without invalidating all those pointers is
				// a problem for another day.
				position := b.fset.Position(c.End()) // Fset is synchronized
				b.endLineToCommentGroup[fileLine{position.Filename, position.Line}] = c
			}
		}

		return nil
	}
	if err := forEachPackageRecursive(pkgs, absorbPkg); err != nil {
		return nil, err
	}

	return append(existingPkgs, pkgs...), nil
}

// alreadyLoaded figures out which of the specified patterns have already been loaded
// and which have not, and returns those respectively.
func (b *Builder) alreadyLoaded(patterns ...string) ([]*packages.Package, []string, error) {
	existingPkgs := make([]*packages.Package, 0, len(patterns))
	netNewPkgs := make([]string, 0, len(patterns))

	// Expand and canonicalize the requested patterns.  This should be fast.
	if pkgPaths, err := b.FindPackages(patterns...); err != nil {
		return nil, nil, err
	} else {
		for _, pkgPath := range pkgPaths {
			ip := importPathString(pkgPath)
			if pkg := b.goPkgs[ip]; pkg != nil {
				existingPkgs = append(existingPkgs, pkg)
			} else {
				netNewPkgs = append(netNewPkgs, pkgPath)
			}
		}
	}
	return existingPkgs, netNewPkgs, nil
}

// forEachPackageRecursive will run the provided function on all of the specified
// packages, and on their imports recursively.  Errors are accumulated and
// returned as via errors.Join.
func forEachPackageRecursive(pkgs []*packages.Package, fn func(pkg *packages.Package) error) error {
	seen := map[string]bool{} // PkgPaths we have already visited
	var errs []error
	for _, pkg := range pkgs {
		errs = append(errs, recursePackage(pkg, fn, seen)...)
	}
	if len(errs) > 0 {
		return errors.Join(errs...)
	}
	return nil
}

func recursePackage(pkg *packages.Package, fn func(pkg *packages.Package) error, seen map[string]bool) []error {
	if seen[pkg.PkgPath] {
		return nil
	}
	var errs []error
	seen[pkg.PkgPath] = true
	if err := fn(pkg); err != nil {
		errs = append(errs, err)
	}
	for _, imp := range pkg.Imports {
		errs = append(errs, recursePackage(imp, fn, seen)...)
	}
	return errs
}

// UserRequestedPackages fetches a list of the user-imported packages.
func (b *Builder) UserRequestedPackages() []string {
	// Iterate packages in a predictable order.
	pkgPaths := make([]string, 0, len(b.userRequested))
	for k := range b.userRequested {
		pkgPaths = append(pkgPaths, string(k))
	}
	sort.Strings(pkgPaths)
	return pkgPaths
}

// NewUniverse finalizes the loaded packages, searches through them for types
// and produces a new Universe. The returned Universe has one types.Package
// entry for each Go package that has been loaded, including all of their
// dependencies, recursively.  It also has one entry, whose key is "", which
// represents "builtin" types.
func (b *Builder) NewUniverse() (types.Universe, error) {
	u := types.Universe{}

	pkgs := []*packages.Package{}
	for _, path := range b.UserRequestedPackages() {
		pkgs = append(pkgs, b.goPkgs[importPathString(path)])
	}
	if err := b.addPkgsToUniverse(pkgs, &u); err != nil {
		return nil, err
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

// packageDir tries to figure out the directory of the specified package.
func packageDir(pkg *packages.Package) (string, error) {
	// Sometimes Module is present but has no Dir, e.g. when it is vendored.
	if pkg.Module != nil && pkg.Module.Dir != "" {
		subdir := strings.TrimPrefix(pkg.PkgPath, pkg.Module.Path)
		return filepath.Join(pkg.Module.Dir, subdir), nil
	}
	if len(pkg.GoFiles) > 0 {
		return filepath.Dir(pkg.GoFiles[0]), nil
	}
	if len(pkg.IgnoredFiles) > 0 {
		return filepath.Dir(pkg.IgnoredFiles[0]), nil
	}
	return "", fmt.Errorf("can't find package dir for %q - no module info and no Go files", pkg.PkgPath)
}

// addPkgsToUniverse adds the packages, and all of their deps, recursively, to
// the universe and (if needed) searches through them for types.
func (b *Builder) addPkgsToUniverse(pkgs []*packages.Package, u *types.Universe) error {
	addOne := func(pkg *packages.Package) error {
		if err := b.addPkgToUniverse(pkg, u); err != nil {
			return err
		}
		return nil
	}
	if err := forEachPackageRecursive(pkgs, addOne); err != nil {
		return err
	}
	return nil
}

// addPkgToUniverse adds one package to the universe and (if needed) searches
// through it for types.
func (b *Builder) addPkgToUniverse(pkg *packages.Package, u *types.Universe) error {
	pkgPath := importPathString(pkg.PkgPath)
	if b.fullyProcessed[pkgPath] {
		return nil
	}

	// We're keeping this package, though we might not fully process it.
	klog.V(5).Infof("addPkgToUniverse %q", pkgPath)

	absPath := ""
	if dir, err := packageDir(pkg); err != nil {
		return err
	} else {
		absPath = dir
	}

	// This will get-or-create the Package.
	gengoPkg := u.Package(string(pkgPath))
	gengoPkg.Path = pkg.PkgPath
	gengoPkg.SourcePath = absPath

	// If the package was not user-requested, we can stop here.
	if !b.userRequested[pkgPath] {
		return nil
	}

	// Mark it as done, so we don't ever re-process it.
	b.fullyProcessed[pkgPath] = true
	gengoPkg.Name = pkg.Name

	// For historical reasons we treat files named "doc.go" specially.
	// TODO: It would be nice to not do this and instead treat package
	// doc-comments as the "global" config place.  This would require changing
	// most generators and input files.
	for _, f := range pkg.Syntax {
		// This gets the filename for the ast.File.  Iterating pkg.GoFiles is
		// documented as unreliable.
		pos := b.fset.Position(f.FileStart)
		if filepath.Base(pos.Filename) == "doc.go" {
			gengoPkg.Comments = []string{}
			for i := range f.Comments {
				gengoPkg.Comments = append(gengoPkg.Comments, splitLines(f.Comments[i].Text())...)
			}
			if f.Doc != nil {
				gengoPkg.DocComments = splitLines(f.Doc.Text())
			}
		}
	}

	// Walk all the types, recursively and save them for later access.
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

	// Add all of this package's imports.
	importedPkgs := []string{}
	for _, imp := range pkg.Imports {
		if err := b.addPkgToUniverse(imp, u); err != nil {
			return err
		}
		importedPkgs = append(importedPkgs, imp.PkgPath)
	}
	sort.Strings(importedPkgs)
	u.AddImports(pkg.PkgPath, importedPkgs...)

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
			Package: "", // This is a magic package name in the Universe.
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
