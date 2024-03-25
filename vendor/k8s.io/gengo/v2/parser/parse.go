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
	gotypes "go/types"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"golang.org/x/tools/go/packages"
	"k8s.io/gengo/v2/types"
	"k8s.io/klog/v2"
)

// Parser lets you add all the go files in all the packages that you care
// about, then constructs the type source data.
type Parser struct {
	// Map of package paths to definitions.  These keys should be canonical
	// Go import paths (example.com/foo/bar) and not local paths (./foo/bar).
	goPkgs map[string]*packages.Package

	// Keep track of which packages were directly requested (as opposed to
	// those which are transitively loaded).
	userRequested map[string]bool

	// Keep track of which packages have already been scanned for types.
	fullyProcessed map[string]bool

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

// New constructs a new Parser.
func New() *Parser {
	return NewWithOptions(Options{})
}

func NewWithOptions(opts Options) *Parser {
	return &Parser{
		goPkgs:                map[string]*packages.Package{},
		userRequested:         map[string]bool{},
		fullyProcessed:        map[string]bool{},
		fset:                  token.NewFileSet(),
		endLineToCommentGroup: map[fileLine]*ast.CommentGroup{},
		buildTags:             opts.BuildTags,
	}
}

// Options holds optional settings for the Parser.
type Options struct {
	// BuildTags is a list of optional tags to be specified when loading
	// packages.
	BuildTags []string
}

// FindPackages expands the provided patterns into a list of Go import-paths,
// much like `go list -find`.
func (p *Parser) FindPackages(patterns ...string) ([]string, error) {
	return p.findPackages(nil, patterns...)
}

// baseCfg is an optional (may be nil) config which might be injected by tests.
func (p *Parser) findPackages(baseCfg *packages.Config, patterns ...string) ([]string, error) {
	toFind := make([]string, 0, len(patterns))
	results := make([]string, 0, len(patterns))
	for _, pat := range patterns {
		if pkg := p.goPkgs[pat]; pkg != nil {
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
		BuildFlags: []string{"-tags", strings.Join(p.buildTags, ",")},
		Tests:      false,
	}
	if baseCfg != nil {
		// This is to support tests, e.g. to inject a fake GOPATH or CWD.
		cfg.Dir = baseCfg.Dir
		cfg.Env = baseCfg.Env
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
func (p *Parser) LoadPackages(patterns ...string) error {
	_, err := p.loadPackages(patterns...)
	return err
}

// LoadPackagesWithConfigForTesting loads and parses the specified Go packages with the
// specified packages.Config as a starting point.  This is for testing, and
// only the .Dir and .Env fields of the Config will be considered.
func (p *Parser) LoadPackagesWithConfigForTesting(cfg *packages.Config, patterns ...string) error {
	_, err := p.loadPackagesWithConfig(cfg, patterns...)
	return err
}

// LoadPackagesTo loads and parses the specified Go packages, and inserts them
// into the specified Universe. It returns the packages which match the
// patterns, but loads all packages and their imports, recursively, into the
// universe.  See NewUniverse for more.
func (p *Parser) LoadPackagesTo(u *types.Universe, patterns ...string) ([]*types.Package, error) {
	// Load Packages.
	pkgs, err := p.loadPackages(patterns...)
	if err != nil {
		return nil, err
	}

	// Load types in all packages (it will internally filter).
	if err := p.addPkgsToUniverse(pkgs, u); err != nil {
		return nil, err
	}

	// Return the results as gengo types.Packages.
	ret := make([]*types.Package, 0, len(pkgs))
	for _, pkg := range pkgs {
		ret = append(ret, u.Package(pkg.PkgPath))
	}

	return ret, nil
}

func (p *Parser) loadPackages(patterns ...string) ([]*packages.Package, error) {
	return p.loadPackagesWithConfig(nil, patterns...)
}

// baseCfg is an optional (may be nil) config which might be injected by tests.
func (p *Parser) loadPackagesWithConfig(baseCfg *packages.Config, patterns ...string) ([]*packages.Package, error) {
	klog.V(5).Infof("loadPackages %q", patterns)

	// Loading packages is slow - only do ones we know we have not already done
	// (e.g. if a tool calls LoadPackages itself).
	existingPkgs, netNewPkgs, err := p.alreadyLoaded(baseCfg, patterns...)
	if err != nil {
		return nil, err
	}
	if vlog := klog.V(5); vlog.Enabled() {
		if len(existingPkgs) > 0 {
			keys := make([]string, 0, len(existingPkgs))
			for _, p := range existingPkgs {
				keys = append(keys, p.PkgPath)
			}
			vlog.Infof("  already have: %q", keys)
		}
		if len(netNewPkgs) > 0 {
			vlog.Infof("  to be loaded: %q", netNewPkgs)
		}
	}

	// If these were not user-requested before, they are now.
	for _, pkg := range existingPkgs {
		if !p.userRequested[pkg.PkgPath] {
			p.userRequested[pkg.PkgPath] = true
		}
	}
	for _, pkg := range netNewPkgs {
		if !p.userRequested[pkg] {
			p.userRequested[pkg] = true
		}
	}

	if len(netNewPkgs) == 0 {
		return existingPkgs, nil
	}

	cfg := packages.Config{
		Mode: packages.NeedName |
			packages.NeedFiles | packages.NeedImports | packages.NeedDeps |
			packages.NeedModule | packages.NeedTypes | packages.NeedSyntax,
		BuildFlags: []string{"-tags", strings.Join(p.buildTags, ",")},
		Fset:       p.fset,
		Tests:      false,
	}
	if baseCfg != nil {
		// This is to support tests, e.g. to inject a fake GOPATH or CWD.
		cfg.Dir = baseCfg.Dir
		cfg.Env = baseCfg.Env
	}

	tBefore := time.Now()
	pkgs, err := packages.Load(&cfg, netNewPkgs...)
	if err != nil {
		return nil, fmt.Errorf("error loading packages: %w", err)
	}
	klog.V(5).Infof("  loaded %d pkg(s) in %v", len(pkgs), time.Since(tBefore))

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
		p.goPkgs[pkg.PkgPath] = pkg

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
				position := p.fset.Position(c.End()) // Fset is synchronized
				p.endLineToCommentGroup[fileLine{position.Filename, position.Line}] = c
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
// baseCfg is an optional (may be nil) config which might be injected by tests.
func (p *Parser) alreadyLoaded(baseCfg *packages.Config, patterns ...string) ([]*packages.Package, []string, error) {
	existingPkgs := make([]*packages.Package, 0, len(patterns))
	netNewPkgs := make([]string, 0, len(patterns))

	// Expand and canonicalize the requested patterns.  This should be fast.
	if pkgPaths, err := p.findPackages(baseCfg, patterns...); err != nil {
		return nil, nil, err
	} else {
		for _, pkgPath := range pkgPaths {
			if pkg := p.goPkgs[pkgPath]; pkg != nil {
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
func (p *Parser) UserRequestedPackages() []string {
	// Iterate packages in a predictable order.
	pkgPaths := make([]string, 0, len(p.userRequested))
	for k := range p.userRequested {
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
func (p *Parser) NewUniverse() (types.Universe, error) {
	u := types.Universe{}

	pkgs := []*packages.Package{}
	for _, path := range p.UserRequestedPackages() {
		pkgs = append(pkgs, p.goPkgs[path])
	}
	if err := p.addPkgsToUniverse(pkgs, &u); err != nil {
		return nil, err
	}

	return u, nil
}

// addCommentsToType takes any accumulated comment lines prior to obj and
// attaches them to the type t.
func (p *Parser) addCommentsToType(obj gotypes.Object, t *types.Type) {
	t.CommentLines = p.docComment(obj.Pos())
	t.SecondClosestCommentLines = p.priorDetachedComment(obj.Pos())
}

// packageDir tries to figure out the directory of the specified package.
func packageDir(pkg *packages.Package) (string, error) {
	// Sometimes Module is present but has no Dir, e.g. when it is vendored.
	if pkg.Module != nil && pkg.Module.Dir != "" {
		// NOTE: this will not work if tests are loaded, because Go mutates the
		// Package.PkgPath.
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
func (p *Parser) addPkgsToUniverse(pkgs []*packages.Package, u *types.Universe) error {
	addOne := func(pkg *packages.Package) error {
		if err := p.addPkgToUniverse(pkg, u); err != nil {
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
func (p *Parser) addPkgToUniverse(pkg *packages.Package, u *types.Universe) error {
	pkgPath := pkg.PkgPath
	if p.fullyProcessed[pkgPath] {
		return nil
	}

	// This will get-or-create the Package.
	gengoPkg := u.Package(pkgPath)

	if gengoPkg.Dir == "" {
		// We're keeping this package, though we might not fully process it.
		if vlog := klog.V(5); vlog.Enabled() {
			why := "user-requested"
			if !p.userRequested[pkgPath] {
				why = "dependency"
			}
			vlog.Infof("addPkgToUniverse %q (%s)", pkgPath, why)
		}

		absPath := ""
		if dir, err := packageDir(pkg); err != nil {
			return err
		} else {
			absPath = dir
		}

		gengoPkg.Path = pkg.PkgPath
		gengoPkg.Dir = absPath
	}

	// If the package was not user-requested, we can stop here.
	if !p.userRequested[pkgPath] {
		return nil
	}

	// Mark it as done, so we don't ever re-process it.
	p.fullyProcessed[pkgPath] = true
	gengoPkg.Name = pkg.Name

	// For historical reasons we treat files named "doc.go" specially.
	// TODO: It would be nice to not do this and instead treat package
	// doc-comments as the "global" config place.  This would require changing
	// most generators and input files.
	for _, f := range pkg.Syntax {
		// This gets the filename for the ast.File.  Iterating pkg.GoFiles is
		// documented as unreliable.
		pos := p.fset.Position(f.FileStart)
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
		switch obj := s.Lookup(n).(type) {
		case *gotypes.TypeName:
			t := p.walkType(*u, nil, obj.Type())
			p.addCommentsToType(obj, t)
		case *gotypes.Func:
			// We only care about functions, not concrete/abstract methods.
			if obj.Type() != nil && obj.Type().(*gotypes.Signature).Recv() == nil {
				t := p.addFunction(*u, nil, obj)
				p.addCommentsToType(obj, t)
			}
		case *gotypes.Var:
			if !obj.IsField() {
				t := p.addVariable(*u, nil, obj)
				p.addCommentsToType(obj, t)
			}
		case *gotypes.Const:
			t := p.addConstant(*u, nil, obj)
			p.addCommentsToType(obj, t)
		default:
			klog.Infof("addPkgToUniverse %q: unhandled object of type %T: %v", pkgPath, obj, obj)
		}
	}

	// Add all of this package's imports.
	importedPkgs := []string{}
	for _, imp := range pkg.Imports {
		if err := p.addPkgToUniverse(imp, u); err != nil {
			return err
		}
		importedPkgs = append(importedPkgs, imp.PkgPath)
	}
	sort.Strings(importedPkgs)
	u.AddImports(pkg.PkgPath, importedPkgs...)

	return nil
}

// If the specified position has a "doc comment", return that.
func (p *Parser) docComment(pos token.Pos) []string {
	// An object's doc comment always ends on the line before the object's own
	// declaration.
	c1 := p.priorCommentLines(pos, 1)
	return splitLines(c1.Text()) // safe even if c1 is nil
}

// If there is a detached (not immediately before a declaration) comment,
// return that.
func (p *Parser) priorDetachedComment(pos token.Pos) []string {
	// An object's doc comment always ends on the line before the object's own
	// declaration.
	c1 := p.priorCommentLines(pos, 1)

	// Using a literal "2" here is brittle in theory (it means literally 2
	// lines), but in practice Go code is gofmt'ed (which elides repeated blank
	// lines), so it works.
	var c2 *ast.CommentGroup
	if c1 == nil {
		c2 = p.priorCommentLines(pos, 2)
	} else {
		c2 = p.priorCommentLines(c1.List[0].Slash, 2)
	}
	return splitLines(c2.Text()) // safe even if c1 is nil
}

// If there's a comment block which ends nlines before pos, return it.
func (p *Parser) priorCommentLines(pos token.Pos, lines int) *ast.CommentGroup {
	position := p.fset.Position(pos)
	key := fileLine{position.Filename, position.Line - lines}
	return p.endLineToCommentGroup[key]
}

func splitLines(str string) []string {
	return strings.Split(strings.TrimRight(str, "\n"), "\n")
}

func goFuncNameToName(in string) types.Name {
	name := strings.TrimPrefix(in, "func ")
	nameParts := strings.Split(name, "(")
	return goNameToName(nameParts[0])
}

func goVarNameToName(in string) types.Name {
	nameParts := strings.Split(in, " ")
	// nameParts[0] is "var".
	// nameParts[2:] is the type of the variable, we ignore it for now.
	return goNameToName(nameParts[1])
}

func goNameToName(in string) types.Name {
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

func (p *Parser) convertSignature(u types.Universe, t *gotypes.Signature) *types.Signature {
	signature := &types.Signature{}
	for i := 0; i < t.Params().Len(); i++ {
		signature.Parameters = append(signature.Parameters, p.walkType(u, nil, t.Params().At(i).Type()))
		signature.ParameterNames = append(signature.ParameterNames, t.Params().At(i).Name())
	}
	for i := 0; i < t.Results().Len(); i++ {
		signature.Results = append(signature.Results, p.walkType(u, nil, t.Results().At(i).Type()))
		signature.ResultNames = append(signature.ResultNames, t.Results().At(i).Name())
	}
	if r := t.Recv(); r != nil {
		signature.Receiver = p.walkType(u, nil, r.Type())
	}
	signature.Variadic = t.Variadic()
	return signature
}

// walkType adds the type, and any necessary child types.
func (p *Parser) walkType(u types.Universe, useName *types.Name, in gotypes.Type) *types.Type {
	// Most of the cases are underlying types of the named type.
	name := goNameToName(in.String())
	if useName != nil {
		name = *useName
	}

	switch t := in.(type) {
	case *gotypes.Struct:
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
				Type:         p.walkType(u, nil, f.Type()),
				CommentLines: p.docComment(f.Pos()),
			}
			out.Members = append(out.Members, m)
		}
		return out
	case *gotypes.Map:
		out := u.Type(name)
		if out.Kind != types.Unknown {
			return out
		}
		out.Kind = types.Map
		out.Elem = p.walkType(u, nil, t.Elem())
		out.Key = p.walkType(u, nil, t.Key())
		return out
	case *gotypes.Pointer:
		out := u.Type(name)
		if out.Kind != types.Unknown {
			return out
		}
		out.Kind = types.Pointer
		out.Elem = p.walkType(u, nil, t.Elem())
		return out
	case *gotypes.Slice:
		out := u.Type(name)
		if out.Kind != types.Unknown {
			return out
		}
		out.Kind = types.Slice
		out.Elem = p.walkType(u, nil, t.Elem())
		return out
	case *gotypes.Array:
		out := u.Type(name)
		if out.Kind != types.Unknown {
			return out
		}
		out.Kind = types.Array
		out.Elem = p.walkType(u, nil, t.Elem())
		out.Len = in.(*gotypes.Array).Len()
		return out
	case *gotypes.Chan:
		out := u.Type(name)
		if out.Kind != types.Unknown {
			return out
		}
		out.Kind = types.Chan
		out.Elem = p.walkType(u, nil, t.Elem())
		// TODO: need to store direction, otherwise raw type name
		// cannot be properly written.
		return out
	case *gotypes.Basic:
		out := u.Type(types.Name{
			Package: "", // This is a magic package name in the Universe.
			Name:    t.Name(),
		})
		if out.Kind != types.Unknown {
			return out
		}
		out.Kind = types.Unsupported
		return out
	case *gotypes.Signature:
		out := u.Type(name)
		if out.Kind != types.Unknown {
			return out
		}
		out.Kind = types.Func
		out.Signature = p.convertSignature(u, t)
		return out
	case *gotypes.Interface:
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
			name := goNameToName(method.String())
			mt := p.walkType(u, &name, method.Type())
			mt.CommentLines = p.docComment(method.Pos())
			out.Methods[method.Name()] = mt
		}
		return out
	case *gotypes.Named:
		var out *types.Type
		switch t.Underlying().(type) {
		case *gotypes.Named, *gotypes.Basic, *gotypes.Map, *gotypes.Slice:
			name := goNameToName(t.String())
			out = u.Type(name)
			if out.Kind != types.Unknown {
				return out
			}
			out.Kind = types.Alias
			out.Underlying = p.walkType(u, nil, t.Underlying())
		default:
			// gotypes package makes everything "named" with an
			// underlying anonymous type--we remove that annoying
			// "feature" for users. This flattens those types
			// together.
			name := goNameToName(t.String())
			if out := u.Type(name); out.Kind != types.Unknown {
				return out // short circuit if we've already made this.
			}
			out = p.walkType(u, &name, t.Underlying())
		}
		// If the underlying type didn't already add methods, add them.
		// (Interface types will have already added methods.)
		if len(out.Methods) == 0 {
			for i := 0; i < t.NumMethods(); i++ {
				if out.Methods == nil {
					out.Methods = map[string]*types.Type{}
				}
				method := t.Method(i)
				name := goNameToName(method.String())
				mt := p.walkType(u, &name, method.Type())
				mt.CommentLines = p.docComment(method.Pos())
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

func (p *Parser) addFunction(u types.Universe, useName *types.Name, in *gotypes.Func) *types.Type {
	name := goFuncNameToName(in.String())
	if useName != nil {
		name = *useName
	}
	out := u.Function(name)
	out.Kind = types.DeclarationOf
	out.Underlying = p.walkType(u, nil, in.Type())
	return out
}

func (p *Parser) addVariable(u types.Universe, useName *types.Name, in *gotypes.Var) *types.Type {
	name := goVarNameToName(in.String())
	if useName != nil {
		name = *useName
	}
	out := u.Variable(name)
	out.Kind = types.DeclarationOf
	out.Underlying = p.walkType(u, nil, in.Type())
	return out
}

func (p *Parser) addConstant(u types.Universe, useName *types.Name, in *gotypes.Const) *types.Type {
	name := goVarNameToName(in.String())
	if useName != nil {
		name = *useName
	}
	out := u.Constant(name)
	out.Kind = types.DeclarationOf
	out.Underlying = p.walkType(u, nil, in.Type())

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
