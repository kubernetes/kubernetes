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
	"go/constant"
	"go/parser"
	"go/token"
	tc "go/types"
	"os"
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
	// Map of package ID to definitions.  These keys should be canonical
	// Go pkg names (example.com/foo/bar) and not local paths (./foo/bar).
	pkgMap map[importPathString]*packages.Package

	// Map of package ID to whether the user requested it or it was from
	// an import.
	userRequested map[importPathString]bool

	// Tracks accumulated parsed files, so we don't have to re-parse.
	fset *token.FileSet

	// If true, include *_test.go
	includeTestFiles bool

	// Tags to use when parsing code.
	buildTags []string

	// All comments from everywhere in every parsed file.
	commentLineLock       sync.Mutex
	endLineToCommentGroup map[fileLine]*ast.CommentGroup
}

// key type for finding comments.
type fileLine struct {
	file string
	line int
}

// New constructs a new builder.
func New(includeTestFiles bool, buildTags ...string) *Builder {
	return &Builder{
		pkgMap:                map[importPathString]*packages.Package{},
		userRequested:         map[importPathString]bool{},
		fset:                  token.NewFileSet(),
		endLineToCommentGroup: map[fileLine]*ast.CommentGroup{},
		includeTestFiles:      includeTestFiles,
		buildTags:             buildTags,
	}
}

//FIXME: comment
func (b *Builder) AddDirs(dirs []string) error {
	_, err := b.loadDirs(dirs)
	return err
}

func (b *Builder) loadDirs(dirs []string) ([]*packages.Package, error) {
	netNewDirs := make([]string, 0, len(dirs))
	existingPkgs := make([]*packages.Package, 0, len(dirs))
	for _, d := range dirs {
		ip := importPathString(d)
		if b.pkgMap[ip] == nil {
			netNewDirs = append(netNewDirs, d)
		} else {
			//FIXME: streamline wrt processPkg - sett8ing in 2 plaaces is ick
			b.userRequested[ip] = true
			existingPkgs = append(existingPkgs, b.pkgMap[ip])
		}
	}
	if len(netNewDirs) == 0 {
		return existingPkgs, nil
	}

	//FIXME: are all these "need" right?
	//FIXME: does -i take directories, pkgs, or both?  e.g. sahould I add leading ./ ?
	cfg := packages.Config{
		Mode:       packages.NeedName | packages.NeedFiles | packages.NeedImports | packages.NeedDeps | packages.NeedModule | packages.NeedTypes | packages.NeedTypesSizes | packages.NeedTypesInfo | packages.NeedSyntax,
		Tests:      b.includeTestFiles,
		BuildFlags: []string{"-tags", strings.Join(b.buildTags, ",")},
		Fset:       b.fset,
		ParseFile: func(fset *token.FileSet, filename string, src []byte) (*ast.File, error) {
			const mode = parser.DeclarationErrors | parser.ParseComments
			p, err := parser.ParseFile(b.fset, filename, src, mode)
			if err != nil {
				return nil, err
			}
			//FIXME: is this the best way?
			for _, c := range p.Comments {
				position := b.fset.Position(c.End())
				b.commentLineLock.Lock()
				//FIXME: only do this if it is user-requested
				b.endLineToCommentGroup[fileLine{position.Filename, position.Line}] = c
				b.commentLineLock.Unlock()
			}
			return p, nil
		},
	}
	klog.V(3).Infof("loading %d packages", len(netNewDirs))
	pkgs, err := packages.Load(&cfg, netNewDirs...)
	if err != nil {
		return nil, fmt.Errorf("error loading packages: %w\n", err)
	}
	klog.V(3).Infof("loaded %d packages", len(pkgs))

	for _, p := range pkgs {
		b.processPkg(p, true)
		//FIXME: err?
	}
	return pkgs, nil
}

//FIXME: this should be "addPkgsTo"?
func (b *Builder) AddDirsTo(dirs []string, u *types.Universe) error {
	pkgs, err := b.loadDirs(dirs)
	if err != nil {
		return err
	}
	//FIXME: should we just do this in loadDirs?
	for _, pkg := range pkgs {
		// FIXME: name or ID?
		pkgPath := importPathString(pkg.ID)
		if b.userRequested[pkgPath] {
			if err := b.findTypesIn(pkgPath, u); err != nil {
				//FIXME: what to do?
				return err
			}
		}
	}
	return nil
}

//FIXME: comment
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
	}
}

// FindPackages fetches a list of the user-imported packages.
// Note that you need to call b.FindTypes() first.
func (b *Builder) FindPackages() []string {
	// Iterate packages in a predictable order.
	pkgPaths := []string{}
	for k := range b.pkgMap {
		pkgPaths = append(pkgPaths, string(k))
	}
	//FIXME: save this index
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

//FIXME: hack - store in Builder
var typesFound = map[importPathString]bool{}

// findTypesIn finalizes the package import and searches through the package
// for types.
func (b *Builder) findTypesIn(pkgPath importPathString, u *types.Universe) error {
	if typesFound[pkgPath] {
		return nil
	}
	typesFound[pkgPath] = true

	klog.V(5).Infof("findTypesIn %s", pkgPath)
	//pkg := b.typeCheckedPackages[pkgPath]
	pkg := b.pkgMap[pkgPath]
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
	if len(pkg.GoFiles) == 0 {
		klog.V(5).Infof("findTypesIn %s: package has no go files", pkgPath)
		return nil
	}
	absPath, _ := filepath.Split(pkg.GoFiles[0])

	// We're keeping this package.  This call will create the record.
	//FIXME: store the pkg instead of these fields
	u.Package(string(pkgPath)).Name = pkg.Name
	u.Package(string(pkgPath)).Path = pkg.ID // FIXME or pkgPath?  see go core vendor weirdness (handled in go2make)
	u.Package(string(pkgPath)).SourcePath = absPath

	for i, f := range pkg.Syntax {
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

	return nil
}

// if there's a comment on the line `lines` before pos, return its text, otherwise "".
func (b *Builder) priorCommentLines(pos token.Pos, lines int) *ast.CommentGroup {
	position := b.fset.Position(pos)
	key := fileLine{position.Filename, position.Line - lines}
	b.commentLineLock.Lock()
	defer b.commentLineLock.Unlock()
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
		//FIXME: why is deepcopy checking .package to be it s own?
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
