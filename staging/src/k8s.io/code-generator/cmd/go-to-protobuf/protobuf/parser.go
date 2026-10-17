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

package protobuf

import (
	"bytes"
	"errors"
	"fmt"
	"go/ast"
	"go/format"
	"go/parser"
	"go/printer"
	"go/token"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"

	customreflect "k8s.io/code-generator/third_party/forked/golang/reflect"
)

func rewriteFile(name string, header []byte, rewriteFn func(*token.FileSet, *ast.File) error) error {
	fset := token.NewFileSet()
	src, err := os.ReadFile(name)
	if err != nil {
		return err
	}
	file, err := parser.ParseFile(fset, name, src, parser.DeclarationErrors|parser.ParseComments)
	if err != nil {
		return err
	}

	if err := rewriteFn(fset, file); err != nil {
		return err
	}

	b := &bytes.Buffer{}
	b.Write(header)
	if err := printer.Fprint(b, fset, file); err != nil {
		return err
	}

	body, err := format.Source(b.Bytes())
	if err != nil {
		return err
	}

	f, err := os.OpenFile(name, os.O_WRONLY|os.O_TRUNC, 0644)
	if err != nil {
		return err
	}
	defer f.Close()
	if _, err := f.Write(body); err != nil {
		return err
	}
	return f.Close()
}

// ExtractFunc extracts information from the provided TypeSpec and returns true if the type should be
// removed from the destination file.
type ExtractFunc func(*ast.TypeSpec) bool

// OptionalFunc returns true if the provided local name is a type that has protobuf.nullable=true
// and should have its marshal functions adjusted to remove the 'Items' accessor.
type OptionalFunc func(name string) bool

func RewriteGeneratedGogoProtobufFile(file string, extractFn ExtractFunc, optionalFn OptionalFunc, header []byte, dropGogo bool) error {
	return rewriteFile(file, header, func(fset *token.FileSet, file *ast.File) error {
		cmap := ast.NewCommentMap(fset, file, file.Comments)

		// must run before generated types are dropped, since it reads their fields
		rewriteAnyFields(file)

		// transform methods that point to optional maps or slices
		for _, d := range file.Decls {
			rewriteOptionalMethods(d, optionalFn)
		}
		if dropGogo {
			// transform references to gogo sort util
			var oldSortImport string
			var usedSort bool
			for _, d := range file.Decls {
				oldSortImport, usedSort = rewriteGogoSortImport(d)
				if usedSort {
					break
				}
			}
			if usedSort {
				for _, d := range file.Decls {
					rewriteGogoSort(d, oldSortImport, "sort")
				}
			}
		}

		// remove types that are already declared
		decls := []ast.Decl{}
		for _, d := range file.Decls {
			if dropExistingTypeDeclarations(d, extractFn) {
				continue
			}
			if dropEmptyImportDeclarations(d) {
				continue
			}
			// remove all but required functions
			if dropGogo && dropUnusedGo(d) {
				continue
			}
			decls = append(decls, d)
		}
		file.Decls = decls

		// remove unmapped comments
		file.Comments = cmap.Filter(file).Comments()
		return nil
	})
}

// rewriteGogoSortImport rewrites an import of "github.com/gogo/protobuf/sortkeys" to "sort",
// and returns the original package alias and true if the rewrite occurred.
// Returns "", false if the decl is not an import decl, or does not contain an import of "github.com/gogo/protobuf/sortkeys",
func rewriteGogoSortImport(decl ast.Decl) (string, bool) {
	t, ok := decl.(*ast.GenDecl)
	if !ok {
		return "", false
	}
	if t.Tok != token.IMPORT {
		return "", false
	}
	for _, s := range t.Specs {
		if spec, ok := s.(*ast.ImportSpec); ok {
			if spec.Path != nil && spec.Path.Value == `"github.com/gogo/protobuf/sortkeys"` {
				// switch gogo sort to stdlib sort
				spec.Path.Value = `"sort"`
				oldName := "sortkeys"
				if spec.Name != nil {
					oldName = spec.Name.Name
				}
				spec.Name = nil
				return oldName, true
			}
		}
	}
	return "", false
}

// rewriteGogoSort walks the AST, replacing use of the oldSortImport package with newSortImport
func rewriteGogoSort(decl ast.Decl, oldSortImport, newSortImport string) {
	t, ok := decl.(*ast.FuncDecl)
	if !ok {
		return
	}
	ast.Walk(replacePackageVisitor{oldPackage: oldSortImport, newPackage: newSortImport}, t.Body)
}

// keepFuncs is an allowlist of top-level func decls we should keep
var keepFuncs = map[string]bool{
	// generated helpers
	"sovGenerated":           true,
	"sozGenerated":           true,
	"skipGenerated":          true,
	"encodeVarintGenerated":  true,
	"valueToStringGenerated": true,

	// unmarshal
	"Reset":     true,
	"Unmarshal": true,

	// marshal
	"Size":                 true,
	"Marshal":              true,
	"MarshalTo":            true,
	"MarshalToSizedBuffer": true,

	// other widely used methods
	"String": true,
}

// keepVars is an allowlist of top-level var decls we should keep
var keepVars = map[string]bool{
	"ErrInvalidLengthGenerated":        true,
	"ErrIntOverflowGenerated":          true,
	"ErrUnexpectedEndOfGroupGenerated": true,
}

// dropUnusedGo returns true if the top-level decl should be dropped.
// Has the following behavior for different decl types:
// * import: decl is rewritten to drop gogo package imports. Returns true if all imports in the decl were gogo imports, false if non-gogo imports remain.
// * var: decl is rewritten to drop vars not in the keepVars allowlist. Returns true if all vars in the decl were removed, false if allowlisted vars remain.
// * const: returns true
// * type: returns true
// * func: returns true if the func is not in the keepFuncs allowlist and should be dropped.
// * other: returns false
func dropUnusedGo(decl ast.Decl) bool {
	switch t := decl.(type) {
	case *ast.GenDecl:
		switch t.Tok {
		case token.IMPORT:
			specs := []ast.Spec{}
			for _, s := range t.Specs {
				if spec, ok := s.(*ast.ImportSpec); ok {
					if spec.Path == nil || !strings.HasPrefix(spec.Path.Value, `"github.com/gogo/protobuf/`) {
						specs = append(specs, spec)
					}
				}
			}
			if len(specs) == 0 {
				return true
			}
			t.Specs = specs
			return false
		case token.CONST:
			// drop all const declarations
			return true
		case token.VAR:
			specs := []ast.Spec{}
			for _, s := range t.Specs {
				if spec, ok := s.(*ast.ValueSpec); ok {
					if keepVars[spec.Names[0].Name] {
						specs = append(specs, spec)
					}
				}
			}
			if len(specs) == 0 {
				return true
			}
			t.Specs = specs
			return false
		case token.TYPE:
			// drop all type declarations
			return true
		}
	case *ast.FuncDecl:
		name := ""
		if t.Name != nil {
			name = t.Name.Name
		}
		return !keepFuncs[name]
	default:
		return false
	}
	return false
}

// rewriteOptionalMethods makes specific mutations to marshaller methods that belong to types identified
// as being "optional" (they may be nil on the wire). This allows protobuf to serialize a map or slice and
// properly discriminate between empty and nil (which is not possible in protobuf).
// TODO: move into upstream gogo-protobuf once https://github.com/gogo/protobuf/issues/181
// has agreement
func rewriteOptionalMethods(decl ast.Decl, isOptional OptionalFunc) {
	if t, ok := decl.(*ast.FuncDecl); ok {
		ident, ptr, ok := receiver(t)
		if !ok {
			return
		}

		// correct initialization of the form `m.Field = &OptionalType{}` to
		// `m.Field = OptionalType{}`
		if t.Name.Name == "Unmarshal" {
			ast.Walk(optionalAssignmentVisitor{fn: isOptional}, t.Body)
		}

		if !isOptional(ident.Name) {
			return
		}

		switch t.Name.Name {
		case "Unmarshal":
			ast.Walk(&optionalItemsVisitor{}, t.Body)
		case "MarshalTo", "Size", "String", "MarshalToSizedBuffer":
			ast.Walk(&optionalItemsVisitor{}, t.Body)
			fallthrough
		case "Marshal":
			// if the method has a pointer receiver, set it back to a normal receiver
			if ptr {
				t.Recv.List[0].Type = ident
			}
		}
	}
}

type optionalAssignmentVisitor struct {
	fn OptionalFunc
}

// Visit walks the provided node, transforming field initializations of the form
// m.Field = &OptionalType{} -> m.Field = OptionalType{}
func (v optionalAssignmentVisitor) Visit(n ast.Node) ast.Visitor {
	if t, ok := n.(*ast.AssignStmt); ok {
		if len(t.Lhs) == 1 && len(t.Rhs) == 1 {
			if !isFieldSelector(t.Lhs[0], "m", "") {
				return nil
			}
			unary, ok := t.Rhs[0].(*ast.UnaryExpr)
			if !ok || unary.Op != token.AND {
				return nil
			}
			composite, ok := unary.X.(*ast.CompositeLit)
			if !ok || composite.Type == nil || len(composite.Elts) != 0 {
				return nil
			}
			if ident, ok := composite.Type.(*ast.Ident); ok && v.fn(ident.Name) {
				t.Rhs[0] = composite
			}
		}
		return nil
	}
	return v
}

type optionalItemsVisitor struct{}

// Visit walks the provided node, looking for specific patterns to transform that match
// the effective outcome of turning struct{ map[x]y || []x } into map[x]y or []x.
func (v *optionalItemsVisitor) Visit(n ast.Node) ast.Visitor {
	switch t := n.(type) {
	case *ast.RangeStmt:
		if isFieldSelector(t.X, "m", "Items") {
			t.X = &ast.Ident{Name: "m"}
		}
	case *ast.AssignStmt:
		if len(t.Lhs) == 1 && len(t.Rhs) == 1 {
			switch lhs := t.Lhs[0].(type) {
			case *ast.IndexExpr:
				if isFieldSelector(lhs.X, "m", "Items") {
					lhs.X = &ast.StarExpr{X: &ast.Ident{Name: "m"}}
				}
			default:
				if isFieldSelector(t.Lhs[0], "m", "Items") {
					t.Lhs[0] = &ast.StarExpr{X: &ast.Ident{Name: "m"}}
				}
			}
			if rhs, ok := t.Rhs[0].(*ast.CallExpr); ok {
				if ident, ok := rhs.Fun.(*ast.Ident); ok && ident.Name == "append" {
					ast.Walk(v, rhs)
					if len(rhs.Args) > 0 {
						if arg, ok := rhs.Args[0].(*ast.Ident); ok {
							if arg.Name == "m" {
								rhs.Args[0] = &ast.StarExpr{X: &ast.Ident{Name: "m"}}
							}
						}
					}
					return nil
				}
			}
		}
	case *ast.IfStmt:
		if cond, ok := t.Cond.(*ast.BinaryExpr); ok {
			if cond.Op == token.EQL {
				if isFieldSelector(cond.X, "m", "Items") && isIdent(cond.Y, "nil") {
					cond.X = &ast.StarExpr{X: &ast.Ident{Name: "m"}}
				}
			}
		}
		if t.Init != nil {
			// Find form:
			// if err := m[len(m.Items)-1].Unmarshal(data[iNdEx:postIndex]); err != nil {
			// 	return err
			// }
			if s, ok := t.Init.(*ast.AssignStmt); ok {
				if call, ok := s.Rhs[0].(*ast.CallExpr); ok {
					if sel, ok := call.Fun.(*ast.SelectorExpr); ok {
						if x, ok := sel.X.(*ast.IndexExpr); ok {
							// m[] -> (*m)[]
							if sel2, ok := x.X.(*ast.SelectorExpr); ok {
								if ident, ok := sel2.X.(*ast.Ident); ok && ident.Name == "m" {
									x.X = &ast.StarExpr{X: &ast.Ident{Name: "m"}}
								}
							}
							// len(m.Items) -> len(*m)
							if bin, ok := x.Index.(*ast.BinaryExpr); ok {
								if call2, ok := bin.X.(*ast.CallExpr); ok && len(call2.Args) == 1 {
									if isFieldSelector(call2.Args[0], "m", "Items") {
										call2.Args[0] = &ast.StarExpr{X: &ast.Ident{Name: "m"}}
									}
								}
							}
						}
					}
				}
			}
		}
	case *ast.IndexExpr:
		if isFieldSelector(t.X, "m", "Items") {
			t.X = &ast.Ident{Name: "m"}
			return nil
		}
	case *ast.CallExpr:
		changed := false
		for i := range t.Args {
			if isFieldSelector(t.Args[i], "m", "Items") {
				t.Args[i] = &ast.Ident{Name: "m"}
				changed = true
			}
		}
		if changed {
			return nil
		}
	}
	return v
}

func isFieldSelector(n ast.Expr, name, field string) bool {
	s, ok := n.(*ast.SelectorExpr)
	if !ok || s.Sel == nil || (field != "" && s.Sel.Name != field) {
		return false
	}
	return isIdent(s.X, name)
}

func isIdent(n ast.Expr, value string) bool {
	ident, ok := n.(*ast.Ident)
	return ok && ident.Name == value
}

func receiver(f *ast.FuncDecl) (ident *ast.Ident, pointer bool, ok bool) {
	if f.Recv == nil || len(f.Recv.List) != 1 {
		return nil, false, false
	}
	switch t := f.Recv.List[0].Type.(type) {
	case *ast.StarExpr:
		identity, ok := t.X.(*ast.Ident)
		if !ok {
			return nil, false, false
		}
		return identity, true, true
	case *ast.Ident:
		return t, false, true
	}
	return nil, false, false
}

// dropExistingTypeDeclarations removes any type declaration for which extractFn returns true. The function
// returns true if the entire declaration should be dropped.
func dropExistingTypeDeclarations(decl ast.Decl, extractFn ExtractFunc) bool {
	if t, ok := decl.(*ast.GenDecl); ok {
		if t.Tok != token.TYPE {
			return false
		}
		specs := []ast.Spec{}
		for _, s := range t.Specs {
			if spec, ok := s.(*ast.TypeSpec); ok {
				if extractFn(spec) {
					continue
				}
				specs = append(specs, spec)
			}
		}
		if len(specs) == 0 {
			return true
		}
		t.Specs = specs
	}
	return false
}

// dropEmptyImportDeclarations strips any generated but no-op imports from the generated code
// to prevent generation from being able to define side-effects.  The function returns true
// if the entire declaration should be dropped.
func dropEmptyImportDeclarations(decl ast.Decl) bool {
	if t, ok := decl.(*ast.GenDecl); ok {
		if t.Tok != token.IMPORT {
			return false
		}
		specs := []ast.Spec{}
		for _, s := range t.Specs {
			if spec, ok := s.(*ast.ImportSpec); ok {
				if spec.Name != nil && spec.Name.Name == "_" {
					continue
				}
				specs = append(specs, spec)
			}
		}
		if len(specs) == 0 {
			return true
		}
		t.Specs = specs
	}
	return false
}

func RewriteTypesWithProtobufStructTags(name string, structTags map[string]map[string]string) error {
	return rewriteFile(name, []byte{}, func(fset *token.FileSet, file *ast.File) error {
		allErrs := []error{}

		// set any new struct tags
		for _, d := range file.Decls {
			if errs := updateStructTags(d, structTags, []string{"protobuf"}); len(errs) > 0 {
				allErrs = append(allErrs, errs...)
			}
		}

		if len(allErrs) > 0 {
			var s string
			for _, err := range allErrs {
				s += err.Error() + "\n"
			}
			return errors.New(s)
		}
		return nil
	})
}

func getFieldName(expr ast.Expr, structname string) (name string, err error) {
	for {
		switch t := expr.(type) {
		case *ast.Ident:
			return t.Name, nil
		case *ast.SelectorExpr:
			return t.Sel.Name, nil
		case *ast.StarExpr:
			expr = t.X
		default:
			return "", fmt.Errorf("unable to get name for tag from struct %q, field %#v", structname, t)
		}
	}
}

func updateStructTags(decl ast.Decl, structTags map[string]map[string]string, toCopy []string) []error {
	var errs []error
	t, ok := decl.(*ast.GenDecl)
	if !ok {
		return nil
	}
	if t.Tok != token.TYPE {
		return nil
	}

	for _, s := range t.Specs {
		spec, ok := s.(*ast.TypeSpec)
		if !ok {
			continue
		}
		typeName := spec.Name.Name
		fieldTags, ok := structTags[typeName]
		if !ok {
			continue
		}
		st, ok := spec.Type.(*ast.StructType)
		if !ok {
			continue
		}

		for i := range st.Fields.List {
			f := st.Fields.List[i]
			var name string
			var err error
			if len(f.Names) == 0 {
				name, err = getFieldName(f.Type, spec.Name.Name)
				if err != nil {
					errs = append(errs, err)
					continue
				}
			} else {
				name = f.Names[0].Name
			}
			value, ok := fieldTags[name]
			if !ok {
				continue
			}
			var tags customreflect.StructTags
			if f.Tag != nil {
				oldTags, err := customreflect.ParseStructTags(strings.Trim(f.Tag.Value, "`"))
				if err != nil {
					errs = append(errs, fmt.Errorf("unable to read struct tag from struct %q, field %q: %v", spec.Name.Name, name, err))
					continue
				}
				tags = oldTags
			}
			for _, name := range toCopy {
				// don't overwrite existing tags
				if tags.Has(name) {
					continue
				}
				// append new tags
				if v := reflect.StructTag(value).Get(name); len(v) > 0 {
					tags = append(tags, customreflect.StructTag{Name: name, Value: v})
				}
			}
			if len(tags) == 0 {
				continue
			}
			if f.Tag == nil {
				f.Tag = &ast.BasicLit{}
			}
			f.Tag.Value = tags.String()
		}
	}
	return errs
}

type replacePackageVisitor struct {
	oldPackage string
	newPackage string
}

// Visit walks the provided node, transforming references to the old package to the new package.
func (v replacePackageVisitor) Visit(n ast.Node) ast.Visitor {
	if e, ok := n.(*ast.SelectorExpr); ok {
		if i, ok := e.X.(*ast.Ident); ok && i.Name == v.oldPackage {
			i.Name = v.newPackage
		}
		return nil
	}
	return v
}

// anyRuntimePackage provides the helpers that interface-typed fields, which
// go-to-protobuf maps to google.protobuf.Any, are marshaled through.
const anyRuntimePackage = "k8s.io/apimachinery/pkg/runtime/protoany"

// anyGoPackages are the Go packages protoc-gen-gogo may emit google.protobuf.Any
// as, depending on the go_package of the any.proto it was given.
var anyGoPackages = map[string]string{
	`"google.golang.org/protobuf/types/known/anypb"`: "anypb",
	`"github.com/gogo/protobuf/types"`:               "types",
}

type anyFieldKind int

const (
	anySingle anyFieldKind = iota + 1
	anyList
)

// rewriteAnyFields rewrites the marshalers protoc-gen-gogo produces for
// google.protobuf.Any fields so they operate on the Go interface the field
// really has, by routing them through anyRuntimePackage. Returns true if the
// file was changed.
func rewriteAnyFields(file *ast.File) bool {
	alias, importSpec := findAnyImport(file)
	if importSpec == nil {
		return false
	}
	fieldsByType := findAnyFields(file, alias)
	if len(fieldsByType) == 0 {
		return false
	}

	for _, d := range file.Decls {
		fn, ok := d.(*ast.FuncDecl)
		if !ok || fn.Body == nil {
			continue
		}
		ident, _, ok := receiver(fn)
		if !ok || len(fn.Recv.List[0].Names) != 1 {
			continue
		}
		fields, ok := fieldsByType[ident.Name]
		if !ok {
			continue
		}
		ast.Walk(&anyFieldVisitor{
			recv:   fn.Recv.List[0].Names[0].Name,
			fields: fields,
			alias:  alias,
		}, fn.Body)
	}

	// The generated struct types are the only remaining users of the Any
	// import, and they are dropped later, so swap it for the runtime helpers.
	importSpec.Name = ast.NewIdent(filepath.Base(anyRuntimePackage))
	importSpec.Path.Value = strconv.Quote(anyRuntimePackage)
	return true
}

func findAnyImport(file *ast.File) (alias string, spec *ast.ImportSpec) {
	for _, d := range file.Decls {
		gd, ok := d.(*ast.GenDecl)
		if !ok || gd.Tok != token.IMPORT {
			continue
		}
		for _, s := range gd.Specs {
			is, ok := s.(*ast.ImportSpec)
			if !ok || is.Path == nil {
				continue
			}
			name, ok := anyGoPackages[is.Path.Value]
			if !ok {
				continue
			}
			if is.Name != nil {
				name = is.Name.Name
			}
			return name, is
		}
	}
	return "", nil
}

// findAnyFields returns, per generated struct, the fields typed *alias.Any or []*alias.Any.
func findAnyFields(file *ast.File, alias string) map[string]map[string]anyFieldKind {
	result := map[string]map[string]anyFieldKind{}
	for _, d := range file.Decls {
		gd, ok := d.(*ast.GenDecl)
		if !ok || gd.Tok != token.TYPE {
			continue
		}
		for _, s := range gd.Specs {
			ts, ok := s.(*ast.TypeSpec)
			if !ok {
				continue
			}
			st, ok := ts.Type.(*ast.StructType)
			if !ok {
				continue
			}
			for _, f := range st.Fields.List {
				var kind anyFieldKind
				switch t := f.Type.(type) {
				case *ast.StarExpr:
					if isAnyTypeExpr(t, alias) {
						kind = anySingle
					}
				case *ast.ArrayType:
					if t.Len == nil && isAnyTypeExpr(t.Elt, alias) {
						kind = anyList
					}
				}
				if kind == 0 {
					continue
				}
				if result[ts.Name.Name] == nil {
					result[ts.Name.Name] = map[string]anyFieldKind{}
				}
				for _, n := range f.Names {
					result[ts.Name.Name][n.Name] = kind
				}
			}
		}
	}
	return result
}

// isAnyTypeExpr matches *alias.Any.
func isAnyTypeExpr(e ast.Expr, alias string) bool {
	star, ok := e.(*ast.StarExpr)
	if !ok {
		return false
	}
	return isFieldSelector(star.X, alias, "Any")
}

// anyFieldVisitor rewrites, within one method of a type with Any fields:
//
//	X.Size()                                    -> protoany.Size(X)
//	X.MarshalToSizedBuffer(b)                   -> protoany.MarshalToSizedBuffer(X, b)
//	X.Unmarshal(b)                              -> protoany.Unmarshal(b, &X)
//	if m.F == nil { m.F = &anypb.Any{} }        -> (removed)
//	append(m.F, &anypb.Any{})                   -> append(m.F, nil)
//	strings.Replace(s, "Any", "anypb.Any", 1)   -> s
//
// where X is m.F, m.F[i], or the value variable of a range over m.F.
type anyFieldVisitor struct {
	recv      string
	fields    map[string]anyFieldKind
	alias     string
	rangeVars map[string]struct{}
}

func (v *anyFieldVisitor) Visit(n ast.Node) ast.Visitor {
	switch t := n.(type) {
	case *ast.BlockStmt:
		t.List = v.dropAnyNilInits(t.List)
	case *ast.CaseClause:
		// Unmarshal's per-field logic lives directly in switch cases.
		t.Body = v.dropAnyNilInits(t.Body)
	case *ast.RangeStmt:
		if v.isField(t.X, anyList) {
			if value, ok := t.Value.(*ast.Ident); ok && value.Name != "_" {
				nested := *v
				nested.rangeVars = map[string]struct{}{value.Name: {}}
				for k := range v.rangeVars {
					nested.rangeVars[k] = struct{}{}
				}
				ast.Walk(&nested, t.Body)
				return nil
			}
		}
	case *ast.CallExpr:
		v.rewriteCall(t)
	}
	return v
}

func (v *anyFieldVisitor) rewriteCall(call *ast.CallExpr) {
	if isIdent(call.Fun, "append") && len(call.Args) == 2 && v.isField(call.Args[0], anyList) && v.isNewAny(call.Args[1]) {
		call.Args[1] = ast.NewIdent("nil")
		return
	}

	sel, ok := call.Fun.(*ast.SelectorExpr)
	if !ok {
		return
	}
	if isIdent(sel.X, "strings") && sel.Sel.Name == "Replace" && len(call.Args) == 4 {
		// gogo's String() qualifies the type name; there is no Any type to name.
		if lit, ok := call.Args[2].(*ast.BasicLit); ok && lit.Value == strconv.Quote(v.alias+".Any") {
			if inner, ok := call.Args[0].(*ast.CallExpr); ok {
				*call = *inner
			}
		}
		return
	}
	if !v.isAnyValue(sel.X) {
		return
	}
	helper := func(name string) ast.Expr {
		return &ast.SelectorExpr{X: ast.NewIdent(filepath.Base(anyRuntimePackage)), Sel: ast.NewIdent(name)}
	}
	switch sel.Sel.Name {
	case "Size":
		call.Fun, call.Args = helper("Size"), []ast.Expr{sel.X}
	case "MarshalToSizedBuffer":
		call.Fun, call.Args = helper("MarshalToSizedBuffer"), append([]ast.Expr{sel.X}, call.Args...)
	case "Unmarshal":
		call.Fun, call.Args = helper("Unmarshal"), append(call.Args, &ast.UnaryExpr{Op: token.AND, X: sel.X})
	}
}

// isField matches recv.F where F is an Any field of the given kind.
func (v *anyFieldVisitor) isField(e ast.Expr, kind anyFieldKind) bool {
	sel, ok := e.(*ast.SelectorExpr)
	if !ok || !isIdent(sel.X, v.recv) {
		return false
	}
	return v.fields[sel.Sel.Name] == kind
}

// isAnyValue matches an expression holding a single Any value.
func (v *anyFieldVisitor) isAnyValue(e ast.Expr) bool {
	switch t := e.(type) {
	case *ast.Ident:
		_, ok := v.rangeVars[t.Name]
		return ok
	case *ast.IndexExpr:
		return v.isField(t.X, anyList)
	default:
		return v.isField(e, anySingle)
	}
}

// isNewAny matches &alias.Any{}.
func (v *anyFieldVisitor) isNewAny(e ast.Expr) bool {
	u, ok := e.(*ast.UnaryExpr)
	if !ok || u.Op != token.AND {
		return false
	}
	lit, ok := u.X.(*ast.CompositeLit)
	return ok && len(lit.Elts) == 0 && isFieldSelector(lit.Type, v.alias, "Any")
}

func (v *anyFieldVisitor) dropAnyNilInits(list []ast.Stmt) []ast.Stmt {
	stmts := list[:0]
	for _, s := range list {
		if !v.isAnyNilInit(s) {
			stmts = append(stmts, s)
		}
	}
	return stmts
}

// isAnyNilInit matches `if m.F == nil { m.F = &alias.Any{} }`, which would
// otherwise assign a concrete Any into the interface field before decoding.
func (v *anyFieldVisitor) isAnyNilInit(s ast.Stmt) bool {
	ifStmt, ok := s.(*ast.IfStmt)
	if !ok || ifStmt.Init != nil || ifStmt.Else != nil || len(ifStmt.Body.List) != 1 {
		return false
	}
	cond, ok := ifStmt.Cond.(*ast.BinaryExpr)
	if !ok || cond.Op != token.EQL || !v.isField(cond.X, anySingle) || !isIdent(cond.Y, "nil") {
		return false
	}
	assign, ok := ifStmt.Body.List[0].(*ast.AssignStmt)
	return ok && len(assign.Lhs) == 1 && len(assign.Rhs) == 1 && v.isField(assign.Lhs[0], anySingle) && v.isNewAny(assign.Rhs[0])
}
