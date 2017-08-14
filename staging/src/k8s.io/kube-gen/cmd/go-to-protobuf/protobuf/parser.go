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
	"io/ioutil"
	"os"
	"reflect"
	"strings"

	customreflect "k8s.io/kube-gen/third_party/forked/golang/reflect"
)

func rewriteFile(name string, header []byte, rewriteFn func(*token.FileSet, *ast.File) error) error {
	fset := token.NewFileSet()
	src, err := ioutil.ReadFile(name)
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

func RewriteGeneratedGogoProtobufFile(name string, extractFn ExtractFunc, optionalFn OptionalFunc, header []byte) error {
	return rewriteFile(name, header, func(fset *token.FileSet, file *ast.File) error {
		cmap := ast.NewCommentMap(fset, file, file.Comments)

		// transform methods that point to optional maps or slices
		for _, d := range file.Decls {
			rewriteOptionalMethods(d, optionalFn)
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
			decls = append(decls, d)
		}
		file.Decls = decls

		// remove unmapped comments
		file.Comments = cmap.Filter(file).Comments()
		return nil
	})
}

// rewriteOptionalMethods makes specific mutations to marshaller methods that belong to types identified
// as being "optional" (they may be nil on the wire). This allows protobuf to serialize a map or slice and
// properly discriminate between empty and nil (which is not possible in protobuf).
// TODO: move into upstream gogo-protobuf once https://github.com/gogo/protobuf/issues/181
//   has agreement
func rewriteOptionalMethods(decl ast.Decl, isOptional OptionalFunc) {
	switch t := decl.(type) {
	case *ast.FuncDecl:
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
		case "MarshalTo", "Size", "String":
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
	switch t := n.(type) {
	case *ast.AssignStmt:
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
			switch rhs := t.Rhs[0].(type) {
			case *ast.CallExpr:
				if ident, ok := rhs.Fun.(*ast.Ident); ok && ident.Name == "append" {
					ast.Walk(v, rhs)
					if len(rhs.Args) > 0 {
						switch arg := rhs.Args[0].(type) {
						case *ast.Ident:
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
		switch cond := t.Cond.(type) {
		case *ast.BinaryExpr:
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
			switch s := t.Init.(type) {
			case *ast.AssignStmt:
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
	switch t := decl.(type) {
	case *ast.GenDecl:
		if t.Tok != token.TYPE {
			return false
		}
		specs := []ast.Spec{}
		for _, s := range t.Specs {
			switch spec := s.(type) {
			case *ast.TypeSpec:
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
	switch t := decl.(type) {
	case *ast.GenDecl:
		if t.Tok != token.IMPORT {
			return false
		}
		specs := []ast.Spec{}
		for _, s := range t.Specs {
			switch spec := s.(type) {
			case *ast.ImportSpec:
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
			if len(f.Names) == 0 {
				switch t := f.Type.(type) {
				case *ast.Ident:
					name = t.Name
				case *ast.SelectorExpr:
					name = t.Sel.Name
				default:
					errs = append(errs, fmt.Errorf("unable to get name for tag from struct %q, field %#v", spec.Name.Name, t))
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
