/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"go/format"
	"io/ioutil"
	"os"
	"reflect"
	"strings"

	"k8s.io/kubernetes/third_party/golang/go/ast"
	"k8s.io/kubernetes/third_party/golang/go/parser"
	"k8s.io/kubernetes/third_party/golang/go/printer"
	"k8s.io/kubernetes/third_party/golang/go/token"
	customreflect "k8s.io/kubernetes/third_party/golang/reflect"
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

func RewriteGeneratedGogoProtobufFile(name string, extractFn ExtractFunc, header []byte) error {
	return rewriteFile(name, header, func(fset *token.FileSet, file *ast.File) error {
		cmap := ast.NewCommentMap(fset, file, file.Comments)

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
