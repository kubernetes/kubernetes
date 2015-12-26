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

package main

import (
	"bytes"
	"go/format"
	"io/ioutil"
	"os"

	"k8s.io/kubernetes/third_party/golang/go/ast"
	//"k8s.io/kubernetes/third_party/golang/go/build"
	"k8s.io/kubernetes/third_party/golang/go/parser"
	"k8s.io/kubernetes/third_party/golang/go/printer"
	"k8s.io/kubernetes/third_party/golang/go/token"
	//tc "k8s.io/kubernetes/third_party/golang/go/types"
)

func RewriteGeneratedGogoProtobufFile(name string, packageName string, typeExistsFn func(string) bool, header []byte) error {
	fset := token.NewFileSet()
	src, err := ioutil.ReadFile(name)
	if err != nil {
		return err
	}
	file, err := parser.ParseFile(fset, name, src, parser.DeclarationErrors|parser.ParseComments)
	if err != nil {
		return err
	}
	cmap := ast.NewCommentMap(fset, file, file.Comments)

	// ensure the package name matches the expected package name
	//file.Name.Name = packageName

	// remove types that are already declared
	decls := []ast.Decl{}
	for _, d := range file.Decls {
		if !dropExistingTypeDeclarations(d, typeExistsFn) {
			decls = append(decls, d)
		}
	}
	file.Decls = decls

	// remove unmapped comments
	file.Comments = cmap.Filter(file).Comments()

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
	_, err = f.Write(body)
	return err
}

func dropExistingTypeDeclarations(decl ast.Decl, existsFn func(string) bool) bool {
	switch t := decl.(type) {
	case *ast.GenDecl:
		if t.Tok != token.TYPE {
			return false
		}
		specs := []ast.Spec{}
		for _, s := range t.Specs {
			switch spec := s.(type) {
			case *ast.TypeSpec:
				if existsFn(spec.Name.Name) {
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
