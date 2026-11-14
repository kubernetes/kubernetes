/*
Copyright 2026 The Kubernetes Authors.

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
	"fmt"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
)

var atomicMergeFunctions = []functionSpec{
	{
		name: "doList", leanName: "listProgram", receiver: "*mergingWalker", receiverID: "w",
		signature:  "func (w *mergingWalker) doList(t *schema.List) (errs ValidationErrors)",
		parameters: []string{"t"}, namedResults: []string{"errs"}, atomicMerge: true,
	},
	{
		name: "doMap", leanName: "mapProgram", receiver: "*mergingWalker", receiverID: "w",
		signature:  "func (w *mergingWalker) doMap(t *schema.Map) (errs ValidationErrors)",
		parameters: []string{"t"}, namedResults: []string{"errs"}, atomicMerge: true,
	},
	{
		name: "doLeaf", leanName: "leafProgram", receiver: "*mergingWalker", receiverID: "w",
		signature: "func (w *mergingWalker) doLeaf()", atomicMerge: true, voidResult: true,
	},
	{
		name: "ruleKeepRHS", leanName: "keepRightProgram", signature: "func(w *mergingWalker)",
		parameters: []string{"w"}, maps: map[string]string{"w": "typed-mutable"},
		standalone: true, initializer: true, atomicMerge: true, voidResult: true,
	},
}

// The initializer is a checked function literal, not a hand-written model of its name.
func importMergeRule(source []byte, function functionSpec) (string, error) {
	positions := token.NewFileSet()
	file, err := parser.ParseFile(positions, "merge.go", source, 0)
	if err != nil {
		return "", err
	}
	// A function call with the same spelling could wrap the closure with unmodeled effects.
	definedType := false
	for _, declaration := range file.Decls {
		group, ok := declaration.(*ast.GenDecl)
		if !ok || group.Tok != token.TYPE {
			continue
		}
		for _, declaration := range group.Specs {
			spec := declaration.(*ast.TypeSpec)
			if spec.Name.Name != "mergeRule" {
				continue
			}
			var typ bytes.Buffer
			if err := format.Node(&typ, positions, spec.Type); err != nil {
				return "", err
			}
			if definedType || spec.Assign.IsValid() || spec.TypeParams != nil || typ.String() != function.signature {
				return "", fmt.Errorf("mergeRule must be the unique checked function type")
			}
			definedType = true
		}
	}
	if !definedType {
		return "", fmt.Errorf("mergeRule is not declared as the checked function type")
	}
	var literal *ast.FuncLit
	for _, declaration := range file.Decls {
		group, ok := declaration.(*ast.GenDecl)
		if !ok || group.Tok != token.VAR {
			continue
		}
		for _, declaration := range group.Specs {
			spec := declaration.(*ast.ValueSpec)
			for _, name := range spec.Names {
				if name.Name != function.name {
					continue
				}
				if literal != nil || len(spec.Names) != 1 || len(spec.Values) != 1 {
					return "", fmt.Errorf("merge rule requires one unique initialized variable")
				}
				conversion, ok := spec.Values[0].(*ast.CallExpr)
				if !ok || len(conversion.Args) != 1 || conversion.Ellipsis.IsValid() {
					return "", fmt.Errorf("merge rule requires a mergeRule function conversion")
				}
				typeName, ok := conversion.Fun.(*ast.Ident)
				if !ok || typeName.Name != "mergeRule" {
					return "", fmt.Errorf("merge rule conversion must name mergeRule")
				}
				literal, ok = conversion.Args[0].(*ast.FuncLit)
				if !ok {
					return "", fmt.Errorf("merge rule must contain its complete function literal")
				}
			}
		}
	}
	if literal == nil {
		return "", fmt.Errorf("merge rule %s was not found", function.name)
	}
	var signature bytes.Buffer
	if err := format.Node(&signature, positions, literal.Type); err != nil {
		return "", err
	}
	if signature.String() != function.signature {
		return "", fmt.Errorf("unsupported merge rule signature: %s", signature.String())
	}
	return renderStatements(literal.Body.List, rootScope(function))
}
