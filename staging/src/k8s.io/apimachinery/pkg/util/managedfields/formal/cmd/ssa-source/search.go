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

// ssa-source imports a restricted Go function's complete syntax tree into Lean.
package main

import (
	_ "embed"
	"fmt"
	"go/ast"
	"go/token"
)

//go:embed go-source-LICENSE
var goSourceLicense string

// Keep Go's overflow-avoiding midpoint idiom explicit in the imported program.
func renderSearchMidpoint(expression *ast.CallExpr, context *scope) (string, error) {
	for _, name := range []string{"int", "uint"} {
		bound, found := context.resolve(name)
		if !found || bound.kind != "builtin" || bound.id != name {
			return "", fmt.Errorf("search midpoint requires the unshadowed %s conversion", name)
		}
	}
	if len(expression.Args) != 1 || expression.Ellipsis.IsValid() {
		return "", fmt.Errorf("search midpoint requires int(uint(i+j) >> 1)")
	}
	shift, ok := expression.Args[0].(*ast.BinaryExpr)
	if !ok || shift.Op != token.SHR {
		return "", fmt.Errorf("search midpoint requires an unsigned right shift")
	}
	one, ok := shift.Y.(*ast.BasicLit)
	if !ok || one.Kind != token.INT || one.Value != "1" {
		return "", fmt.Errorf("search midpoint must shift by exactly one bit")
	}
	unsigned, ok := shift.X.(*ast.CallExpr)
	if !ok || len(unsigned.Args) != 1 || unsigned.Ellipsis.IsValid() {
		return "", fmt.Errorf("search midpoint requires uint(i+j)")
	}
	name, ok := unsigned.Fun.(*ast.Ident)
	if !ok || name.Name != "uint" {
		return "", fmt.Errorf("search midpoint requires unsigned conversion before shifting")
	}
	sum, ok := unsigned.Args[0].(*ast.BinaryExpr)
	if !ok || sum.Op != token.ADD {
		return "", fmt.Errorf("search midpoint requires the sum of its bounds")
	}
	left, err := renderExpr(sum.X, context)
	if err != nil {
		return "", err
	}
	right, err := renderExpr(sum.Y, context)
	if err != nil {
		return "", err
	}
	return "(.unsignedMidpoint " + left + " " + right + ")", nil
}

// Arbitrary callback captures or effects have no meaning in the pure search interpreter.
func renderPathComparator(expression *ast.FuncLit) (string, error) {
	params, results := expression.Type.Params, expression.Type.Results
	if expression.Type.TypeParams != nil || params == nil || len(params.List) != 1 || len(params.List[0].Names) != 2 ||
		results == nil || len(results.List) != 1 || len(results.List[0].Names) != 0 || len(expression.Body.List) != 1 {
		return "", fmt.Errorf("search callback must have exactly two PathElement parameters and one int result")
	}
	paramType, ok := params.List[0].Type.(*ast.Ident)
	if !ok || paramType.Name != "PathElement" {
		return "", fmt.Errorf("search callback parameters must be PathElement")
	}
	resultType, ok := results.List[0].Type.(*ast.Ident)
	if !ok || resultType.Name != "int" {
		return "", fmt.Errorf("search callback result must be int")
	}
	first, second := params.List[0].Names[0].Name, params.List[0].Names[1].Name
	if first == second || first == "_" || second == "_" {
		return "", fmt.Errorf("search callback requires distinct named parameters")
	}
	returned, ok := expression.Body.List[0].(*ast.ReturnStmt)
	if !ok || len(returned.Results) != 1 {
		return "", fmt.Errorf("search callback must directly return the PathElement comparison")
	}
	call, ok := returned.Results[0].(*ast.CallExpr)
	if !ok || len(call.Args) != 1 || call.Ellipsis.IsValid() {
		return "", fmt.Errorf("search callback must directly compare its two parameters")
	}
	method, ok := call.Fun.(*ast.SelectorExpr)
	if !ok || method.Sel.Name != "Compare" {
		return "", fmt.Errorf("search callback must call PathElement.Compare")
	}
	receiver, receiverOK := method.X.(*ast.Ident)
	argument, argumentOK := call.Args[0].(*ast.Ident)
	if !receiverOK || !argumentOK || receiver.Name != first || argument.Name != second {
		return "", fmt.Errorf("search callback must compare its first parameter with its second parameter")
	}
	return "(.pathComparator)", nil
}
