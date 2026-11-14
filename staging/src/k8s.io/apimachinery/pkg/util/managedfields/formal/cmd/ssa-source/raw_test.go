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
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestRemovalRetainsCompleteStructuralBodies(t *testing.T) {
	directory, err := exec.Command("go", "list", "-f", "{{.Dir}}", "sigs.k8s.io/structured-merge-diff/v6/typed").Output()
	if err != nil {
		t.Fatal(err)
	}
	source, err := os.ReadFile(filepath.Join(strings.TrimSpace(string(directory)), "remove.go"))
	if err != nil {
		t.Fatal(err)
	}
	fragments := [][]string{
		{"CompositeLit", "removingWalker", "NewFreelistAllocator", "resolveSchema", "NewValueInterface"},
		{"Unstructured", "ReturnStmt"},
		{"IsList", "AsListUsing", "DeferStmt", "ForStmt", "continue", "removeItemsWithSchema", "newItems"},
		{"IsMap", "AsMapUsing", "RangeStmt", "FuncLit", "newMap", "removeItemsWithSchema", "WithPrefix"},
	}
	for index, function := range atomicRemoveFunctions {
		program, err := importFunction(source, function)
		if err != nil {
			t.Fatal(err)
		}
		for _, fragment := range fragments[index] {
			if !strings.Contains(program, fragment) {
				t.Fatalf("%s dropped %s", function.name, fragment)
			}
		}
	}
}

func TestStructuralSyntaxPreservesClosureEffectsAndOperators(t *testing.T) {
	render := func(source string) string {
		t.Helper()
		node, err := parser.ParseExpr(source)
		if err != nil {
			t.Fatal(err)
		}
		result, err := renderRaw(reflect.ValueOf(node))
		if err != nil {
			t.Fatal(err)
		}
		return result
	}
	original := render(`func(k string) bool { if k == "x" { target[k] = nil }; return true }`)
	for _, source := range []string{
		`func(k string) bool { if k != "x" { target[k] = nil }; return true }`,
		`func(k string) bool { if k == "x" { other[k] = nil }; return true }`,
		`func(k string) bool { if k == "x" { target[k] = nil }; return false }`,
	} {
		if render(source) == original {
			t.Fatalf("lost closure syntax: %s", source)
		}
	}
	if render(`fn(items...)`) == render(`fn(items)`) {
		t.Fatal("lost variadic call marker")
	}
	if render(`(x + y) * z`) == render(`x + (y * z)`) {
		t.Fatal("lost expression grouping")
	}
	if render(`func(k string) bool { // position/comment change
        if k == "x" { target[k] = nil }; return true }`) != original {
		t.Fatal("structural syntax depends on concrete source positions or comments")
	}
	for _, node := range []any{42, struct{ Hidden int }{1}} {
		if _, err := renderRaw(reflect.ValueOf(node)); err == nil {
			t.Fatalf("accepted unknown AST kind %T", node)
		}
	}
	result, err := renderRaw(reflect.ValueOf(&ast.BasicLit{Kind: token.INT, Value: "9007199254740993"}))
	if err != nil || !strings.Contains(result, "9007199254740993") {
		t.Fatalf("numeric token changed: %v", err)
	}
}
