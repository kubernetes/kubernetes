/*
Copyright 2019 The Kubernetes Authors.

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

// This tool is for checking conformance e2e tests follow the requirements
// which is https://github.com/kubernetes/community/blob/master/contributors/devel/sig-architecture/conformance-tests.md#conformance-test-requirements
package main

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
)

// callName returns "pkg.Func" for a call of that form, or "".
func callName(call *ast.CallExpr) string {
	sel, ok := call.Fun.(*ast.SelectorExpr)
	if !ok {
		return ""
	}
	pkg, ok := sel.X.(*ast.Ident)
	if !ok {
		return ""
	}
	return pkg.Name + "." + sel.Sel.Name
}

func isSkip(name string) bool {
	// [LinuxOnly] tests may skip on Windows nodes.
	if name == "e2eskipper.SkipIfNodeOSDistroIs" {
		return false
	}
	return name == "ginkgo.Skip" || strings.HasPrefix(name, "e2eskipper.Skip")
}

// checkFile reports the requirement: it works for all providers (e.g., no SkipIfProviderIs/SkipUnlessProviderIs calls).
// It looks at the body of each conformance test and at the BeforeEach nodes that run before it.
func checkFile(fset *token.FileSet, file *ast.File) []string {
	var problems []string
	report := func(test *ast.CallExpr, n ast.Node) {
		ast.Inspect(n, func(n ast.Node) bool {
			if call, ok := n.(*ast.CallExpr); ok {
				if name := callName(call); isSkip(name) {
					problems = append(problems, fmt.Sprintf("%v: conformance test must not be skipped, %s is called at line %d",
						fset.Position(test.Pos()), name, fset.Position(call.Pos()).Line))
				}
			}
			return true
		})
	}

	// setups are the BeforeEach nodes of the enclosing containers.
	var walk func(n ast.Node, setups []*ast.CallExpr)
	walk = func(n ast.Node, setups []*ast.CallExpr) {
		ast.Inspect(n, func(n ast.Node) bool {
			call, ok := n.(*ast.CallExpr)
			if !ok || len(call.Args) == 0 {
				return true
			}
			body, ok := call.Args[len(call.Args)-1].(*ast.FuncLit)
			if !ok {
				return true
			}
			if callName(call) == "framework.ConformanceIt" {
				report(call, body)
				for _, setup := range setups {
					report(call, setup)
				}
				return false
			}
			// Any other call that takes a function is treated as a container.
			inner := setups
			for _, stmt := range body.Body.List {
				if expr, ok := stmt.(*ast.ExprStmt); ok {
					if c, ok := expr.X.(*ast.CallExpr); ok && strings.HasSuffix(callName(c), "BeforeEach") {
						inner = append(inner[:len(inner):len(inner)], c)
					}
				}
			}
			walk(body.Body, inner)
			return false
		})
	}
	walk(file, nil)
	return problems
}

func checkDir(root string) ([]string, error) {
	var problems []string
	fset := token.NewFileSet()
	err := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() || !strings.HasSuffix(path, ".go") {
			return err
		}
		file, err := parser.ParseFile(fset, path, nil, 0)
		if err != nil {
			return err
		}
		problems = append(problems, checkFile(fset, file)...)
		return nil
	})
	return problems, err
}

func main() {
	if len(os.Args) != 2 {
		fmt.Fprintln(os.Stderr, "Usage: check_conformance_test_requirements [e2e-test-path]")
		os.Exit(1)
	}
	problems, err := checkDir(os.Args[1])
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
	for _, problem := range problems {
		fmt.Fprintln(os.Stderr, problem)
	}
	if len(problems) > 0 {
		os.Exit(1)
	}
}
