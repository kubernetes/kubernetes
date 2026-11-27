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
	"go/types"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
)

func isConformance(name string) bool { return name == "framework.ConformanceIt" }

func isSkip(name string) bool {
	// [LinuxOnly] tests may skip on Windows nodes.
	if name == "e2eskipper.SkipIfNodeOSDistroIs" {
		return false
	}
	return name == "ginkgo.Skip" || strings.HasPrefix(name, "e2eskipper.Skip")
}

// find returns the calls under n to functions that match.
func find(n ast.Node, match func(name string) bool) []token.Pos {
	var found []token.Pos
	ast.Inspect(n, func(n ast.Node) bool {
		if call, ok := n.(*ast.CallExpr); ok && match(types.ExprString(call.Fun)) {
			found = append(found, call.Pos())
		}
		return true
	})
	return found
}

// checkFile checks the requirement: it works for all providers (e.g., no SkipIfProviderIs/SkipUnlessProviderIs calls).
// It returns the skip calls in conformance tests and in the BeforeEach nodes that run before them.
func checkFile(file *ast.File) []token.Pos {
	var skips []token.Pos
	ast.Inspect(file, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok || len(call.Args) == 0 {
			return true
		}
		fn, ok := call.Args[len(call.Args)-1].(*ast.FuncLit)
		if !ok {
			return true
		}
		if isConformance(types.ExprString(call.Fun)) {
			skips = append(skips, find(fn, isSkip)...)
			return false
		}
		// Any other call that takes a function may be a container.
		if len(find(fn, isConformance)) == 0 {
			return false
		}
		for _, stmt := range fn.Body.List {
			if expr, ok := stmt.(*ast.ExprStmt); ok && strings.HasPrefix(types.ExprString(expr.X), "ginkgo.BeforeEach(") {
				skips = append(skips, find(expr, isSkip)...)
			}
		}
		return true
	})
	return skips
}

func main() {
	if len(os.Args) != 2 {
		fmt.Fprintln(os.Stderr, "Usage: check_conformance_test_requirements [e2e-test-path]")
		os.Exit(1)
	}
	failed := false
	fset := token.NewFileSet()
	err := filepath.WalkDir(os.Args[1], func(path string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() || !strings.HasSuffix(path, ".go") {
			return err
		}
		file, err := parser.ParseFile(fset, path, nil, 0)
		if err != nil {
			return err
		}
		for _, pos := range checkFile(file) {
			fmt.Fprintf(os.Stderr, "%v: Conformance test should not call any e2eskipper.Skip*()\n", fset.Position(pos))
			failed = true
		}
		return nil
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
	if failed {
		os.Exit(1)
	}
}
