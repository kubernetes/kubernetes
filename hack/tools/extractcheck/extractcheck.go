/*
Copyright 2023 The Kubernetes Authors.

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
	"flag"
	"fmt"
	"go/ast"
	"go/types"
	"path"
	"regexp"
	"strings"

	"golang.org/x/tools/go/analysis"
)

type config struct {
	matches []*regexp.Regexp
}

func newConfig() config {
	var matches []*regexp.Regexp
	excludeRegex := []string{
		"k8s.io/kubernetes/test/integration/client/client_test.go",
		"k8s.io/kubernetes/test/integration/client/dynamic_client_test.go",
		"k8s.io/client-go/applyconfigurations/.*",
	}

	for _, f := range excludeRegex {
		re, err := regexp.Compile(f)
		if err != nil {
			panic(err)
		}
		matches = append(matches, re)
	}
	return config{matches}
}

func (c config) isEnable(filename string) bool {
	for _, r := range c.matches {
		if matchFullString(filename, r) {
			return false
		}
	}
	return true
}

func Analyser() *analysis.Analyzer {
	flags := flag.NewFlagSet("", flag.ExitOnError)
	return &analysis.Analyzer{
		Name: "extractcheck",
		Doc:  "Tool to check SSA extract calls.",
		Run: func(pass *analysis.Pass) (interface{}, error) {
			return run(pass)
		},
		Flags: *flags,
	}
}

func run(pass *analysis.Pass) (interface{}, error) {
	c := newConfig()
	for _, file := range pass.Files {
		filename := pass.Pkg.Path() + "/" + path.Base(pass.Fset.Position(file.Package).Filename)
		if !c.isEnable(filename) {
			continue
		}
		ast.Inspect(file, func(n ast.Node) bool {
			switch n := n.(type) {
			case *ast.CallExpr:
				// We are interested in function calls, as we want to detect calls to
				//   * k8s.io/apimachinery/pkg/util/managedfields.ExtractInto()
				//   * k8s.io/client-go/applyconfigurations.Extract*()
				checkForFunctionExpr(n, pass)
			}

			return true
		})
	}
	return nil, nil
}

// checkForFunctionExpr checks if the call expression matches SSA Extract-model.
func checkForFunctionExpr(fexpr *ast.CallExpr, pass *analysis.Pass) {
	fun := fexpr.Fun

	if selExpr, ok := fun.(*ast.SelectorExpr); ok {
		if isExtractManagedFields(selExpr, pass) || isExtractApplyConfigrations(selExpr, pass) {
			pass.Report(analysis.Diagnostic{
				Pos:     fexpr.Pos(),
				Message: fmt.Sprintf("Extract function %q should not be used because managedFields was removed in kube-controller-manager/kube-scheduler", selExpr.Sel.Name),
			})
		}
	}
}

// isExtractManagedFields returns true if the type of the expression is
// k8s.io/client-go/applyconfigurations.ExtractInto.
func isExtractManagedFields(expr *ast.SelectorExpr, pass *analysis.Pass) bool {
	fName := expr.Sel.Name
	return strings.HasPrefix(fName, "ExtractInto") && fromPackage(expr.Sel, "k8s.io/apimachinery/pkg/util/managedfields", pass)
}

// isExtractApplyConfigrations returns true if the type of the expression is
// k8s.io/client-go/applyconfigurations.Extract* (e.g. ExtractPod)
func isExtractApplyConfigrations(expr *ast.SelectorExpr, pass *analysis.Pass) bool {
	fName := expr.Sel.Name
	return strings.HasPrefix(fName, "Extract") && fromPackage(expr.Sel, "k8s.io/client-go/applyconfigurations", pass)
}

// fromPackage checks whether an expression is an identifier that refers
// to a package under target path.
func fromPackage(expr ast.Expr, packagePath string, pass *analysis.Pass) bool {
	if ident, ok := expr.(*ast.Ident); ok {
		if object, ok := pass.TypesInfo.Uses[ident]; ok {
			switch object := object.(type) {
			case *types.Func:
				pkg := object.Pkg()
				if strings.HasPrefix(pkg.Path(), packagePath) {
					return true
				}
			}
		}
	}

	return false
}

func matchFullString(str string, re *regexp.Regexp) bool {
	loc := re.FindStringIndex(str)
	if loc == nil {
		// No match at all.
		return false
	}
	if loc[1]-loc[0] < len(str) {
		// Only matches a substring.
		return false
	}
	return true
}
