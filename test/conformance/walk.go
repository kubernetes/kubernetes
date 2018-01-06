/*
Copyright 2017 The Kubernetes Authors.

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

// Package main provides a tool that scans kubernetes e2e test source code
// looking for conformance test declarations, which it emits on stdout.  It
// also looks for legacy, manually added "[Conformance]" tags and reports an
// error if it finds any.
//
// This approach is not air tight, but it will serve our purpose as a
// pre-submit check.
package main

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"
)

type visitor struct {
	FileSet *token.FileSet
}

func newVisitor() *visitor {
	return &visitor{
		FileSet: token.NewFileSet(),
	}
}

func (v *visitor) isConformanceCall(call *ast.CallExpr) bool {
	switch fun := call.Fun.(type) {
	case *ast.SelectorExpr:
		if fun.Sel != nil {
			return fun.Sel.Name == "ConformanceIt"
		}
	}
	return false
}

func (v *visitor) isLegacyItCall(call *ast.CallExpr) bool {
	switch fun := call.Fun.(type) {
	case *ast.Ident:
		if fun.Name != "It" {
			return false
		}
		if len(call.Args) < 1 {
			v.failf(call, "Not enough arguments to It()")
		}
	default:
		return false
	}

	switch arg := call.Args[0].(type) {
	case *ast.BasicLit:
		if arg.Kind != token.STRING {
			v.failf(arg, "Unexpected non-string argument to It()")
		}
		if strings.Contains(arg.Value, "[Conformance]") {
			return true
		}
	default:
		// non-literal argument to It()... we just ignore these even though they could be a way to "sneak in" a conformance test
	}

	return false
}

func (v *visitor) failf(expr ast.Expr, format string, a ...interface{}) {
	msg := fmt.Sprintf(format, a...)
	fmt.Fprintf(os.Stderr, "ERROR at %v: %s\n", v.FileSet.Position(expr.Pos()), msg)
	os.Exit(65)
}

func (v *visitor) emit(arg ast.Expr) {
	switch at := arg.(type) {
	case *ast.BasicLit:
		if at.Kind != token.STRING {
			v.failf(at, "framework.ConformanceIt() called with non-string argument")
			return
		}
		fmt.Printf("%s: %s\n", v.FileSet.Position(at.Pos()).Filename, at.Value)
	default:
		v.failf(at, "framework.ConformanceIt() called with non-literal argument")
		fmt.Fprintf(os.Stderr, "ERROR: non-literal argument %v at %v\n", arg, v.FileSet.Position(arg.Pos()))
	}
}

// Visit visits each node looking for either calls to framework.ConformanceIt,
// which it will emit in its list of conformance tests, or legacy calls to
// It() with a manually embedded [Conformance] tag, which it will complain
// about.
func (v *visitor) Visit(node ast.Node) (w ast.Visitor) {
	switch t := node.(type) {
	case *ast.CallExpr:
		if v.isConformanceCall(t) {
			v.emit(t.Args[0])
		} else if v.isLegacyItCall(t) {
			v.failf(t, "Using It() with manual [Conformance] tag is no longer allowed.  Use framework.ConformanceIt() instead.")
			return nil
		}
	}
	return v
}

func scandir(dir string) {
	v := newVisitor()
	pkg, err := parser.ParseDir(v.FileSet, dir, nil, 0)
	if err != nil {
		panic(err)
	}

	for _, p := range pkg {
		ast.Walk(v, p)
	}
}

func scanfile(path string) {
	v := newVisitor()
	file, err := parser.ParseFile(v.FileSet, path, nil, 0)
	if err != nil {
		panic(err)
	}

	ast.Walk(v, file)
}

func main() {
	args := os.Args[1:]
	if len(args) < 1 {
		fmt.Fprintf(os.Stderr, "USAGE: %s <DIR or FILE> [...]\n", os.Args[0])
		os.Exit(64)
	}

	for _, arg := range args {
		filepath.Walk(arg, func(path string, info os.FileInfo, err error) error {
			if err != nil {
				return err
			}
			if info.IsDir() {
				scandir(path)
			} else {
				// TODO(mml): Remove this once we have all-go-srcs build rules.  See https://github.com/kubernetes/repo-infra/pull/45
				if strings.HasSuffix(path, ".go") {
					scanfile(path)
				}
			}
			return nil
		})
	}
}
