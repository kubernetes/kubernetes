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
	"bytes"
	"crypto/sha256"
	"flag"
	"fmt"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"os"
	"strconv"
	"strings"
)

const function = "addBackDanglingItems"

func quoted(value string) (string, error) {
	for _, char := range value {
		if char < 32 || char > 126 {
			return "", fmt.Errorf("unsupported non-ASCII Go string %q", value)
		}
	}
	return strconv.Quote(value), nil
}

func expr(expression ast.Expr, known map[string]bool) (string, error) {
	switch e := expression.(type) {
	case *ast.Ident:
		if !known[e.Name] {
			return "", fmt.Errorf("unsupported global or unbound Go name %q", e.Name)
		}
		name, err := quoted(e.Name)
		return "(.var " + name + ")", err
	case *ast.BasicLit:
		if e.Kind != token.STRING {
			return "", fmt.Errorf("unsupported Go literal %s", e.Value)
		}
		value, err := strconv.Unquote(e.Value)
		if err != nil {
			return "", err
		}
		text, err := quoted(value)
		return "(.text " + text + ")", err
	case *ast.SelectorExpr:
		receiver, err := expr(e.X, known)
		if err != nil {
			return "", err
		}
		field, err := quoted(e.Sel.Name)
		return "(.select " + receiver + " " + field + ")", err
	case *ast.CallExpr:
		method, ok := e.Fun.(*ast.SelectorExpr)
		if !ok || e.Ellipsis.IsValid() {
			return "", fmt.Errorf("unsupported Go call %T", e.Fun)
		}
		receiver, err := expr(method.X, known)
		if err != nil {
			return "", err
		}
		args, err := expressions(e.Args, known)
		if err != nil {
			return "", err
		}
		name, err := quoted(method.Sel.Name)
		return "(.call " + receiver + " " + name + " " + args + ")", err
	case *ast.BinaryExpr:
		nilName, ok := e.Y.(*ast.Ident)
		if e.Op != token.NEQ || !ok || nilName.Name != "nil" {
			return "", fmt.Errorf("unsupported Go comparison %s", e.Op)
		}
		value, err := expr(e.X, known)
		return "(.neNil " + value + ")", err
	default:
		return "", fmt.Errorf("unsupported Go expression %T", expression)
	}
}

func expressions(input []ast.Expr, known map[string]bool) (string, error) {
	items := make([]string, 0, len(input))
	for _, input := range input {
		item, err := expr(input, known)
		if err != nil {
			return "", err
		}
		items = append(items, item)
	}
	return "[" + strings.Join(items, ", ") + "]", nil
}

func statements(input []ast.Stmt, branch bool, known map[string]bool) (string, error) {
	items := make([]string, 0, len(input))
	for _, statement := range input {
		switch s := statement.(type) {
		case *ast.AssignStmt:
			if branch || (s.Tok != token.ASSIGN && s.Tok != token.DEFINE) {
				return "", fmt.Errorf("unsupported Go assignment or block scope")
			}
			names := make([]string, 0, len(s.Lhs))
			bindings := make([]string, 0, len(s.Lhs))
			for _, target := range s.Lhs {
				name, ok := target.(*ast.Ident)
				if !ok || name.Name == "_" {
					return "", fmt.Errorf("unsupported Go assignment target %T", target)
				}
				if s.Tok == token.ASSIGN && !known[name.Name] {
					return "", fmt.Errorf("assignment to unbound Go name %q", name.Name)
				}
				text, err := quoted(name.Name)
				if err != nil {
					return "", err
				}
				names = append(names, text)
				bindings = append(bindings, name.Name)
			}
			values, err := expressions(s.Rhs, known)
			if err != nil {
				return "", err
			}
			for _, name := range bindings {
				known[name] = true
			}
			items = append(items, "(.assign ["+strings.Join(names, ", ")+"] "+values+")")
		case *ast.IfStmt:
			if s.Init != nil || s.Else != nil {
				return "", fmt.Errorf("unsupported Go if initializer or else")
			}
			condition, err := expr(s.Cond, known)
			if err != nil {
				return "", err
			}
			body, err := statements(s.Body.List, true, known)
			if err != nil {
				return "", err
			}
			items = append(items, "(.branch "+condition+" "+body+")")
		case *ast.ReturnStmt:
			if len(s.Results) == 0 {
				return "", fmt.Errorf("unsupported Go named return")
			}
			values, err := expressions(s.Results, known)
			if err != nil {
				return "", err
			}
			items = append(items, "(.ret "+values+")")
		default:
			return "", fmt.Errorf("unsupported Go statement %T", statement)
		}
	}
	return "[\n  " + strings.Join(items, ",\n  ") + "\n]", nil
}

func importProgram(source []byte) (string, error) {
	positions := token.NewFileSet()
	file, err := parser.ParseFile(positions, "update.go", source, 0)
	if err != nil {
		return "", err
	}
	var target *ast.FuncDecl
	for _, declaration := range file.Decls {
		if candidate, ok := declaration.(*ast.FuncDecl); ok && candidate.Name.Name == function {
			if target != nil {
				return "", fmt.Errorf("duplicate source function %s", function)
			}
			target = candidate
		}
	}
	if target == nil || target.Body == nil {
		return "", fmt.Errorf("source function %s was not found", function)
	}
	var signature bytes.Buffer
	if err := format.Node(&signature, positions, &ast.FuncDecl{Recv: target.Recv, Name: target.Name, Type: target.Type}); err != nil {
		return "", err
	}
	const expected = "func (s *Updater) addBackDanglingItems(merged, pruned *typed.TypedValue, lastSet fieldpath.VersionedSet) (*typed.TypedValue, error)"
	if strings.TrimSpace(signature.String()) != expected {
		return "", fmt.Errorf("unsupported source function signature: %s", signature.String())
	}
	return statements(target.Body.List, false, map[string]bool{
		"s": true, "merged": true, "pruned": true, "lastSet": true, "fmt": true, "nil": true,
	})
}

func run() error {
	sourcePath := flag.String("source", "", "Go update.go source (required)")
	checkPath := flag.String("check", "", "compare generated output with this Lean file")
	flag.Parse()
	if *sourcePath == "" || flag.NArg() != 0 {
		return fmt.Errorf("supply -source and no positional arguments")
	}
	source, err := os.ReadFile(*sourcePath)
	if err != nil {
		return err
	}
	program, err := importProgram(source)
	if err != nil {
		return err
	}
	generated := fmt.Sprintf(`/-
Copyright 2026 The Kubernetes Authors.
SPDX-License-Identifier: Apache-2.0

Generated by cmd/ssa-source. Do not edit.
Source: vendor/sigs.k8s.io/structured-merge-diff/v6/merge/update.go
SHA-256: %x
-/
import SSA.GoSubset

namespace SSA.GoSource

def danglingProgram : List GoSubset.Stmt := %s

end SSA.GoSource
`, sha256.Sum256(source), program)
	if *checkPath == "" {
		_, err = fmt.Print(generated)
		return err
	}
	expected, err := os.ReadFile(*checkPath)
	if err != nil {
		return err
	}
	if string(expected) != generated {
		return fmt.Errorf("%s is stale; regenerate it from %s with cmd/ssa-source", *checkPath, *sourcePath)
	}
	fmt.Println("Imported Lean program matches the complete Go function AST and source digest.")
	return nil
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
