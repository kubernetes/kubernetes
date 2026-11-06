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

type functionSpec struct {
	name       string
	leanName   string
	signature  string
	parameters []string
	maps       map[string]string
	elements   bool
}

var elementFunctions = []functionSpec{
	{
		name: "Difference", leanName: "differenceProgram",
		signature:  "func (s *PathElementSet) Difference(s2 *PathElementSet) *PathElementSet",
		parameters: []string{"s2"}, maps: map[string]string{"s2": "element-input"}, elements: true,
	},
}

var functions = []functionSpec{
	{
		name: "addBackDanglingItems", leanName: "danglingProgram",
		signature:  "func (s *Updater) addBackDanglingItems(merged, pruned *typed.TypedValue, lastSet fieldpath.VersionedSet) (*typed.TypedValue, error)",
		parameters: []string{"merged", "pruned", "lastSet"},
	},
	{
		name: "addBackOwnedItemsForVersion", leanName: "ownedVersionProgram",
		signature:  "func (s *Updater) addBackOwnedItemsForVersion(merged, pruned *typed.TypedValue, version fieldpath.APIVersion, managed *fieldpath.Set) (*typed.TypedValue, *typed.TypedValue, error)",
		parameters: []string{"merged", "pruned", "version", "managed"},
	},
	{
		name: "prune", leanName: "pruneProgram",
		signature:  "func (s *Updater) prune(merged *typed.TypedValue, managers fieldpath.ManagedFields, applyingManager string, lastSet fieldpath.VersionedSet) (*typed.TypedValue, error)",
		parameters: []string{"merged", "managers", "applyingManager", "lastSet"},
		maps:       map[string]string{"managers": "managed"},
	},
	{
		name: "addBackOwnedItems", leanName: "ownedItemsProgram",
		signature:  "func (s *Updater) addBackOwnedItems(merged, pruned *typed.TypedValue, prunedVersion fieldpath.APIVersion, managedFields fieldpath.ManagedFields, applyingManager string) (*typed.TypedValue, error)",
		parameters: []string{"merged", "pruned", "prunedVersion", "managedFields", "applyingManager"},
		maps:       map[string]string{"managedFields": "managed"},
	},
	{
		name: "Apply", leanName: "applyProgram",
		signature:  "func (s *Updater) Apply(liveObject, configObject *typed.TypedValue, version fieldpath.APIVersion, managers fieldpath.ManagedFields, manager string, force bool) (*typed.TypedValue, fieldpath.ManagedFields, error)",
		parameters: []string{"liveObject", "configObject", "version", "managers", "manager", "force"},
		maps:       map[string]string{"managers": "managed-write"},
	},
}

func quoted(value string) (string, error) {
	for _, char := range value {
		if char < 32 || char > 126 {
			return "", fmt.Errorf("unsupported non-ASCII Go string %q", value)
		}
	}
	return strconv.Quote(value), nil
}

func importFunction(source []byte, function functionSpec) (string, error) {
	positions := token.NewFileSet()
	file, err := parser.ParseFile(positions, "update.go", source, 0)
	if err != nil {
		return "", err
	}
	var target *ast.FuncDecl
	for _, declaration := range file.Decls {
		if candidate, ok := declaration.(*ast.FuncDecl); ok && candidate.Name.Name == function.name {
			if target != nil {
				return "", fmt.Errorf("duplicate source function %s", function.name)
			}
			target = candidate
		}
	}
	if target == nil || target.Body == nil {
		return "", fmt.Errorf("source function %s was not found", function.name)
	}
	var signature bytes.Buffer
	if err := format.Node(&signature, positions, &ast.FuncDecl{Recv: target.Recv, Name: target.Name, Type: target.Type}); err != nil {
		return "", err
	}
	if strings.TrimSpace(signature.String()) != function.signature {
		return "", fmt.Errorf("unsupported source function signature: %s", signature.String())
	}
	return renderStatements(target.Body.List, rootScope(function))
}

func importProgram(source []byte) (string, error) {
	return importFunction(source, functions[0])
}

func validateImports(source []byte) error {
	file, err := parser.ParseFile(token.NewFileSet(), "update.go", source, parser.ImportsOnly)
	if err != nil {
		return err
	}
	if file.Name.Name != "merge" {
		return fmt.Errorf("unsupported Go source package %q", file.Name.Name)
	}
	expected := map[string]string{
		"fmt":       "fmt",
		"typed":     "sigs.k8s.io/structured-merge-diff/v6/typed",
		"fieldpath": "sigs.k8s.io/structured-merge-diff/v6/fieldpath",
		"value":     "sigs.k8s.io/structured-merge-diff/v6/value",
	}
	for _, imported := range file.Imports {
		path, err := strconv.Unquote(imported.Path.Value)
		if err != nil {
			return err
		}
		name := path[strings.LastIndex(path, "/")+1:]
		if imported.Name != nil {
			name = imported.Name.Name
		}
		if required, ok := expected[name]; ok {
			if path != required {
				return fmt.Errorf("Go import %q binds %q instead of %q", name, path, required)
			}
			delete(expected, name)
		}
	}
	if len(expected) != 0 {
		return fmt.Errorf("Go source is missing required import bindings: %v", expected)
	}
	return nil
}

func run() error {
	sourcePath := flag.String("source", "", "Go update.go source, or element.go with -elements (required)")
	checkPath := flag.String("check", "", "compare generated output with this Lean file")
	elements := flag.Bool("elements", false, "import fieldpath element.go instead of merge update.go")
	flag.Parse()
	if *sourcePath == "" || flag.NArg() != 0 {
		return fmt.Errorf("supply -source and no positional arguments")
	}
	source, err := os.ReadFile(*sourcePath)
	if err != nil {
		return err
	}
	selected := functions
	namespace, description := "GoSource", "merge/update.go"
	if *elements {
		file, err := parser.ParseFile(token.NewFileSet(), "element.go", source, parser.PackageClauseOnly)
		if err != nil {
			return err
		}
		if file.Name.Name != "fieldpath" {
			return fmt.Errorf("unsupported element-set source package %q", file.Name.Name)
		}
		selected = elementFunctions
		namespace, description = "ElementSource", "fieldpath/element.go"
	} else {
		if err := validateImports(source); err != nil {
			return err
		}
	}
	var programs strings.Builder
	for _, function := range selected {
		program, err := importFunction(source, function)
		if err != nil {
			return err
		}
		fmt.Fprintf(&programs, "def %s : List GoSubset.Stmt := %s\n\n", function.leanName, program)
	}
	generated := fmt.Sprintf(`/-
Copyright 2026 The Kubernetes Authors.
SPDX-License-Identifier: Apache-2.0

Generated by cmd/ssa-source. Do not edit.
Source: vendor/sigs.k8s.io/structured-merge-diff/v6/%s
SHA-256: %x
-/
import SSA.GoSubset

namespace SSA.%s

%send SSA.%s
`, description, sha256.Sum256(source), namespace, programs.String(), namespace)
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
	fmt.Printf("Imported Lean programs match %d complete Go function ASTs and source digest.\n", len(selected))
	return nil
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
