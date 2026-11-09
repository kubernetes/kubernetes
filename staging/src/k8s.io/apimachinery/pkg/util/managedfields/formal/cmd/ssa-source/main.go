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
	"reflect"
	"strconv"
	"strings"
)

type functionSpec struct {
	name         string
	leanName     string
	signature    string
	parameters   []string
	maps         map[string]string
	elements     bool
	updater      bool
	sets         bool
	receiver     string
	receiverID   string
	search       bool
	standalone   bool
	atomicFields bool
	namedResults []string
	atomicMerge  bool
	voidResult   bool
	initializer  bool
	rawSyntax    bool
}

var searchFunctions = []functionSpec{{
	name: "BinarySearchFunc", leanName: "binarySearchProgram", standalone: true,
	signature:  "func BinarySearchFunc[S ~[]E, E, T any](x S, target T, cmp func(E, T) int) (int, bool)",
	parameters: []string{"x", "target", "cmp"}, maps: map[string]string{"x": "search-input", "cmp": "search-comparator"},
	elements: true, search: true,
}}

var hasFunctions = []functionSpec{
	{
		name: "Has", leanName: "hasProgram", receiver: "*PathElementSet",
		signature: "func (s *PathElementSet) Has(pe PathElement) bool", parameters: []string{"pe"},
		elements: true, search: true,
	},
	{
		name: "Less", leanName: "lessProgram", receiver: "PathElement", receiverID: "e",
		signature: "func (e PathElement) Less(rhs PathElement) bool", parameters: []string{"rhs"},
		elements: true, search: true,
	},
}

var setFunctions = []functionSpec{
	{
		name: "Difference", leanName: "setDifferenceProgram", receiver: "*Set",
		signature: "func (s *Set) Difference(s2 *Set) *Set", parameters: []string{"s2"},
		maps: map[string]string{"s2": "set-input"}, elements: true, sets: true,
	},
	{
		name: "Difference", leanName: "nodeDifferenceProgram", receiver: "*SetNodeMap",
		signature: "func (s *SetNodeMap) Difference(s2 *Set) *SetNodeMap", parameters: []string{"s2"},
		maps: map[string]string{"s2": "set-input"}, elements: true, sets: true,
	},
	{
		name: "RecursiveDifference", leanName: "recursiveSetProgram", receiver: "*Set",
		signature: "func (s *Set) RecursiveDifference(s2 *Set) *Set", parameters: []string{"s2"},
		maps: map[string]string{"s2": "set-input"}, elements: true, sets: true,
	},
	{
		name: "RecursiveDifference", leanName: "recursiveNodeProgram", receiver: "*SetNodeMap",
		signature: "func (s *SetNodeMap) RecursiveDifference(s2 *Set) *SetNodeMap", parameters: []string{"s2"},
		maps: map[string]string{"s2": "set-input"}, elements: true, sets: true,
	},
	{
		name: "Filter", leanName: "excludeFilterProgram", receiver: "excludeFilter", receiverID: "t",
		signature: "func (t excludeFilter) Filter(set *Set) *Set", parameters: []string{"set"},
		maps: map[string]string{"set": "set-input"}, elements: true, sets: true,
	},
}

var updaterFunctions = []functionSpec{{
	name: "update", leanName: "updateProgram",
	signature:  "func (s *Updater) update(oldObject, newObject *typed.TypedValue, version fieldpath.APIVersion, managers fieldpath.ManagedFields, workflow string, force bool) (fieldpath.ManagedFields, *typed.Comparison, error)",
	parameters: []string{"oldObject", "newObject", "version", "managers", "workflow", "force"},
	maps:       map[string]string{"managers": "managed-write"}, updater: true,
}}

var elementFunctions = []functionSpec{
	{
		name: "Difference", leanName: "differenceProgram",
		signature:  "func (s *PathElementSet) Difference(s2 *PathElementSet) *PathElementSet",
		parameters: []string{"s2"}, maps: map[string]string{"s2": "element-input"}, elements: true,
	},
	{
		name: "Union", leanName: "unionProgram",
		signature:  "func (s *PathElementSet) Union(s2 *PathElementSet) *PathElementSet",
		parameters: []string{"s2"}, maps: map[string]string{"s2": "element-input"}, elements: true,
	},
	{
		name: "Intersection", leanName: "intersectionProgram",
		signature:  "func (s *PathElementSet) Intersection(s2 *PathElementSet) *PathElementSet",
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
	if function.initializer {
		return importMergeRule(source, function)
	}
	positions := token.NewFileSet()
	file, err := parser.ParseFile(positions, "update.go", source, 0)
	if err != nil {
		return "", err
	}
	var target *ast.FuncDecl
	for _, declaration := range file.Decls {
		if candidate, ok := declaration.(*ast.FuncDecl); ok && candidate.Name.Name == function.name {
			if function.standalone && candidate.Recv != nil {
				continue
			}
			if function.receiver != "" {
				if candidate.Recv == nil || len(candidate.Recv.List) != 1 {
					continue
				}
				var receiver bytes.Buffer
				if err := format.Node(&receiver, positions, candidate.Recv.List[0].Type); err != nil {
					return "", err
				}
				if receiver.String() != function.receiver {
					continue
				}
			}
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
	if function.rawSyntax {
		if err := validateRawBindings(target); err != nil {
			return "", err
		}
		return renderRaw(reflect.ValueOf(target.Body))
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
	return validateImportBindings(file.Imports, expected)
}

func validateImportBindings(imports []*ast.ImportSpec, expected map[string]string) error {
	for _, imported := range imports {
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

func validateSearchImport(source []byte) error {
	file, err := parser.ParseFile(token.NewFileSet(), "element.go", source, parser.ImportsOnly)
	if err != nil {
		return err
	}
	found := false
	for _, imported := range file.Imports {
		path, err := strconv.Unquote(imported.Path.Value)
		if err != nil {
			return err
		}
		name := path[strings.LastIndex(path, "/")+1:]
		if imported.Name != nil {
			name = imported.Name.Name
		}
		if name != "slices" {
			continue
		}
		if path != "slices" || found {
			return fmt.Errorf("Has requires exactly one standard-library slices import")
		}
		found = true
	}
	if !found {
		return fmt.Errorf("Has is missing the standard-library slices import")
	}
	return nil
}

func run() error {
	sourcePath := flag.String("source", "", "selected Go source file (required)")
	checkPath := flag.String("check", "", "compare generated output with this Lean file")
	elements := flag.Bool("elements", false, "import fieldpath element.go instead of merge update.go")
	updater := flag.Bool("updater", false, "import the complete ownership-update function with the updater interpreter")
	sets := flag.Bool("sets", false, "import field-set Difference, RecursiveDifference and exclude-filter bodies from fieldpath/set.go")
	search := flag.Bool("search", false, "import Go standard-library slices.BinarySearchFunc")
	has := flag.Bool("has", false, "import PathElementSet.Has and PathElement.Less")
	atomicFields := flag.Bool("atomic-fields", false, "import the complete typed field-set map/list visitors")
	schemaLookup := flag.Bool("schema-lookup", false, "import complete Schema.Resolve and no-override lookup bodies")
	schemaDispatch := flag.Bool("schema-dispatch", false, "import complete schema resolution, atom deduction and handler dispatch bodies")
	atomicRemove := flag.Bool("atomic-remove", false, "import complete structural typed removal/extraction ASTs")
	atomicMerge := flag.Bool("atomic-merge", false, "import complete typed atomic merge dispatch and leaf replacement")
	flag.Parse()
	if *sourcePath == "" || flag.NArg() != 0 {
		return fmt.Errorf("supply -source and no positional arguments")
	}
	modes := 0
	for _, active := range []bool{*elements, *updater, *sets, *search, *has, *atomicFields, *atomicMerge, *atomicRemove, *schemaDispatch, *schemaLookup} {
		if active {
			modes++
		}
	}
	if modes > 1 {
		return fmt.Errorf("select at most one source mode: -elements, -updater, -sets, -search, -has, -atomic-fields, -atomic-merge, -atomic-remove, -schema-dispatch or -schema-lookup")
	}
	source, err := os.ReadFile(*sourcePath)
	if err != nil {
		return err
	}
	selected := functions
	namespace, description := "GoSource", "merge/update.go"
	if *elements || *sets || *has || *search || *atomicFields || *atomicMerge || *atomicRemove || *schemaDispatch || *schemaLookup {
		file, err := parser.ParseFile(token.NewFileSet(), "element.go", source, parser.PackageClauseOnly)
		if err != nil {
			return err
		}
		pkg := "fieldpath"
		if *search {
			pkg = "slices"
		}
		if *atomicFields || *atomicMerge || *atomicRemove || *schemaDispatch || *schemaLookup {
			pkg = "typed"
		}
		if *schemaLookup {
			pkg = "schema"
		}
		if file.Name.Name != pkg {
			return fmt.Errorf("unsupported source package %q; expected %q", file.Name.Name, pkg)
		}
		selected = elementFunctions
		namespace, description = "ElementSource", "fieldpath/element.go"
		if *sets {
			selected = setFunctions
			namespace, description = "SetSource", "fieldpath/set.go"
		}
		if *has {
			selected = hasFunctions
			namespace, description = "HasSource", "fieldpath/element.go"
			if err := validateSearchImport(source); err != nil {
				return err
			}
		}
		if *search {
			selected = searchFunctions
			namespace, description = "SearchSource", "slices/sort.go"
		}
		if *atomicFields {
			selected = atomicFieldFunctions
			namespace, description = "AtomicFieldSource", "typed/tofieldset.go"
			if err := validateAtomicFieldImports(source); err != nil {
				return err
			}
		}
		if *atomicMerge {
			selected = atomicMergeFunctions
			namespace, description = "AtomicMergeSource", "typed/merge.go"
			if err := validateAtomicFieldImports(source); err != nil {
				return err
			}
		}
	} else {
		if err := validateImports(source); err != nil {
			return err
		}
		if *updater {
			selected = updaterFunctions
			namespace = "UpdaterSource"
		}
	}
	if *atomicRemove {
		selected = atomicRemoveFunctions
		namespace, description = "AtomicRemoveSource", "typed/remove.go"
		if err := validateAtomicFieldImports(source); err != nil {
			return err
		}
	}
	if *schemaDispatch {
		selected = schemaDispatchFunctions
		namespace, description = "SchemaDispatchSource", "typed/helpers.go"
		if err := validateAtomicFieldImports(source, "errors", "fmt", "strings"); err != nil {
			return err
		}
	}
	if *schemaLookup {
		selected = schemaLookupFunctions
		namespace, description = "SchemaLookupSource", "schema/elements.go"
		file, err := parser.ParseFile(token.NewFileSet(), "elements.go", source, parser.ImportsOnly)
		if err != nil {
			return err
		}
		if err := validateImportBindings(file.Imports, map[string]string{"sync": "sync", "atomic": "sync/atomic"}); err != nil {
			return err
		}
	}
	programType, importModule := "List GoSubset.Stmt", "GoSubset"
	if *atomicRemove || *schemaDispatch || *schemaLookup {
		programType, importModule = "RawGoSyntax.Node", "RawGoSyntax"
	}
	var programs strings.Builder
	for _, function := range selected {
		program, err := importFunction(source, function)
		if err != nil {
			return err
		}
		fmt.Fprintf(&programs, "def %s : %s := %s\n\n", function.leanName, programType, program)
	}
	licenseID, originNotice := "Apache-2.0", ""
	sourceLabel := "vendor/sigs.k8s.io/structured-merge-diff/v6/" + description
	if *search {
		sourceLabel = "Go standard library " + description
		licenseID = "Apache-2.0 AND BSD-3-Clause"
		originNotice = strings.TrimPrefix(strings.SplitN(string(source), "\n", 2)[0], "// ") + "\n\n" + goSourceLicense
	}
	generated := fmt.Sprintf(`/-
Copyright 2026 The Kubernetes Authors.
SPDX-License-Identifier: %s
%s
Generated by cmd/ssa-source. Do not edit.
Source: %s
SHA-256: %x
-/
import SSA.%s

namespace SSA.%s

%send SSA.%s
`, licenseID, originNotice, sourceLabel, sha256.Sum256(source), importModule, namespace, programs.String(), namespace)
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
