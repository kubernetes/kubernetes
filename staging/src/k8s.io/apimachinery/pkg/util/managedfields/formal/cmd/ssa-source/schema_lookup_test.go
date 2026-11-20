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
	"go/parser"
	"go/token"
	"strings"
	"testing"
)

func TestSchemaLookupRetainsCompleteBodies(t *testing.T) {
	source := selectedSMDSource(t, "schema", "elements.go")
	fragments := [][]string{
		{"NamedType", "FindNamedType", "StarExpr", "Inlined", "ReturnStmt"},
		{"ElementRelationship", "resolveNoOverrides", "DeferStmt", "Unlock", "resolvedTypes", "IndexExpr", "SwitchStmt", "CopyInto", "mapCopy", "listCopy", "DeclStmt"},
		{"Fields", "ElementType", "Unions", "ElementRelationship", "Load", "Store", "Once", "FuncLit", "dst", "mm"},
	}
	for index, function := range schemaLookupFunctions {
		program, err := importFunction(source, function)
		if err != nil {
			t.Fatalf("%s: %v", function.name, err)
		}
		for _, fragment := range fragments[index] {
			if !strings.Contains(program, fragment) {
				t.Fatalf("%s dropped %q", function.name, fragment)
			}
		}
	}
}

func TestSchemaLookupChecksSyncBindings(t *testing.T) {
	source := selectedSMDSource(t, "schema", "elements.go")
	for _, test := range []struct{ old, new string }{
		{`"sync"`, `locking "sync"`},
		{`"sync/atomic"`, `atomic "example.invalid/atomic"`},
	} {
		mutated := strings.Replace(string(source), test.old, test.new, 1)
		if mutated == string(source) {
			t.Fatalf("missing import mutation: %s", test.old)
		}
		file, err := parser.ParseFile(token.NewFileSet(), "elements.go", mutated, parser.ImportsOnly)
		if err != nil {
			t.Fatal(err)
		}
		if err := validateImportBindings(file.Imports, map[string]string{"sync": "sync", "atomic": "sync/atomic"}); err == nil {
			t.Fatalf("accepted changed schema synchronization binding: %s", test.new)
		}
	}
}

func TestSchemaLookupRejectsChangedSignatureAndGlobalCache(t *testing.T) {
	function := schemaLookupFunctions[1]
	for _, source := range []string{
		"package schema\n" + strings.Replace(function.signature, "tr TypeRef", "tr *TypeRef", 1) + " { return Atom{}, true }",
		"package schema\nvar cache map[TypeRef]Atom\n" + function.signature + " { result, ok := cache[tr]; return result, ok }",
	} {
		if _, err := importFunction([]byte(source), function); err == nil {
			t.Fatalf("accepted changed schema signature or global cache: %s", source)
		}
	}
}
