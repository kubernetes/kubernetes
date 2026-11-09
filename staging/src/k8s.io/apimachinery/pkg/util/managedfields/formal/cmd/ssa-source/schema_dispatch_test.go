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
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func selectedSMDSource(t *testing.T, packageName, filename string) []byte {
	t.Helper()
	directory, err := exec.Command("go", "list", "-f", "{{.Dir}}", "sigs.k8s.io/structured-merge-diff/v6/"+packageName).Output()
	if err != nil {
		t.Fatal(err)
	}
	source, err := os.ReadFile(filepath.Join(strings.TrimSpace(string(directory)), filename))
	if err != nil {
		t.Fatal(err)
	}
	return source
}

func TestSchemaDispatchRetainsCompleteBodies(t *testing.T) {
	source := selectedSMDSource(t, "typed", "helpers.go")
	fragments := [][]string{
		{"Resolve", "deduceAtom", "handleAtom", "errorf", "NamedType", "UnaryExpr"},
		{"SwitchStmt", "CaseClause", "IsFloat", "IsInt", "IsString", "IsBool", "IsList", "IsMap", "CompositeLit"},
		{"SwitchStmt", "doMap", "doScalar", "doList", "errorf", "NamedType"},
	}
	for index, function := range schemaDispatchFunctions {
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

func TestSchemaDispatchChecksStandardAndDependencyBindings(t *testing.T) {
	source := selectedSMDSource(t, "typed", "helpers.go")
	if err := validateAtomicFieldImports(source, "errors", "fmt", "strings"); err != nil {
		t.Fatal(err)
	}
	for _, mutation := range []struct{ old, new string }{
		{`"fmt"`, `format "fmt"`},
		{`"strings"`, `"other.example/strings"`},
		{`"sigs.k8s.io/structured-merge-diff/v6/schema"`, `schema "other.example/schema"`},
	} {
		mutated := strings.Replace(string(source), mutation.old, mutation.new, 1)
		if mutated == string(source) {
			t.Fatalf("missing import mutation %s", mutation.old)
		}
		if err := validateAtomicFieldImports([]byte(mutated), "errors", "fmt", "strings"); err == nil {
			t.Fatalf("accepted changed schema binding: %s", mutation.new)
		}
	}
}

func TestStructuralStringConstantsRetainTokenAndDecodedValue(t *testing.T) {
	token := `"a\x62\u0063"`
	expression, err := parser.ParseExpr(token)
	if err != nil {
		t.Fatal(err)
	}
	syntax, err := renderRaw(reflect.ValueOf(expression))
	if err != nil {
		t.Fatal(err)
	}
	original, err := quoted(token)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(syntax, original) || !strings.Contains(syntax, `("DecodedString",(.atom "abc"))`) {
		t.Fatalf("lost string token or Go decoding: %s", syntax)
	}
}

func TestRawFramesRejectGlobalsAndShadowing(t *testing.T) {
	function := schemaDispatchFunctions[1]
	for _, test := range []struct{ declaration, body string }{
		{"var v value.Value", "if v.IsMap() { return schema.Atom{} }; return atom"},
		{`const name = "global"`, "_ = name; return atom"},
		{"", "{ val := val; _ = val }; return atom"},
		{"", "nil := val; _ = nil; return atom"},
	} {
		source := []byte("package typed\n" + test.declaration + "\n" + function.signature + " { " + test.body + " }\n")
		if _, err := importFunction(source, function); err == nil {
			t.Fatalf("accepted global/shadowed frame variable: %s", source)
		}
	}
}
