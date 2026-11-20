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
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func standardSearchSource(t *testing.T) []byte {
	t.Helper()
	source, err := os.ReadFile(filepath.Join(runtime.GOROOT(), "src", "slices", "sort.go"))
	if err != nil {
		t.Fatal(err)
	}
	return source
}

func TestSearchImportsCompleteStandardLibraryBody(t *testing.T) {
	program, err := importFunction(standardSearchSource(t), searchFunctions[0])
	if err != nil {
		t.Fatal(err)
	}
	for _, required := range []string{"(.unsignedMidpoint", "(.addNumbers", `"Invoke"`, "(.andAlso", "(.equalScalars", `"h$1"`} {
		if !strings.Contains(program, required) {
			t.Fatalf("standard search import dropped %q: %s", required, program)
		}
	}
}

func TestSearchRejectsUnsafeMidpointAndAliases(t *testing.T) {
	source := string(standardSearchSource(t))
	start := strings.Index(source, "func BinarySearchFunc[")
	end := start + strings.Index(source[start:], "\n}\n") + 2
	original := source[start:end]
	for name, change := range map[string][2]string{
		"signed shift":        {"int(uint(i+j) >> 1)", "int((i+j) >> 1)"},
		"wrong shift":         {"int(uint(i+j) >> 1)", "int(uint(i+j) >> 2)"},
		"product":             {"int(uint(i+j) >> 1)", "int(uint(i*j) >> 1)"},
		"shadow int":          {"h := int(uint(i+j) >> 1)", "int := 0; h := int(uint(i+j) >> 1)"},
		"shadow uint":         {"h := int(uint(i+j) >> 1)", "uint := 0; h := int(uint(i+j) >> 1)"},
		"array alias":         {"n := len(x)", "alias := x; n := len(alias)"},
		"comparator alias":    {"n := len(x)", "alias := cmp; n := len(x)"},
		"undeclared receiver": {"n := len(x)", "n := len(s)"},
	} {
		t.Run(name, func(t *testing.T) {
			if strings.Count(original, change[0]) != 1 {
				t.Fatalf("test mutation is not unique: %q", change[0])
			}
			changed := source[:start] + strings.Replace(original, change[0], change[1], 1) + source[end:]
			if _, err := importFunction([]byte(changed), searchFunctions[0]); err == nil {
				t.Fatalf("unsupported search arithmetic or alias was accepted: %s", name)
			}
		})
	}
}

func hasSource(body string) []byte {
	return []byte("package fieldpath\nimport \"slices\"\n" + hasFunctions[0].signature + ` {
_, found := slices.BinarySearchFunc(s.members, pe, func(a, b PathElement) int { ` + body + ` })
return found
}`)
}

func TestHasImportsOnlyClosedPathComparator(t *testing.T) {
	source := hasSource("return a.Compare(b)")
	if err := validateSearchImport(source); err != nil {
		t.Fatal(err)
	}
	program, err := importFunction(source, hasFunctions[0])
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(program, "(.pathComparator)") || !strings.Contains(program, `"BinarySearchFunc"`) {
		t.Fatalf("Has import lost the standard search or its callback: %s", program)
	}
	for _, body := range []string{"return b.Compare(a)", "return a.Compare(pe)", "a = b; return a.Compare(b)", "return a.Compare(b)+1"} {
		if _, err := importFunction(hasSource(body), hasFunctions[0]); err == nil {
			t.Fatalf("unmodeled comparison callback accepted: %s", body)
		}
	}
	for _, badImport := range []string{`import slices "example.com/slices"`, `import renamed "slices"`, `import . "slices"`} {
		changed := strings.Replace(string(source), `import "slices"`, badImport, 1)
		if err := validateSearchImport([]byte(changed)); err == nil {
			t.Fatalf("unrelated or unresolved standard search import accepted: %s", badImport)
		}
	}
}
