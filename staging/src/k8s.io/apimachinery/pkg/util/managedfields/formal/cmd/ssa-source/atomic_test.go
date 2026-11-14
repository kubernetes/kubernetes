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
	"strings"
	"testing"
)

func atomicFieldSource(function functionSpec, body string) []byte {
	return []byte(`package typed
import (
  "sigs.k8s.io/structured-merge-diff/v6/fieldpath"
  "sigs.k8s.io/structured-merge-diff/v6/schema"
  "sigs.k8s.io/structured-merge-diff/v6/value"
)
` + function.signature + " {\n" + body + "\n}\n")
}

func TestAtomicFieldImportsKeepEffectsCleanupAndUnvisitedTail(t *testing.T) {
	for _, function := range atomicFieldFunctions {
		container, visitor := "list", "visitListItems"
		if function.name == "doMap" {
			container, visitor = "map", "visitMapItems"
		}
		body := "resource, _ := " + container + `Value(v.allocator, v.value)
if resource != nil { defer v.allocator.Free(resource) }
if t.ElementRelationship == schema.Atomic { v.set.Insert(v.path); return nil }
if resource == nil { return nil }
errs = v.` + visitor + `(t, resource)
return errs`
		source := atomicFieldSource(function, body)
		if err := validateAtomicFieldImports(source); err != nil {
			t.Fatal(err)
		}
		program, err := importFunction(source, function)
		if err != nil {
			t.Fatal(err)
		}
		for _, fragment := range []string{"(.deferCall", "(.effect", `"Insert"`, `"ElementRelationship"`, visitor, `"errs"`} {
			if !strings.Contains(program, fragment) {
				t.Fatalf("typed import dropped %q: %s", fragment, program)
			}
		}
		for _, prefix := range []string{"alias := v;\n", "v.path = nil;\n", "v.set = nil;\n", "v.allocator = nil;\n", "errs = unknown();\n"} {
			if _, err := importFunction(atomicFieldSource(function, prefix+body), function); err == nil {
				t.Fatalf("typed import accepted unsupported input mutation, alias or global: %s", prefix)
			}
		}
		wrong := strings.Replace(string(source), `"sigs.k8s.io/structured-merge-diff/v6/schema"`, `schema "example.com/schema"`, 1)
		if err := validateAtomicFieldImports([]byte(wrong)); err == nil {
			t.Fatal("typed import accepted a different schema package")
		}
	}
}
