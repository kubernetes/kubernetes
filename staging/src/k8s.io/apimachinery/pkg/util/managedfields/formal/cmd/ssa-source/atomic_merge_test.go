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
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func selectedMergeSource(t *testing.T) []byte {
	t.Helper()
	directory, err := exec.Command("go", "list", "-f", "{{.Dir}}", "sigs.k8s.io/structured-merge-diff/v6/typed").Output()
	if err != nil {
		t.Fatal(err)
	}
	source, err := os.ReadFile(filepath.Join(strings.TrimSpace(string(directory)), "merge.go"))
	if err != nil {
		t.Fatal(err)
	}
	return source
}

func TestAtomicMergeImportsCompleteDispatchLeafAndRule(t *testing.T) {
	source := selectedMergeSource(t)
	for index, function := range atomicMergeFunctions {
		program, err := importFunction(source, function)
		if err != nil {
			t.Fatalf("%s: %v", function.name, err)
		}
		fragments := [][]string{
			{"derefList", "(.deferCall", "Length", "visitListItems", "emptyPromoteToLeaf", "doLeaf"},
			{"derefMap", "(.deferCall", "Empty", "visitMapItems", "(.concatenate", "doLeaf"},
			{"inLeaf", "(.ret [])", "(.fieldSet", `"rule"`},
			{"(.branchElse", "Unstructured", "(.addressOf", `"v$1"`, `"v$2"`, `"out"`},
		}
		for _, fragment := range fragments[index] {
			if !strings.Contains(program, fragment) {
				t.Fatalf("%s dropped %q: %s", function.name, fragment, program)
			}
		}
	}
}

func TestAtomicMergeRejectsMutableAliasesAndWrongRuleInitializers(t *testing.T) {
	function := atomicMergeFunctions[3]
	wrap := func(initializer string) []byte {
		return []byte("package typed\ntype mergeRule func(w *mergingWalker)\nvar ruleKeepRHS = " + initializer + "\n")
	}
	for _, initializer := range []string{
		"otherRule", "otherType(func(w *mergingWalker) {})", "mergeRule(otherRule)",
		"mergeRule(func(w mergingWalker) {})", "mergeRule(func(other *mergingWalker) {})",
		"mergeRule(func(w *mergingWalker) { alias := w; })",
		"mergeRule(func(w *mergingWalker) { w.lhs = nil })",
		"mergeRule(func(w *mergingWalker) { w.rhs = nil })",
		"mergeRule(func(w *mergingWalker) { w.allocator = nil })",
		"mergeRule(func(w *mergingWalker) { w.out = unbound })",
	} {
		if _, err := importFunction(wrap(initializer), function); err == nil {
			t.Fatalf("accepted unsupported rule or walker mutation: %s", initializer)
		}
	}
	valid := "mergeRule(func(w *mergingWalker) { if w.rhs != nil { v := w.rhs.Unstructured(); w.out = &v } })"
	if _, err := importFunction(wrap(valid), function); err != nil {
		t.Fatal(err)
	}
	duplicate := append(wrap(valid), []byte("var ruleKeepRHS = "+valid+"\n")...)
	if _, err := importFunction(duplicate, function); err == nil {
		t.Fatal("accepted duplicate initialized merge rules")
	}
	wrapped := strings.Replace(string(wrap(valid)), "type mergeRule func(w *mergingWalker)",
		"func mergeRule(f func(w *mergingWalker)) func(w *mergingWalker) { return f }", 1)
	if _, err := importFunction([]byte(wrapped), function); err == nil {
		t.Fatal("accepted a potentially effectful wrapper in place of the function type conversion")
	}
}
