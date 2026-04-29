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

package celtest

import (
	"strings"
	"testing"
)

func TestEvalMutation_EvaluatesMutationsIndependentlyOfMatchConditions(t *testing.T) {
	e, err := NewEvaluator()
	if err != nil {
		t.Fatalf("NewEvaluator() error: %v", err)
	}

	policy := &AdmissionPolicy{
		MatchConditions: []MatchCondition{{Path: "spec.matchConditions[0]", Name: "not-system", Expression: "false"}},
		Mutations: []Mutation{{
			Path:       "spec.mutations[0]",
			PatchType:  "ApplyConfiguration",
			Expression: "Object{metadata: Object.metadata{labels: {'mutated': 'true'}}}",
		}},
	}

	input := &AdmissionInput{
		Object: map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "Pod",
			"metadata": map[string]interface{}{
				"name":      "test-pod",
				"namespace": "kube-system",
			},
		},
	}

	result, err := e.EvalMutation(policy, input)
	if err != nil {
		t.Fatalf("EvalMutation() error: %v", err)
	}
	if len(result.Patches) != 1 {
		t.Fatalf("got %d patches, want 1", len(result.Patches))
	}
}

func TestEvalMutation_ApplyConfiguration(t *testing.T) {
	e, err := NewEvaluator()
	if err != nil {
		t.Fatalf("NewEvaluator() error: %v", err)
	}

	policy := &AdmissionPolicy{
		Variables: []Variable{
			{Name: "newName", Expression: "'mutated-pod'"},
		},
		Mutations: []Mutation{
			{
				Path:       "spec.mutations[0]",
				PatchType:  "ApplyConfiguration",
				Expression: "Object{metadata: Object.metadata{name: variables.newName}}",
			},
		},
	}
	input := &AdmissionInput{
		Object: map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "Pod",
			"metadata":   map[string]interface{}{"name": "original-pod"},
		},
	}

	result, err := e.EvalMutation(policy, input)
	if err != nil {
		t.Fatalf("EvalMutation() error: %v", err)
	}
	if len(result.Patches) != 1 {
		t.Fatalf("got %d patches, want 1", len(result.Patches))
	}
	if result.Patches[0].Error != nil {
		t.Fatalf("patch error: %v", result.Patches[0].Error)
	}
	if result.Patches[0].PatchType != "ApplyConfiguration" {
		t.Errorf("patchType = %q, want ApplyConfiguration", result.Patches[0].PatchType)
	}
	applyConfig, ok := result.Patches[0].Value.(map[string]interface{})
	if !ok {
		t.Fatalf("patch value type = %T, want map[string]interface{}", result.Patches[0].Value)
	}
	metadata, ok := applyConfig["metadata"].(map[string]interface{})
	if !ok {
		t.Fatalf("metadata type = %T, want map[string]interface{}", applyConfig["metadata"])
	}
	if metadata["name"] != "mutated-pod" {
		t.Errorf("metadata.name = %v, want %q", metadata["name"], "mutated-pod")
	}
	if result.Cost <= 0 {
		t.Errorf("Cost = %d, want > 0", result.Cost)
	}
}

func TestEvalMutation_JSONPatch(t *testing.T) {
	e, err := NewEvaluator()
	if err != nil {
		t.Fatalf("NewEvaluator() error: %v", err)
	}

	policy := &AdmissionPolicy{
		Mutations: []Mutation{
			{
				Path:       "spec.mutations[0]",
				PatchType:  "JSONPatch",
				Expression: "[JSONPatch{op: \"replace\", path: \"/metadata/name\", value: \"new-name\"}]",
			},
		},
	}
	input := &AdmissionInput{
		Object: map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "Pod",
			"metadata":   map[string]interface{}{"name": "old-name"},
		},
	}

	result, err := e.EvalMutation(policy, input)
	if err != nil {
		t.Fatalf("EvalMutation() error: %v", err)
	}
	if len(result.Patches) != 1 {
		t.Fatalf("got %d patches, want 1", len(result.Patches))
	}
	if result.Patches[0].Error != nil {
		t.Fatalf("patch error: %v", result.Patches[0].Error)
	}
	// JSONPatch expressions return a CEL list of JSONPatch values;
	// verify the result is a non-empty slice.
	patchOps, ok := result.Patches[0].Value.([]interface{})
	if !ok {
		// CEL may return []ref.Val for typed JSONPatch objects; accept any non-nil slice.
		if result.Patches[0].Value == nil {
			t.Fatal("patch value should not be nil")
		}
	} else if len(patchOps) != 1 {
		t.Errorf("got %d JSON patch ops, want 1", len(patchOps))
	}
}

func TestEvalMutation_RejectsInvalidPatchReturnTypes(t *testing.T) {
	e, err := NewEvaluator()
	if err != nil {
		t.Fatalf("NewEvaluator() error: %v", err)
	}

	input := &AdmissionInput{
		Object: map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "Pod",
			"metadata":   map[string]interface{}{"name": "test"},
		},
	}
	tests := []struct {
		name      string
		patchType string
		expr      string
	}{
		{name: "apply configuration must return Object", patchType: "ApplyConfiguration", expr: "true"},
		{name: "json patch must return list of JSONPatch", patchType: "JSONPatch", expr: "Object{}"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			policy := &AdmissionPolicy{
				Mutations: []Mutation{
					{
						Path:       "spec.mutations[0]",
						PatchType:  tt.patchType,
						Expression: tt.expr,
					},
				},
			}
			_, err := e.EvalMutation(policy, input)
			if err == nil {
				t.Fatal("expected EvalMutation() to reject invalid patch return type")
			}
			if !strings.Contains(err.Error(), "spec.mutations[0]") {
				t.Errorf("error should reference mutation path, got: %v", err)
			}
		})
	}
}

func TestEvalMutation_WithoutPatchTypesRejectsPatchExpressions(t *testing.T) {
	e, err := NewEvaluator(WithoutPatchTypes())
	if err != nil {
		t.Fatalf("NewEvaluator() error: %v", err)
	}

	policy := &AdmissionPolicy{
		Mutations: []Mutation{
			{
				Path:       "spec.mutations[0]",
				PatchType:  "ApplyConfiguration",
				Expression: "Object{}",
			},
		},
	}
	input := &AdmissionInput{
		Object: map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "Pod",
			"metadata":   map[string]interface{}{"name": "test"},
		},
	}

	_, err = e.EvalMutation(policy, input)
	if err == nil {
		t.Fatal("expected EvalMutation() to reject patch expressions when patch types are disabled")
	}
}

func TestEvalMutation_MultiplePatches(t *testing.T) {
	e, err := NewEvaluator()
	if err != nil {
		t.Fatalf("NewEvaluator() error: %v", err)
	}

	policy := &AdmissionPolicy{
		Mutations: []Mutation{
			{
				Path:       "spec.mutations[0]",
				PatchType:  "ApplyConfiguration",
				Expression: "Object{metadata: Object.metadata{name: \"first\"}}",
			},
			{
				Path:       "spec.mutations[1]",
				PatchType:  "ApplyConfiguration",
				Expression: "Object{metadata: Object.metadata{name: \"second\"}}",
			},
		},
	}
	input := &AdmissionInput{
		Object: map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "Pod",
			"metadata":   map[string]interface{}{"name": "original"},
		},
	}

	result, err := e.EvalMutation(policy, input)
	if err != nil {
		t.Fatalf("EvalMutation() error: %v", err)
	}
	if len(result.Patches) != 2 {
		t.Fatalf("got %d patches, want 2", len(result.Patches))
	}
	wantNames := []string{"first", "second"}
	for i, patch := range result.Patches {
		if patch.Error != nil {
			t.Errorf("patch[%d] error: %v", i, patch.Error)
			continue
		}
		applyConfig, ok := patch.Value.(map[string]interface{})
		if !ok {
			t.Errorf("patch[%d] value type = %T, want map[string]interface{}", i, patch.Value)
			continue
		}
		metadata, ok := applyConfig["metadata"].(map[string]interface{})
		if !ok {
			t.Errorf("patch[%d] metadata type = %T, want map[string]interface{}", i, applyConfig["metadata"])
			continue
		}
		if metadata["name"] != wantNames[i] {
			t.Errorf("patch[%d] metadata.name = %v, want %q", i, metadata["name"], wantNames[i])
		}
	}
}

func TestEvalMutation_CostLimitIsPerMutation(t *testing.T) {
	e, err := NewEvaluator(WithCostLimit(1000))
	if err != nil {
		t.Fatalf("NewEvaluator() error: %v", err)
	}

	policy := &AdmissionPolicy{
		Mutations: []Mutation{
			{
				Path:       "spec.mutations[0]",
				PatchType:  "ApplyConfiguration",
				Expression: "Object{metadata: Object.metadata{name: string([1,2,3,4,5,6,7,8,9,10].map(x, [1,2,3,4,5,6,7,8,9,10].map(y, x * y)).size())}}",
			},
			{
				Path:       "spec.mutations[1]",
				PatchType:  "ApplyConfiguration",
				Expression: "Object{metadata: Object.metadata{name: \"second\"}}",
			},
		},
	}
	input := &AdmissionInput{
		Object: map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "Pod",
			"metadata":   map[string]interface{}{"name": "test-pod"},
		},
	}

	result, err := e.EvalMutation(policy, input)
	if err != nil {
		t.Fatalf("EvalMutation() error: %v", err)
	}
	if len(result.Patches) != 2 {
		t.Fatalf("got %d patches, want 2", len(result.Patches))
	}
	if result.Patches[0].Error == nil {
		t.Fatal("expected first patch to fail after exceeding the cost budget")
	}
	if result.Patches[1].Error != nil {
		t.Fatalf("expected second patch to get a fresh cost budget, got error: %v", result.Patches[1].Error)
	}
}

func TestEvalMutationByPath(t *testing.T) {
	e, err := NewEvaluator()
	if err != nil {
		t.Fatalf("NewEvaluator() error: %v", err)
	}

	policy := &AdmissionPolicy{
		Variables: []Variable{
			{Name: "target", Expression: "'selected'"},
		},
		Mutations: []Mutation{
			{
				Path:       "spec.mutations[0]",
				PatchType:  "ApplyConfiguration",
				Expression: "Object{metadata: Object.metadata{name: \"first\"}}",
			},
			{
				Path:       "spec.mutations[1]",
				PatchType:  "ApplyConfiguration",
				Expression: "Object{metadata: Object.metadata{name: variables.target}}",
			},
		},
	}
	input := &AdmissionInput{
		Object: map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "Pod",
			"metadata":   map[string]interface{}{"name": "original"},
		},
	}

	t.Run("select second mutation", func(t *testing.T) {
		val, err := e.EvalMutationByPath(policy, MutationSelector{Path: "spec.mutations[1]"}, input)
		if err != nil {
			t.Fatalf("EvalMutationByPath() error: %v", err)
		}
		applyConfig, ok := val.(map[string]interface{})
		if !ok {
			t.Fatalf("value type = %T, want map[string]interface{}", val)
		}
		metadata, ok := applyConfig["metadata"].(map[string]interface{})
		if !ok {
			t.Fatalf("metadata type = %T, want map[string]interface{}", applyConfig["metadata"])
		}
		if metadata["name"] != "selected" {
			t.Errorf("metadata.name = %v, want %q", metadata["name"], "selected")
		}
	})

	t.Run("not found", func(t *testing.T) {
		_, err := e.EvalMutationByPath(policy, MutationSelector{Path: "spec.mutations[99]"}, input)
		if err == nil {
			t.Error("expected error for missing path")
		}
	})

	t.Run("empty path", func(t *testing.T) {
		_, err := e.EvalMutationByPath(policy, MutationSelector{}, input)
		if err == nil {
			t.Error("expected error for empty path")
		}
	})
}

func TestEvalMutation_CompilationError(t *testing.T) {
	// Verify that EvalMutation returns an error (not a result with per-patch errors)
	// for compilation failures, symmetric with EvalAdmission.
	e, err := NewEvaluator()
	if err != nil {
		t.Fatalf("NewEvaluator() error: %v", err)
	}

	policy := &AdmissionPolicy{
		Mutations: []Mutation{
			{
				Path:       "spec.mutations[0]",
				PatchType:  "ApplyConfiguration",
				Expression: "Object{metadata: Object.metadata{name: 'valid'}}",
			},
			{
				Path:       "spec.mutations[1]",
				PatchType:  "ApplyConfiguration",
				Expression: "???invalid",
			},
		},
	}
	input := &AdmissionInput{
		Object: map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "Pod",
			"metadata":   map[string]interface{}{"name": "test"},
		},
	}

	_, err = e.EvalMutation(policy, input)
	if err == nil {
		t.Error("EvalMutation() expected error when any mutation has compilation error")
	}
	if !strings.Contains(err.Error(), "spec.mutations[1]") {
		t.Errorf("error should reference the failing mutation path, got: %v", err)
	}
}
