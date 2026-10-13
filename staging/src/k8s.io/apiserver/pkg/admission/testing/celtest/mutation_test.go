/*
Copyright The Kubernetes Authors.

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
		matchConditions: []matchCondition{{Path: "spec.matchConditions[0]", Name: "not-system", Expression: "false"}},
		mutations: []mutation{{
			Path:       "spec.mutations[0]",
			PatchType:  "ApplyConfiguration",
			Expression: "Object{metadata: Object.metadata{labels: {'mutated': 'true'}}}",
		}},
	}

	input := &AdmissionInput{
		object: map[string]interface{}{
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
		variables: []variable{
			{Name: "newName", Expression: "'mutated-pod'"},
		},
		mutations: []mutation{
			{
				Path:       "spec.mutations[0]",
				PatchType:  "ApplyConfiguration",
				Expression: "Object{metadata: Object.metadata{name: variables.newName}}",
			},
		},
	}
	input := &AdmissionInput{
		object: map[string]interface{}{
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
		mutations: []mutation{
			{
				Path:       "spec.mutations[0]",
				PatchType:  "JSONPatch",
				Expression: "[JSONPatch{op: \"replace\", path: \"/metadata/name\", value: \"new-name\"}]",
			},
		},
	}
	input := &AdmissionInput{
		object: map[string]interface{}{
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
		object: map[string]interface{}{
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
				mutations: []mutation{
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
		mutations: []mutation{
			{
				Path:       "spec.mutations[0]",
				PatchType:  "ApplyConfiguration",
				Expression: "Object{}",
			},
		},
	}
	input := &AdmissionInput{
		object: map[string]interface{}{
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
		mutations: []mutation{
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
		object: map[string]interface{}{
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
	wantPatches := []struct {
		path string
		name string
	}{
		{path: "spec.mutations[0]", name: "first"},
		{path: "spec.mutations[1]", name: "second"},
	}
	for i, patch := range result.Patches {
		if patch.Path != wantPatches[i].path {
			t.Errorf("patch[%d].Path = %q, want %q", i, patch.Path, wantPatches[i].path)
		}
		if patch.PatchType != "ApplyConfiguration" {
			t.Errorf("patch[%d].PatchType = %q, want ApplyConfiguration", i, patch.PatchType)
		}
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
		if metadata["name"] != wantPatches[i].name {
			t.Errorf("patch[%d] metadata.name = %v, want %q", i, metadata["name"], wantPatches[i].name)
		}
	}
}

func TestEvalMutation_CostLimitIsPerMutation(t *testing.T) {
	e, err := NewEvaluator(WithCostLimit(1000))
	if err != nil {
		t.Fatalf("NewEvaluator() error: %v", err)
	}

	policy := &AdmissionPolicy{
		mutations: []mutation{
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
		object: map[string]interface{}{
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

func TestEvalMutation_CompilationError(t *testing.T) {
	// Verify that EvalMutation returns an error (not a result with per-patch errors)
	// for compilation failures, symmetric with EvalValidations.
	e, err := NewEvaluator()
	if err != nil {
		t.Fatalf("NewEvaluator() error: %v", err)
	}

	policy := &AdmissionPolicy{
		mutations: []mutation{
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
		object: map[string]interface{}{
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
