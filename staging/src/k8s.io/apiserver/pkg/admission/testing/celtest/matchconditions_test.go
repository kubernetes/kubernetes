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
	"testing"

	corev1 "k8s.io/api/core/v1"
)

func TestEvalMatchConditions(t *testing.T) {
	e, err := NewEvaluator()
	if err != nil {
		t.Fatalf("NewEvaluator() error: %v", err)
	}

	input := &AdmissionInput{
		object: map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "Pod",
			"metadata": map[string]interface{}{
				"name":      "test-pod",
				"namespace": "default",
			},
		},
		params: map[string]interface{}{
			"data": map[string]interface{}{"enabled": "true"},
		},
	}

	t.Run("all match", func(t *testing.T) {
		policy := &AdmissionPolicy{
			matchConditions: []matchCondition{
				{Path: "spec.matchConditions[0]", Name: "params-enabled", Expression: "params.data.enabled == 'true'"},
				{Path: "spec.matchConditions[1]", Name: "not-system", Expression: "object.metadata.namespace != 'kube-system'"},
			},
		}
		policy.setHasParams(true)

		result, err := e.EvalMatchConditions(policy, input)
		if err != nil {
			t.Fatalf("EvalMatchConditions() error: %v", err)
		}
		if len(result.Conditions) != 2 {
			t.Fatalf("got %d conditions, want 2", len(result.Conditions))
		}
		wantConditions := []struct {
			path       string
			name       string
			expression string
			value      interface{}
		}{
			{path: "spec.matchConditions[0]", name: "params-enabled", expression: "params.data.enabled == 'true'", value: true},
			{path: "spec.matchConditions[1]", name: "not-system", expression: "object.metadata.namespace != 'kube-system'", value: true},
		}
		for index, condition := range result.Conditions {
			if condition.Path != wantConditions[index].path {
				t.Errorf("condition[%d].Path = %q, want %q", index, condition.Path, wantConditions[index].path)
			}
			if condition.Name != wantConditions[index].name {
				t.Errorf("condition[%d].Name = %q, want %q", index, condition.Name, wantConditions[index].name)
			}
			if condition.Expression != wantConditions[index].expression {
				t.Errorf("condition[%d].Expression = %q, want %q", index, condition.Expression, wantConditions[index].expression)
			}
			if condition.Error != nil {
				t.Fatalf("condition[%d] error: %v", index, condition.Error)
			}
			if condition.Value != wantConditions[index].value {
				t.Fatalf("condition[%d] value = %v, want %v", index, condition.Value, wantConditions[index].value)
			}
		}
	})

	t.Run("false condition", func(t *testing.T) {
		policy := &AdmissionPolicy{
			matchConditions: []matchCondition{
				{Path: "spec.matchConditions[0]", Name: "system-only", Expression: "object.metadata.namespace == 'kube-system'"},
			},
		}

		result, err := e.EvalMatchConditions(policy, input)
		if err != nil {
			t.Fatalf("EvalMatchConditions() error: %v", err)
		}
		if len(result.Conditions) != 1 {
			t.Fatalf("got %d conditions, want 1", len(result.Conditions))
		}
		if result.Conditions[0].Error != nil {
			t.Fatalf("condition error: %v", result.Conditions[0].Error)
		}
		if result.Conditions[0].Value != false {
			t.Fatalf("condition value = %v, want false", result.Conditions[0].Value)
		}
	})

	t.Run("runtime error is reported per condition", func(t *testing.T) {
		policy := &AdmissionPolicy{
			matchConditions: []matchCondition{
				{Path: "spec.matchConditions[0]", Name: "bad", Expression: "1 / 0 == 0"},
			},
		}

		result, err := e.EvalMatchConditions(policy, input)
		if err != nil {
			t.Fatalf("EvalMatchConditions() error: %v", err)
		}
		if len(result.Conditions) != 1 || result.Conditions[0].Error == nil {
			t.Fatalf("expected per-condition error, got %#v", result.Conditions)
		}
	})

	t.Run("compile error", func(t *testing.T) {
		policy := &AdmissionPolicy{
			matchConditions: []matchCondition{
				{Path: "spec.matchConditions[0]", Name: "bad", Expression: "object.metadata.name =="},
			},
		}

		if _, err := e.EvalMatchConditions(policy, input); err == nil {
			t.Fatal("expected compile error")
		}
	})

	t.Run("patch types are rejected", func(t *testing.T) {
		policy := &AdmissionPolicy{
			matchConditions: []matchCondition{
				{Path: "spec.matchConditions[0]", Name: "patch-type", Expression: patchTypeBoolExpression},
			},
		}

		if _, err := e.EvalMatchConditions(policy, input); err == nil {
			t.Fatal("expected match condition evaluation to reject mutation patch types")
		}
	})

	t.Run("namespaceObject is not bound", func(t *testing.T) {
		policy := &AdmissionPolicy{
			matchConditions: []matchCondition{
				{Path: "spec.matchConditions[0]", Name: "no-namespace", Expression: "namespaceObject == null"},
			},
		}
		inputWithNamespace := *input
		inputWithNamespace.namespace = &corev1.Namespace{}

		result, err := e.EvalMatchConditions(policy, &inputWithNamespace)
		if err != nil {
			t.Fatalf("EvalMatchConditions() error: %v", err)
		}
		if len(result.Conditions) != 1 {
			t.Fatalf("got %d conditions, want 1", len(result.Conditions))
		}
		if result.Conditions[0].Error != nil {
			t.Fatalf("condition error: %v", result.Conditions[0].Error)
		}
		if result.Conditions[0].Value != true {
			t.Fatalf("condition value = %v, want true", result.Conditions[0].Value)
		}
	})

	t.Run("single condition result includes name", func(t *testing.T) {
		policy := &AdmissionPolicy{
			matchConditions: []matchCondition{
				{Path: "spec.matchConditions[0]", Name: "params-enabled", Expression: "params.data.enabled == 'true'"},
			},
		}
		policy.setHasParams(true)

		result, err := e.EvalMatchConditions(policy, input)
		if err != nil {
			t.Fatalf("EvalMatchConditions() error: %v", err)
		}
		if len(result.Conditions) != 1 {
			t.Fatalf("got %d conditions, want 1", len(result.Conditions))
		}
		if result.Conditions[0].Name != "params-enabled" {
			t.Errorf("condition name = %q, want params-enabled", result.Conditions[0].Name)
		}
		if result.Conditions[0].Value != true {
			t.Errorf("condition value = %v, want true", result.Conditions[0].Value)
		}
	})

	t.Run("duplicate names return distinct paths", func(t *testing.T) {
		policy := &AdmissionPolicy{
			matchConditions: []matchCondition{
				{Path: "webhooks[0].matchConditions[0]", Name: "same-name", Expression: "false"},
				{Path: "webhooks[1].matchConditions[0]", Name: "same-name", Expression: "true"},
			},
		}

		result, err := e.EvalMatchConditions(policy, input)
		if err != nil {
			t.Fatalf("EvalMatchConditions() error: %v", err)
		}
		if len(result.Conditions) != 2 {
			t.Fatalf("got %d conditions, want 2", len(result.Conditions))
		}
		wantPaths := []string{"webhooks[0].matchConditions[0]", "webhooks[1].matchConditions[0]"}
		wantValues := []interface{}{false, true}
		for index, condition := range result.Conditions {
			if condition.Name != "same-name" {
				t.Errorf("condition[%d].Name = %q, want same-name", index, condition.Name)
			}
			if condition.Path != wantPaths[index] {
				t.Errorf("condition[%d].Path = %q, want %q", index, condition.Path, wantPaths[index])
			}
			if condition.Value != wantValues[index] {
				t.Errorf("condition[%d].Value = %v, want %v", index, condition.Value, wantValues[index])
			}
		}
	})
}
