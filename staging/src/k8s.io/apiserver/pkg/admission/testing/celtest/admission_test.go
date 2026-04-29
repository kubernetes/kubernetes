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
	admissionv1 "k8s.io/api/admission/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"testing"
)

func TestEvalAdmission_Allowed(t *testing.T) {
	e, err := NewEvaluator()
	if err != nil {
		t.Fatalf("NewEvaluator() error: %v", err)
	}

	policy := &AdmissionPolicy{
		Validations: []Validation{
			{Path: "validations[0]", Expression: "object.metadata.name == 'allowed-name'"},
		},
	}
	input := &AdmissionInput{
		Object: map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "Pod",
			"metadata":   map[string]interface{}{"name": "allowed-name"},
		},
	}

	result, err := e.EvalAdmission(policy, input)
	if err != nil {
		t.Fatalf("EvalAdmission() error: %v", err)
	}
	if !result.Allowed {
		t.Errorf("EvalAdmission() Allowed = false, want true; violations: %s", result.FormatViolations())
	}
	if result.Cost <= 0 {
		t.Errorf("EvalAdmission() Cost = %d, want > 0", result.Cost)
	}
}

func TestEvalAdmission_Denied(t *testing.T) {
	e, err := NewEvaluator()
	if err != nil {
		t.Fatalf("NewEvaluator() error: %v", err)
	}

	policy := &AdmissionPolicy{
		Validations: []Validation{
			{
				Path:       "validations[0]",
				Expression: "object.metadata.name != 'bad-name'",
				Message:    "name must not be bad-name",
			},
		},
	}
	input := &AdmissionInput{
		Object: map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "Pod",
			"metadata":   map[string]interface{}{"name": "bad-name"},
		},
	}

	result, err := e.EvalAdmission(policy, input)
	if err != nil {
		t.Fatalf("EvalAdmission() error: %v", err)
	}
	if result.Allowed {
		t.Error("EvalAdmission() Allowed = true, want false")
	}
	if len(result.Violations) != 1 {
		t.Fatalf("EvalAdmission() got %d violations, want 1", len(result.Violations))
	}
	if result.Violations[0].Message != "name must not be bad-name" {
		t.Errorf("violation message = %q, want %q", result.Violations[0].Message, "name must not be bad-name")
	}
}

func TestEvalAdmission_MessageExpression(t *testing.T) {
	e, err := NewEvaluator()
	if err != nil {
		t.Fatalf("NewEvaluator() error: %v", err)
	}

	policy := &AdmissionPolicy{
		Validations: []Validation{
			{
				Path:              "validations[0]",
				Expression:        "false",
				Message:           "static fallback",
				MessageExpression: "'denied: ' + object.metadata.name",
			},
		},
	}
	input := &AdmissionInput{
		Object: map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "Pod",
			"metadata":   map[string]interface{}{"name": "my-pod"},
		},
	}

	result, err := e.EvalAdmission(policy, input)
	if err != nil {
		t.Fatalf("EvalAdmission() error: %v", err)
	}
	if result.Allowed {
		t.Error("EvalAdmission() Allowed = true, want false")
	}
	if len(result.Violations) != 1 {
		t.Fatalf("got %d violations, want 1", len(result.Violations))
	}
	if result.Violations[0].Message != "denied: my-pod" {
		t.Errorf("violation message = %q, want %q", result.Violations[0].Message, "denied: my-pod")
	}
	if result.Violations[0].MessageError != nil {
		t.Errorf("unexpected MessageError: %v", result.Violations[0].MessageError)
	}
}

func TestEvalAdmission_MessageExpressionError(t *testing.T) {
	e, err := NewEvaluator()
	if err != nil {
		t.Fatalf("NewEvaluator() error: %v", err)
	}

	policy := &AdmissionPolicy{
		Validations: []Validation{
			{
				Path:              "validations[0]",
				Expression:        "false",
				Message:           "static fallback",
				MessageExpression: "object.nonexistent.deeply.nested",
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

	result, err := e.EvalAdmission(policy, input)
	if err != nil {
		t.Fatalf("EvalAdmission() error: %v", err)
	}
	if len(result.Violations) != 1 {
		t.Fatalf("got %d violations, want 1", len(result.Violations))
	}
	if result.Violations[0].MessageError == nil {
		t.Error("expected MessageError to be non-nil for invalid messageExpression")
	}
	// Static message should still be populated as fallback
	if result.Violations[0].Message != "static fallback" {
		t.Errorf("violation message = %q, want %q", result.Violations[0].Message, "static fallback")
	}
}

func TestEvalAdmission_MessageExpressionDoesNotDeclareAuthorizer(t *testing.T) {
	e, err := NewEvaluator()
	if err != nil {
		t.Fatalf("NewEvaluator() error: %v", err)
	}

	policy := &AdmissionPolicy{
		Validations: []Validation{
			{
				Path:              "validations[0]",
				Expression:        "false",
				Message:           "static fallback",
				MessageExpression: "authorizer.requestResource.check('get').allowed() ? 'allowed' : 'denied'",
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

	result, err := e.EvalAdmission(policy, input)
	if err != nil {
		t.Fatalf("EvalAdmission() error: %v", err)
	}
	if len(result.Violations) != 1 {
		t.Fatalf("got %d violations, want 1", len(result.Violations))
	}
	if result.Violations[0].MessageError == nil {
		t.Fatal("expected MessageError when messageExpression references authorizer")
	}
	if result.Violations[0].Message != "static fallback" {
		t.Errorf("violation message = %q, want fallback", result.Violations[0].Message)
	}

	_, err = e.EvalValidation(policy, ValidationSelector{Path: "validations[0]", Part: "messageExpression"}, input)
	if err == nil {
		t.Fatal("expected EvalValidation messageExpression to reject authorizer")
	}
}

func TestEvalAdmission_Variables(t *testing.T) {
	e, err := NewEvaluator()
	if err != nil {
		t.Fatalf("NewEvaluator() error: %v", err)
	}

	policy := &AdmissionPolicy{
		Variables: []Variable{
			{Name: "podName", Expression: "object.metadata.name"},
			{Name: "isAllowed", Expression: "variables.podName == 'good-pod'"},
		},
		Validations: []Validation{
			{Path: "validations[0]", Expression: "variables.isAllowed"},
		},
	}

	t.Run("allowed", func(t *testing.T) {
		input := &AdmissionInput{
			Object: map[string]interface{}{
				"apiVersion": "v1",
				"kind":       "Pod",
				"metadata":   map[string]interface{}{"name": "good-pod"},
			},
		}
		result, err := e.EvalAdmission(policy, input)
		if err != nil {
			t.Fatalf("EvalAdmission() error: %v", err)
		}
		if !result.Allowed {
			t.Errorf("expected Allowed=true, got violations: %s", result.FormatViolations())
		}
	})

	t.Run("denied", func(t *testing.T) {
		input := &AdmissionInput{
			Object: map[string]interface{}{
				"apiVersion": "v1",
				"kind":       "Pod",
				"metadata":   map[string]interface{}{"name": "bad-pod"},
			},
		}
		result, err := e.EvalAdmission(policy, input)
		if err != nil {
			t.Fatalf("EvalAdmission() error: %v", err)
		}
		if result.Allowed {
			t.Error("expected Allowed=false")
		}
	})
}

func TestEvalVariable(t *testing.T) {
	e, err := NewEvaluator()
	if err != nil {
		t.Fatalf("NewEvaluator() error: %v", err)
	}

	policy := &AdmissionPolicy{
		Variables: []Variable{
			{Name: "podName", Expression: "object.metadata.name"},
			{Name: "nameLen", Expression: "size(variables.podName)"},
		},
	}
	input := &AdmissionInput{
		Object: map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "Pod",
			"metadata":   map[string]interface{}{"name": "hello"},
		},
	}

	t.Run("first variable", func(t *testing.T) {
		val, err := e.EvalVariable(policy, "podName", input)
		if err != nil {
			t.Fatalf("EvalVariable() error: %v", err)
		}
		if val != "hello" {
			t.Errorf("EvalVariable() = %v, want %q", val, "hello")
		}
	})

	t.Run("chained variable", func(t *testing.T) {
		val, err := e.EvalVariable(policy, "nameLen", input)
		if err != nil {
			t.Fatalf("EvalVariable() error: %v", err)
		}
		if val != int64(5) {
			t.Errorf("EvalVariable() = %v (%T), want 5 (int64)", val, val)
		}
	})

	t.Run("not found", func(t *testing.T) {
		_, err := e.EvalVariable(policy, "nonexistent", input)
		if err == nil {
			t.Error("EvalVariable() expected error for missing variable")
		}
	})
}

func TestEvalValidation(t *testing.T) {
	e, err := NewEvaluator()
	if err != nil {
		t.Fatalf("NewEvaluator() error: %v", err)
	}

	policy := &AdmissionPolicy{
		Validations: []Validation{
			{
				Path:              "validations[0]",
				Expression:        "object.metadata.name == 'test'",
				Message:           "name must be test",
				MessageExpression: "'got: ' + object.metadata.name",
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

	t.Run("expression part", func(t *testing.T) {
		val, err := e.EvalValidation(policy, ValidationSelector{Path: "validations[0]"}, input)
		if err != nil {
			t.Fatalf("EvalValidation() error: %v", err)
		}
		if val != true {
			t.Errorf("EvalValidation() = %v, want true", val)
		}
	})

	t.Run("messageExpression part", func(t *testing.T) {
		val, err := e.EvalValidation(policy, ValidationSelector{Path: "validations[0]", Part: "messageExpression"}, input)
		if err != nil {
			t.Fatalf("EvalValidation() error: %v", err)
		}
		if val != "got: test" {
			t.Errorf("EvalValidation() = %v, want %q", val, "got: test")
		}
	})

	t.Run("empty path", func(t *testing.T) {
		_, err := e.EvalValidation(policy, ValidationSelector{}, input)
		if err == nil {
			t.Error("expected error for empty path")
		}
	})

	t.Run("not found path", func(t *testing.T) {
		_, err := e.EvalValidation(policy, ValidationSelector{Path: "validations[99]"}, input)
		if err == nil {
			t.Error("expected error for missing path")
		}
	})

	t.Run("unsupported part", func(t *testing.T) {
		_, err := e.EvalValidation(policy, ValidationSelector{Path: "validations[0]", Part: "bogus"}, input)
		if err == nil {
			t.Error("expected error for unsupported part")
		}
	})
}

func TestEvalAdmission_EvaluatesValidationsIndependentlyOfMatchConditions(t *testing.T) {
	e, err := NewEvaluator()
	if err != nil {
		t.Fatalf("NewEvaluator() error: %v", err)
	}

	policy := &AdmissionPolicy{
		MatchConditions: []MatchCondition{{Path: "spec.matchConditions[0]", Name: "not-system", Expression: "false"}},
		Validations:     []Validation{{Path: "spec.validations[0]", Expression: "false", Message: "validation still evaluated"}},
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

	result, err := e.EvalAdmission(policy, input)
	if err != nil {
		t.Fatalf("EvalAdmission() error: %v", err)
	}
	if result.Allowed {
		t.Fatal("expected validation to be evaluated and deny")
	}
	if len(result.Violations) != 1 {
		t.Fatalf("got %d violations, want 1", len(result.Violations))
	}
}

func TestEvalAdmission_RequestFields(t *testing.T) {
	e, err := NewEvaluator()
	if err != nil {
		t.Fatalf("NewEvaluator() error: %v", err)
	}

	policy := &AdmissionPolicy{
		Validations: []Validation{
			{Path: "validations[0]", Expression: "request.operation == 'CREATE'"},
		},
	}
	input := &AdmissionInput{
		Object: map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "Pod",
			"metadata":   map[string]interface{}{"name": "test"},
		},
		Request: &admissionv1.AdmissionRequest{
			Operation: admissionv1.Create,
			Kind:      metav1.GroupVersionKind{Version: "v1", Kind: "Pod"},
			Resource:  metav1.GroupVersionResource{Version: "v1", Resource: "pods"},
		},
	}

	result, err := e.EvalAdmission(policy, input)
	if err != nil {
		t.Fatalf("EvalAdmission() error: %v", err)
	}
	if !result.Allowed {
		t.Errorf("expected Allowed=true, got violations: %s", result.FormatViolations())
	}
}

func TestEvalAdmission_CompilationError(t *testing.T) {
	e, err := NewEvaluator()
	if err != nil {
		t.Fatalf("NewEvaluator() error: %v", err)
	}

	policy := &AdmissionPolicy{
		Validations: []Validation{
			{Path: "validations[0]", Expression: "???invalid"},
		},
	}
	input := &AdmissionInput{
		Object: map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "Pod",
			"metadata":   map[string]interface{}{"name": "test"},
		},
	}

	_, err = e.EvalAdmission(policy, input)
	if err == nil {
		t.Error("EvalAdmission() expected compilation error")
	}
}

func TestEvalAdmission_RejectsPatchTypes(t *testing.T) {
	e, err := NewEvaluator()
	if err != nil {
		t.Fatalf("NewEvaluator() error: %v", err)
	}

	policy := &AdmissionPolicy{
		Validations: []Validation{{Path: "validations[0]", Expression: patchTypeBoolExpression}},
	}
	input := &AdmissionInput{
		Object: map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "Pod",
			"metadata":   map[string]interface{}{"name": "test"},
		},
	}

	if _, err := e.EvalAdmission(policy, input); err == nil {
		t.Fatal("expected validation evaluation to reject mutation patch types")
	}
	if _, err := e.EvalValidation(policy, ValidationSelector{Path: "validations[0]"}, input); err == nil {
		t.Fatal("expected single validation evaluation to reject mutation patch types")
	}
}

func TestEvalAdmission_MultipleValidations(t *testing.T) {
	e, err := NewEvaluator()
	if err != nil {
		t.Fatalf("NewEvaluator() error: %v", err)
	}

	policy := &AdmissionPolicy{
		Validations: []Validation{
			{Path: "validations[0]", Expression: "true"},
			{Path: "validations[1]", Expression: "false", Message: "always fails"},
			{Path: "validations[2]", Expression: "true"},
		},
	}
	input := &AdmissionInput{
		Object: map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "Pod",
			"metadata":   map[string]interface{}{"name": "test"},
		},
	}

	result, err := e.EvalAdmission(policy, input)
	if err != nil {
		t.Fatalf("EvalAdmission() error: %v", err)
	}
	if result.Allowed {
		t.Error("expected Allowed=false when one validation fails")
	}
	if len(result.Violations) != 1 {
		t.Fatalf("got %d violations, want 1", len(result.Violations))
	}
	if result.Violations[0].Message != "always fails" {
		t.Errorf("violation message = %q, want %q", result.Violations[0].Message, "always fails")
	}
}

func TestPreambleVariables(t *testing.T) {
	e, err := NewEvaluator(
		WithPreambleVariables(
			Variable{Name: "alwaysTrue", Expression: "true"},
		),
	)
	if err != nil {
		t.Fatalf("NewEvaluator() error: %v", err)
	}

	policy := &AdmissionPolicy{
		Validations: []Validation{
			{Path: "validations[0]", Expression: "variables.alwaysTrue"},
		},
	}
	input := &AdmissionInput{
		Object: map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "Pod",
			"metadata":   map[string]interface{}{"name": "test"},
		},
	}

	result, err := e.EvalAdmission(policy, input)
	if err != nil {
		t.Fatalf("EvalAdmission() error: %v", err)
	}
	if !result.Allowed {
		t.Errorf("expected Allowed=true with preamble variable, got violations: %s", result.FormatViolations())
	}
}

func TestSetHasParams(t *testing.T) {
	e, err := NewEvaluator()
	if err != nil {
		t.Fatalf("NewEvaluator() error: %v", err)
	}

	t.Run("params enabled", func(t *testing.T) {
		policy := &AdmissionPolicy{
			Validations: []Validation{
				{Path: "validations[0]", Expression: "params.data.maxReplicas == '5'"},
			},
		}
		policy.SetHasParams(true)

		input := &AdmissionInput{
			Object: map[string]interface{}{
				"apiVersion": "v1",
				"kind":       "Pod",
				"metadata":   map[string]interface{}{"name": "test"},
			},
			Params: map[string]interface{}{
				"data": map[string]interface{}{"maxReplicas": "5"},
			},
		}

		result, err := e.EvalAdmission(policy, input)
		if err != nil {
			t.Fatalf("EvalAdmission() error: %v", err)
		}
		if !result.Allowed {
			t.Errorf("expected Allowed=true, got violations: %s", result.FormatViolations())
		}
	})

	t.Run("params disabled causes compilation error", func(t *testing.T) {
		policy := &AdmissionPolicy{
			Validations: []Validation{
				{Path: "validations[0]", Expression: "params.data.maxReplicas == '5'"},
			},
		}
		policy.SetHasParams(false)

		input := &AdmissionInput{
			Object: map[string]interface{}{
				"apiVersion": "v1",
				"kind":       "Pod",
				"metadata":   map[string]interface{}{"name": "test"},
			},
		}

		_, err := e.EvalAdmission(policy, input)
		if err == nil {
			t.Error("expected compilation error when params is disabled but expression references params")
		}
	})

	t.Run("default enables params", func(t *testing.T) {
		// Manually constructed policy without SetHasParams should default to params enabled.
		policy := &AdmissionPolicy{
			Validations: []Validation{
				{Path: "validations[0]", Expression: "params.data.key == 'val'"},
			},
		}
		input := &AdmissionInput{
			Object: map[string]interface{}{
				"apiVersion": "v1",
				"kind":       "Pod",
				"metadata":   map[string]interface{}{"name": "test"},
			},
			Params: map[string]interface{}{
				"data": map[string]interface{}{"key": "val"},
			},
		}

		result, err := e.EvalAdmission(policy, input)
		if err != nil {
			t.Fatalf("EvalAdmission() error: %v", err)
		}
		if !result.Allowed {
			t.Errorf("expected Allowed=true with default params, got violations: %s", result.FormatViolations())
		}
	})
}

func TestEvalAdmission_OldObject(t *testing.T) {
	e, err := NewEvaluator()
	if err != nil {
		t.Fatalf("NewEvaluator() error: %v", err)
	}

	policy := &AdmissionPolicy{
		Validations: []Validation{
			{
				Path:       "validations[0]",
				Expression: "object.metadata.name != oldObject.metadata.name",
				Message:    "name must change on update",
			},
		},
	}

	t.Run("names differ", func(t *testing.T) {
		input := &AdmissionInput{
			Object: map[string]interface{}{
				"apiVersion": "v1",
				"kind":       "Pod",
				"metadata":   map[string]interface{}{"name": "new-name"},
			},
			OldObject: map[string]interface{}{
				"apiVersion": "v1",
				"kind":       "Pod",
				"metadata":   map[string]interface{}{"name": "old-name"},
			},
			Request: &admissionv1.AdmissionRequest{
				Operation: admissionv1.Update,
				Kind:      metav1.GroupVersionKind{Version: "v1", Kind: "Pod"},
				Resource:  metav1.GroupVersionResource{Version: "v1", Resource: "pods"},
			},
		}

		result, err := e.EvalAdmission(policy, input)
		if err != nil {
			t.Fatalf("EvalAdmission() error: %v", err)
		}
		if !result.Allowed {
			t.Errorf("expected Allowed=true, got violations: %s", result.FormatViolations())
		}
	})

	t.Run("names same", func(t *testing.T) {
		input := &AdmissionInput{
			Object: map[string]interface{}{
				"apiVersion": "v1",
				"kind":       "Pod",
				"metadata":   map[string]interface{}{"name": "same-name"},
			},
			OldObject: map[string]interface{}{
				"apiVersion": "v1",
				"kind":       "Pod",
				"metadata":   map[string]interface{}{"name": "same-name"},
			},
			Request: &admissionv1.AdmissionRequest{
				Operation: admissionv1.Update,
				Kind:      metav1.GroupVersionKind{Version: "v1", Kind: "Pod"},
				Resource:  metav1.GroupVersionResource{Version: "v1", Resource: "pods"},
			},
		}

		result, err := e.EvalAdmission(policy, input)
		if err != nil {
			t.Fatalf("EvalAdmission() error: %v", err)
		}
		if result.Allowed {
			t.Error("expected Allowed=false when names are the same")
		}
		if len(result.Violations) != 1 {
			t.Fatalf("got %d violations, want 1", len(result.Violations))
		}
		if result.Violations[0].Message != "name must change on update" {
			t.Errorf("violation message = %q, want %q", result.Violations[0].Message, "name must change on update")
		}
	})

	t.Run("inferred update operation", func(t *testing.T) {
		// When both object and oldObject are set without explicit request, operation should be inferred as UPDATE.
		input := &AdmissionInput{
			Object: map[string]interface{}{
				"apiVersion": "v1",
				"kind":       "Pod",
				"metadata":   map[string]interface{}{"name": "new-name"},
			},
			OldObject: map[string]interface{}{
				"apiVersion": "v1",
				"kind":       "Pod",
				"metadata":   map[string]interface{}{"name": "old-name"},
			},
		}

		policyOp := &AdmissionPolicy{
			Validations: []Validation{
				{Path: "validations[0]", Expression: "request.operation == 'UPDATE'"},
			},
		}

		result, err := e.EvalAdmission(policyOp, input)
		if err != nil {
			t.Fatalf("EvalAdmission() error: %v", err)
		}
		if !result.Allowed {
			t.Errorf("expected inferred UPDATE operation, got violations: %s", result.FormatViolations())
		}
	})
}
