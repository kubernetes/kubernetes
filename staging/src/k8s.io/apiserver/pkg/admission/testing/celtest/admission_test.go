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

	admissionv1 "k8s.io/api/admission/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
)

type unconvertibleRuntimeObject struct {
	metav1.TypeMeta `json:",inline"`
	Values          []uint64 `json:"values,omitempty"`
}

func (o *unconvertibleRuntimeObject) DeepCopyObject() runtime.Object {
	if o == nil {
		return nil
	}
	out := *o
	out.Values = append([]uint64(nil), o.Values...)
	return &out
}

func TestEvalValidations_Allowed(t *testing.T) {
	e, err := NewEvaluator()
	if err != nil {
		t.Fatalf("NewEvaluator() error: %v", err)
	}

	policy := &AdmissionPolicy{
		validations: []validation{
			{Path: "validations[0]", Expression: "object.metadata.name == 'allowed-name'"},
		},
	}
	input := &AdmissionInput{
		object: map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "Pod",
			"metadata":   map[string]interface{}{"name": "allowed-name"},
		},
	}

	result, err := e.EvalValidations(policy, input)
	if err != nil {
		t.Fatalf("EvalValidations() error: %v", err)
	}
	if !result.Allowed {
		t.Errorf("EvalValidations() Allowed = false, want true; violations: %s", result.FormatViolations())
	}
	if result.Cost <= 0 {
		t.Errorf("EvalValidations() Cost = %d, want > 0", result.Cost)
	}
}

func TestEvalValidations_TypedAdmissionInput(t *testing.T) {
	e, err := NewEvaluator()
	if err != nil {
		t.Fatalf("NewEvaluator() error: %v", err)
	}

	policy := &AdmissionPolicy{
		validations: []validation{
			{Path: "validations[0]", Expression: "object.kind == 'Pod'"},
			{Path: "validations[1]", Expression: "request.kind.kind == 'Pod'"},
			{Path: "validations[2]", Expression: "request.resource.resource == 'pods'"},
			{Path: "validations[3]", Expression: "object.metadata.name == 'typed-pod'"},
			{Path: "validations[4]", Expression: "object.metadata.namespace == 'default'"},
			{Path: "validations[5]", Expression: "params.kind == 'ConfigMap'"},
			{Path: "validations[6]", Expression: "params.data.requiredTeam == object.metadata.labels.team"},
		},
	}
	policy.setHasParams(true)

	input := &AdmissionInput{
		object: &corev1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "typed-pod",
				Namespace: "default",
				Labels:    map[string]string{"team": "platform"},
			},
			Spec: corev1.PodSpec{
				Containers: []corev1.Container{{Name: "app", Image: "registry.k8s.io/pause:3.10"}},
			},
		},
		params: &corev1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{Name: "policy-config"},
			Data:       map[string]string{"requiredTeam": "platform"},
		},
	}

	result, err := e.EvalValidations(policy, input)
	if err != nil {
		t.Fatalf("EvalValidations() error: %v", err)
	}
	if !result.Allowed {
		t.Fatalf("expected Allowed=true for typed input, got violations: %s", result.FormatViolations())
	}
}

func TestEvalValidations_TypedOldObject(t *testing.T) {
	e, err := NewEvaluator()
	if err != nil {
		t.Fatalf("NewEvaluator() error: %v", err)
	}

	policy := &AdmissionPolicy{
		validations: []validation{
			{Path: "validations[0]", Expression: "request.operation == 'UPDATE'"},
			{Path: "validations[1]", Expression: "oldObject.metadata.labels.version == 'old'"},
			{Path: "validations[2]", Expression: "object.metadata.labels.version == 'new'"},
		},
	}
	input := &AdmissionInput{
		object: &corev1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:   "typed-pod",
				Labels: map[string]string{"version": "new"},
			},
		},
		oldObject: &corev1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:   "typed-pod",
				Labels: map[string]string{"version": "old"},
			},
		},
	}

	result, err := e.EvalValidations(policy, input)
	if err != nil {
		t.Fatalf("EvalValidations() error: %v", err)
	}
	if !result.Allowed {
		t.Fatalf("expected typed oldObject update to allow, got violations: %s", result.FormatViolations())
	}
}

func TestEvalValidations_TypedAdmissionInputConversionErrors(t *testing.T) {
	e, err := NewEvaluator()
	if err != nil {
		t.Fatalf("NewEvaluator() error: %v", err)
	}

	badObject := func() *unconvertibleRuntimeObject {
		return &unconvertibleRuntimeObject{
			TypeMeta: metav1.TypeMeta{APIVersion: "example.com/v1", Kind: "Bad"},
			Values:   []uint64{^uint64(0)},
		}
	}
	tests := []struct {
		name string
		in   *AdmissionInput
		want string
	}{
		{
			name: "object",
			in:   &AdmissionInput{object: badObject()},
			want: "converting object",
		},
		{
			name: "oldObject",
			in:   &AdmissionInput{oldObject: badObject()},
			want: "converting oldObject",
		},
		{
			name: "params",
			in:   &AdmissionInput{params: badObject()},
			want: "converting params",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := e.EvalExpression("true", tt.in)
			if err == nil {
				t.Fatal("expected conversion error")
			}
			if !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("error = %q, want to contain %q", err.Error(), tt.want)
			}
		})
	}
}

func TestEvalValidations_Denied(t *testing.T) {
	e, err := NewEvaluator()
	if err != nil {
		t.Fatalf("NewEvaluator() error: %v", err)
	}

	policy := &AdmissionPolicy{
		validations: []validation{
			{
				Path:       "validations[0]",
				Expression: "object.metadata.name != 'bad-name'",
				Message:    "name must not be bad-name",
			},
		},
	}
	input := &AdmissionInput{
		object: map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "Pod",
			"metadata":   map[string]interface{}{"name": "bad-name"},
		},
	}

	result, err := e.EvalValidations(policy, input)
	if err != nil {
		t.Fatalf("EvalValidations() error: %v", err)
	}
	if result.Allowed {
		t.Error("EvalValidations() Allowed = true, want false")
	}
	if len(result.Violations) != 1 {
		t.Fatalf("EvalValidations() got %d violations, want 1", len(result.Violations))
	}
	if result.Violations[0].Message != "name must not be bad-name" {
		t.Errorf("violation message = %q, want %q", result.Violations[0].Message, "name must not be bad-name")
	}
}

func TestEvalValidations_MessageExpression(t *testing.T) {
	e, err := NewEvaluator()
	if err != nil {
		t.Fatalf("NewEvaluator() error: %v", err)
	}

	policy := &AdmissionPolicy{
		validations: []validation{
			{
				Path:              "validations[0]",
				Expression:        "false",
				Message:           "static fallback",
				MessageExpression: "'denied: ' + object.metadata.name",
			},
		},
	}
	input := &AdmissionInput{
		object: map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "Pod",
			"metadata":   map[string]interface{}{"name": "my-pod"},
		},
	}

	result, err := e.EvalValidations(policy, input)
	if err != nil {
		t.Fatalf("EvalValidations() error: %v", err)
	}
	if result.Allowed {
		t.Error("EvalValidations() Allowed = true, want false")
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

func TestEvalValidations_MessageExpressionFallback(t *testing.T) {
	e, err := NewEvaluator()
	if err != nil {
		t.Fatalf("NewEvaluator() error: %v", err)
	}

	tests := []struct {
		name              string
		messageExpression string
		want              string
	}{
		{name: "trimmed", messageExpression: "'  denied  '", want: "denied"},
		{name: "empty falls back", messageExpression: "'   '", want: "static fallback"},
		{name: "multiline falls back", messageExpression: "'hello\\nthere'", want: "static fallback"},
		{name: "oversized falls back", messageExpression: "'" + strings.Repeat("x", 5*1024+1) + "'", want: "static fallback"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			policy := &AdmissionPolicy{
				validations: []validation{{
					Path:              "validations[0]",
					Expression:        "false",
					Message:           "static fallback",
					MessageExpression: tt.messageExpression,
				}},
			}
			input := &AdmissionInput{
				object: map[string]interface{}{
					"apiVersion": "v1",
					"kind":       "Pod",
					"metadata":   map[string]interface{}{"name": "my-pod"},
				},
			}

			result, err := e.EvalValidations(policy, input)
			if err != nil {
				t.Fatalf("EvalValidations() error: %v", err)
			}
			if len(result.Violations) != 1 {
				t.Fatalf("got %d violations, want 1", len(result.Violations))
			}
			if result.Violations[0].Message != tt.want {
				t.Errorf("violation message = %q, want %q", result.Violations[0].Message, tt.want)
			}
		})
	}
}

func TestEvalValidations_MessageExpressionError(t *testing.T) {
	e, err := NewEvaluator()
	if err != nil {
		t.Fatalf("NewEvaluator() error: %v", err)
	}

	policy := &AdmissionPolicy{
		validations: []validation{
			{
				Path:              "validations[0]",
				Expression:        "false",
				Message:           "static fallback",
				MessageExpression: "object.nonexistent.deeply.nested",
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

	result, err := e.EvalValidations(policy, input)
	if err != nil {
		t.Fatalf("EvalValidations() error: %v", err)
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

func TestEvalValidations_MessageExpressionDoesNotDeclareAuthorizer(t *testing.T) {
	e, err := NewEvaluator()
	if err != nil {
		t.Fatalf("NewEvaluator() error: %v", err)
	}

	policy := &AdmissionPolicy{
		validations: []validation{
			{
				Path:              "validations[0]",
				Expression:        "false",
				Message:           "static fallback",
				MessageExpression: "authorizer.requestResource.check('get').allowed() ? 'allowed' : 'denied'",
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

	result, err := e.EvalValidations(policy, input)
	if err != nil {
		t.Fatalf("EvalValidations() error: %v", err)
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

}

func TestEvalValidations_Variables(t *testing.T) {
	e, err := NewEvaluator()
	if err != nil {
		t.Fatalf("NewEvaluator() error: %v", err)
	}

	policy := &AdmissionPolicy{
		variables: []variable{
			{Name: "podName", Expression: "object.metadata.name"},
			{Name: "isAllowed", Expression: "variables.podName == 'good-pod'"},
		},
		validations: []validation{
			{Path: "validations[0]", Expression: "variables.isAllowed"},
		},
	}

	t.Run("allowed", func(t *testing.T) {
		input := &AdmissionInput{
			object: map[string]interface{}{
				"apiVersion": "v1",
				"kind":       "Pod",
				"metadata":   map[string]interface{}{"name": "good-pod"},
			},
		}
		result, err := e.EvalValidations(policy, input)
		if err != nil {
			t.Fatalf("EvalValidations() error: %v", err)
		}
		if !result.Allowed {
			t.Errorf("expected Allowed=true, got violations: %s", result.FormatViolations())
		}
	})

	t.Run("denied", func(t *testing.T) {
		input := &AdmissionInput{
			object: map[string]interface{}{
				"apiVersion": "v1",
				"kind":       "Pod",
				"metadata":   map[string]interface{}{"name": "bad-pod"},
			},
		}
		result, err := e.EvalValidations(policy, input)
		if err != nil {
			t.Fatalf("EvalValidations() error: %v", err)
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
		variables: []variable{
			{Name: "podName", Expression: "object.metadata.name"},
			{Name: "nameLen", Expression: "size(variables.podName)"},
		},
	}
	input := &AdmissionInput{
		object: map[string]interface{}{
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

func TestEvalValidations_EvaluatesValidationsIndependentlyOfMatchConditions(t *testing.T) {
	e, err := NewEvaluator()
	if err != nil {
		t.Fatalf("NewEvaluator() error: %v", err)
	}

	policy := &AdmissionPolicy{
		matchConditions: []matchCondition{{Path: "spec.matchConditions[0]", Name: "not-system", Expression: "false"}},
		validations:     []validation{{Path: "spec.validations[0]", Expression: "false", Message: "validation still evaluated"}},
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

	result, err := e.EvalValidations(policy, input)
	if err != nil {
		t.Fatalf("EvalValidations() error: %v", err)
	}
	if result.Allowed {
		t.Fatal("expected validation to be evaluated and deny")
	}
	if len(result.Violations) != 1 {
		t.Fatalf("got %d violations, want 1", len(result.Violations))
	}
}

func TestEvalValidations_RequestFields(t *testing.T) {
	e, err := NewEvaluator()
	if err != nil {
		t.Fatalf("NewEvaluator() error: %v", err)
	}

	policy := &AdmissionPolicy{
		validations: []validation{
			{Path: "validations[0]", Expression: "request.operation == 'CREATE'"},
		},
	}
	input := &AdmissionInput{
		object: map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "Pod",
			"metadata":   map[string]interface{}{"name": "test"},
		},
		request: &admissionv1.AdmissionRequest{
			Operation: admissionv1.Create,
			Kind:      metav1.GroupVersionKind{Version: "v1", Kind: "Pod"},
			Resource:  metav1.GroupVersionResource{Version: "v1", Resource: "pods"},
		},
	}

	result, err := e.EvalValidations(policy, input)
	if err != nil {
		t.Fatalf("EvalValidations() error: %v", err)
	}
	if !result.Allowed {
		t.Errorf("expected Allowed=true, got violations: %s", result.FormatViolations())
	}
}

func TestEvalValidations_CompilationError(t *testing.T) {
	e, err := NewEvaluator()
	if err != nil {
		t.Fatalf("NewEvaluator() error: %v", err)
	}

	policy := &AdmissionPolicy{
		validations: []validation{
			{Path: "validations[0]", Expression: "???invalid"},
		},
	}
	input := &AdmissionInput{
		object: map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "Pod",
			"metadata":   map[string]interface{}{"name": "test"},
		},
	}

	_, err = e.EvalValidations(policy, input)
	if err == nil {
		t.Error("EvalValidations() expected compilation error")
	}
}

func TestEvalValidations_RejectsPatchTypes(t *testing.T) {
	e, err := NewEvaluator()
	if err != nil {
		t.Fatalf("NewEvaluator() error: %v", err)
	}

	policy := &AdmissionPolicy{
		validations: []validation{{Path: "validations[0]", Expression: patchTypeBoolExpression}},
	}
	input := &AdmissionInput{
		object: map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "Pod",
			"metadata":   map[string]interface{}{"name": "test"},
		},
	}

	if _, err := e.EvalValidations(policy, input); err == nil {
		t.Fatal("expected validation evaluation to reject mutation patch types")
	}
}

func TestEvalValidations_MultipleValidations(t *testing.T) {
	e, err := NewEvaluator()
	if err != nil {
		t.Fatalf("NewEvaluator() error: %v", err)
	}

	policy := &AdmissionPolicy{
		validations: []validation{
			{Path: "validations[0]", Expression: "true"},
			{Path: "validations[1]", Expression: "false", Message: "always fails"},
			{Path: "validations[2]", Expression: "true"},
		},
	}
	input := &AdmissionInput{
		object: map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "Pod",
			"metadata":   map[string]interface{}{"name": "test"},
		},
	}

	result, err := e.EvalValidations(policy, input)
	if err != nil {
		t.Fatalf("EvalValidations() error: %v", err)
	}
	if result.Allowed {
		t.Error("expected Allowed=false when one validation fails")
	}
	if len(result.Violations) != 1 {
		t.Fatalf("got %d violations, want 1", len(result.Violations))
	}
	if result.Violations[0].Expression != "false" {
		t.Errorf("violation expression = %q, want false", result.Violations[0].Expression)
	}
	if result.Violations[0].Message != "always fails" {
		t.Errorf("violation message = %q, want %q", result.Violations[0].Message, "always fails")
	}
}

func TestPreambleVariables(t *testing.T) {
	e, err := NewEvaluator(
		WithPreambleVariables(PreambleVariable{Name: "alwaysTrue", Expression: "true"}),
	)
	if err != nil {
		t.Fatalf("NewEvaluator() error: %v", err)
	}

	policy := &AdmissionPolicy{
		validations: []validation{
			{Path: "validations[0]", Expression: "variables.alwaysTrue"},
		},
	}
	input := &AdmissionInput{
		object: map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "Pod",
			"metadata":   map[string]interface{}{"name": "test"},
		},
	}

	result, err := e.EvalValidations(policy, input)
	if err != nil {
		t.Fatalf("EvalValidations() error: %v", err)
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
			validations: []validation{
				{Path: "validations[0]", Expression: "params.data.maxReplicas == '5'"},
			},
		}
		policy.setHasParams(true)

		input := &AdmissionInput{
			object: map[string]interface{}{
				"apiVersion": "v1",
				"kind":       "Pod",
				"metadata":   map[string]interface{}{"name": "test"},
			},
			params: map[string]interface{}{
				"data": map[string]interface{}{"maxReplicas": "5"},
			},
		}

		result, err := e.EvalValidations(policy, input)
		if err != nil {
			t.Fatalf("EvalValidations() error: %v", err)
		}
		if !result.Allowed {
			t.Errorf("expected Allowed=true, got violations: %s", result.FormatViolations())
		}
	})

	t.Run("params disabled causes compilation error", func(t *testing.T) {
		policy := &AdmissionPolicy{
			validations: []validation{
				{Path: "validations[0]", Expression: "params.data.maxReplicas == '5'"},
			},
		}
		policy.setHasParams(false)

		input := &AdmissionInput{
			object: map[string]interface{}{
				"apiVersion": "v1",
				"kind":       "Pod",
				"metadata":   map[string]interface{}{"name": "test"},
			},
		}

		_, err := e.EvalValidations(policy, input)
		if err == nil {
			t.Error("expected compilation error when params is disabled but expression references params")
		}
	})

	t.Run("default enables params", func(t *testing.T) {
		// Manually constructed policy without setHasParams should default to params enabled.
		policy := &AdmissionPolicy{
			validations: []validation{
				{Path: "validations[0]", Expression: "params.data.key == 'val'"},
			},
		}
		input := &AdmissionInput{
			object: map[string]interface{}{
				"apiVersion": "v1",
				"kind":       "Pod",
				"metadata":   map[string]interface{}{"name": "test"},
			},
			params: map[string]interface{}{
				"data": map[string]interface{}{"key": "val"},
			},
		}

		result, err := e.EvalValidations(policy, input)
		if err != nil {
			t.Fatalf("EvalValidations() error: %v", err)
		}
		if !result.Allowed {
			t.Errorf("expected Allowed=true with default params, got violations: %s", result.FormatViolations())
		}
	})
}

func TestEvalValidations_OldObject(t *testing.T) {
	e, err := NewEvaluator()
	if err != nil {
		t.Fatalf("NewEvaluator() error: %v", err)
	}

	policy := &AdmissionPolicy{
		validations: []validation{
			{
				Path:       "validations[0]",
				Expression: "object.metadata.name != oldObject.metadata.name",
				Message:    "name must change on update",
			},
		},
	}

	t.Run("names differ", func(t *testing.T) {
		input := &AdmissionInput{
			object: map[string]interface{}{
				"apiVersion": "v1",
				"kind":       "Pod",
				"metadata":   map[string]interface{}{"name": "new-name"},
			},
			oldObject: map[string]interface{}{
				"apiVersion": "v1",
				"kind":       "Pod",
				"metadata":   map[string]interface{}{"name": "old-name"},
			},
			request: &admissionv1.AdmissionRequest{
				Operation: admissionv1.Update,
				Kind:      metav1.GroupVersionKind{Version: "v1", Kind: "Pod"},
				Resource:  metav1.GroupVersionResource{Version: "v1", Resource: "pods"},
			},
		}

		result, err := e.EvalValidations(policy, input)
		if err != nil {
			t.Fatalf("EvalValidations() error: %v", err)
		}
		if !result.Allowed {
			t.Errorf("expected Allowed=true, got violations: %s", result.FormatViolations())
		}
	})

	t.Run("names same", func(t *testing.T) {
		input := &AdmissionInput{
			object: map[string]interface{}{
				"apiVersion": "v1",
				"kind":       "Pod",
				"metadata":   map[string]interface{}{"name": "same-name"},
			},
			oldObject: map[string]interface{}{
				"apiVersion": "v1",
				"kind":       "Pod",
				"metadata":   map[string]interface{}{"name": "same-name"},
			},
			request: &admissionv1.AdmissionRequest{
				Operation: admissionv1.Update,
				Kind:      metav1.GroupVersionKind{Version: "v1", Kind: "Pod"},
				Resource:  metav1.GroupVersionResource{Version: "v1", Resource: "pods"},
			},
		}

		result, err := e.EvalValidations(policy, input)
		if err != nil {
			t.Fatalf("EvalValidations() error: %v", err)
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
			object: map[string]interface{}{
				"apiVersion": "v1",
				"kind":       "Pod",
				"metadata":   map[string]interface{}{"name": "new-name"},
			},
			oldObject: map[string]interface{}{
				"apiVersion": "v1",
				"kind":       "Pod",
				"metadata":   map[string]interface{}{"name": "old-name"},
			},
		}

		policyOp := &AdmissionPolicy{
			validations: []validation{
				{Path: "validations[0]", Expression: "request.operation == 'UPDATE'"},
			},
		}

		result, err := e.EvalValidations(policyOp, input)
		if err != nil {
			t.Fatalf("EvalValidations() error: %v", err)
		}
		if !result.Allowed {
			t.Errorf("expected inferred UPDATE operation, got violations: %s", result.FormatViolations())
		}
	})
}
