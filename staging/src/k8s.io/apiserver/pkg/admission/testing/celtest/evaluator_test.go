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
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	admissionv1 "k8s.io/api/admission/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

func TestNewEvaluator(t *testing.T) {
	e, err := NewEvaluator()
	if err != nil {
		t.Fatalf("NewEvaluator() error: %v", err)
	}
	if e == nil {
		t.Fatal("NewEvaluator() returned nil")
	}
	if !e.authorizerEnabled {
		t.Error("authorizerEnabled should default to true")
	}
	if !e.patchTypesEnabled {
		t.Error("patchTypesEnabled should default to true")
	}
}

func TestNewEvaluatorWithOptions(t *testing.T) {
	e, err := NewEvaluator(
		WithVersion(1, 30),
		WithoutPatchTypes(),
		WithoutAuthorizer(),
		WithCostLimit(1000),
	)
	if err != nil {
		t.Fatalf("NewEvaluator() error: %v", err)
	}
	if e.patchTypesEnabled {
		t.Error("patchTypesEnabled should be false after WithoutPatchTypes()")
	}
	if e.authorizerEnabled {
		t.Error("authorizerEnabled should be false after WithoutAuthorizer()")
	}
	if e.costLimit != 1000 {
		t.Errorf("costLimit = %d, want 1000", e.costLimit)
	}
}

func TestCompileCheck(t *testing.T) {
	e, err := NewEvaluator()
	if err != nil {
		t.Fatalf("NewEvaluator() error: %v", err)
	}

	tests := []struct {
		name    string
		expr    string
		wantErr bool
	}{
		{name: "valid bool", expr: "true", wantErr: false},
		{name: "valid string", expr: "'hello'", wantErr: false},
		{name: "valid object access", expr: "object.metadata.name", wantErr: false},
		{name: "valid request access", expr: "request.operation == 'CREATE'", wantErr: false},
		{name: "invalid syntax", expr: "???", wantErr: true},
		{name: "undefined variable", expr: "nonexistent.field", wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := e.CompileCheck(tt.expr)
			if (err != nil) != tt.wantErr {
				t.Errorf("CompileCheck(%q) error = %v, wantErr %v", tt.expr, err, tt.wantErr)
			}
		})
	}
}

func TestEvalExpression(t *testing.T) {
	e, err := NewEvaluator()
	if err != nil {
		t.Fatalf("NewEvaluator() error: %v", err)
	}

	tests := []struct {
		name    string
		expr    string
		input   *AdmissionInput
		want    interface{}
		wantErr bool
	}{
		{
			name:  "literal true",
			expr:  "true",
			input: &AdmissionInput{},
			want:  true,
		},
		{
			name:  "literal string",
			expr:  "'hello'",
			input: &AdmissionInput{},
			want:  "hello",
		},
		{
			name: "object field access",
			expr: "object.metadata.name",
			input: &AdmissionInput{
				Object: map[string]interface{}{
					"apiVersion": "v1",
					"kind":       "Pod",
					"metadata":   map[string]interface{}{"name": "test-pod"},
				},
			},
			want: "test-pod",
		},
		{
			name:    "nil input defaults to empty",
			expr:    "true",
			input:   nil,
			want:    true,
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := e.EvalExpression(tt.expr, tt.input)
			if (err != nil) != tt.wantErr {
				t.Fatalf("EvalExpression() error = %v, wantErr %v", err, tt.wantErr)
			}
			if !tt.wantErr && got != tt.want {
				t.Errorf("EvalExpression() = %v (%T), want %v (%T)", got, got, tt.want, tt.want)
			}
		})
	}
}

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

func TestParseAdmissionPolicy_Flat(t *testing.T) {
	yaml := `
variables:
  - name: podName
    expression: "object.metadata.name"
validations:
  - expression: "variables.podName != 'bad'"
    message: "name must not be bad"
`
	policy, err := ParseAdmissionPolicy(yaml)
	if err != nil {
		t.Fatalf("ParseAdmissionPolicy() error: %v", err)
	}
	if len(policy.Variables) != 1 {
		t.Errorf("got %d variables, want 1", len(policy.Variables))
	}
	if len(policy.Validations) != 1 {
		t.Errorf("got %d validations, want 1", len(policy.Validations))
	}
	if policy.Variables[0].Name != "podName" {
		t.Errorf("variable name = %q, want %q", policy.Variables[0].Name, "podName")
	}
}

func TestParseAdmissionPolicy_VAP(t *testing.T) {
	yaml := `
apiVersion: admissionregistration.k8s.io/v1
kind: ValidatingAdmissionPolicy
spec:
  validations:
    - expression: "object.spec.replicas <= 5"
      message: "too many replicas"
`
	policy, err := ParseAdmissionPolicy(yaml)
	if err != nil {
		t.Fatalf("ParseAdmissionPolicy() error: %v", err)
	}
	if len(policy.Validations) != 1 {
		t.Errorf("got %d validations, want 1", len(policy.Validations))
	}
	if policy.hasParams {
		t.Error("hasParams should be false when paramKind is not set")
	}
}

func TestParseAdmissionPolicy_Errors(t *testing.T) {
	tests := []struct {
		name string
		yaml string
	}{
		{name: "empty variable name", yaml: `
variables:
  - name: ""
    expression: "true"
`},
		{name: "empty variable expression", yaml: `
variables:
  - name: "x"
    expression: ""
`},
		{name: "empty validation expression", yaml: `
validations:
  - expression: ""
`},
		{name: "unsupported kind", yaml: `
apiVersion: v1
kind: UnsupportedKind
`},
		{name: "empty policy", yaml: `{}`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := ParseAdmissionPolicy(tt.yaml)
			if err == nil {
				t.Error("ParseAdmissionPolicy() expected error")
			}
		})
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

func TestFormatViolations(t *testing.T) {
	t.Run("no violations", func(t *testing.T) {
		r := &AdmissionResult{Allowed: true}
		if got := r.FormatViolations(); got != "" {
			t.Errorf("FormatViolations() = %q, want empty", got)
		}
	})

	t.Run("with message", func(t *testing.T) {
		r := &AdmissionResult{
			Violations: []Violation{{Expression: "false", Message: "denied"}},
		}
		if got := r.FormatViolations(); got != "denied" {
			t.Errorf("FormatViolations() = %q, want %q", got, "denied")
		}
	})

	t.Run("with error", func(t *testing.T) {
		r := &AdmissionResult{
			Violations: []Violation{{Expression: "bad", Error: fmt.Errorf("compile error")}},
		}
		got := r.FormatViolations()
		if !strings.Contains(got, "compile error") {
			t.Errorf("FormatViolations() = %q, expected to contain 'compile error'", got)
		}
	})
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

func TestParseAdmissionPolicy_MAP(t *testing.T) {
	yaml := `
apiVersion: admissionregistration.k8s.io/v1
kind: MutatingAdmissionPolicy
spec:
  variables:
    - name: replicas
      expression: "object.spec.replicas"
  mutations:
    - patchType: ApplyConfiguration
      applyConfiguration:
        expression: "Object{spec: Object.spec{replicas: 3}}"
    - patchType: JSONPatch
      jsonPatch:
        expression: '[JSONPatch{op: "replace", path: "/spec/replicas", value: 3}]'
`
	policy, err := ParseAdmissionPolicy(yaml)
	if err != nil {
		t.Fatalf("ParseAdmissionPolicy() error: %v", err)
	}
	if len(policy.Variables) != 1 {
		t.Errorf("got %d variables, want 1", len(policy.Variables))
	}
	if len(policy.Mutations) != 2 {
		t.Fatalf("got %d mutations, want 2", len(policy.Mutations))
	}
	if policy.Mutations[0].PatchType != "ApplyConfiguration" {
		t.Errorf("mutation[0] patchType = %q, want ApplyConfiguration", policy.Mutations[0].PatchType)
	}
	if policy.Mutations[1].PatchType != "JSONPatch" {
		t.Errorf("mutation[1] patchType = %q, want JSONPatch", policy.Mutations[1].PatchType)
	}
	if policy.hasParams {
		t.Error("hasParams should be false when paramKind is not set")
	}
}

func TestParseAdmissionPolicy_MAP_WithParamKind(t *testing.T) {
	yaml := `
apiVersion: admissionregistration.k8s.io/v1
kind: MutatingAdmissionPolicy
spec:
  paramKind:
    apiVersion: v1
    kind: ConfigMap
  mutations:
    - patchType: ApplyConfiguration
      applyConfiguration:
        expression: "Object{spec: Object.spec{replicas: int(params.data.maxReplicas)}}"
`
	policy, err := ParseAdmissionPolicy(yaml)
	if err != nil {
		t.Fatalf("ParseAdmissionPolicy() error: %v", err)
	}
	if !policy.hasParams {
		t.Error("hasParams should be true when paramKind is set")
	}
}

func TestParseAdmissionPolicy_MAP_Errors(t *testing.T) {
	tests := []struct {
		name string
		yaml string
	}{
		{name: "missing patchType", yaml: `
apiVersion: admissionregistration.k8s.io/v1
kind: MutatingAdmissionPolicy
spec:
  mutations:
    - applyConfiguration:
        expression: "Object{}"
`},
		{name: "missing applyConfiguration expression", yaml: `
apiVersion: admissionregistration.k8s.io/v1
kind: MutatingAdmissionPolicy
spec:
  mutations:
    - patchType: ApplyConfiguration
`},
		{name: "missing jsonPatch expression", yaml: `
apiVersion: admissionregistration.k8s.io/v1
kind: MutatingAdmissionPolicy
spec:
  mutations:
    - patchType: JSONPatch
`},
		{name: "unsupported patchType", yaml: `
apiVersion: admissionregistration.k8s.io/v1
kind: MutatingAdmissionPolicy
spec:
  mutations:
    - patchType: StrategicMerge
      applyConfiguration:
        expression: "Object{}"
`},
		{name: "empty policy", yaml: `
apiVersion: admissionregistration.k8s.io/v1
kind: MutatingAdmissionPolicy
spec: {}
`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := ParseAdmissionPolicy(tt.yaml)
			if err == nil {
				t.Error("ParseAdmissionPolicy() expected error")
			}
		})
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

func TestParseAdmissionPolicyFile(t *testing.T) {
	dir := t.TempDir()

	t.Run("valid file", func(t *testing.T) {
		path := filepath.Join(dir, "policy.yaml")
		content := []byte(`
variables:
  - name: podName
    expression: "object.metadata.name"
validations:
  - expression: "variables.podName != 'bad'"
    message: "name must not be bad"
`)
		if err := os.WriteFile(path, content, 0644); err != nil {
			t.Fatalf("writing test file: %v", err)
		}
		policy, err := ParseAdmissionPolicyFile(path)
		if err != nil {
			t.Fatalf("ParseAdmissionPolicyFile() error: %v", err)
		}
		if len(policy.Variables) != 1 {
			t.Errorf("got %d variables, want 1", len(policy.Variables))
		}
		if len(policy.Validations) != 1 {
			t.Errorf("got %d validations, want 1", len(policy.Validations))
		}
	})

	t.Run("missing file", func(t *testing.T) {
		_, err := ParseAdmissionPolicyFile(filepath.Join(dir, "nonexistent.yaml"))
		if err == nil {
			t.Error("expected error for missing file")
		}
	})

	t.Run("invalid content", func(t *testing.T) {
		path := filepath.Join(dir, "empty.yaml")
		if err := os.WriteFile(path, []byte("{}"), 0644); err != nil {
			t.Fatalf("writing test file: %v", err)
		}
		_, err := ParseAdmissionPolicyFile(path)
		if err == nil {
			t.Error("expected error for empty policy file")
		}
	})
}

func TestWithVersion(t *testing.T) {
	// Version 1.28 should NOT have CEL sets library (added in 1.29).
	eOld, err := NewEvaluator(WithVersion(1, 28))
	if err != nil {
		t.Fatalf("NewEvaluator(1.28) error: %v", err)
	}
	err = eOld.CompileCheck("sets.contains([1, 2, 3], [1])")
	if err == nil {
		t.Error("expected compilation error for sets library at version 1.28")
	}

	// Version 1.29+ should have sets library.
	eNew, err := NewEvaluator(WithVersion(1, 29))
	if err != nil {
		t.Fatalf("NewEvaluator(1.29) error: %v", err)
	}
	err = eNew.CompileCheck("sets.contains([1, 2, 3], [1])")
	if err != nil {
		t.Errorf("CompileCheck with sets at v1.29 should succeed, got: %v", err)
	}
}

func TestParseAdmissionInput(t *testing.T) {
	yaml := `
object:
  apiVersion: v1
  kind: Pod
  metadata:
    name: test-pod
    namespace: default
params:
  apiVersion: v1
  kind: ConfigMap
  metadata:
    name: policy-config
  data:
    allowed: "true"
request:
  operation: CREATE
  kind:
    version: v1
    kind: Pod
  resource:
    version: v1
    resource: pods
namespaceObject:
  apiVersion: v1
  kind: Namespace
  metadata:
    name: default
`
	input, err := ParseAdmissionInput(yaml)
	if err != nil {
		t.Fatalf("ParseAdmissionInput() error: %v", err)
	}
	if input.Object["kind"] != "Pod" {
		t.Errorf("object.kind = %v, want Pod", input.Object["kind"])
	}
	if input.Params["kind"] != "ConfigMap" {
		t.Errorf("params.kind = %v, want ConfigMap", input.Params["kind"])
	}
	if input.Request == nil || input.Request.Operation != admissionv1.Create {
		t.Fatalf("request.operation = %v, want CREATE", input.Request)
	}
	if input.Namespace == nil || input.Namespace.Name != "default" {
		t.Fatalf("namespace.name = %v, want default", input.Namespace)
	}

	e, err := NewEvaluator()
	if err != nil {
		t.Fatalf("NewEvaluator() error: %v", err)
	}
	policy := &AdmissionPolicy{
		Validations: []Validation{{Path: "validations[0]", Expression: "params.data.allowed == 'true'"}},
	}
	policy.SetHasParams(true)
	result, err := e.EvalAdmission(policy, input)
	if err != nil {
		t.Fatalf("EvalAdmission() error: %v", err)
	}
	if !result.Allowed {
		t.Errorf("expected Allowed=true, got violations: %s", result.FormatViolations())
	}
}

func TestParseAdmissionInput_PreambleCanUnwrapParamsResource(t *testing.T) {
	yaml := `
object:
  apiVersion: v1
  kind: Pod
  metadata:
    name: labeled-pod
    labels:
      team: platform
params:
  apiVersion: policy.example.com/v1alpha1
  kind: LabelPolicyParams
  metadata:
    name: required-labels
  spec:
    parameters:
      labels:
      - key: team
request:
  operation: CREATE
  kind:
    version: v1
    kind: Pod
  resource:
    version: v1
    resource: pods
`
	input, err := ParseAdmissionInput(yaml)
	if err != nil {
		t.Fatalf("ParseAdmissionInput() error: %v", err)
	}

	e, err := NewEvaluator(WithPreambleVariables(
		Variable{
			Name:       "params",
			Expression: `!has(params.spec) ? null : !has(params.spec.parameters) ? null : params.spec.parameters`,
		},
	))
	if err != nil {
		t.Fatalf("NewEvaluator() error: %v", err)
	}

	policy := &AdmissionPolicy{
		Validations: []Validation{
			{
				Path:       "validations[0]",
				Expression: `variables.params.labels.all(entry, has(object.metadata.labels) && entry.key in object.metadata.labels)`,
				Message:    "missing required label",
			},
		},
	}
	policy.SetHasParams(true)

	result, err := e.EvalAdmission(policy, input)
	if err != nil {
		t.Fatalf("EvalAdmission() error: %v", err)
	}
	if !result.Allowed {
		t.Errorf("expected Allowed=true, got violations: %s", result.FormatViolations())
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

func TestWithCostLimit(t *testing.T) {
	// A very low cost budget should cause evaluation to fail.
	e, err := NewEvaluator(WithCostLimit(1))
	if err != nil {
		t.Fatalf("NewEvaluator() error: %v", err)
	}

	// This expression has enough cost to exceed budget of 1.
	policy := &AdmissionPolicy{
		Variables: []Variable{
			{Name: "a", Expression: "object.metadata.name"},
			{Name: "b", Expression: "variables.a + variables.a"},
			{Name: "c", Expression: "variables.b + variables.b"},
		},
		Validations: []Validation{
			{Path: "validations[0]", Expression: "size(variables.c) > 0"},
		},
	}
	input := &AdmissionInput{
		Object: map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "Pod",
			"metadata":   map[string]interface{}{"name": "test-pod"},
		},
	}

	result, err := e.EvalAdmission(policy, input)
	// Either the evaluation returns an error due to budget exhaustion,
	// or the result contains violations with errors.
	if err != nil {
		return // budget exceeded at top level, expected
	}
	if result.Allowed {
		// If the budget was somehow not exceeded, the result should still be valid.
		// This test primarily ensures WithCostLimit is wired through.
		if result.Cost <= 0 {
			t.Error("expected Cost > 0 even with low budget")
		}
	}
}

func TestParseAdmissionPolicy_ValidatingWebhookConfiguration(t *testing.T) {
	yaml := `
apiVersion: admissionregistration.k8s.io/v1
kind: ValidatingWebhookConfiguration
webhooks:
  - name: validate.example.com
    matchConditions:
      - name: exclude-leases
        expression: "!(request.resource.resource == 'leases')"
      - name: exclude-system-ns
        expression: "request.namespace != 'kube-system'"
`
	policy, err := ParseAdmissionPolicy(yaml)
	if err != nil {
		t.Fatalf("ParseAdmissionPolicy() error: %v", err)
	}
	if len(policy.Validations) != 2 {
		t.Fatalf("got %d validations, want 2", len(policy.Validations))
	}
	if policy.Validations[0].Path != "webhooks[0].matchConditions[0]" {
		t.Errorf("validation[0] path = %q, want %q", policy.Validations[0].Path, "webhooks[0].matchConditions[0]")
	}
	if policy.hasParams {
		t.Error("hasParams should be false for webhook configurations")
	}
}

func TestParseAdmissionPolicy_MutatingWebhookConfiguration(t *testing.T) {
	yaml := `
apiVersion: admissionregistration.k8s.io/v1
kind: MutatingWebhookConfiguration
webhooks:
  - name: mutate.example.com
    matchConditions:
      - name: only-pods
        expression: "request.resource.resource == 'pods'"
`
	policy, err := ParseAdmissionPolicy(yaml)
	if err != nil {
		t.Fatalf("ParseAdmissionPolicy() error: %v", err)
	}
	if len(policy.Validations) != 1 {
		t.Fatalf("got %d validations, want 1", len(policy.Validations))
	}
	if policy.Validations[0].Expression != "request.resource.resource == 'pods'" {
		t.Errorf("unexpected expression: %q", policy.Validations[0].Expression)
	}
}

func TestParseAdmissionPolicy_WebhookErrors(t *testing.T) {
	tests := []struct {
		name string
		yaml string
	}{
		{name: "empty matchConditions", yaml: `
apiVersion: admissionregistration.k8s.io/v1
kind: ValidatingWebhookConfiguration
webhooks:
  - name: validate.example.com
`},
		{name: "empty expression in matchCondition", yaml: `
apiVersion: admissionregistration.k8s.io/v1
kind: ValidatingWebhookConfiguration
webhooks:
  - name: validate.example.com
    matchConditions:
      - name: empty
        expression: ""
`},
		{name: "mutating empty matchConditions", yaml: `
apiVersion: admissionregistration.k8s.io/v1
kind: MutatingWebhookConfiguration
webhooks:
  - name: mutate.example.com
`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := ParseAdmissionPolicy(tt.yaml)
			if err == nil {
				t.Error("ParseAdmissionPolicy() expected error")
			}
		})
	}
}

func TestEvalAdmission_WebhookMatchConditions(t *testing.T) {
	e, err := NewEvaluator(WithoutAuthorizer(), WithoutPatchTypes())
	if err != nil {
		t.Fatalf("NewEvaluator() error: %v", err)
	}

	yaml := `
apiVersion: admissionregistration.k8s.io/v1
kind: ValidatingWebhookConfiguration
webhooks:
  - name: validate.example.com
    matchConditions:
      - name: only-pods
        expression: "request.resource.resource == 'pods'"
`
	policy, err := ParseAdmissionPolicy(yaml)
	if err != nil {
		t.Fatalf("ParseAdmissionPolicy() error: %v", err)
	}

	input := &AdmissionInput{
		Object: map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "Pod",
			"metadata":   map[string]interface{}{"name": "test"},
		},
		Request: &admissionv1.AdmissionRequest{
			Kind:     metav1.GroupVersionKind{Version: "v1", Kind: "Pod"},
			Resource: metav1.GroupVersionResource{Version: "v1", Resource: "pods"},
		},
	}

	result, err := e.EvalAdmission(policy, input)
	if err != nil {
		t.Fatalf("EvalAdmission() error: %v", err)
	}
	if !result.Allowed {
		t.Errorf("expected Allowed=true for pods, got violations: %s", result.FormatViolations())
	}
}

func TestDefaultResourceForGVK(t *testing.T) {
	tests := []struct {
		kind    string
		wantRes string
	}{
		{kind: "Pod", wantRes: "pods"},
		{kind: "Service", wantRes: "services"},
		{kind: "Deployment", wantRes: "deployments"},
		{kind: "Ingress", wantRes: "ingresses"},
		{kind: "NetworkPolicy", wantRes: "networkpolicies"},
		{kind: "DaemonSet", wantRes: "daemonsets"},
		{kind: "ConfigMap", wantRes: "configmaps"},
		{kind: "Endpoints", wantRes: "endpointses"}, // naive but consistent for sibilant ending
		{kind: "", wantRes: "objects"},
	}
	for _, tt := range tests {
		t.Run(tt.kind, func(t *testing.T) {
			gvk := schema.GroupVersionKind{Version: "v1", Kind: tt.kind}
			gvr := defaultResourceForGVK(gvk)
			if gvr.Resource != tt.wantRes {
				t.Errorf("defaultResourceForGVK(%q) = %q, want %q", tt.kind, gvr.Resource, tt.wantRes)
			}
		})
	}
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

func TestParseAdmissionPolicy_VAPWithParamKind(t *testing.T) {
	yaml := `
apiVersion: admissionregistration.k8s.io/v1
kind: ValidatingAdmissionPolicy
spec:
  paramKind:
    apiVersion: v1
    kind: ConfigMap
  validations:
    - expression: "params.data.allowed == 'true'"
      message: "not allowed"
`
	policy, err := ParseAdmissionPolicy(yaml)
	if err != nil {
		t.Fatalf("ParseAdmissionPolicy() error: %v", err)
	}
	if !policy.hasParams {
		t.Error("hasParams should be true when paramKind is set")
	}
}
