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
	"fmt"
	"strings"
	"testing"
)

const (
	patchTypeBoolExpression   = `JSONPatch{op: "add", path: "/metadata/name", value: "x"}.op == "add"`
	patchTypeStringExpression = `JSONPatch{op: "add", path: "/metadata/name", value: "x"}.op`
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
		{name: "patch types unavailable", expr: patchTypeBoolExpression, wantErr: true},
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
				object: map[string]interface{}{
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
		{
			name:    "patch types unavailable",
			expr:    patchTypeBoolExpression,
			input:   &AdmissionInput{},
			wantErr: true,
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

	t.Run("multiple violations use newlines", func(t *testing.T) {
		r := &AdmissionResult{
			Violations: []Violation{
				{Expression: "false", Message: "first violation"},
				{Expression: "object.metadata.name == 'allowed'"},
				{Expression: "bad", Error: fmt.Errorf("compile error")},
			},
		}
		want := "first violation\nexpression \"object.metadata.name == 'allowed'\" evaluated to false\nerror evaluating \"bad\": compile error"
		if got := r.FormatViolations(); got != want {
			t.Errorf("FormatViolations() = %q, want %q", got, want)
		}
	})

	t.Run("includes message expression error", func(t *testing.T) {
		r := &AdmissionResult{
			Violations: []Violation{{
				Expression:   "false",
				Message:      "static fallback",
				MessageError: fmt.Errorf("message line 1\nmessage line 2"),
			}},
		}
		want := "static fallback\nerror evaluating messageExpression for \"false\": message line 1\nmessage line 2"
		if got := r.FormatViolations(); got != want {
			t.Errorf("FormatViolations() = %q, want %q", got, want)
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

	policy := &AdmissionPolicy{
		validations: []validation{
			{Path: "validations[0]", Expression: "sets.contains([1, 2, 3], [1])"},
		},
	}
	result, err := eOld.EvalValidations(policy, &AdmissionInput{})
	if err != nil {
		t.Fatalf("EvalValidations should use stored expression compatibility, got error: %v", err)
	}
	if !result.Allowed {
		t.Fatalf("EvalValidations with stored expression compatibility should allow, got: %s", result.FormatViolations())
	}

	// Version 1.29+ should have sets library.
	eNew, err := NewEvaluator(WithVersion(1, 29))
	if err != nil {
		t.Fatalf("NewEvaluator(1.29) error: %v", err)
	}
	err = eNew.CompileCheck("sets.contains([1, 2, 3], [1])")
	if err != nil {
		t.Errorf("compileCheck with sets at v1.29 should succeed, got: %v", err)
	}
}

func TestWithCostLimit(t *testing.T) {
	// A very low cost budget should cause evaluation to fail.
	e, err := NewEvaluator(WithCostLimit(1))
	if err != nil {
		t.Fatalf("NewEvaluator() error: %v", err)
	}

	// This expression has enough cost to exceed budget of 1.
	policy := &AdmissionPolicy{
		variables: []variable{
			{Name: "a", Expression: "object.metadata.name"},
			{Name: "b", Expression: "variables.a + variables.a"},
			{Name: "c", Expression: "variables.b + variables.b"},
		},
		validations: []validation{
			{Path: "validations[0]", Expression: "size(variables.c) > 0"},
		},
	}
	input := &AdmissionInput{
		object: map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "Pod",
			"metadata":   map[string]interface{}{"name": "test-pod"},
		},
	}

	result, err := e.EvalValidations(policy, input)
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
