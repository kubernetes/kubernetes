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
	"testing"
)

func TestEvalAuditAnnotations(t *testing.T) {
	e, err := NewEvaluator()
	if err != nil {
		t.Fatalf("NewEvaluator() error: %v", err)
	}

	policy := &AdmissionPolicy{
		AuditAnnotations: []AuditAnnotation{
			{Path: "spec.auditAnnotations[0]", Key: "pod-name", ValueExpression: "string(object.metadata.name)"},
			{Path: "spec.auditAnnotations[1]", Key: "empty", ValueExpression: "''"},
			{Path: "spec.auditAnnotations[2]", Key: "null", ValueExpression: "null"},
		},
	}
	input := &AdmissionInput{
		Object: map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "Pod",
			"metadata":   map[string]interface{}{"name": "audit-pod"},
		},
	}

	result, err := e.EvalAuditAnnotations(policy, input)
	if err != nil {
		t.Fatalf("EvalAuditAnnotations() error: %v", err)
	}
	if len(result.Annotations) != 3 {
		t.Fatalf("got %d annotations, want 3", len(result.Annotations))
	}
	if result.Annotations[0].Value != "audit-pod" {
		t.Errorf("annotation[0] = %#v, want audit-pod", result.Annotations[0])
	}
	if result.Annotations[1].Value != "" {
		t.Errorf("annotation[1] value = %v, want empty string", result.Annotations[1].Value)
	}
	if result.Annotations[2].Value != nil {
		t.Errorf("annotation[2] value = %v, want nil", result.Annotations[2].Value)
	}

	value, err := e.EvalAuditAnnotation(policy, AuditAnnotationSelector{Path: "spec.auditAnnotations[0]"}, input)
	if err != nil {
		t.Fatalf("EvalAuditAnnotation() error: %v", err)
	}
	if value != "audit-pod" {
		t.Errorf("EvalAuditAnnotation() = %v, want audit-pod", value)
	}
}

func TestEvalAuditAnnotations_CompileError(t *testing.T) {
	e, err := NewEvaluator()
	if err != nil {
		t.Fatalf("NewEvaluator() error: %v", err)
	}

	input := &AdmissionInput{
		Object: map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "Pod",
			"metadata":   map[string]interface{}{"name": "test-pod"},
		},
	}

	policy := &AdmissionPolicy{
		Validations:      []Validation{{Path: "spec.validations[0]", Expression: "true"}},
		AuditAnnotations: []AuditAnnotation{{Path: "spec.auditAnnotations[0]", Key: "bad", ValueExpression: "1"}},
	}

	if _, err := e.EvalAuditAnnotations(policy, input); err == nil {
		t.Fatal("expected audit annotation compile error")
	}

	result, err := e.EvalAdmission(policy, input)
	if err != nil {
		t.Fatalf("EvalAdmission() error: %v", err)
	}
	if !result.Allowed {
		t.Fatalf("EvalAdmission should only evaluate validations, got violations: %s", result.FormatViolations())
	}
}

func TestEvalAuditAnnotations_RejectsPatchTypes(t *testing.T) {
	e, err := NewEvaluator()
	if err != nil {
		t.Fatalf("NewEvaluator() error: %v", err)
	}

	policy := &AdmissionPolicy{
		AuditAnnotations: []AuditAnnotation{{Path: "spec.auditAnnotations[0]", Key: "patch", ValueExpression: patchTypeStringExpression}},
	}
	input := &AdmissionInput{
		Object: map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "Pod",
			"metadata":   map[string]interface{}{"name": "test-pod"},
		},
	}

	if _, err := e.EvalAuditAnnotations(policy, input); err == nil {
		t.Fatal("expected audit annotation evaluation to reject mutation patch types")
	}
	if _, err := e.EvalAuditAnnotation(policy, AuditAnnotationSelector{Path: "spec.auditAnnotations[0]"}, input); err == nil {
		t.Fatal("expected single audit annotation evaluation to reject mutation patch types")
	}
}
