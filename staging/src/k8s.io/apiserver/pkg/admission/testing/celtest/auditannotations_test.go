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
	"context"
	"testing"

	"k8s.io/apiserver/pkg/authorization/authorizer"
)

func TestEvalAuditAnnotations(t *testing.T) {
	e, err := NewEvaluator()
	if err != nil {
		t.Fatalf("NewEvaluator() error: %v", err)
	}

	policy := &AdmissionPolicy{
		auditAnnotations: []auditAnnotation{
			{Path: "spec.auditAnnotations[0]", Key: "pod-name", ValueExpression: "string(object.metadata.name)"},
			{Path: "spec.auditAnnotations[1]", Key: "empty", ValueExpression: "''"},
			{Path: "spec.auditAnnotations[2]", Key: "null", ValueExpression: "null"},
		},
	}
	input := &AdmissionInput{
		object: map[string]interface{}{
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
	wantAnnotations := []struct {
		path            string
		key             string
		valueExpression string
		value           interface{}
	}{
		{path: "spec.auditAnnotations[0]", key: "pod-name", valueExpression: "string(object.metadata.name)", value: "audit-pod"},
		{path: "spec.auditAnnotations[1]", key: "empty", valueExpression: "''", value: ""},
		{path: "spec.auditAnnotations[2]", key: "null", valueExpression: "null", value: nil},
	}
	for i, want := range wantAnnotations {
		annotation := result.Annotations[i]
		if annotation.Path != want.path {
			t.Errorf("annotation[%d].Path = %q, want %q", i, annotation.Path, want.path)
		}
		if annotation.Key != want.key {
			t.Errorf("annotation[%d].Key = %q, want %q", i, annotation.Key, want.key)
		}
		if annotation.ValueExpression != want.valueExpression {
			t.Errorf("annotation[%d].ValueExpression = %q, want %q", i, annotation.ValueExpression, want.valueExpression)
		}
		if annotation.Value != want.value {
			t.Errorf("annotation[%d].Value = %#v, want %#v", i, annotation.Value, want.value)
		}
	}
}

func TestEvalAuditAnnotations_CompileError(t *testing.T) {
	e, err := NewEvaluator()
	if err != nil {
		t.Fatalf("NewEvaluator() error: %v", err)
	}

	input := &AdmissionInput{
		object: map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "Pod",
			"metadata":   map[string]interface{}{"name": "test-pod"},
		},
	}

	policy := &AdmissionPolicy{
		validations:      []validation{{Path: "spec.validations[0]", Expression: "true"}},
		auditAnnotations: []auditAnnotation{{Path: "spec.auditAnnotations[0]", Key: "bad", ValueExpression: "1"}},
	}

	if _, err := e.EvalAuditAnnotations(policy, input); err == nil {
		t.Fatal("expected audit annotation compile error")
	}

	result, err := e.EvalValidations(policy, input)
	if err != nil {
		t.Fatalf("EvalValidations() error: %v", err)
	}
	if !result.Allowed {
		t.Fatalf("EvalValidations should only evaluate validations, got violations: %s", result.FormatViolations())
	}
}

func TestEvalAuditAnnotations_RejectsPatchTypes(t *testing.T) {
	e, err := NewEvaluator()
	if err != nil {
		t.Fatalf("NewEvaluator() error: %v", err)
	}

	policy := &AdmissionPolicy{
		auditAnnotations: []auditAnnotation{{Path: "spec.auditAnnotations[0]", Key: "patch", ValueExpression: patchTypeStringExpression}},
	}
	input := &AdmissionInput{
		object: map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "Pod",
			"metadata":   map[string]interface{}{"name": "test-pod"},
		},
	}

	if _, err := e.EvalAuditAnnotations(policy, input); err == nil {
		t.Fatal("expected audit annotation evaluation to reject mutation patch types")
	}
}

func TestEvalAuditAnnotations_StripsAuthorizerBinding(t *testing.T) {
	e, err := NewEvaluator()
	if err != nil {
		t.Fatalf("NewEvaluator() error: %v", err)
	}

	allowAll := authorizer.AuthorizerFunc(func(_ context.Context, _ authorizer.Attributes) (authorizer.Decision, string, error) {
		return authorizer.DecisionAllow, "", nil
	})

	policy := &AdmissionPolicy{
		auditAnnotations: []auditAnnotation{
			{
				Path:            "spec.auditAnnotations[0]",
				Key:             "auth",
				ValueExpression: "authorizer.requestResource.check('get').allowed() ? 'allowed' : 'denied'",
			},
		},
	}
	input := &AdmissionInput{
		object: map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "Pod",
			"metadata":   map[string]interface{}{"name": "audit-pod"},
		},
		authorizer: allowAll,
	}

	bulkResult, err := e.EvalAuditAnnotations(policy, input)
	if err != nil {
		t.Fatalf("EvalAuditAnnotations() error: %v", err)
	}
	if len(bulkResult.Annotations) != 1 {
		t.Fatalf("got %d annotations, want 1", len(bulkResult.Annotations))
	}
	if bulkResult.Annotations[0].Error == nil {
		t.Fatalf("EvalAuditAnnotations: expected runtime error from stripped authorizer binding, got value %#v", bulkResult.Annotations[0].Value)
	}
}
