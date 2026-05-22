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
	admissionv1 "k8s.io/api/admission/v1"
	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"testing"
)

func TestParseAndEvalVAPMatchConditionsAndAuditAnnotations(t *testing.T) {
	e, err := NewEvaluator()
	if err != nil {
		t.Fatalf("NewEvaluator() error: %v", err)
	}

	policyYAML := `{"apiVersion":"admissionregistration.k8s.io/v1","kind":"ValidatingAdmissionPolicy","spec":{"paramKind":{"apiVersion":"v1","kind":"ConfigMap"},"matchConditions":[{"name":"enabled","expression":"params.data.enabled == 'true'"}],"validations":[{"expression":"true"}],"auditAnnotations":[{"key":"pod-name","valueExpression":"string(object.metadata.name)"}]}}`
	policy, err := ParseAdmissionPolicy(policyYAML)
	if err != nil {
		t.Fatalf("ParseAdmissionPolicy() error: %v", err)
	}

	input := &AdmissionInput{
		Object: map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "Pod",
			"metadata":   map[string]interface{}{"name": "test-pod"},
		},
		Params: map[string]interface{}{
			"data": map[string]interface{}{"enabled": "true"},
		},
	}

	result, err := e.EvalAdmission(policy, input)
	if err != nil {
		t.Fatalf("EvalAdmission() error: %v", err)
	}
	if !result.Allowed {
		t.Fatalf("expected admission validations to allow: %s", result.FormatViolations())
	}

	matchResult, err := e.EvalMatchConditions(policy, input)
	if err != nil {
		t.Fatalf("EvalMatchConditions() error: %v", err)
	}
	if len(matchResult.Conditions) != 1 || matchResult.Conditions[0].Value != true {
		t.Fatalf("unexpected matchConditions: %#v", matchResult.Conditions)
	}

	auditResult, err := e.EvalAuditAnnotations(policy, input)
	if err != nil {
		t.Fatalf("EvalAuditAnnotations() error: %v", err)
	}
	if len(auditResult.Annotations) != 1 || auditResult.Annotations[0].Value != "test-pod" {
		t.Fatalf("unexpected audit annotations: %#v", auditResult.Annotations)
	}
}

func TestTypedPolicyAndInputTable(t *testing.T) {
	e, err := NewEvaluator()
	if err != nil {
		t.Fatalf("NewEvaluator() error: %v", err)
	}

	policy, err := NewFromValidatingAdmissionPolicy(&admissionregistrationv1.ValidatingAdmissionPolicy{
		Spec: admissionregistrationv1.ValidatingAdmissionPolicySpec{
			ParamKind: &admissionregistrationv1.ParamKind{APIVersion: "v1", Kind: "ConfigMap"},
			Validations: []admissionregistrationv1.Validation{{
				Expression: "has(object.metadata.labels) && object.metadata.labels.team == params.data.requiredTeam",
				Message:    "pod team label must match policy params",
			}},
		},
	})
	if err != nil {
		t.Fatalf("NewFromValidatingAdmissionPolicy() error: %v", err)
	}

	tests := []struct {
		name    string
		labels  map[string]string
		allowed bool
	}{
		{name: "matching label", labels: map[string]string{"team": "platform"}, allowed: true},
		{name: "different label", labels: map[string]string{"team": "frontend"}, allowed: false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := e.EvalAdmission(policy, &AdmissionInput{
				Object: &corev1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						Name:   "sample",
						Labels: tt.labels,
					},
				},
				Params: &corev1.ConfigMap{
					ObjectMeta: metav1.ObjectMeta{Name: "policy-params"},
					Data:       map[string]string{"requiredTeam": "platform"},
				},
			})
			if err != nil {
				t.Fatalf("EvalAdmission() error: %v", err)
			}
			if result.Allowed != tt.allowed {
				t.Fatalf("Allowed = %v, want %v; violations: %s", result.Allowed, tt.allowed, result.FormatViolations())
			}
		})
	}
}

func TestParseAndEvalMAPMatchConditions(t *testing.T) {
	e, err := NewEvaluator()
	if err != nil {
		t.Fatalf("NewEvaluator() error: %v", err)
	}

	policyYAML := `{"apiVersion":"admissionregistration.k8s.io/v1","kind":"MutatingAdmissionPolicy","spec":{"matchConditions":[{"name":"only-pods","expression":"object.kind == 'Pod'"}],"mutations":[{"patchType":"ApplyConfiguration","applyConfiguration":{"expression":"Object{metadata: Object.metadata{labels: {'mutated': 'true'}}}"}}]}}`
	policy, err := ParseAdmissionPolicy(policyYAML)
	if err != nil {
		t.Fatalf("ParseAdmissionPolicy() error: %v", err)
	}

	t.Run("false match condition still allows direct mutation evaluation", func(t *testing.T) {
		input := &AdmissionInput{
			Object: map[string]interface{}{
				"apiVersion": "v1",
				"kind":       "ConfigMap",
				"metadata":   map[string]interface{}{"name": "not-a-pod"},
			},
		}
		matchResult, err := e.EvalMatchConditions(policy, input)
		if err != nil {
			t.Fatalf("EvalMatchConditions() error: %v", err)
		}
		if len(matchResult.Conditions) != 1 || matchResult.Conditions[0].Value != false {
			t.Fatalf("unexpected matchConditions: %#v", matchResult.Conditions)
		}

		result, err := e.EvalMutation(policy, input)
		if err != nil {
			t.Fatalf("EvalMutation() error: %v", err)
		}
		if len(result.Patches) != 1 {
			t.Fatalf("got %d patches, want 1", len(result.Patches))
		}
	})

	t.Run("true match condition", func(t *testing.T) {
		input := &AdmissionInput{
			Object: map[string]interface{}{
				"apiVersion": "v1",
				"kind":       "Pod",
				"metadata":   map[string]interface{}{"name": "pod"},
			},
		}
		matchResult, err := e.EvalMatchConditions(policy, input)
		if err != nil {
			t.Fatalf("EvalMatchConditions() error: %v", err)
		}
		if len(matchResult.Conditions) != 1 || matchResult.Conditions[0].Value != true {
			t.Fatalf("unexpected matchConditions: %#v", matchResult.Conditions)
		}

		result, err := e.EvalMutation(policy, input)
		if err != nil {
			t.Fatalf("EvalMutation() error: %v", err)
		}
		if len(result.Patches) != 1 {
			t.Fatalf("got %d patches, want 1", len(result.Patches))
		}
	})
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
