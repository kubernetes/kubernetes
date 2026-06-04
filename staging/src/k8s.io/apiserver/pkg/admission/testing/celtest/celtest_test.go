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

package celtest_test

import (
	"testing"

	admissionv1 "k8s.io/api/admission/v1"
	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/admission/testing/celtest"
)

func TestPublicAdmissionAPI(t *testing.T) {
	evaluator, err := celtest.NewEvaluator(celtest.WithPreambleVariables(celtest.PreambleVariable{Name: "objectName", Expression: "object.metadata.name"}))
	if err != nil {
		t.Fatalf("NewEvaluator() error: %v", err)
	}

	policy, err := celtest.NewFromValidatingAdmissionPolicy(&admissionregistrationv1.ValidatingAdmissionPolicy{
		Spec: admissionregistrationv1.ValidatingAdmissionPolicySpec{
			ParamKind: &admissionregistrationv1.ParamKind{APIVersion: "v1", Kind: "ConfigMap"},
			Variables: []admissionregistrationv1.Variable{{
				Name:       "requiredTeam",
				Expression: "params.data.requiredTeam",
			}},
			MatchConditions: []admissionregistrationv1.MatchCondition{{
				Name:       "only-pods",
				Expression: "object.kind == 'Pod'",
			}},
			Validations: []admissionregistrationv1.Validation{{
				Expression: "variables.objectName == 'sample' && object.metadata.labels.team == variables.requiredTeam",
			}},
		},
	})
	if err != nil {
		t.Fatalf("NewFromValidatingAdmissionPolicy() error: %v", err)
	}

	input := celtest.NewAdmissionInput().
		SetObject(&corev1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:   "sample",
				Labels: map[string]string{"team": "platform"},
			},
		}).
		SetParams(&corev1.ConfigMap{Data: map[string]string{"requiredTeam": "platform"}})

	if err := evaluator.CompileCheck("variables.objectName == 'sample'"); err != nil {
		t.Fatalf("CompileCheck() error: %v", err)
	}

	value, err := evaluator.EvalExpression("variables.objectName", input)
	if err != nil {
		t.Fatalf("EvalExpression() error: %v", err)
	}
	if value != "sample" {
		t.Fatalf("EvalExpression() = %v, want %q", value, "sample")
	}

	value, err = evaluator.EvalVariable(policy, "requiredTeam", input)
	if err != nil {
		t.Fatalf("EvalVariable() error: %v", err)
	}
	if value != "platform" {
		t.Fatalf("EvalVariable() = %v, want %q", value, "platform")
	}

	matchResult, err := evaluator.EvalMatchConditions(policy, input)
	if err != nil {
		t.Fatalf("EvalMatchConditions() error: %v", err)
	}
	if len(matchResult.Conditions) != 1 || matchResult.Conditions[0].Value != true {
		t.Fatalf("unexpected matchConditions: %#v", matchResult.Conditions)
	}

	result, err := evaluator.EvalValidations(policy, input)
	if err != nil {
		t.Fatalf("EvalValidations() error: %v", err)
	}
	if !result.Allowed {
		t.Fatalf("expected Allowed=true, got violations: %s", result.FormatViolations())
	}
}

func TestValidatingAdmissionPolicyPublicAPI(t *testing.T) {
	evaluator, err := celtest.NewEvaluator()
	if err != nil {
		t.Fatalf("NewEvaluator() error: %v", err)
	}

	policyYAML := `{"apiVersion":"admissionregistration.k8s.io/v1","kind":"ValidatingAdmissionPolicy","spec":{"paramKind":{"apiVersion":"v1","kind":"ConfigMap"},"matchConditions":[{"name":"enabled","expression":"params.data.enabled == 'true'"}],"validations":[{"expression":"true"}],"auditAnnotations":[{"key":"pod-name","valueExpression":"string(object.metadata.name)"}]}}`
	policy, err := celtest.ParseAdmissionPolicy(policyYAML)
	if err != nil {
		t.Fatalf("ParseAdmissionPolicy() error: %v", err)
	}

	input := celtest.NewAdmissionInput().
		SetUnstructuredObject(map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "Pod",
			"metadata":   map[string]interface{}{"name": "test-pod"},
		}).
		SetUnstructuredParams(map[string]interface{}{
			"data": map[string]interface{}{"enabled": "true"},
		})

	result, err := evaluator.EvalValidations(policy, input)
	if err != nil {
		t.Fatalf("EvalValidations() error: %v", err)
	}
	if !result.Allowed {
		t.Fatalf("expected admission validations to allow: %s", result.FormatViolations())
	}

	matchResult, err := evaluator.EvalMatchConditions(policy, input)
	if err != nil {
		t.Fatalf("EvalMatchConditions() error: %v", err)
	}
	if len(matchResult.Conditions) != 1 || matchResult.Conditions[0].Value != true {
		t.Fatalf("unexpected matchConditions: %#v", matchResult.Conditions)
	}

	auditResult, err := evaluator.EvalAuditAnnotations(policy, input)
	if err != nil {
		t.Fatalf("EvalAuditAnnotations() error: %v", err)
	}
	if len(auditResult.Annotations) != 1 || auditResult.Annotations[0].Value != "test-pod" {
		t.Fatalf("unexpected audit annotations: %#v", auditResult.Annotations)
	}
}

func TestTypedPolicyAndInputTable(t *testing.T) {
	evaluator, err := celtest.NewEvaluator()
	if err != nil {
		t.Fatalf("NewEvaluator() error: %v", err)
	}

	policy, err := celtest.NewFromValidatingAdmissionPolicy(&admissionregistrationv1.ValidatingAdmissionPolicy{
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
			input := celtest.NewAdmissionInput().
				SetObject(&corev1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						Name:   "sample",
						Labels: tt.labels,
					},
				}).
				SetParams(&corev1.ConfigMap{
					ObjectMeta: metav1.ObjectMeta{Name: "policy-params"},
					Data:       map[string]string{"requiredTeam": "platform"},
				})

			result, err := evaluator.EvalValidations(policy, input)
			if err != nil {
				t.Fatalf("EvalValidations() error: %v", err)
			}
			if result.Allowed != tt.allowed {
				t.Fatalf("Allowed = %v, want %v; violations: %s", result.Allowed, tt.allowed, result.FormatViolations())
			}
		})
	}
}

func TestMutatingAdmissionPolicyPublicAPI(t *testing.T) {
	evaluator, err := celtest.NewEvaluator()
	if err != nil {
		t.Fatalf("NewEvaluator() error: %v", err)
	}

	policyYAML := `{"apiVersion":"admissionregistration.k8s.io/v1","kind":"MutatingAdmissionPolicy","spec":{"matchConditions":[{"name":"only-pods","expression":"object.kind == 'Pod'"}],"mutations":[{"patchType":"ApplyConfiguration","applyConfiguration":{"expression":"Object{metadata: Object.metadata{labels: {'mutated': 'true'}}}"}}]}}`
	policy, err := celtest.ParseAdmissionPolicy(policyYAML)
	if err != nil {
		t.Fatalf("ParseAdmissionPolicy() error: %v", err)
	}
	verifyMutatedLabelPatch := func(t *testing.T, result *celtest.MutationResult) {
		t.Helper()
		if len(result.Patches) != 1 {
			t.Fatalf("got %d patches, want 1", len(result.Patches))
		}
		patch := result.Patches[0]
		if patch.Path != "spec.mutations[0]" {
			t.Fatalf("patch path = %q, want spec.mutations[0]", patch.Path)
		}
		if patch.PatchType != string(admissionregistrationv1.PatchTypeApplyConfiguration) {
			t.Fatalf("patch type = %q, want ApplyConfiguration", patch.PatchType)
		}
		if patch.Error != nil {
			t.Fatalf("patch error: %v", patch.Error)
		}
		applyConfig, ok := patch.Value.(map[string]interface{})
		if !ok {
			t.Fatalf("patch value type = %T, want map[string]interface{}", patch.Value)
		}
		metadata, ok := applyConfig["metadata"].(map[string]interface{})
		if !ok {
			t.Fatalf("metadata type = %T, want map[string]interface{}", applyConfig["metadata"])
		}
		labels, ok := metadata["labels"].(map[string]interface{})
		if !ok {
			t.Fatalf("labels type = %T, want map[string]interface{}", metadata["labels"])
		}
		if labels["mutated"] != "true" {
			t.Fatalf("mutated label = %v, want true", labels["mutated"])
		}
	}

	t.Run("false match condition still allows direct mutation evaluation", func(t *testing.T) {
		input := celtest.NewAdmissionInput().SetUnstructuredObject(map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "ConfigMap",
			"metadata":   map[string]interface{}{"name": "not-a-pod"},
		})
		matchResult, err := evaluator.EvalMatchConditions(policy, input)
		if err != nil {
			t.Fatalf("EvalMatchConditions() error: %v", err)
		}
		if len(matchResult.Conditions) != 1 || matchResult.Conditions[0].Value != false {
			t.Fatalf("unexpected matchConditions: %#v", matchResult.Conditions)
		}

		result, err := evaluator.EvalMutation(policy, input)
		if err != nil {
			t.Fatalf("EvalMutation() error: %v", err)
		}
		verifyMutatedLabelPatch(t, result)
	})

	t.Run("true match condition", func(t *testing.T) {
		input := celtest.NewAdmissionInput().SetUnstructuredObject(map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "Pod",
			"metadata":   map[string]interface{}{"name": "pod"},
		})
		matchResult, err := evaluator.EvalMatchConditions(policy, input)
		if err != nil {
			t.Fatalf("EvalMatchConditions() error: %v", err)
		}
		if len(matchResult.Conditions) != 1 || matchResult.Conditions[0].Value != true {
			t.Fatalf("unexpected matchConditions: %#v", matchResult.Conditions)
		}

		result, err := evaluator.EvalMutation(policy, input)
		if err != nil {
			t.Fatalf("EvalMutation() error: %v", err)
		}
		verifyMutatedLabelPatch(t, result)
	})
}

func TestValidatingWebhookConfigurationPublicAPI(t *testing.T) {
	evaluator, err := celtest.NewEvaluator(celtest.WithoutAuthorizer(), celtest.WithoutPatchTypes())
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
	policy, err := celtest.ParseAdmissionPolicy(yaml)
	if err != nil {
		t.Fatalf("ParseAdmissionPolicy() error: %v", err)
	}

	input := celtest.NewAdmissionInput().
		SetUnstructuredObject(map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "Pod",
			"metadata":   map[string]interface{}{"name": "test"},
		}).
		SetRequest(&admissionv1.AdmissionRequest{
			Kind:     metav1.GroupVersionKind{Version: "v1", Kind: "Pod"},
			Resource: metav1.GroupVersionResource{Version: "v1", Resource: "pods"},
		})

	matchResult, err := evaluator.EvalMatchConditions(policy, input)
	if err != nil {
		t.Fatalf("EvalMatchConditions() error: %v", err)
	}
	if len(matchResult.Conditions) != 1 {
		t.Fatalf("got %d match conditions, want 1", len(matchResult.Conditions))
	}
	if matchResult.Conditions[0].Value != true {
		t.Fatalf("match condition value = %v, want true", matchResult.Conditions[0].Value)
	}

}
