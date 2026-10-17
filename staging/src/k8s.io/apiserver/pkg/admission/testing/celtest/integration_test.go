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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// preambleVars returns example preamble variables that a policy framework may
// inject before policy-specific variables.
func preambleVars() Option {
	return WithPreambleVariables(
		PreambleVariable{Name: "anyObject", Expression: "object"},
		PreambleVariable{Name: "params", Expression: "params"},
		PreambleVariable{Name: "isUpdate", Expression: "has(request.operation) && request.operation == 'UPDATE'"},
	)
}

// TestIntegration_ValidatingAdmissionPolicy tests end-to-end evaluation of a
// ValidatingAdmissionPolicy YAML document including variables, validations,
// and messageExpression.
func TestIntegration_ValidatingAdmissionPolicy(t *testing.T) {
	e, err := NewEvaluator()
	if err != nil {
		t.Fatalf("NewEvaluator() error: %v", err)
	}

	policyYAML := `
apiVersion: admissionregistration.k8s.io/v1
kind: ValidatingAdmissionPolicy
metadata:
  name: require-labels
spec:
  paramKind:
    apiVersion: v1
    kind: ConfigMap
  variables:
    - name: requiredLabel
      expression: "params.data.requiredLabel"
    - name: hasLabel
      expression: "has(object.metadata.labels) && variables.requiredLabel in object.metadata.labels"
  validations:
    - expression: "variables.hasLabel"
      message: "missing required label"
      messageExpression: "'object ' + object.metadata.name + ' is missing required label: ' + variables.requiredLabel"
`
	policy, err := ParseAdmissionPolicy(policyYAML)
	if err != nil {
		t.Fatalf("ParseAdmissionPolicy() error: %v", err)
	}

	t.Run("allowed when label present", func(t *testing.T) {
		input := &AdmissionInput{
			object: map[string]interface{}{
				"apiVersion": "v1",
				"kind":       "Pod",
				"metadata": map[string]interface{}{
					"name":   "good-pod",
					"labels": map[string]interface{}{"team": "backend"},
				},
			},
			params: map[string]interface{}{
				"data": map[string]interface{}{"requiredLabel": "team"},
			},
		}
		result, err := e.EvalValidations(policy, input)
		if err != nil {
			t.Fatalf("EvalValidations() error: %v", err)
		}
		if !result.Allowed {
			t.Errorf("expected Allowed=true, violations: %s", result.FormatViolations())
		}
	})

	t.Run("denied when label missing", func(t *testing.T) {
		input := &AdmissionInput{
			object: map[string]interface{}{
				"apiVersion": "v1",
				"kind":       "Pod",
				"metadata":   map[string]interface{}{"name": "bad-pod"},
			},
			params: map[string]interface{}{
				"data": map[string]interface{}{"requiredLabel": "team"},
			},
		}
		result, err := e.EvalValidations(policy, input)
		if err != nil {
			t.Fatalf("EvalValidations() error: %v", err)
		}
		if result.Allowed {
			t.Fatal("expected Allowed=false")
		}
		if len(result.Violations) != 1 {
			t.Fatalf("got %d violations, want 1", len(result.Violations))
		}
		if !strings.Contains(result.Violations[0].Message, "bad-pod") {
			t.Errorf("violation message = %q, expected to contain 'bad-pod'", result.Violations[0].Message)
		}
		if !strings.Contains(result.Violations[0].Message, "team") {
			t.Errorf("violation message = %q, expected to contain 'team'", result.Violations[0].Message)
		}
	})

	t.Run("eval individual variable", func(t *testing.T) {
		input := &AdmissionInput{
			object: map[string]interface{}{
				"apiVersion": "v1",
				"kind":       "Pod",
				"metadata": map[string]interface{}{
					"name":   "pod",
					"labels": map[string]interface{}{"team": "backend"},
				},
			},
			params: map[string]interface{}{
				"data": map[string]interface{}{"requiredLabel": "team"},
			},
		}
		val, err := e.EvalVariable(policy, "hasLabel", input)
		if err != nil {
			t.Fatalf("EvalVariable() error: %v", err)
		}
		if val != true {
			t.Errorf("hasLabel = %v, want true", val)
		}
	})
}

// TestIntegration_MutatingAdmissionPolicy tests end-to-end evaluation of a
// MutatingAdmissionPolicy YAML document with ApplyConfiguration and JSONPatch mutations.
func TestIntegration_MutatingAdmissionPolicy(t *testing.T) {
	e, err := NewEvaluator()
	if err != nil {
		t.Fatalf("NewEvaluator() error: %v", err)
	}

	policyYAML := `
apiVersion: admissionregistration.k8s.io/v1
kind: MutatingAdmissionPolicy
metadata:
  name: inject-sidecar-label
spec:
  variables:
    - name: needsLabel
      expression: "!has(object.metadata.labels) || !('sidecar-injected' in object.metadata.labels)"
  mutations:
    - patchType: ApplyConfiguration
      applyConfiguration:
        expression: |
          variables.needsLabel ? Object{
            metadata: Object.metadata{
              labels: {"sidecar-injected": "true"}
            }
          } : Object{}
`
	policy, err := ParseAdmissionPolicy(policyYAML)
	if err != nil {
		t.Fatalf("ParseAdmissionPolicy() error: %v", err)
	}

	t.Run("mutation produces patch when label missing", func(t *testing.T) {
		input := &AdmissionInput{
			object: map[string]interface{}{
				"apiVersion": "v1",
				"kind":       "Pod",
				"metadata":   map[string]interface{}{"name": "no-labels"},
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
		applyConfig, ok := result.Patches[0].Value.(map[string]interface{})
		if !ok {
			t.Fatalf("patch value type = %T, want map[string]interface{}", result.Patches[0].Value)
		}
		metadata, ok := applyConfig["metadata"].(map[string]interface{})
		if !ok {
			t.Fatalf("metadata type = %T, want map[string]interface{}", applyConfig["metadata"])
		}
		labels, ok := metadata["labels"].(map[string]interface{})
		if !ok {
			t.Fatalf("labels type = %T, want map[string]interface{}", metadata["labels"])
		}
		if labels["sidecar-injected"] != "true" {
			t.Errorf("labels[sidecar-injected] = %v, want 'true'", labels["sidecar-injected"])
		}
	})

	t.Run("variable evaluation", func(t *testing.T) {
		input := &AdmissionInput{
			object: map[string]interface{}{
				"apiVersion": "v1",
				"kind":       "Pod",
				"metadata": map[string]interface{}{
					"name":   "has-label",
					"labels": map[string]interface{}{"sidecar-injected": "true"},
				},
			},
		}
		val, err := e.EvalVariable(policy, "needsLabel", input)
		if err != nil {
			t.Fatalf("EvalVariable() error: %v", err)
		}
		if val != false {
			t.Errorf("needsLabel = %v, want false (label already present)", val)
		}
	})
}

// TestIntegration_WebhookMatchConditions tests evaluation of CEL matchConditions
// from both ValidatingWebhookConfiguration and MutatingWebhookConfiguration.
func TestIntegration_WebhookMatchConditions(t *testing.T) {
	e, err := NewEvaluator(WithoutAuthorizer())
	if err != nil {
		t.Fatalf("NewEvaluator() error: %v", err)
	}

	t.Run("ValidatingWebhookConfiguration", func(t *testing.T) {
		yaml := `
apiVersion: admissionregistration.k8s.io/v1
kind: ValidatingWebhookConfiguration
metadata:
  name: test-webhook
webhooks:
  - name: validate.example.com
    admissionReviewVersions: ["v1"]
    sideEffects: None
    clientConfig:
      url: "https://example.com/validate"
    matchConditions:
      - name: not-kube-system
        expression: "object.metadata.namespace != 'kube-system'"
      - name: is-create
        expression: "request.operation == 'CREATE'"
`
		policy, err := ParseAdmissionPolicy(yaml)
		if err != nil {
			t.Fatalf("ParseAdmissionPolicy() error: %v", err)
		}
		if len(policy.matchConditions) != 2 {
			t.Fatalf("got %d matchConditions, want 2", len(policy.matchConditions))
		}

		input := &AdmissionInput{
			object: map[string]interface{}{
				"apiVersion": "v1",
				"kind":       "Pod",
				"metadata": map[string]interface{}{
					"name":      "test",
					"namespace": "default",
				},
			},
			request: &admissionv1.AdmissionRequest{
				Operation: admissionv1.Create,
				Kind:      metav1.GroupVersionKind{Version: "v1", Kind: "Pod"},
				Resource:  metav1.GroupVersionResource{Version: "v1", Resource: "pods"},
			},
		}

		result, err := e.EvalMatchConditions(policy, input)
		if err != nil {
			t.Fatalf("EvalMatchConditions() error: %v", err)
		}
		if len(result.Conditions) != 2 {
			t.Fatalf("got %d conditions, want 2", len(result.Conditions))
		}
		for index, condition := range result.Conditions {
			if condition.Error != nil {
				t.Fatalf("condition[%d] error: %v", index, condition.Error)
			}
			if condition.Value != true {
				t.Errorf("condition[%d] value = %v, want true", index, condition.Value)
			}
		}
	})

	t.Run("MutatingWebhookConfiguration", func(t *testing.T) {
		yaml := `
apiVersion: admissionregistration.k8s.io/v1
kind: MutatingWebhookConfiguration
metadata:
  name: test-mutating-webhook
webhooks:
  - name: mutate.example.com
    admissionReviewVersions: ["v1"]
    sideEffects: None
    clientConfig:
      url: "https://example.com/mutate"
    matchConditions:
      - name: skip-system-ns
        expression: "!object.metadata.namespace.startsWith('kube-')"
`
		policy, err := ParseAdmissionPolicy(yaml)
		if err != nil {
			t.Fatalf("ParseAdmissionPolicy() error: %v", err)
		}

		// The matchCondition expression evaluates to false for kube-system.
		input := &AdmissionInput{
			object: map[string]interface{}{
				"apiVersion": "v1",
				"kind":       "Pod",
				"metadata": map[string]interface{}{
					"name":      "test",
					"namespace": "kube-system",
				},
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
			t.Errorf("condition value = %v, want false", result.Conditions[0].Value)
		}
	})
}

// TestIntegration_FlatPolicy tests the flat variables/validations format used
// by projects that store policy expressions without a resource wrapper.
func TestIntegration_FlatPolicy(t *testing.T) {
	e, err := NewEvaluator()
	if err != nil {
		t.Fatalf("NewEvaluator() error: %v", err)
	}

	flatYAML := `
variables:
  - name: replicas
    expression: "has(object.spec.replicas) ? object.spec.replicas : 1"
  - name: maxAllowed
    expression: "5"
validations:
  - expression: "variables.replicas <= variables.maxAllowed"
    messageExpression: "'replicas ' + string(variables.replicas) + ' exceeds max ' + string(variables.maxAllowed)"
`
	policy, err := ParseAdmissionPolicy(flatYAML)
	if err != nil {
		t.Fatalf("ParseAdmissionPolicy() error: %v", err)
	}

	t.Run("allowed", func(t *testing.T) {
		input := &AdmissionInput{
			object: map[string]interface{}{
				"apiVersion": "apps/v1",
				"kind":       "Deployment",
				"metadata":   map[string]interface{}{"name": "small-deploy"},
				"spec":       map[string]interface{}{"replicas": int64(3)},
			},
		}
		result, err := e.EvalValidations(policy, input)
		if err != nil {
			t.Fatalf("EvalValidations() error: %v", err)
		}
		if !result.Allowed {
			t.Errorf("expected Allowed=true: %s", result.FormatViolations())
		}
	})

	t.Run("denied with dynamic message", func(t *testing.T) {
		input := &AdmissionInput{
			object: map[string]interface{}{
				"apiVersion": "apps/v1",
				"kind":       "Deployment",
				"metadata":   map[string]interface{}{"name": "big-deploy"},
				"spec":       map[string]interface{}{"replicas": int64(10)},
			},
		}
		result, err := e.EvalValidations(policy, input)
		if err != nil {
			t.Fatalf("EvalValidations() error: %v", err)
		}
		if result.Allowed {
			t.Fatal("expected Allowed=false")
		}
		if !strings.Contains(result.Violations[0].Message, "10") {
			t.Errorf("message %q should contain actual replica count", result.Violations[0].Message)
		}
	})
}

// TestIntegration_PreambleVariables tests using WithPreambleVariables and then
// evaluating a policy that references those variables.
func TestIntegration_PreambleVariables(t *testing.T) {
	e, err := NewEvaluator(preambleVars())
	if err != nil {
		t.Fatalf("NewEvaluator() error: %v", err)
	}

	policy := &AdmissionPolicy{
		variables: []variable{
			{Name: "containers", Expression: `has(variables.anyObject.spec.containers) ? variables.anyObject.spec.containers : []`},
			{Name: "initContainers", Expression: `has(variables.anyObject.spec.initContainers) ? variables.anyObject.spec.initContainers : []`},
			{Name: "allContainers", Expression: `variables.containers + variables.initContainers`},
			{Name: "badContainers", Expression: `
				variables.allContainers.filter(container,
					has(container.securityContext) && has(container.securityContext.privileged) && container.securityContext.privileged
				).map(container, "Privileged container is not allowed: " + container.name + ", securityContext.privileged: true")`},
		},
		validations: []validation{
			{
				Path:              "validations[0]",
				Expression:        "variables.isUpdate || size(variables.badContainers) == 0",
				MessageExpression: `variables.badContainers.join(", ")`,
			},
		},
	}
	policy.setHasParams(true)

	t.Run("allowed - no privileged containers", func(t *testing.T) {
		input := &AdmissionInput{
			object: map[string]interface{}{
				"apiVersion": "v1",
				"kind":       "Pod",
				"metadata":   map[string]interface{}{"name": "safe-pod"},
				"spec": map[string]interface{}{
					"containers": []interface{}{
						map[string]interface{}{
							"name":  "app",
							"image": "nginx:latest",
						},
					},
				},
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
			t.Errorf("expected Allowed=true: %s", result.FormatViolations())
		}
	})

	t.Run("denied - privileged container", func(t *testing.T) {
		input := &AdmissionInput{
			object: map[string]interface{}{
				"apiVersion": "v1",
				"kind":       "Pod",
				"metadata":   map[string]interface{}{"name": "bad-pod"},
				"spec": map[string]interface{}{
					"containers": []interface{}{
						map[string]interface{}{
							"name":  "app",
							"image": "nginx:latest",
							"securityContext": map[string]interface{}{
								"privileged": true,
							},
						},
					},
				},
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
		if result.Allowed {
			t.Fatal("expected Allowed=false for privileged container")
		}
		if len(result.Violations) != 1 {
			t.Fatalf("got %d violations, want 1", len(result.Violations))
		}
		if !strings.Contains(result.Violations[0].Message, "Privileged container is not allowed") {
			t.Errorf("violation message = %q, expected to contain 'Privileged container is not allowed'", result.Violations[0].Message)
		}
		if !strings.Contains(result.Violations[0].Message, "app") {
			t.Errorf("violation message = %q, expected to contain container name 'app'", result.Violations[0].Message)
		}
	})

	t.Run("allowed on UPDATE - privileged container allowed", func(t *testing.T) {
		input := &AdmissionInput{
			object: map[string]interface{}{
				"apiVersion": "v1",
				"kind":       "Pod",
				"metadata":   map[string]interface{}{"name": "privileged-pod"},
				"spec": map[string]interface{}{
					"containers": []interface{}{
						map[string]interface{}{
							"name":  "app",
							"image": "nginx:latest",
							"securityContext": map[string]interface{}{
								"privileged": true,
							},
						},
					},
				},
			},
			oldObject: map[string]interface{}{
				"apiVersion": "v1",
				"kind":       "Pod",
				"metadata":   map[string]interface{}{"name": "privileged-pod"},
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
			t.Errorf("expected Allowed=true on UPDATE: %s", result.FormatViolations())
		}
	})

	t.Run("denied - privileged init container", func(t *testing.T) {
		input := &AdmissionInput{
			object: map[string]interface{}{
				"apiVersion": "v1",
				"kind":       "Pod",
				"metadata":   map[string]interface{}{"name": "init-pod"},
				"spec": map[string]interface{}{
					"containers": []interface{}{
						map[string]interface{}{"name": "app", "image": "nginx:latest"},
					},
					"initContainers": []interface{}{
						map[string]interface{}{
							"name":  "init-priv",
							"image": "busybox:latest",
							"securityContext": map[string]interface{}{
								"privileged": true,
							},
						},
					},
				},
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
		if result.Allowed {
			t.Fatal("expected Allowed=false for privileged init container")
		}
		if !strings.Contains(result.Violations[0].Message, "init-priv") {
			t.Errorf("message = %q, expected to contain 'init-priv'", result.Violations[0].Message)
		}
	})

	t.Run("eval individual variable", func(t *testing.T) {
		input := &AdmissionInput{
			object: map[string]interface{}{
				"apiVersion": "v1",
				"kind":       "Pod",
				"metadata":   map[string]interface{}{"name": "pod"},
				"spec": map[string]interface{}{
					"containers": []interface{}{
						map[string]interface{}{"name": "c1", "image": "img1"},
						map[string]interface{}{"name": "c2", "image": "img2"},
					},
				},
			},
			request: &admissionv1.AdmissionRequest{
				Operation: admissionv1.Create,
				Kind:      metav1.GroupVersionKind{Version: "v1", Kind: "Pod"},
				Resource:  metav1.GroupVersionResource{Version: "v1", Resource: "pods"},
			},
		}
		val, err := e.EvalVariable(policy, "containers", input)
		if err != nil {
			t.Fatalf("EvalVariable(containers) error: %v", err)
		}
		containers, ok := val.([]interface{})
		if !ok {
			t.Fatalf("containers type = %T, want []interface{}", val)
		}
		if len(containers) != 2 {
			t.Errorf("containers length = %d, want 2", len(containers))
		}
	})
}

// TestIntegration_CompileCheck tests that compileCheck catches syntax and type
// errors without needing input data.
func TestIntegration_CompileCheck(t *testing.T) {
	e, err := NewEvaluator()
	if err != nil {
		t.Fatalf("NewEvaluator() error: %v", err)
	}

	validExprs := []string{
		"object.metadata.name == 'test'",
		"has(object.metadata.labels)",
		"request.operation == 'CREATE'",
		"oldObject == null",
		"size(object.metadata.name) > 0",
		"object.metadata.name.startsWith('prefix-')",
	}
	for _, expr := range validExprs {
		if err := e.CompileCheck(expr); err != nil {
			t.Errorf("CompileCheck(%q) unexpected error: %v", expr, err)
		}
	}

	invalidExprs := []string{
		"???",
		"nonexistent.field",
	}
	for _, expr := range invalidExprs {
		if err := e.CompileCheck(expr); err == nil {
			t.Errorf("CompileCheck(%q) expected error", expr)
		}
	}
}

// TestIntegration_CostTracking verifies that CEL cost is tracked across
// variables and validations and that WithCostLimit enforces the budget.
func TestIntegration_CostTracking(t *testing.T) {
	t.Run("cost is reported", func(t *testing.T) {
		e, err := NewEvaluator()
		if err != nil {
			t.Fatalf("NewEvaluator() error: %v", err)
		}
		policy := &AdmissionPolicy{
			validations: []validation{
				{Path: "v0", Expression: "true"},
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
		if result.Cost < 0 {
			t.Errorf("expected non-negative cost, got %d", result.Cost)
		}
	})

	t.Run("cost limit exceeded", func(t *testing.T) {
		e, err := NewEvaluator(WithCostLimit(1))
		if err != nil {
			t.Fatalf("NewEvaluator() error: %v", err)
		}
		policy := &AdmissionPolicy{
			validations: []validation{
				{Path: "v0", Expression: "object.metadata.name == 'test'"},
				{Path: "v1", Expression: "object.metadata.name == 'test'"},
				{Path: "v2", Expression: "object.metadata.name == 'test'"},
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
			t.Error("expected cost budget exceeded error")
		}
	})
}

// TestIntegration_EvalExpression tests standalone expression evaluation.
func TestIntegration_EvalExpression(t *testing.T) {
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
				"namespace": "production",
				"labels": map[string]interface{}{
					"app":  "web",
					"tier": "frontend",
				},
			},
		},
	}

	tests := []struct {
		name string
		expr string
		want interface{}
	}{
		{name: "string field", expr: "object.metadata.name", want: "test-pod"},
		{name: "has check", expr: "has(object.metadata.labels)", want: true},
		{name: "map access", expr: "object.metadata.labels.app", want: "web"},
		{name: "string method", expr: "object.metadata.namespace.startsWith('prod')", want: true},
		{name: "arithmetic", expr: "size(object.metadata.name)", want: int64(8)},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := e.EvalExpression(tt.expr, input)
			if err != nil {
				t.Fatalf("EvalExpression(%q) error: %v", tt.expr, err)
			}
			if got != tt.want {
				t.Errorf("EvalExpression(%q) = %v (%T), want %v (%T)", tt.expr, got, got, tt.want, tt.want)
			}
		})
	}
}
