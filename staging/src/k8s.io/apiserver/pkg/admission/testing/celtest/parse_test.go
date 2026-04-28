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
	"os"
	"path/filepath"
	"testing"
)

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
  matchConditions:
    - name: skip-system-namespaces
      expression: "request.namespace != 'kube-system'"
  validations:
    - expression: "object.spec.replicas <= 5"
      message: "too many replicas"
  auditAnnotations:
    - key: replicas
      valueExpression: "string(object.spec.replicas)"
`
	policy, err := ParseAdmissionPolicy(yaml)
	if err != nil {
		t.Fatalf("ParseAdmissionPolicy() error: %v", err)
	}
	if len(policy.MatchConditions) != 1 {
		t.Fatalf("got %d matchConditions, want 1", len(policy.MatchConditions))
	}
	if policy.MatchConditions[0].Path != "spec.matchConditions[0]" {
		t.Errorf("matchCondition[0] path = %q, want spec.matchConditions[0]", policy.MatchConditions[0].Path)
	}
	if len(policy.Validations) != 1 {
		t.Errorf("got %d validations, want 1", len(policy.Validations))
	}
	if len(policy.AuditAnnotations) != 1 {
		t.Errorf("got %d auditAnnotations, want 1", len(policy.AuditAnnotations))
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

func TestParseAdmissionPolicy_MAP(t *testing.T) {
	yaml := `{"apiVersion":"admissionregistration.k8s.io/v1","kind":"MutatingAdmissionPolicy","spec":{"matchConditions":[{"name":"only-pods","expression":"request.resource.resource == 'pods'"}],"variables":[{"name":"replicas","expression":"object.spec.replicas"}],"mutations":[{"patchType":"ApplyConfiguration","applyConfiguration":{"expression":"Object{spec: Object.spec{replicas: 3}}"}},{"patchType":"JSONPatch","jsonPatch":{"expression":"[JSONPatch{op: \"replace\", path: \"/spec/replicas\", value: 3}]"}}]}}`
	policy, err := ParseAdmissionPolicy(yaml)
	if err != nil {
		t.Fatalf("ParseAdmissionPolicy() error: %v", err)
	}
	if len(policy.Variables) != 1 {
		t.Errorf("got %d variables, want 1", len(policy.Variables))
	}
	if len(policy.MatchConditions) != 1 {
		t.Errorf("got %d matchConditions, want 1", len(policy.MatchConditions))
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
	if len(policy.MatchConditions) != 2 {
		t.Fatalf("got %d matchConditions, want 2", len(policy.MatchConditions))
	}
	if policy.MatchConditions[0].Path != "webhooks[0].matchConditions[0]" {
		t.Errorf("matchCondition[0] path = %q, want %q", policy.MatchConditions[0].Path, "webhooks[0].matchConditions[0]")
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
	if len(policy.MatchConditions) != 1 {
		t.Fatalf("got %d matchConditions, want 1", len(policy.MatchConditions))
	}
	if policy.MatchConditions[0].Expression != "request.resource.resource == 'pods'" {
		t.Errorf("unexpected expression: %q", policy.MatchConditions[0].Expression)
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
