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
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	admissionv1 "k8s.io/api/admission/v1"
	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
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
	object := input.Object.(map[string]interface{})
	if object["kind"] != "Pod" {
		t.Errorf("object.kind = %v, want Pod", object["kind"])
	}
	params := input.Params.(map[string]interface{})
	if params["kind"] != "ConfigMap" {
		t.Errorf("params.kind = %v, want ConfigMap", params["kind"])
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

func TestNewFromTypedAdmissionPoliciesMatchesYAMLParsers(t *testing.T) {
	tests := []struct {
		name       string
		yaml       string
		fromTyped  func() (*AdmissionPolicy, error)
		checkParam bool
		wantParams bool
	}{
		{
			name: "ValidatingAdmissionPolicy",
			yaml: `
apiVersion: admissionregistration.k8s.io/v1
kind: ValidatingAdmissionPolicy
spec:
  paramKind:
    apiVersion: v1
    kind: ConfigMap
  variables:
    - name: podName
      expression: "object.metadata.name"
  matchConditions:
    - name: only-pods
      expression: "request.resource.resource == 'pods'"
  validations:
    - expression: "variables.podName != 'bad'"
      message: "name must not be bad"
      messageExpression: "'bad name: ' + variables.podName"
  auditAnnotations:
    - key: pod-name
      valueExpression: "variables.podName"
`,
			fromTyped: func() (*AdmissionPolicy, error) {
				return NewFromValidatingAdmissionPolicy(&admissionregistrationv1.ValidatingAdmissionPolicy{
					Spec: admissionregistrationv1.ValidatingAdmissionPolicySpec{
						ParamKind: &admissionregistrationv1.ParamKind{APIVersion: "v1", Kind: "ConfigMap"},
						Variables: []admissionregistrationv1.Variable{{
							Name:       "podName",
							Expression: "object.metadata.name",
						}},
						MatchConditions: []admissionregistrationv1.MatchCondition{{
							Name:       "only-pods",
							Expression: "request.resource.resource == 'pods'",
						}},
						Validations: []admissionregistrationv1.Validation{{
							Expression:        "variables.podName != 'bad'",
							Message:           "name must not be bad",
							MessageExpression: "'bad name: ' + variables.podName",
						}},
						AuditAnnotations: []admissionregistrationv1.AuditAnnotation{{
							Key:             "pod-name",
							ValueExpression: "variables.podName",
						}},
					},
				})
			},
			checkParam: true,
			wantParams: true,
		},
		{
			name: "MutatingAdmissionPolicy",
			yaml: `
apiVersion: admissionregistration.k8s.io/v1
kind: MutatingAdmissionPolicy
spec:
  paramKind:
    apiVersion: v1
    kind: ConfigMap
  variables:
    - name: replicas
      expression: "object.spec.replicas"
  matchConditions:
    - name: only-deployments
      expression: "request.resource.resource == 'deployments'"
  mutations:
    - patchType: ApplyConfiguration
      applyConfiguration:
        expression: "Object{spec: Object.spec{replicas: int(params.data.replicas)}}"
    - patchType: JSONPatch
      jsonPatch:
        expression: "[JSONPatch{op: 'replace', path: '/spec/replicas', value: 3}]"
`,
			fromTyped: func() (*AdmissionPolicy, error) {
				return NewFromMutatingAdmissionPolicy(&admissionregistrationv1.MutatingAdmissionPolicy{
					Spec: admissionregistrationv1.MutatingAdmissionPolicySpec{
						ParamKind: &admissionregistrationv1.ParamKind{APIVersion: "v1", Kind: "ConfigMap"},
						Variables: []admissionregistrationv1.Variable{{
							Name:       "replicas",
							Expression: "object.spec.replicas",
						}},
						MatchConditions: []admissionregistrationv1.MatchCondition{{
							Name:       "only-deployments",
							Expression: "request.resource.resource == 'deployments'",
						}},
						Mutations: []admissionregistrationv1.Mutation{
							{
								PatchType:          admissionregistrationv1.PatchTypeApplyConfiguration,
								ApplyConfiguration: &admissionregistrationv1.ApplyConfiguration{Expression: "Object{spec: Object.spec{replicas: int(params.data.replicas)}}"},
							},
							{
								PatchType: admissionregistrationv1.PatchTypeJSONPatch,
								JSONPatch: &admissionregistrationv1.JSONPatch{Expression: "[JSONPatch{op: 'replace', path: '/spec/replicas', value: 3}]"},
							},
						},
					},
				})
			},
			checkParam: true,
			wantParams: true,
		},
		{
			name: "ValidatingWebhookConfiguration",
			yaml: `
apiVersion: admissionregistration.k8s.io/v1
kind: ValidatingWebhookConfiguration
webhooks:
  - name: validate.example.com
    matchConditions:
      - name: exclude-leases
        expression: "request.resource.resource != 'leases'"
`,
			fromTyped: func() (*AdmissionPolicy, error) {
				return NewFromValidatingWebhookConfiguration(&admissionregistrationv1.ValidatingWebhookConfiguration{
					Webhooks: []admissionregistrationv1.ValidatingWebhook{{
						Name: "validate.example.com",
						MatchConditions: []admissionregistrationv1.MatchCondition{{
							Name:       "exclude-leases",
							Expression: "request.resource.resource != 'leases'",
						}},
					}},
				})
			},
			checkParam: true,
			wantParams: false,
		},
		{
			name: "MutatingWebhookConfiguration",
			yaml: `
apiVersion: admissionregistration.k8s.io/v1
kind: MutatingWebhookConfiguration
webhooks:
  - name: mutate.example.com
    matchConditions:
      - name: only-pods
        expression: "request.resource.resource == 'pods'"
`,
			fromTyped: func() (*AdmissionPolicy, error) {
				return NewFromMutatingWebhookConfiguration(&admissionregistrationv1.MutatingWebhookConfiguration{
					Webhooks: []admissionregistrationv1.MutatingWebhook{{
						Name: "mutate.example.com",
						MatchConditions: []admissionregistrationv1.MatchCondition{{
							Name:       "only-pods",
							Expression: "request.resource.resource == 'pods'",
						}},
					}},
				})
			},
			checkParam: true,
			wantParams: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fromYAML, err := ParseAdmissionPolicy(tt.yaml)
			if err != nil {
				t.Fatalf("ParseAdmissionPolicy() error: %v", err)
			}
			fromTyped, err := tt.fromTyped()
			if err != nil {
				t.Fatalf("typed constructor error: %v", err)
			}
			if !reflect.DeepEqual(fromTyped, fromYAML) {
				t.Fatalf("typed constructor policy differs from YAML parser:\ntyped: %#v\nYAML:  %#v", fromTyped, fromYAML)
			}
			if tt.checkParam && fromTyped.hasParams != tt.wantParams {
				t.Fatalf("hasParams = %v, want %v", fromTyped.hasParams, tt.wantParams)
			}
		})
	}
}

func TestNewFromTypedAdmissionPolicyErrors(t *testing.T) {
	tests := []struct {
		name    string
		create  func() (*AdmissionPolicy, error)
		wantErr string
	}{
		{
			name:    "nil ValidatingAdmissionPolicy",
			create:  func() (*AdmissionPolicy, error) { return NewFromValidatingAdmissionPolicy(nil) },
			wantErr: "ValidatingAdmissionPolicy is nil",
		},
		{
			name: "empty ValidatingAdmissionPolicy",
			create: func() (*AdmissionPolicy, error) {
				return NewFromValidatingAdmissionPolicy(&admissionregistrationv1.ValidatingAdmissionPolicy{})
			},
			wantErr: "ValidatingAdmissionPolicy does not contain CEL expressions",
		},
		{
			name:    "nil MutatingAdmissionPolicy",
			create:  func() (*AdmissionPolicy, error) { return NewFromMutatingAdmissionPolicy(nil) },
			wantErr: "MutatingAdmissionPolicy is nil",
		},
		{
			name: "empty MutatingAdmissionPolicy",
			create: func() (*AdmissionPolicy, error) {
				return NewFromMutatingAdmissionPolicy(&admissionregistrationv1.MutatingAdmissionPolicy{})
			},
			wantErr: "MutatingAdmissionPolicy does not contain CEL expressions",
		},
		{
			name:    "nil ValidatingWebhookConfiguration",
			create:  func() (*AdmissionPolicy, error) { return NewFromValidatingWebhookConfiguration(nil) },
			wantErr: "ValidatingWebhookConfiguration is nil",
		},
		{
			name: "empty ValidatingWebhookConfiguration",
			create: func() (*AdmissionPolicy, error) {
				return NewFromValidatingWebhookConfiguration(&admissionregistrationv1.ValidatingWebhookConfiguration{})
			},
			wantErr: "ValidatingWebhookConfiguration does not contain CEL expressions",
		},
		{
			name:    "nil MutatingWebhookConfiguration",
			create:  func() (*AdmissionPolicy, error) { return NewFromMutatingWebhookConfiguration(nil) },
			wantErr: "MutatingWebhookConfiguration is nil",
		},
		{
			name: "empty MutatingWebhookConfiguration",
			create: func() (*AdmissionPolicy, error) {
				return NewFromMutatingWebhookConfiguration(&admissionregistrationv1.MutatingWebhookConfiguration{})
			},
			wantErr: "MutatingWebhookConfiguration does not contain CEL expressions",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := tt.create()
			if err == nil {
				t.Fatal("expected error")
			}
			if !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("error = %q, want to contain %q", err.Error(), tt.wantErr)
			}
		})
	}
}
