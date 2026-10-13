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

package loader

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestLoadValidatingManifestsFromDirectory(t *testing.T) {
	tests := []struct {
		name           string
		files          map[string]string
		wantPolicies   int
		wantBindings   int
		wantErr        bool
		wantErrContain string
	}{
		{
			name: "load policy and binding",
			files: map[string]string{
				"policy.yaml": `
apiVersion: admissionregistration.k8s.io/v1
kind: ValidatingAdmissionPolicy
metadata:
  name: test-policy.static.k8s.io
spec:
  matchConstraints:
    resourceRules:
    - apiGroups: [""]
      apiVersions: ["v1"]
      operations: ["CREATE"]
      resources: ["pods"]
  validations:
  - expression: "object.spec.containers.all(c, c.image.startsWith('allowed/'))"
`,
				"binding.yaml": `
apiVersion: admissionregistration.k8s.io/v1
kind: ValidatingAdmissionPolicyBinding
metadata:
  name: test-binding.static.k8s.io
spec:
  policyName: test-policy.static.k8s.io
  validationActions:
  - Deny
`,
			},
			wantPolicies: 1,
			wantBindings: 1,
			wantErr:      false,
		},
		{
			name: "binding references non-existent policy",
			files: map[string]string{
				"binding.yaml": `
apiVersion: admissionregistration.k8s.io/v1
kind: ValidatingAdmissionPolicyBinding
metadata:
  name: test-binding.static.k8s.io
spec:
  policyName: non-existent-policy.static.k8s.io
  validationActions:
  - Deny
`,
			},
			wantErr:        true,
			wantErrContain: "does not exist in the manifest directory",
		},
		{
			name:         "empty directory",
			files:        map[string]string{},
			wantPolicies: 0,
			wantBindings: 0,
			wantErr:      false,
		},
		{
			name: "policy without name",
			files: map[string]string{
				"policy.yaml": `
apiVersion: admissionregistration.k8s.io/v1
kind: ValidatingAdmissionPolicy
spec:
  matchConstraints:
    resourceRules:
    - apiGroups: [""]
      apiVersions: ["v1"]
      operations: ["CREATE"]
      resources: ["pods"]
`,
			},
			wantErr:        true,
			wantErrContain: "name",
		},
		{
			name: "multi-document yaml with policy and binding",
			files: map[string]string{
				"multi-doc.yaml": `
apiVersion: admissionregistration.k8s.io/v1
kind: ValidatingAdmissionPolicy
metadata:
  name: multidoc-policy.static.k8s.io
spec:
  matchConstraints:
    resourceRules:
    - apiGroups: [""]
      apiVersions: ["v1"]
      operations: ["CREATE"]
      resources: ["pods"]
  validations:
  - expression: "true"
---
apiVersion: admissionregistration.k8s.io/v1
kind: ValidatingAdmissionPolicyBinding
metadata:
  name: multidoc-binding.static.k8s.io
spec:
  policyName: multidoc-policy.static.k8s.io
  validationActions:
  - Deny
`,
			},
			wantPolicies: 1,
			wantBindings: 1,
			wantErr:      false,
		},
		{
			name: "policy name missing static suffix",
			files: map[string]string{
				"policy.yaml": `
apiVersion: admissionregistration.k8s.io/v1
kind: ValidatingAdmissionPolicy
metadata:
  name: no-suffix
spec:
  matchConstraints:
    resourceRules:
    - apiGroups: [""]
      apiVersions: ["v1"]
      operations: ["CREATE"]
      resources: ["pods"]
  validations:
  - expression: "true"
`,
			},
			wantErr:        true,
			wantErrContain: "must have a name ending with",
		},
		{
			name: "duplicate policy names",
			files: map[string]string{
				"01-policy.yaml": `
apiVersion: admissionregistration.k8s.io/v1
kind: ValidatingAdmissionPolicy
metadata:
  name: dup.static.k8s.io
spec:
  matchConstraints:
    resourceRules:
    - apiGroups: [""]
      apiVersions: ["v1"]
      operations: ["CREATE"]
      resources: ["pods"]
  validations:
  - expression: "true"
`,
				"02-policy.yaml": `
apiVersion: admissionregistration.k8s.io/v1
kind: ValidatingAdmissionPolicy
metadata:
  name: dup.static.k8s.io
spec:
  matchConstraints:
    resourceRules:
    - apiGroups: [""]
      apiVersions: ["v1"]
      operations: ["CREATE"]
      resources: ["pods"]
  validations:
  - expression: "true"
`,
			},
			wantErr:        true,
			wantErrContain: "duplicate",
		},
		{
			name: "policy with paramKind",
			files: map[string]string{
				"policy.yaml": `
apiVersion: admissionregistration.k8s.io/v1
kind: ValidatingAdmissionPolicy
metadata:
  name: param-policy.static.k8s.io
spec:
  paramKind:
    apiVersion: v1
    kind: ConfigMap
  matchConstraints:
    resourceRules:
    - apiGroups: [""]
      apiVersions: ["v1"]
      operations: ["CREATE"]
      resources: ["pods"]
  validations:
  - expression: "true"
`,
			},
			wantErr:        true,
			wantErrContain: "spec.paramKind is not supported",
		},
		{
			name: "binding with paramRef",
			files: map[string]string{
				"policy.yaml": `
apiVersion: admissionregistration.k8s.io/v1
kind: ValidatingAdmissionPolicy
metadata:
  name: ref-policy.static.k8s.io
spec:
  matchConstraints:
    resourceRules:
    - apiGroups: [""]
      apiVersions: ["v1"]
      operations: ["CREATE"]
      resources: ["pods"]
  validations:
  - expression: "true"
`,
				"binding.yaml": `
apiVersion: admissionregistration.k8s.io/v1
kind: ValidatingAdmissionPolicyBinding
metadata:
  name: ref-binding.static.k8s.io
spec:
  policyName: ref-policy.static.k8s.io
  paramRef:
    name: my-config
    namespace: default
  validationActions:
  - Deny
`,
			},
			wantErr:        true,
			wantErrContain: "paramRef",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create temp directory
			dir := t.TempDir()

			// Create files
			for name, content := range tt.files {
				path := filepath.Join(dir, name)
				if err := os.WriteFile(path, []byte(content), 0644); err != nil {
					t.Fatalf("failed to write file %s: %v", name, err)
				}
			}

			// Load manifests
			result, err := LoadValidatingManifestsFromDirectory(dir)

			if tt.wantErr {
				if err == nil {
					t.Error("expected error but got none")
				} else if tt.wantErrContain != "" {
					if !strings.Contains(err.Error(), tt.wantErrContain) {
						t.Errorf("expected error containing %q, got %q", tt.wantErrContain, err.Error())
					}
				}
				return
			}

			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if len(result.Policies) != tt.wantPolicies {
				t.Errorf("expected %d policies, got %d", tt.wantPolicies, len(result.Policies))
			}

			if len(result.Bindings) != tt.wantBindings {
				t.Errorf("expected %d bindings, got %d", tt.wantBindings, len(result.Bindings))
			}
		})
	}
}

func TestLoadValidatingManifestsFromDirectory_ListTypes(t *testing.T) {
	tests := []struct {
		name             string
		files            map[string]string
		wantErr          bool
		errContains      string
		wantPolicyCount  int
		wantBindingCount int
	}{
		{
			name: "ValidatingAdmissionPolicyList",
			files: map[string]string{
				"list.yaml": `
apiVersion: admissionregistration.k8s.io/v1
kind: ValidatingAdmissionPolicyList
items:
- metadata:
    name: list-policy-1.static.k8s.io
  spec:
    failurePolicy: Fail
    matchConstraints:
      resourceRules:
      - apiGroups: [""]
        apiVersions: ["v1"]
        operations: ["CREATE"]
        resources: ["pods"]
    validations:
    - expression: "true"
- metadata:
    name: list-policy-2.static.k8s.io
  spec:
    failurePolicy: Fail
    matchConstraints:
      resourceRules:
      - apiGroups: [""]
        apiVersions: ["v1"]
        operations: ["CREATE"]
        resources: ["pods"]
    validations:
    - expression: "true"
`,
			},
			wantPolicyCount: 2,
		},
		{
			name: "ValidatingAdmissionPolicyBindingList",
			files: map[string]string{
				"list.yaml": `
apiVersion: admissionregistration.k8s.io/v1
kind: ValidatingAdmissionPolicyList
items:
- metadata:
    name: ref-policy.static.k8s.io
  spec:
    failurePolicy: Fail
    matchConstraints:
      resourceRules:
      - apiGroups: [""]
        apiVersions: ["v1"]
        operations: ["CREATE"]
        resources: ["pods"]
    validations:
    - expression: "true"
---
apiVersion: admissionregistration.k8s.io/v1
kind: ValidatingAdmissionPolicyBindingList
items:
- metadata:
    name: list-binding.static.k8s.io
  spec:
    policyName: ref-policy.static.k8s.io
    validationActions:
    - Deny
`,
			},
			wantPolicyCount:  1,
			wantBindingCount: 1,
		},
		{
			name: "v1.List with policy and binding",
			files: map[string]string{
				"list.yaml": `
apiVersion: v1
kind: List
items:
- apiVersion: admissionregistration.k8s.io/v1
  kind: ValidatingAdmissionPolicy
  metadata:
    name: v1list-policy.static.k8s.io
  spec:
    failurePolicy: Fail
    matchConstraints:
      resourceRules:
      - apiGroups: [""]
        apiVersions: ["v1"]
        operations: ["CREATE"]
        resources: ["pods"]
    validations:
    - expression: "true"
- apiVersion: admissionregistration.k8s.io/v1
  kind: ValidatingAdmissionPolicyBinding
  metadata:
    name: v1list-binding.static.k8s.io
  spec:
    policyName: v1list-policy.static.k8s.io
    validationActions:
    - Deny
`,
			},
			wantPolicyCount:  1,
			wantBindingCount: 1,
		},
		{
			name: "v1.List with unsupported type",
			files: map[string]string{
				"list.yaml": `
apiVersion: v1
kind: List
items:
- apiVersion: admissionregistration.k8s.io/v1
  kind: ValidatingAdmissionPolicy
  metadata:
    name: valid.static.k8s.io
  spec:
    failurePolicy: Fail
    matchConstraints:
      resourceRules:
      - apiGroups: [""]
        apiVersions: ["v1"]
        operations: ["CREATE"]
        resources: ["pods"]
    validations:
    - expression: "true"
- apiVersion: v1
  kind: ConfigMap
  metadata:
    name: not-a-policy
`,
			},
			wantErr:     true,
			errContains: "failed to decode list item",
		},
		{
			name: "binding policyName missing static suffix",
			files: map[string]string{
				"binding.yaml": `
apiVersion: admissionregistration.k8s.io/v1
kind: ValidatingAdmissionPolicyBinding
metadata:
  name: bad-ref.static.k8s.io
spec:
  policyName: some-policy-without-suffix
  validationActions:
  - Deny
`,
			},
			wantErr:     true,
			errContains: "must end with",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			for name, content := range tt.files {
				if err := os.WriteFile(filepath.Join(dir, name), []byte(content), 0644); err != nil {
					t.Fatalf("Failed to write file: %v", err)
				}
			}
			result, err := LoadValidatingManifestsFromDirectory(dir)
			if tt.wantErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				if tt.errContains != "" && !strings.Contains(err.Error(), tt.errContains) {
					t.Errorf("expected error containing %q, got: %v", tt.errContains, err)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got := len(result.Policies); got != tt.wantPolicyCount {
				t.Errorf("Policies count = %d, want %d", got, tt.wantPolicyCount)
			}
			if got := len(result.Bindings); got != tt.wantBindingCount {
				t.Errorf("Bindings count = %d, want %d", got, tt.wantBindingCount)
			}
		})
	}
}
