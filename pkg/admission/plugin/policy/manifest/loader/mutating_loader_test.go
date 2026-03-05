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

func TestLoadMutatingManifestsFromDirectory(t *testing.T) {
	tests := []struct {
		name             string
		files            map[string]string
		wantErr          bool
		errContains      string
		wantPolicyCount  int
		wantBindingCount int
	}{
		{
			name:  "empty directory",
			files: map[string]string{},
		},
		{
			name: "load policy and binding",
			files: map[string]string{
				"policy.yaml": `
apiVersion: admissionregistration.k8s.io/v1
kind: MutatingAdmissionPolicy
metadata:
  name: test-mutate.static.k8s.io
spec:
  reinvocationPolicy: Never
  failurePolicy: Fail
  matchConstraints:
    resourceRules:
    - apiGroups: [""]
      apiVersions: ["v1"]
      operations: ["CREATE"]
      resources: ["pods"]
  mutations:
  - patchType: ApplyConfiguration
    applyConfiguration:
      expression: "Object{spec: Object.spec{serviceAccountName: \"default\"}}"
---
apiVersion: admissionregistration.k8s.io/v1
kind: MutatingAdmissionPolicyBinding
metadata:
  name: test-mutate-binding.static.k8s.io
spec:
  policyName: test-mutate.static.k8s.io
`,
			},
			wantPolicyCount:  1,
			wantBindingCount: 1,
		},
		{
			name: "binding references non-existent policy",
			files: map[string]string{
				"binding.yaml": `
apiVersion: admissionregistration.k8s.io/v1
kind: MutatingAdmissionPolicyBinding
metadata:
  name: orphan-binding.static.k8s.io
spec:
  policyName: does-not-exist.static.k8s.io
`,
			},
			wantErr:     true,
			errContains: "does not exist",
		},
		{
			name: "policy without name",
			files: map[string]string{
				"policy.yaml": `
apiVersion: admissionregistration.k8s.io/v1
kind: MutatingAdmissionPolicy
metadata:
  name: ""
spec:
  reinvocationPolicy: Never
  matchConstraints:
    resourceRules:
    - apiGroups: [""]
      apiVersions: ["v1"]
      operations: ["CREATE"]
      resources: ["pods"]
  mutations:
  - patchType: ApplyConfiguration
    applyConfiguration:
      expression: "Object{}"
`,
			},
			wantErr:     true,
			errContains: "name",
		},
		{
			name: "policy name missing static suffix",
			files: map[string]string{
				"policy.yaml": `
apiVersion: admissionregistration.k8s.io/v1
kind: MutatingAdmissionPolicy
metadata:
  name: no-suffix
spec:
  reinvocationPolicy: Never
  matchConstraints:
    resourceRules:
    - apiGroups: [""]
      apiVersions: ["v1"]
      operations: ["CREATE"]
      resources: ["pods"]
  mutations:
  - patchType: ApplyConfiguration
    applyConfiguration:
      expression: "Object{}"
`,
			},
			wantErr:     true,
			errContains: "must have a name ending with",
		},
		{
			name: "duplicate policy names",
			files: map[string]string{
				"01-policy.yaml": `
apiVersion: admissionregistration.k8s.io/v1
kind: MutatingAdmissionPolicy
metadata:
  name: dup.static.k8s.io
spec:
  reinvocationPolicy: Never
  matchConstraints:
    resourceRules:
    - apiGroups: [""]
      apiVersions: ["v1"]
      operations: ["CREATE"]
      resources: ["pods"]
  mutations:
  - patchType: ApplyConfiguration
    applyConfiguration:
      expression: "Object{}"
`,
				"02-policy.yaml": `
apiVersion: admissionregistration.k8s.io/v1
kind: MutatingAdmissionPolicy
metadata:
  name: dup.static.k8s.io
spec:
  reinvocationPolicy: Never
  matchConstraints:
    resourceRules:
    - apiGroups: [""]
      apiVersions: ["v1"]
      operations: ["CREATE"]
      resources: ["pods"]
  mutations:
  - patchType: ApplyConfiguration
    applyConfiguration:
      expression: "Object{}"
`,
			},
			wantErr:     true,
			errContains: "duplicate",
		},
		{
			name: "policy with paramKind",
			files: map[string]string{
				"policy.yaml": `
apiVersion: admissionregistration.k8s.io/v1
kind: MutatingAdmissionPolicy
metadata:
  name: with-param.static.k8s.io
spec:
  reinvocationPolicy: Never
  paramKind:
    apiVersion: v1
    kind: ConfigMap
  matchConstraints:
    resourceRules:
    - apiGroups: [""]
      apiVersions: ["v1"]
      operations: ["CREATE"]
      resources: ["pods"]
  mutations:
  - patchType: ApplyConfiguration
    applyConfiguration:
      expression: "Object{}"
`,
			},
			wantErr:     true,
			errContains: "paramKind is not supported",
		},
		{
			name: "binding with paramRef",
			files: map[string]string{
				"policy.yaml": `
apiVersion: admissionregistration.k8s.io/v1
kind: MutatingAdmissionPolicy
metadata:
  name: ref-policy.static.k8s.io
spec:
  reinvocationPolicy: Never
  matchConstraints:
    resourceRules:
    - apiGroups: [""]
      apiVersions: ["v1"]
      operations: ["CREATE"]
      resources: ["pods"]
  mutations:
  - patchType: ApplyConfiguration
    applyConfiguration:
      expression: "Object{}"
---
apiVersion: admissionregistration.k8s.io/v1
kind: MutatingAdmissionPolicyBinding
metadata:
  name: with-param.static.k8s.io
spec:
  policyName: ref-policy.static.k8s.io
  paramRef:
    name: some-configmap
`,
			},
			wantErr:     true,
			errContains: "paramRef",
		},
		{
			name: "binding policyName missing static suffix",
			files: map[string]string{
				"binding.yaml": `
apiVersion: admissionregistration.k8s.io/v1
kind: MutatingAdmissionPolicyBinding
metadata:
  name: bad-ref.static.k8s.io
spec:
  policyName: some-policy-without-suffix
`,
			},
			wantErr:     true,
			errContains: "must end with",
		},
		{
			name: "unsupported type in directory",
			files: map[string]string{
				"wrong.yaml": `
apiVersion: admissionregistration.k8s.io/v1
kind: ValidatingAdmissionPolicy
metadata:
  name: wrong-type.static.k8s.io
spec:
  validations:
  - expression: "true"
`,
			},
			wantErr:     true,
			errContains: "unsupported resource type",
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
			result, err := LoadMutatingManifestsFromDirectory(dir)
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

func TestLoadMutatingManifestsFromDirectory_ListTypes(t *testing.T) {
	tests := []struct {
		name             string
		files            map[string]string
		wantErr          bool
		errContains      string
		wantPolicyCount  int
		wantBindingCount int
	}{
		{
			name: "MutatingAdmissionPolicyList",
			files: map[string]string{
				"list.yaml": `
apiVersion: admissionregistration.k8s.io/v1
kind: MutatingAdmissionPolicyList
items:
- metadata:
    name: list-policy-1.static.k8s.io
  spec:
    reinvocationPolicy: Never
    failurePolicy: Fail
    matchConstraints:
      resourceRules:
      - apiGroups: [""]
        apiVersions: ["v1"]
        operations: ["CREATE"]
        resources: ["pods"]
    mutations:
    - patchType: ApplyConfiguration
      applyConfiguration:
        expression: "Object{}"
- metadata:
    name: list-policy-2.static.k8s.io
  spec:
    reinvocationPolicy: Never
    failurePolicy: Fail
    matchConstraints:
      resourceRules:
      - apiGroups: [""]
        apiVersions: ["v1"]
        operations: ["CREATE"]
        resources: ["pods"]
    mutations:
    - patchType: ApplyConfiguration
      applyConfiguration:
        expression: "Object{}"
`,
			},
			wantPolicyCount: 2,
		},
		{
			name: "v1.List with policy and binding",
			files: map[string]string{
				"list.yaml": `
apiVersion: v1
kind: List
items:
- apiVersion: admissionregistration.k8s.io/v1
  kind: MutatingAdmissionPolicy
  metadata:
    name: v1list-policy.static.k8s.io
  spec:
    reinvocationPolicy: Never
    failurePolicy: Fail
    matchConstraints:
      resourceRules:
      - apiGroups: [""]
        apiVersions: ["v1"]
        operations: ["CREATE"]
        resources: ["pods"]
    mutations:
    - patchType: ApplyConfiguration
      applyConfiguration:
        expression: "Object{}"
- apiVersion: admissionregistration.k8s.io/v1
  kind: MutatingAdmissionPolicyBinding
  metadata:
    name: v1list-binding.static.k8s.io
  spec:
    policyName: v1list-policy.static.k8s.io
`,
			},
			wantPolicyCount:  1,
			wantBindingCount: 1,
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
			result, err := LoadMutatingManifestsFromDirectory(dir)
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
