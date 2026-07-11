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

	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
)

var (
	testScheme = runtime.NewScheme()
	testCodecs serializer.CodecFactory
)

func init() {
	utilruntime.Must(admissionregistrationv1.AddToScheme(testScheme))
	testScheme.AddUnversionedTypes(metav1.SchemeGroupVersion, &metav1.List{}, &metav1.Status{})
	testCodecs = serializer.NewCodecFactory(testScheme, serializer.EnableStrict)
}

func acceptTestPolicy(obj runtime.Object) ([]*admissionregistrationv1.ValidatingAdmissionPolicy, error) {
	switch p := obj.(type) {
	case *admissionregistrationv1.ValidatingAdmissionPolicy:
		return []*admissionregistrationv1.ValidatingAdmissionPolicy{p}, nil
	case *admissionregistrationv1.ValidatingAdmissionPolicyList:
		items := make([]*admissionregistrationv1.ValidatingAdmissionPolicy, len(p.Items))
		for i := range p.Items {
			items[i] = &p.Items[i]
		}
		return items, nil
	default:
		return nil, ErrUnrecognizedType
	}
}

func acceptTestBinding(obj runtime.Object) ([]*admissionregistrationv1.ValidatingAdmissionPolicyBinding, error) {
	switch b := obj.(type) {
	case *admissionregistrationv1.ValidatingAdmissionPolicyBinding:
		return []*admissionregistrationv1.ValidatingAdmissionPolicyBinding{b}, nil
	case *admissionregistrationv1.ValidatingAdmissionPolicyBindingList:
		items := make([]*admissionregistrationv1.ValidatingAdmissionPolicyBinding, len(b.Items))
		for i := range b.Items {
			items[i] = &b.Items[i]
		}
		return items, nil
	default:
		return nil, ErrUnrecognizedType
	}
}

func TestLoadPolicyManifests(t *testing.T) {
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
  - expression: "true"
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
		},
		{
			name:         "empty directory",
			files:        map[string]string{},
			wantPolicies: 0,
			wantBindings: 0,
		},
		{
			name: "binding references non-existent policy",
			files: map[string]string{
				"binding.yaml": `
apiVersion: admissionregistration.k8s.io/v1
kind: ValidatingAdmissionPolicyBinding
metadata:
  name: orphan.static.k8s.io
spec:
  policyName: does-not-exist.static.k8s.io
  validationActions:
  - Deny
`,
			},
			wantErr:        true,
			wantErrContain: "does not exist",
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
			name: "unsupported resource type",
			files: map[string]string{
				"wrong.yaml": `
apiVersion: v1
kind: ConfigMap
metadata:
  name: not-a-policy
`,
			},
			wantErr:        true,
			wantErrContain: "error loading",
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
			wantPolicies: 1,
			wantBindings: 1,
		},
		{
			name: "policy with paramKind rejected",
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
			wantErrContain: "paramKind is not supported",
		},
		{
			name: "binding with paramRef rejected",
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
			wantErr:        true,
			wantErrContain: "must end with",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			for name, content := range tt.files {
				if err := os.WriteFile(filepath.Join(dir, name), []byte(content), 0644); err != nil {
					t.Fatalf("failed to write file %s: %v", name, err)
				}
			}

			policies, bindings, _, err := LoadPolicyManifests(
				dir,
				testCodecs.UniversalDeserializer(),
				acceptTestPolicy,
				acceptTestBinding,
				func(b *admissionregistrationv1.ValidatingAdmissionPolicyBinding) string {
					return b.Spec.PolicyName
				},
			)

			if tt.wantErr {
				if err == nil {
					t.Error("expected error but got none")
				} else if tt.wantErrContain != "" && !strings.Contains(err.Error(), tt.wantErrContain) {
					t.Errorf("expected error containing %q, got %q", tt.wantErrContain, err.Error())
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if len(policies) != tt.wantPolicies {
				t.Errorf("expected %d policies, got %d", tt.wantPolicies, len(policies))
			}
			if len(bindings) != tt.wantBindings {
				t.Errorf("expected %d bindings, got %d", tt.wantBindings, len(bindings))
			}
		})
	}
}

func TestValidateBindingReferences(t *testing.T) {
	policies := []*admissionregistrationv1.ValidatingAdmissionPolicy{
		{ObjectMeta: metav1.ObjectMeta{Name: "policy-a"}},
		{ObjectMeta: metav1.ObjectMeta{Name: "policy-b"}},
	}
	goodBindings := []*admissionregistrationv1.ValidatingAdmissionPolicyBinding{
		{ObjectMeta: metav1.ObjectMeta{Name: "b1"}, Spec: admissionregistrationv1.ValidatingAdmissionPolicyBindingSpec{PolicyName: "policy-a"}},
		{ObjectMeta: metav1.ObjectMeta{Name: "b2"}, Spec: admissionregistrationv1.ValidatingAdmissionPolicyBindingSpec{PolicyName: "policy-b"}},
	}
	badBindings := []*admissionregistrationv1.ValidatingAdmissionPolicyBinding{
		{ObjectMeta: metav1.ObjectMeta{Name: "b1"}, Spec: admissionregistrationv1.ValidatingAdmissionPolicyBindingSpec{PolicyName: "policy-missing"}},
	}

	getBindPolicy := func(b *admissionregistrationv1.ValidatingAdmissionPolicyBinding) string {
		return b.Spec.PolicyName
	}

	if err := validateBindingReferences(policies, goodBindings, getBindPolicy); err != nil {
		t.Errorf("unexpected error for valid bindings: %v", err)
	}

	if err := validateBindingReferences(policies, badBindings, getBindPolicy); err == nil {
		t.Error("expected error for binding referencing non-existent policy")
	}
}
