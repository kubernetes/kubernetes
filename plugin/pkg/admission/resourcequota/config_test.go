/*
Copyright 2019 The Kubernetes Authors.

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

package resourcequota

import (
	"bytes"
	"reflect"
	"strings"
	"testing"

	corev1 "k8s.io/api/core/v1"
	resourcequotaapi "k8s.io/kubernetes/plugin/pkg/admission/resourcequota/apis/resourcequota"
)

func TestLoadConfiguration(t *testing.T) {
	testcases := []struct {
		name         string
		input        string
		expectErr    string
		expectConfig *resourcequotaapi.Configuration
	}{
		{
			name:      "empty",
			input:     ``,
			expectErr: `'Kind' is missing`,
		},
		{
			name:      "unknown type",
			input:     `{"kind":"Unknown","apiVersion":"v1"}`,
			expectErr: `no kind "Unknown" is registered`,
		},
		{
			name: "valid v1alpha1 config",
			input: `
kind: Configuration
apiVersion: resourcequota.admission.k8s.io/v1alpha1
limitedResources:
- apiGroup: ""
  resource: persistentvolumeclaims
  matchContains:
  - .storageclass.storage.k8s.io/requests.storage
- apiGroup: ""
  resource: pods
  matchScopes:
  - scopeName: PriorityClass
    operator: In
    values:
    - cluster-services
`,
			expectConfig: &resourcequotaapi.Configuration{
				LimitedResources: []resourcequotaapi.LimitedResource{
					{APIGroup: "", Resource: "persistentvolumeclaims", MatchContains: []string{".storageclass.storage.k8s.io/requests.storage"}},
					{APIGroup: "", Resource: "pods", MatchScopes: []corev1.ScopedResourceSelectorRequirement{
						{ScopeName: "PriorityClass", Operator: "In", Values: []string{"cluster-services"}},
					},
					},
				}},
		},
		{
			name: "valid v1beta1 config",
			input: `
kind: Configuration
apiVersion: resourcequota.admission.k8s.io/v1beta1
limitedResources:
- apiGroup: ""
  resource: persistentvolumeclaims
  matchContains:
  - .storageclass.storage.k8s.io/requests.storage
- apiGroup: ""
  resource: pods
  matchScopes:
  - scopeName: PriorityClass
    operator: In
    values:
    - cluster-services
`,
			expectConfig: &resourcequotaapi.Configuration{
				LimitedResources: []resourcequotaapi.LimitedResource{
					{APIGroup: "", Resource: "persistentvolumeclaims", MatchContains: []string{".storageclass.storage.k8s.io/requests.storage"}},
					{APIGroup: "", Resource: "pods", MatchScopes: []corev1.ScopedResourceSelectorRequirement{
						{ScopeName: "PriorityClass", Operator: "In", Values: []string{"cluster-services"}},
					},
					},
				}},
		},
		{
			name: "valid v1 config",
			input: `
kind: ResourceQuotaConfiguration
apiVersion: apiserver.config.k8s.io/v1
limitedResources:
- apiGroup: ""
  resource: persistentvolumeclaims
  matchContains:
  - .storageclass.storage.k8s.io/requests.storage
- apiGroup: ""
  resource: pods
  matchScopes:
  - scopeName: PriorityClass
    operator: In
    values:
    - cluster-services
`,
			expectConfig: &resourcequotaapi.Configuration{
				LimitedResources: []resourcequotaapi.LimitedResource{
					{APIGroup: "", Resource: "persistentvolumeclaims", MatchContains: []string{".storageclass.storage.k8s.io/requests.storage"}},
					{APIGroup: "", Resource: "pods", MatchScopes: []corev1.ScopedResourceSelectorRequirement{
						{ScopeName: "PriorityClass", Operator: "In", Values: []string{"cluster-services"}},
					},
					},
				}},
		},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			config, err := LoadConfiguration(bytes.NewBuffer([]byte(tc.input)))

			if len(tc.expectErr) > 0 {
				if err == nil {
					t.Fatal("expected err, got none")
				}
				if !strings.Contains(err.Error(), tc.expectErr) {
					t.Fatalf("expected err containing %q, got %v", tc.expectErr, err)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(config, tc.expectConfig) {
				t.Fatalf("expected\n%#v\ngot\n%#v", tc.expectConfig, config)
			}
		})
	}
}
