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

package v1

import (
	"testing"

	"github.com/google/go-cmp/cmp"

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/runtime"
	utilpointer "k8s.io/utils/pointer"
)

func TestDefaults(t *testing.T) {
	scheme := runtime.NewScheme()
	AddToScheme(scheme)
	tests := []struct {
		name     string
		original *CustomResourceDefinition
		expected *CustomResourceDefinition
	}{
		{
			name:     "empty",
			original: &CustomResourceDefinition{},
			expected: &CustomResourceDefinition{
				Spec: CustomResourceDefinitionSpec{
					Conversion: &CustomResourceConversion{Strategy: NoneConverter},
				},
			},
		},
		{
			name: "conversion defaults",
			original: &CustomResourceDefinition{
				Spec: CustomResourceDefinitionSpec{
					Scope: NamespaceScoped,
					Conversion: &CustomResourceConversion{
						Strategy: WebhookConverter,
						Webhook: &WebhookConversion{
							ClientConfig: &WebhookClientConfig{
								Service: &ServiceReference{},
							},
						},
					},
				},
			},
			expected: &CustomResourceDefinition{
				Spec: CustomResourceDefinitionSpec{
					Scope: NamespaceScoped,
					Conversion: &CustomResourceConversion{
						Strategy: WebhookConverter,
						Webhook: &WebhookConversion{
							ClientConfig: &WebhookClientConfig{
								Service: &ServiceReference{Port: utilpointer.Int32Ptr(443)},
							},
						},
					},
				},
			},
		},
		{
			name: "storage status defaults",
			original: &CustomResourceDefinition{
				Spec: CustomResourceDefinitionSpec{
					Scope:      NamespaceScoped,
					Conversion: &CustomResourceConversion{Strategy: NoneConverter},
					Versions: []CustomResourceDefinitionVersion{
						{Name: "v1", Storage: false, Served: true},
						{Name: "v2", Storage: true, Served: true},
						{Name: "v3", Storage: false, Served: true},
					},
				},
			},
			expected: &CustomResourceDefinition{
				Spec: CustomResourceDefinitionSpec{
					Scope:      NamespaceScoped,
					Conversion: &CustomResourceConversion{Strategy: NoneConverter},
					Versions: []CustomResourceDefinitionVersion{
						{Name: "v1", Storage: false, Served: true},
						{Name: "v2", Storage: true, Served: true},
						{Name: "v3", Storage: false, Served: true},
					},
				},
				Status: CustomResourceDefinitionStatus{
					StoredVersions: []string{"v2"},
				},
			},
		},
		{
			name: "version defaults",
			original: &CustomResourceDefinition{
				Spec: CustomResourceDefinitionSpec{
					Scope:      NamespaceScoped,
					Conversion: &CustomResourceConversion{Strategy: NoneConverter},
					Versions: []CustomResourceDefinitionVersion{
						{Name: "v1", Storage: true},
					},
				},
			},
			expected: &CustomResourceDefinition{
				Spec: CustomResourceDefinitionSpec{
					Scope:      NamespaceScoped,
					Conversion: &CustomResourceConversion{Strategy: NoneConverter},
					Versions: []CustomResourceDefinitionVersion{
						{Name: "v1", Storage: true},
					},
				},
				Status: CustomResourceDefinitionStatus{
					StoredVersions: []string{"v1"},
				},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			original := test.original
			expected := test.expected
			scheme.Default(original)
			if !apiequality.Semantic.DeepEqual(original, expected) {
				t.Errorf("expected vs got:\n%s", cmp.Diff(test.expected, original))
			}
		})
	}
}
