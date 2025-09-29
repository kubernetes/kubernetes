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

package v1beta1

import (
	"testing"

	"github.com/google/go-cmp/cmp"

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/utils/ptr"
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
					Scope:                 NamespaceScoped,
					Conversion:            &CustomResourceConversion{Strategy: NoneConverter},
					PreserveUnknownFields: ptr.To(true),
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
						WebhookClientConfig: &WebhookClientConfig{
							Service: &ServiceReference{},
						},
					},
					PreserveUnknownFields: ptr.To(true),
				},
			},
			expected: &CustomResourceDefinition{
				Spec: CustomResourceDefinitionSpec{
					Scope: NamespaceScoped,
					Conversion: &CustomResourceConversion{
						Strategy:                 WebhookConverter,
						ConversionReviewVersions: []string{"v1beta1"},
						WebhookClientConfig: &WebhookClientConfig{
							Service: &ServiceReference{Port: ptr.To[int32](443)},
						},
					},
					PreserveUnknownFields: ptr.To(true),
				},
			},
		},
		{
			name: "storage status defaults",
			original: &CustomResourceDefinition{
				Spec: CustomResourceDefinitionSpec{
					Scope:                 NamespaceScoped,
					Conversion:            &CustomResourceConversion{Strategy: NoneConverter},
					PreserveUnknownFields: ptr.To(true),
					Versions: []CustomResourceDefinitionVersion{
						{Name: "v1", Storage: false, Served: true},
						{Name: "v2", Storage: true, Served: true},
						{Name: "v3", Storage: false, Served: true},
					},
				},
			},
			expected: &CustomResourceDefinition{
				Spec: CustomResourceDefinitionSpec{
					Scope:                 NamespaceScoped,
					Conversion:            &CustomResourceConversion{Strategy: NoneConverter},
					PreserveUnknownFields: ptr.To(true),
					Version:               "v1",
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
					Scope:                 NamespaceScoped,
					Conversion:            &CustomResourceConversion{Strategy: NoneConverter},
					PreserveUnknownFields: ptr.To(true),
					Version:               "v1",
				},
			},
			expected: &CustomResourceDefinition{
				Spec: CustomResourceDefinitionSpec{
					Scope:                 NamespaceScoped,
					Conversion:            &CustomResourceConversion{Strategy: NoneConverter},
					PreserveUnknownFields: ptr.To(true),
					Version:               "v1",
					Versions: []CustomResourceDefinitionVersion{
						{Name: "v1", Storage: true, Served: true},
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
