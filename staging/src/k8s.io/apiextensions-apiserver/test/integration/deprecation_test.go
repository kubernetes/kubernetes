/*
Copyright 2020 The Kubernetes Authors.

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

package integration

import (
	"context"
	"reflect"
	"testing"

	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	"k8s.io/apiextensions-apiserver/test/integration/fixtures"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"
	"k8s.io/utils/pointer"
)

var deprecationFixture = &apiextensionsv1.CustomResourceDefinition{
	ObjectMeta: metav1.ObjectMeta{Name: "deps.tests.example.com"},
	Spec: apiextensionsv1.CustomResourceDefinitionSpec{
		Group: "tests.example.com",
		Versions: []apiextensionsv1.CustomResourceDefinitionVersion{
			{
				Name:       "v1alpha1",
				Served:     true,
				Deprecated: true,
				Schema:     &apiextensionsv1.CustomResourceValidation{OpenAPIV3Schema: &apiextensionsv1.JSONSchemaProps{Type: "object"}},
			},
			{
				Name:               "v1alpha2",
				Served:             true,
				Deprecated:         true,
				DeprecationWarning: pointer.StringPtr("custom deprecation warning"),
				Schema:             &apiextensionsv1.CustomResourceValidation{OpenAPIV3Schema: &apiextensionsv1.JSONSchemaProps{Type: "object"}},
			},
			{
				Name:   "v1beta3",
				Served: true,
				Schema: &apiextensionsv1.CustomResourceValidation{OpenAPIV3Schema: &apiextensionsv1.JSONSchemaProps{Type: "object"}},
			},
			{
				Name:   "v1beta2",
				Served: true,
				Schema: &apiextensionsv1.CustomResourceValidation{OpenAPIV3Schema: &apiextensionsv1.JSONSchemaProps{Type: "object"}},
			},
			{
				Name:    "v1",
				Served:  false,
				Storage: true,
				Schema:  &apiextensionsv1.CustomResourceValidation{OpenAPIV3Schema: &apiextensionsv1.JSONSchemaProps{Type: "object"}},
			},
		},
		Names: apiextensionsv1.CustomResourceDefinitionNames{
			Plural:   "deps",
			Singular: "dep",
			Kind:     "Dep",
			ListKind: "DepList",
		},
		Scope: apiextensionsv1.ClusterScoped,
	},
}

func TestCustomResourceDeprecation(t *testing.T) {
	tearDown, config, _, err := fixtures.StartDefaultServer(t)
	if err != nil {
		t.Fatal(err)
	}
	defer tearDown()

	apiExtensionsClient, err := clientset.NewForConfig(config)
	if err != nil {
		t.Fatal(err)
	}

	handler := &warningHandler{}
	config.WarningHandler = handler
	dynamicClient, err := dynamic.NewForConfig(config)
	if err != nil {
		t.Fatal(err)
	}

	crd := deprecationFixture.DeepCopy()
	if _, err := fixtures.CreateNewV1CustomResourceDefinition(crd, apiExtensionsClient, dynamicClient); err != nil {
		t.Fatal(err)
	}

	testcases := []struct {
		name    string
		version string
		want    []string
	}{
		{
			name:    "default",
			version: "v1alpha1",
			want:    []string{"tests.example.com/v1alpha1 Dep is deprecated; use tests.example.com/v1beta3 Dep"},
		},
		{
			name:    "custom",
			version: "v1alpha2",
			want:    []string{"custom deprecation warning"},
		},
		{
			name:    "older non-deprecated",
			version: "v1beta2",
			want:    nil,
		},
		{
			name:    "newest non-deprecated",
			version: "v1beta3",
			want:    nil,
		},
	}
	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			handler.warnings = nil
			resource := schema.GroupVersionResource{Group: "tests.example.com", Version: tc.version, Resource: "deps"}
			_, err := dynamicClient.Resource(resource).List(context.TODO(), metav1.ListOptions{})
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(handler.warnings, tc.want) {
				t.Errorf("expected %v, got %v", tc.want, handler.warnings)
			}
		})
	}
}

type warningHandler struct {
	warnings []string
}

func (w *warningHandler) HandleWarningHeader(code int, agent string, text string) {
	w.warnings = append(w.warnings, text)
}
