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
	"reflect"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/utils/pointer"
)

func TestConversion(t *testing.T) {
	testcases := []struct {
		Name      string
		In        runtime.Object
		Out       runtime.Object
		ExpectOut runtime.Object
		ExpectErr string
	}{
		// Versions
		{
			Name:      "internal to v1, no versions",
			In:        &apiextensions.CustomResourceDefinition{},
			Out:       &CustomResourceDefinition{},
			ExpectOut: &CustomResourceDefinition{},
		},
		{
			Name: "internal to v1, top-level version",
			In: &apiextensions.CustomResourceDefinition{
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Version: "v1",
				},
			},
			Out: &CustomResourceDefinition{},
			ExpectOut: &CustomResourceDefinition{
				Spec: CustomResourceDefinitionSpec{
					Versions: []CustomResourceDefinitionVersion{{Name: "v1", Served: true, Storage: true}},
				},
			},
		},
		{
			Name: "internal to v1, multiple versions",
			In: &apiextensions.CustomResourceDefinition{
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{Name: "v1", Served: true, Storage: true},
						{Name: "v2", Served: false, Storage: false},
					},
				},
			},
			Out: &CustomResourceDefinition{},
			ExpectOut: &CustomResourceDefinition{
				Spec: CustomResourceDefinitionSpec{
					Versions: []CustomResourceDefinitionVersion{
						{Name: "v1", Served: true, Storage: true},
						{Name: "v2", Served: false, Storage: false},
					},
				},
			},
		},
		{
			Name: "v1 to internal, no versions",
			In:   &CustomResourceDefinition{},
			Out:  &apiextensions.CustomResourceDefinition{},
			ExpectOut: &apiextensions.CustomResourceDefinition{
				Spec: apiextensions.CustomResourceDefinitionSpec{
					PreserveUnknownFields: pointer.BoolPtr(false),
				},
			},
		},
		{
			Name: "v1 to internal, single version",
			In: &CustomResourceDefinition{
				Spec: CustomResourceDefinitionSpec{
					Versions: []CustomResourceDefinitionVersion{{Name: "v1", Served: true, Storage: true}},
				},
			},
			Out: &apiextensions.CustomResourceDefinition{},
			ExpectOut: &apiextensions.CustomResourceDefinition{
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Version:               "v1",
					Versions:              []apiextensions.CustomResourceDefinitionVersion{{Name: "v1", Served: true, Storage: true}},
					PreserveUnknownFields: pointer.BoolPtr(false),
				},
			},
		},
		{
			Name: "v1 to internal, multiple versions",
			In: &CustomResourceDefinition{
				Spec: CustomResourceDefinitionSpec{
					Versions: []CustomResourceDefinitionVersion{
						{Name: "v1", Served: true, Storage: true},
						{Name: "v2", Served: false, Storage: false},
					},
				},
			},
			Out: &apiextensions.CustomResourceDefinition{},
			ExpectOut: &apiextensions.CustomResourceDefinition{
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Version: "v1",
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{Name: "v1", Served: true, Storage: true},
						{Name: "v2", Served: false, Storage: false},
					},
					PreserveUnknownFields: pointer.BoolPtr(false),
				},
			},
		},
		// Validation
		{
			Name: "internal to v1, top-level validation moves to per-version",
			In: &apiextensions.CustomResourceDefinition{
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Version:    "v1",
					Validation: &apiextensions.CustomResourceValidation{OpenAPIV3Schema: &apiextensions.JSONSchemaProps{Type: "object"}},
				},
			},
			Out: &CustomResourceDefinition{},
			ExpectOut: &CustomResourceDefinition{
				Spec: CustomResourceDefinitionSpec{
					Versions: []CustomResourceDefinitionVersion{
						{Name: "v1", Served: true, Storage: true, Schema: &CustomResourceValidation{OpenAPIV3Schema: &JSONSchemaProps{Type: "object"}}},
					},
				},
			},
		},
		{
			Name: "internal to v1, per-version validation is preserved",
			In: &apiextensions.CustomResourceDefinition{
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{Name: "v1", Served: true, Storage: true, Schema: &apiextensions.CustomResourceValidation{OpenAPIV3Schema: &apiextensions.JSONSchemaProps{Description: "v1", Type: "object"}}},
						{Name: "v2", Served: false, Storage: false, Schema: &apiextensions.CustomResourceValidation{OpenAPIV3Schema: &apiextensions.JSONSchemaProps{Description: "v2", Type: "object"}}},
					},
				},
			},
			Out: &CustomResourceDefinition{},
			ExpectOut: &CustomResourceDefinition{
				Spec: CustomResourceDefinitionSpec{
					Versions: []CustomResourceDefinitionVersion{
						{Name: "v1", Served: true, Storage: true, Schema: &CustomResourceValidation{OpenAPIV3Schema: &JSONSchemaProps{Description: "v1", Type: "object"}}},
						{Name: "v2", Served: false, Storage: false, Schema: &CustomResourceValidation{OpenAPIV3Schema: &JSONSchemaProps{Description: "v2", Type: "object"}}},
					},
				},
			},
		},
		{
			Name: "v1 to internal, identical validation moves to top-level",
			In: &CustomResourceDefinition{
				Spec: CustomResourceDefinitionSpec{
					Versions: []CustomResourceDefinitionVersion{
						{Name: "v1", Served: true, Storage: true, Schema: &CustomResourceValidation{OpenAPIV3Schema: &JSONSchemaProps{Type: "object"}}},
						{Name: "v2", Served: true, Storage: false, Schema: &CustomResourceValidation{OpenAPIV3Schema: &JSONSchemaProps{Type: "object"}}},
					},
				},
			},
			Out: &apiextensions.CustomResourceDefinition{},
			ExpectOut: &apiextensions.CustomResourceDefinition{
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Version: "v1",
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{Name: "v1", Served: true, Storage: true},
						{Name: "v2", Served: true, Storage: false},
					},
					Validation:            &apiextensions.CustomResourceValidation{OpenAPIV3Schema: &apiextensions.JSONSchemaProps{Type: "object"}},
					PreserveUnknownFields: pointer.BoolPtr(false),
				},
			},
		},
		{
			Name: "v1 to internal, distinct validation remains per-version",
			In: &CustomResourceDefinition{
				Spec: CustomResourceDefinitionSpec{
					Versions: []CustomResourceDefinitionVersion{
						{Name: "v1", Served: true, Storage: true, Schema: &CustomResourceValidation{OpenAPIV3Schema: &JSONSchemaProps{Description: "v1", Type: "object"}}},
						{Name: "v2", Served: true, Storage: false, Schema: &CustomResourceValidation{OpenAPIV3Schema: &JSONSchemaProps{Description: "v2", Type: "object"}}},
					},
				},
			},
			Out: &apiextensions.CustomResourceDefinition{},
			ExpectOut: &apiextensions.CustomResourceDefinition{
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Version: "v1",
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{Name: "v1", Served: true, Storage: true, Schema: &apiextensions.CustomResourceValidation{OpenAPIV3Schema: &apiextensions.JSONSchemaProps{Description: "v1", Type: "object"}}},
						{Name: "v2", Served: true, Storage: false, Schema: &apiextensions.CustomResourceValidation{OpenAPIV3Schema: &apiextensions.JSONSchemaProps{Description: "v2", Type: "object"}}},
					},
					PreserveUnknownFields: pointer.BoolPtr(false),
				},
			},
		},
		// Subresources
		{
			Name: "internal to v1, top-level subresources moves to per-version",
			In: &apiextensions.CustomResourceDefinition{
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Version:      "v1",
					Subresources: &apiextensions.CustomResourceSubresources{Scale: &apiextensions.CustomResourceSubresourceScale{SpecReplicasPath: "spec.replicas"}},
				},
			},
			Out: &CustomResourceDefinition{},
			ExpectOut: &CustomResourceDefinition{
				Spec: CustomResourceDefinitionSpec{
					Versions: []CustomResourceDefinitionVersion{
						{Name: "v1", Served: true, Storage: true, Subresources: &CustomResourceSubresources{Scale: &CustomResourceSubresourceScale{SpecReplicasPath: "spec.replicas"}}},
					},
				},
			},
		},
		{
			Name: "internal to v1, per-version subresources is preserved",
			In: &apiextensions.CustomResourceDefinition{
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{Name: "v1", Served: true, Storage: true, Subresources: &apiextensions.CustomResourceSubresources{Scale: &apiextensions.CustomResourceSubresourceScale{SpecReplicasPath: "spec.replicas1"}}},
						{Name: "v2", Served: false, Storage: false, Subresources: &apiextensions.CustomResourceSubresources{Scale: &apiextensions.CustomResourceSubresourceScale{SpecReplicasPath: "spec.replicas2"}}},
					},
				},
			},
			Out: &CustomResourceDefinition{},
			ExpectOut: &CustomResourceDefinition{
				Spec: CustomResourceDefinitionSpec{
					Versions: []CustomResourceDefinitionVersion{
						{Name: "v1", Served: true, Storage: true, Subresources: &CustomResourceSubresources{Scale: &CustomResourceSubresourceScale{SpecReplicasPath: "spec.replicas1"}}},
						{Name: "v2", Served: false, Storage: false, Subresources: &CustomResourceSubresources{Scale: &CustomResourceSubresourceScale{SpecReplicasPath: "spec.replicas2"}}},
					},
				},
			},
		},
		{
			Name: "v1 to internal, identical subresources moves to top-level",
			In: &CustomResourceDefinition{
				Spec: CustomResourceDefinitionSpec{
					Versions: []CustomResourceDefinitionVersion{
						{Name: "v1", Served: true, Storage: true, Subresources: &CustomResourceSubresources{Scale: &CustomResourceSubresourceScale{SpecReplicasPath: "spec.replicas"}}},
						{Name: "v2", Served: true, Storage: false, Subresources: &CustomResourceSubresources{Scale: &CustomResourceSubresourceScale{SpecReplicasPath: "spec.replicas"}}},
					},
				},
			},
			Out: &apiextensions.CustomResourceDefinition{},
			ExpectOut: &apiextensions.CustomResourceDefinition{
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Version: "v1",
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{Name: "v1", Served: true, Storage: true},
						{Name: "v2", Served: true, Storage: false},
					},
					Subresources:          &apiextensions.CustomResourceSubresources{Scale: &apiextensions.CustomResourceSubresourceScale{SpecReplicasPath: "spec.replicas"}},
					PreserveUnknownFields: pointer.BoolPtr(false),
				},
			},
		},
		{
			Name: "v1 to internal, distinct subresources remains per-version",
			In: &CustomResourceDefinition{
				Spec: CustomResourceDefinitionSpec{
					Versions: []CustomResourceDefinitionVersion{
						{Name: "v1", Served: true, Storage: true, Subresources: &CustomResourceSubresources{Scale: &CustomResourceSubresourceScale{SpecReplicasPath: "spec.replicas1"}}},
						{Name: "v2", Served: true, Storage: false, Subresources: &CustomResourceSubresources{Scale: &CustomResourceSubresourceScale{SpecReplicasPath: "spec.replicas2"}}},
					},
				},
			},
			Out: &apiextensions.CustomResourceDefinition{},
			ExpectOut: &apiextensions.CustomResourceDefinition{
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Version: "v1",
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{Name: "v1", Served: true, Storage: true, Subresources: &apiextensions.CustomResourceSubresources{Scale: &apiextensions.CustomResourceSubresourceScale{SpecReplicasPath: "spec.replicas1"}}},
						{Name: "v2", Served: true, Storage: false, Subresources: &apiextensions.CustomResourceSubresources{Scale: &apiextensions.CustomResourceSubresourceScale{SpecReplicasPath: "spec.replicas2"}}},
					},
					PreserveUnknownFields: pointer.BoolPtr(false),
				},
			},
		},
		// Additional Printer Columns
		{
			Name: "internal to v1, top-level printer columns moves to per-version",
			In: &apiextensions.CustomResourceDefinition{
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Version:                  "v1",
					AdditionalPrinterColumns: []apiextensions.CustomResourceColumnDefinition{{Name: "column1"}},
				},
			},
			Out: &CustomResourceDefinition{},
			ExpectOut: &CustomResourceDefinition{
				Spec: CustomResourceDefinitionSpec{
					Versions: []CustomResourceDefinitionVersion{
						{Name: "v1", Served: true, Storage: true, AdditionalPrinterColumns: []CustomResourceColumnDefinition{{Name: "column1"}}},
					},
				},
			},
		},
		{
			Name: "internal to v1, per-version printer columns is preserved",
			In: &apiextensions.CustomResourceDefinition{
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{Name: "v1", Served: true, Storage: true, AdditionalPrinterColumns: []apiextensions.CustomResourceColumnDefinition{{Name: "column1"}}},
						{Name: "v2", Served: false, Storage: false, AdditionalPrinterColumns: []apiextensions.CustomResourceColumnDefinition{{Name: "column2"}}},
					},
				},
			},
			Out: &CustomResourceDefinition{},
			ExpectOut: &CustomResourceDefinition{
				Spec: CustomResourceDefinitionSpec{
					Versions: []CustomResourceDefinitionVersion{
						{Name: "v1", Served: true, Storage: true, AdditionalPrinterColumns: []CustomResourceColumnDefinition{{Name: "column1"}}},
						{Name: "v2", Served: false, Storage: false, AdditionalPrinterColumns: []CustomResourceColumnDefinition{{Name: "column2"}}},
					},
				},
			},
		},
		{
			Name: "v1 to internal, identical printer columns moves to top-level",
			In: &CustomResourceDefinition{
				Spec: CustomResourceDefinitionSpec{
					Versions: []CustomResourceDefinitionVersion{
						{Name: "v1", Served: true, Storage: true, AdditionalPrinterColumns: []CustomResourceColumnDefinition{{Name: "column1"}}},
						{Name: "v2", Served: true, Storage: false, AdditionalPrinterColumns: []CustomResourceColumnDefinition{{Name: "column1"}}},
					},
				},
			},
			Out: &apiextensions.CustomResourceDefinition{},
			ExpectOut: &apiextensions.CustomResourceDefinition{
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Version: "v1",
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{Name: "v1", Served: true, Storage: true},
						{Name: "v2", Served: true, Storage: false},
					},
					AdditionalPrinterColumns: []apiextensions.CustomResourceColumnDefinition{{Name: "column1"}},
					PreserveUnknownFields:    pointer.BoolPtr(false),
				},
			},
		},
		{
			Name: "v1 to internal, distinct printer columns remains per-version",
			In: &CustomResourceDefinition{
				Spec: CustomResourceDefinitionSpec{
					Versions: []CustomResourceDefinitionVersion{
						{Name: "v1", Served: true, Storage: true, AdditionalPrinterColumns: []CustomResourceColumnDefinition{{Name: "column1"}}},
						{Name: "v2", Served: true, Storage: false, AdditionalPrinterColumns: []CustomResourceColumnDefinition{{Name: "column2"}}},
					},
				},
			},
			Out: &apiextensions.CustomResourceDefinition{},
			ExpectOut: &apiextensions.CustomResourceDefinition{
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Version: "v1",
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{Name: "v1", Served: true, Storage: true, AdditionalPrinterColumns: []apiextensions.CustomResourceColumnDefinition{{Name: "column1"}}},
						{Name: "v2", Served: true, Storage: false, AdditionalPrinterColumns: []apiextensions.CustomResourceColumnDefinition{{Name: "column2"}}},
					},
					PreserveUnknownFields: pointer.BoolPtr(false),
				},
			},
		},
		// webhook conversion config
		{
			Name: "internal to v1, no webhook client config",
			In: &apiextensions.CustomResourceDefinition{
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Conversion: &apiextensions.CustomResourceConversion{},
				},
			},
			Out: &CustomResourceDefinition{},
			ExpectOut: &CustomResourceDefinition{
				Spec: CustomResourceDefinitionSpec{
					Conversion: &CustomResourceConversion{},
				},
			},
		},
		{
			Name: "internal to v1, webhook client config",
			In: &apiextensions.CustomResourceDefinition{
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Conversion: &apiextensions.CustomResourceConversion{
						WebhookClientConfig: &apiextensions.WebhookClientConfig{URL: pointer.StringPtr("http://example.com")},
					},
				},
			},
			Out: &CustomResourceDefinition{},
			ExpectOut: &CustomResourceDefinition{
				Spec: CustomResourceDefinitionSpec{
					Conversion: &CustomResourceConversion{
						Webhook: &WebhookConversion{
							ClientConfig: &WebhookClientConfig{URL: pointer.StringPtr("http://example.com")},
						},
					},
				},
			},
		},
		{
			Name: "internal to v1, webhook versions",
			In: &apiextensions.CustomResourceDefinition{
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Conversion: &apiextensions.CustomResourceConversion{
						ConversionReviewVersions: []string{"v1"},
					},
				},
			},
			Out: &CustomResourceDefinition{},
			ExpectOut: &CustomResourceDefinition{
				Spec: CustomResourceDefinitionSpec{
					Conversion: &CustomResourceConversion{
						Webhook: &WebhookConversion{
							ConversionReviewVersions: []string{"v1"},
						},
					},
				},
			},
		},
		{
			Name: "v1 to internal, no webhook client config",
			In: &CustomResourceDefinition{
				Spec: CustomResourceDefinitionSpec{
					Conversion: &CustomResourceConversion{},
				},
			},
			Out: &apiextensions.CustomResourceDefinition{},
			ExpectOut: &apiextensions.CustomResourceDefinition{
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Conversion:            &apiextensions.CustomResourceConversion{},
					PreserveUnknownFields: pointer.BoolPtr(false),
				},
			},
		},
		{
			Name: "v1 to internal, webhook client config",
			In: &CustomResourceDefinition{
				Spec: CustomResourceDefinitionSpec{
					Conversion: &CustomResourceConversion{
						Webhook: &WebhookConversion{
							ClientConfig: &WebhookClientConfig{URL: pointer.StringPtr("http://example.com")},
						},
					},
				},
			},
			Out: &apiextensions.CustomResourceDefinition{},
			ExpectOut: &apiextensions.CustomResourceDefinition{
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Conversion: &apiextensions.CustomResourceConversion{
						WebhookClientConfig: &apiextensions.WebhookClientConfig{URL: pointer.StringPtr("http://example.com")},
					},
					PreserveUnknownFields: pointer.BoolPtr(false),
				},
			},
		},
		{
			Name: "v1 to internal, webhook versions",
			In: &CustomResourceDefinition{
				Spec: CustomResourceDefinitionSpec{
					Conversion: &CustomResourceConversion{
						Webhook: &WebhookConversion{
							ConversionReviewVersions: []string{"v1"},
						},
					},
				},
			},
			Out: &apiextensions.CustomResourceDefinition{},
			ExpectOut: &apiextensions.CustomResourceDefinition{
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Conversion: &apiextensions.CustomResourceConversion{
						ConversionReviewVersions: []string{"v1"},
					},
					PreserveUnknownFields: pointer.BoolPtr(false),
				},
			},
		},
	}

	scheme := runtime.NewScheme()
	// add internal and external types
	if err := apiextensions.AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}
	if err := AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}

	for _, tc := range testcases {
		t.Run(tc.Name, func(t *testing.T) {
			err := scheme.Convert(tc.In, tc.Out, nil)
			if err != nil {
				if len(tc.ExpectErr) == 0 {
					t.Fatalf("unexpected error %v", err)
				}
				if !strings.Contains(err.Error(), tc.ExpectErr) {
					t.Fatalf("expected error %s, got %v", tc.ExpectErr, err)
				}
				return
			}
			if len(tc.ExpectErr) > 0 {
				t.Fatalf("expected error %s, got none", tc.ExpectErr)
			}
			if !reflect.DeepEqual(tc.Out, tc.ExpectOut) {
				t.Fatalf("unexpected result:\n %s", cmp.Diff(tc.ExpectOut, tc.Out))
			}
		})
	}
}

func TestJSONConversion(t *testing.T) {
	nilJSON := apiextensions.JSON(nil)
	nullJSON := apiextensions.JSON("null")
	stringJSON := apiextensions.JSON("foo")
	boolJSON := apiextensions.JSON(true)
	sliceJSON := apiextensions.JSON([]string{"foo", "bar", "baz"})

	testCases := map[string]struct {
		input    *apiextensions.JSONSchemaProps
		expected *JSONSchemaProps
	}{
		"nil": {
			input: &apiextensions.JSONSchemaProps{
				Default: nil,
			},
			expected: &JSONSchemaProps{},
		},
		"aliased nil": {
			input: &apiextensions.JSONSchemaProps{
				Default: &nilJSON,
			},
			expected: &JSONSchemaProps{},
		},
		"null": {
			input: &apiextensions.JSONSchemaProps{
				Default: &nullJSON,
			},
			expected: &JSONSchemaProps{
				Default: &JSON{
					Raw: []byte(`"null"`),
				},
			},
		},
		"string": {
			input: &apiextensions.JSONSchemaProps{
				Default: &stringJSON,
			},
			expected: &JSONSchemaProps{
				Default: &JSON{
					Raw: []byte(`"foo"`),
				},
			},
		},
		"bool": {
			input: &apiextensions.JSONSchemaProps{
				Default: &boolJSON,
			},
			expected: &JSONSchemaProps{
				Default: &JSON{
					Raw: []byte(`true`),
				},
			},
		},
		"slice": {
			input: &apiextensions.JSONSchemaProps{
				Default: &sliceJSON,
			},
			expected: &JSONSchemaProps{
				Default: &JSON{
					Raw: []byte(`["foo","bar","baz"]`),
				},
			},
		},
	}

	scheme := runtime.NewScheme()

	// add internal and external types
	if err := apiextensions.AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}
	if err := AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}

	for k, tc := range testCases {
		external := &JSONSchemaProps{}
		if err := scheme.Convert(tc.input, external, nil); err != nil {
			t.Errorf("%s: unexpected error: %v", k, err)
		}

		if !reflect.DeepEqual(external, tc.expected) {
			t.Errorf("%s: expected\n\t%#v, got \n\t%#v", k, tc.expected, external)
		}
	}
}
