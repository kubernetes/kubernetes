/*
Copyright 2022 The Kubernetes Authors.

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

package defaulting

import (
	"context"
	"strings"
	"testing"

	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	structuralschema "k8s.io/apiextensions-apiserver/pkg/apiserver/schema"
	apiextensionsfeatures "k8s.io/apiextensions-apiserver/pkg/features"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/component-base/featuregate"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/utils/ptr"
)

func jsonPtr(x interface{}) *apiextensions.JSON {
	ret := apiextensions.JSON(x)
	return &ret
}

func TestDefaultValidationWithCostBudget(t *testing.T) {
	tests := []struct {
		name     string
		input    apiextensions.CustomResourceValidation
		features []featuregate.Feature
	}{
		{
			name: "default cel validation",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type: "object",
					Properties: map[string]apiextensions.JSONSchemaProps{
						"embedded": {
							Type: "object",
							Properties: map[string]apiextensions.JSONSchemaProps{
								"metadata": {
									Type:              "object",
									XEmbeddedResource: true,
									Properties: map[string]apiextensions.JSONSchemaProps{
										"name": {
											Type: "string",
											XValidations: apiextensions.ValidationRules{
												{
													Rule: "self == 'singleton'",
												},
											},
											Default: jsonPtr("singleton"),
										},
									},
								},
							},
						},
						"value": {
							Type: "string",
							XValidations: apiextensions.ValidationRules{
								{
									Rule: "self.startsWith('kube')",
								},
							},
							Default: jsonPtr("kube-everything"),
						},
						"object": {
							Type: "object",
							Properties: map[string]apiextensions.JSONSchemaProps{
								"field1": {
									Type: "integer",
								},
								"field2": {
									Type: "integer",
								},
							},
							XValidations: apiextensions.ValidationRules{
								{
									Rule: "self.field1 < self.field2",
								},
							},
							Default: jsonPtr(map[string]interface{}{"field1": 1, "field2": 2}),
						},
					},
				},
			},
		},
	}

	for _, tt := range tests {
		ctx := context.TODO()
		t.Run(tt.name, func(t *testing.T) {
			for _, f := range tt.features {
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, f, true)
			}

			schema := tt.input.OpenAPIV3Schema
			ss, err := structuralschema.NewStructural(schema)
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}

			f := NewRootObjectFunc().WithTypeMeta(metav1.TypeMeta{APIVersion: "validation/v1", Kind: "Validation"})

			// cost budget is large enough to pass all validation rules
			allErrs, err, _ := validate(ctx, field.NewPath("test"), ss, ss, f, false, false, 10)
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}

			for _, valErr := range allErrs {
				t.Errorf("unexpected error: %v", valErr)
			}

			// cost budget exceeded for the first validation rule
			allErrs, err, _ = validate(ctx, field.NewPath("test"), ss, ss, f, false, false, 0)
			meet := 0
			for _, er := range allErrs {
				if er.Type == field.ErrorTypeInvalid && strings.Contains(er.Error(), "validation failed due to running out of cost budget, no further validation rules will be run") {
					meet += 1
				}
			}
			if meet != 1 {
				t.Errorf("expected to get cost budget exceed error once but got %v cost budget exceed error", meet)
			}
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}

			// cost budget exceeded for the last validation rule
			allErrs, err, _ = validate(ctx, field.NewPath("test"), ss, ss, f, false, false, 9)
			meet = 0
			for _, er := range allErrs {
				if er.Type == field.ErrorTypeInvalid && strings.Contains(er.Error(), "validation failed due to running out of cost budget, no further validation rules will be run") {
					meet += 1
				}
			}
			if meet != 1 {
				t.Errorf("expected to get cost budget exceed error once but got %v cost budget exceed error", meet)
			}
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
		})
	}
}

func TestDefaultValidationWithOptionalOldSelf(t *testing.T) {
	tests := []struct {
		name   string
		input  apiextensions.CustomResourceValidation
		errors []string
	}{
		{
			name: "invalid default",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type: "object",
					Properties: map[string]apiextensions.JSONSchemaProps{
						"defaultFailsRatcheting": {
							Type:    "string",
							Default: jsonPtr("default"),
							XValidations: apiextensions.ValidationRules{
								{
									Rule:            "oldSelf.hasValue()",
									OptionalOldSelf: ptr.To(true),
									Message:         "foobarErrorMessage",
								},
							},
						},
					},
				},
			},
			errors: []string{"foobarErrorMessage"},
		},
		{
			name: "valid default",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type: "object",
					Properties: map[string]apiextensions.JSONSchemaProps{
						"defaultFailsRatcheting": {
							Type:    "string",
							Default: jsonPtr("default"),
							XValidations: apiextensions.ValidationRules{
								{
									Rule:            "oldSelf.orValue(self) == self",
									OptionalOldSelf: ptr.To(true),
									Message:         "foobarErrorMessage",
								},
							},
						},
					},
				},
			},
			errors: []string{},
		},
	}

	for _, tt := range tests {
		ctx := context.TODO()
		t.Run(tt.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, apiextensionsfeatures.CRDValidationRatcheting, true)
			schema := tt.input.OpenAPIV3Schema
			ss, err := structuralschema.NewStructural(schema)
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}

			f := NewRootObjectFunc().WithTypeMeta(metav1.TypeMeta{APIVersion: "validation/v1", Kind: "Validation"})

			// cost budget is large enough to pass all validation rules
			allErrs, err, _ := validate(ctx, field.NewPath("test"), ss, ss, f, false, false, 10)
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}

			for _, err := range allErrs {
				found := false
				for _, expected := range tt.errors {
					if strings.Contains(err.Error(), expected) {
						found = true
						break
					}
				}
				if !found {
					t.Errorf("unexpected error: %v", err)
				}
			}

			for _, expected := range tt.errors {
				found := false
				for _, err := range allErrs {
					if strings.Contains(err.Error(), expected) {
						found = true
						break
					}
				}
				if !found {
					t.Errorf("expected error: %v", expected)
				}
			}

		})
	}
}
