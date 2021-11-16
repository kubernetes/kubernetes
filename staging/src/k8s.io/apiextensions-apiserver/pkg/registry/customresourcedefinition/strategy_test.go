/*
Copyright 2017 The Kubernetes Authors.

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

package customresourcedefinition

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/validation"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/utils/pointer"
)

func strPtr(in string) *string {
	return &in
}

func TestValidateAPIApproval(t *testing.T) {
	okFn := func(t *testing.T, errors field.ErrorList) {
		t.Helper()
		if len(errors) > 0 {
			t.Fatal(errors)
		}
	}

	tests := []struct {
		name string

		group              string
		annotationValue    string
		oldAnnotationValue *string
		validateError      func(t *testing.T, errors field.ErrorList)
	}{
		{
			name:            "ignore non-k8s group",
			group:           "other.io",
			annotationValue: "invalid",
			validateError:   okFn,
		},
		{
			name:            "invalid annotation create",
			group:           "sigs.k8s.io",
			annotationValue: "invalid",
			validateError: func(t *testing.T, errors field.ErrorList) {
				t.Helper()
				if len(errors) == 0 {
					t.Fatal("expected errors, got none")
				}
				if e, a := `metadata.annotations[api-approved.kubernetes.io]: Invalid value: "invalid": protected groups must have approval annotation "api-approved.kubernetes.io" with either a URL or a reason starting with "unapproved", see https://github.com/kubernetes/enhancements/pull/1111`, errors.ToAggregate().Error(); e != a {
					t.Fatal(errors)
				}
			},
		},
		{
			name:               "invalid annotation update",
			group:              "sigs.k8s.io",
			annotationValue:    "invalid",
			oldAnnotationValue: strPtr("invalid"),
			validateError:      okFn,
		},
		{
			name:               "invalid annotation to missing",
			group:              "sigs.k8s.io",
			annotationValue:    "",
			oldAnnotationValue: strPtr("invalid"),
			validateError: func(t *testing.T, errors field.ErrorList) {
				t.Helper()
				if len(errors) == 0 {
					t.Fatal("expected errors, got none")
				}
				if e, a := `metadata.annotations[api-approved.kubernetes.io]: Required value: protected groups must have approval annotation "api-approved.kubernetes.io", see https://github.com/kubernetes/enhancements/pull/1111`, errors.ToAggregate().Error(); e != a {
					t.Fatal(errors)
				}
			},
		},
		{
			name:               "missing to invalid annotation",
			group:              "sigs.k8s.io",
			annotationValue:    "invalid",
			oldAnnotationValue: strPtr(""),
			validateError: func(t *testing.T, errors field.ErrorList) {
				t.Helper()
				if len(errors) == 0 {
					t.Fatal("expected errors, got none")
				}
				if e, a := `metadata.annotations[api-approved.kubernetes.io]: Invalid value: "invalid": protected groups must have approval annotation "api-approved.kubernetes.io" with either a URL or a reason starting with "unapproved", see https://github.com/kubernetes/enhancements/pull/1111`, errors.ToAggregate().Error(); e != a {
					t.Fatal(errors)
				}
			},
		},
		{
			name:            "missing annotation",
			group:           "sigs.k8s.io",
			annotationValue: "",
			validateError: func(t *testing.T, errors field.ErrorList) {
				t.Helper()
				if len(errors) == 0 {
					t.Fatal("expected errors, got none")
				}
				if e, a := `metadata.annotations[api-approved.kubernetes.io]: Required value: protected groups must have approval annotation "api-approved.kubernetes.io", see https://github.com/kubernetes/enhancements/pull/1111`, errors.ToAggregate().Error(); e != a {
					t.Fatal(errors)
				}
			},
		},
		{
			name:               "missing annotation update",
			group:              "sigs.k8s.io",
			annotationValue:    "",
			oldAnnotationValue: strPtr(""),
			validateError:      okFn,
		},
		{
			name:            "url",
			group:           "sigs.k8s.io",
			annotationValue: "https://github.com/kubernetes/kubernetes/pull/79724",
			validateError:   okFn,
		},
		{
			name:            "unapproved",
			group:           "sigs.k8s.io",
			annotationValue: "unapproved, other reason",
			validateError:   okFn,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			crd := &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "foos." + test.group, Annotations: map[string]string{v1beta1.KubeAPIApprovedAnnotation: test.annotationValue}, ResourceVersion: "1"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:    test.group,
					Scope:    apiextensions.NamespaceScoped,
					Version:  "v1",
					Versions: []apiextensions.CustomResourceDefinitionVersion{{Name: "v1", Storage: true, Served: true}},
					Names:    apiextensions.CustomResourceDefinitionNames{Plural: "foos", Singular: "foo", Kind: "Foo", ListKind: "FooList"},
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{Type: "object", XPreserveUnknownFields: pointer.BoolPtr(true)},
					},
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"v1"},
				},
			}
			var oldCRD *apiextensions.CustomResourceDefinition
			if test.oldAnnotationValue != nil {
				oldCRD = &apiextensions.CustomResourceDefinition{
					ObjectMeta: metav1.ObjectMeta{Name: "foos." + test.group, Annotations: map[string]string{v1beta1.KubeAPIApprovedAnnotation: *test.oldAnnotationValue}, ResourceVersion: "1"},
					Spec: apiextensions.CustomResourceDefinitionSpec{
						Group:    test.group,
						Scope:    apiextensions.NamespaceScoped,
						Version:  "v1",
						Versions: []apiextensions.CustomResourceDefinitionVersion{{Name: "v1", Storage: true, Served: true}},
						Names:    apiextensions.CustomResourceDefinitionNames{Plural: "foos", Singular: "foo", Kind: "Foo", ListKind: "FooList"},
						Validation: &apiextensions.CustomResourceValidation{
							OpenAPIV3Schema: &apiextensions.JSONSchemaProps{Type: "object", XPreserveUnknownFields: pointer.BoolPtr(true)},
						},
					},
					Status: apiextensions.CustomResourceDefinitionStatus{
						StoredVersions: []string{"v1"},
					},
				}
			}

			var actual field.ErrorList
			if oldCRD == nil {
				actual = validation.ValidateCustomResourceDefinition(crd)
			} else {
				actual = validation.ValidateCustomResourceDefinitionUpdate(crd, oldCRD)
			}
			test.validateError(t, actual)
		})
	}
}

// TestDropDisabledFields tests if the drop functionality is working fine or not with feature gate switch
func TestDropDisabledFields(t *testing.T) {
	testCases := []struct {
		name               string
		enableXValidations bool
		crd                *apiextensions.CustomResourceDefinition
		oldCRD             *apiextensions.CustomResourceDefinition
		expectedCRD        *apiextensions.CustomResourceDefinition
	}{
		{
			name:               "For creation, FG disabled, no XValidations, no field drop",
			enableXValidations: false,
			crd:                &apiextensions.CustomResourceDefinition{},
			oldCRD:             nil,
			expectedCRD:        &apiextensions.CustomResourceDefinition{},
		},
		{
			name:               "For creation, FG disabled, empty XValidations, no field drop",
			enableXValidations: false,
			crd: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "foos.sigs.k8s.io", Annotations: map[string]string{v1beta1.KubeAPIApprovedAnnotation: "valid"}, ResourceVersion: "1"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Validation: &apiextensions.CustomResourceValidation{},
				},
			},
			oldCRD: nil,
			expectedCRD: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "foos.sigs.k8s.io", Annotations: map[string]string{v1beta1.KubeAPIApprovedAnnotation: "valid"}, ResourceVersion: "1"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Validation: &apiextensions.CustomResourceValidation{},
				},
			},
		},
		{
			name:               "For creation, FG disabled, set XValidations, drop XValidations",
			enableXValidations: false,
			crd: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "foos.sigs.k8s.io", Annotations: map[string]string{v1beta1.KubeAPIApprovedAnnotation: "valid"}, ResourceVersion: "1"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
							XValidations: apiextensions.ValidationRules{
								{
									Rule:    "size(self) > 0",
									Message: "openAPIV3Schema should contain more than 0 element.",
								},
							},
							Dependencies: apiextensions.JSONSchemaDependencies{
								"test": apiextensions.JSONSchemaPropsOrStringArray{
									Schema: &apiextensions.JSONSchemaProps{
										Type: "object",
										XValidations: apiextensions.ValidationRules{
											{
												Rule:    "size(self) > 0",
												Message: "size of scoped field should be greater than 0.",
											},
										},
									},
								},
							},
							Properties: map[string]apiextensions.JSONSchemaProps{
								"subRule": {
									Type: "object",
									XValidations: apiextensions.ValidationRules{
										{
											Rule:    "isTest == true",
											Message: "isTest should be true.",
										},
									},
									Properties: map[string]apiextensions.JSONSchemaProps{
										"isTest": {
											Type: "boolean",
										},
									},
								},
							},
						},
					},
				},
			},
			oldCRD: nil,
			expectedCRD: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "foos.sigs.k8s.io", Annotations: map[string]string{v1beta1.KubeAPIApprovedAnnotation: "valid"}, ResourceVersion: "1"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
							Dependencies: apiextensions.JSONSchemaDependencies{
								"test": apiextensions.JSONSchemaPropsOrStringArray{
									Schema: &apiextensions.JSONSchemaProps{
										Type: "object",
									},
								},
							},
							Properties: map[string]apiextensions.JSONSchemaProps{
								"subRule": {
									Type: "object",
									Properties: map[string]apiextensions.JSONSchemaProps{
										"isTest": {
											Type: "boolean",
										},
									},
								},
							},
						},
					},
				},
			},
		},
		{
			name:               "For creation, FG enabled, set XValidations, update with XValidations",
			enableXValidations: true,
			crd: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "foos.sigs.k8s.io", Annotations: map[string]string{v1beta1.KubeAPIApprovedAnnotation: "valid"}, ResourceVersion: "1"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
							XValidations: apiextensions.ValidationRules{
								{
									Rule:    "size(self) > 0",
									Message: "openAPIV3Schema should contain more than 0 element.",
								},
							},
							Dependencies: apiextensions.JSONSchemaDependencies{
								"test": apiextensions.JSONSchemaPropsOrStringArray{
									Schema: &apiextensions.JSONSchemaProps{
										Type: "object",
										XValidations: apiextensions.ValidationRules{
											{
												Rule:    "size(self) > 0",
												Message: "size of scoped field should be greater than 0.",
											},
										},
									},
								},
							},
							Properties: map[string]apiextensions.JSONSchemaProps{
								"subRule": {
									Type: "object",
									XValidations: apiextensions.ValidationRules{
										{
											Rule:    "isTest == true",
											Message: "isTest should be true.",
										},
									},
									Properties: map[string]apiextensions.JSONSchemaProps{
										"isTest": {
											Type: "boolean",
										},
									},
								},
							},
						},
					},
				},
			},
			oldCRD: nil,
			expectedCRD: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "foos.sigs.k8s.io", Annotations: map[string]string{v1beta1.KubeAPIApprovedAnnotation: "valid"}, ResourceVersion: "1"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
							XValidations: apiextensions.ValidationRules{
								{
									Rule:    "size(self) > 0",
									Message: "openAPIV3Schema should contain more than 0 element.",
								},
							},
							Dependencies: apiextensions.JSONSchemaDependencies{
								"test": apiextensions.JSONSchemaPropsOrStringArray{
									Schema: &apiextensions.JSONSchemaProps{
										Type: "object",
										XValidations: apiextensions.ValidationRules{
											{
												Rule:    "size(self) > 0",
												Message: "size of scoped field should be greater than 0.",
											},
										},
									},
								},
							},
							Properties: map[string]apiextensions.JSONSchemaProps{
								"subRule": {
									Type: "object",
									XValidations: apiextensions.ValidationRules{
										{
											Rule:    "isTest == true",
											Message: "isTest should be true.",
										},
									},
									Properties: map[string]apiextensions.JSONSchemaProps{
										"isTest": {
											Type: "boolean",
										},
									},
								},
							},
						},
					},
				},
			},
		},
		{
			name:               "For update, FG disabled, oldCRD XValidation in use, don't drop XValidations",
			enableXValidations: false,
			crd: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "foos.sigs.k8s.io", Annotations: map[string]string{v1beta1.KubeAPIApprovedAnnotation: "valid"}, ResourceVersion: "1"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
							XValidations: apiextensions.ValidationRules{
								{
									Rule:    "size(self) > 0",
									Message: "openAPIV3Schema should contain more than 0 element.",
								},
							},
							Dependencies: apiextensions.JSONSchemaDependencies{
								"test": apiextensions.JSONSchemaPropsOrStringArray{
									Schema: &apiextensions.JSONSchemaProps{
										Type: "object",
										XValidations: apiextensions.ValidationRules{
											{
												Rule:    "size(self) > 0",
												Message: "size of scoped field should be greater than 0.",
											},
										},
									},
								},
							},
							Properties: map[string]apiextensions.JSONSchemaProps{
								"subRule": {
									Type: "object",
									XValidations: apiextensions.ValidationRules{
										{
											Rule:    "isTest == true",
											Message: "isTest should be true.",
										},
									},
									Properties: map[string]apiextensions.JSONSchemaProps{
										"isTest": {
											Type: "boolean",
										},
									},
								},
							},
						},
					},
				},
			},
			oldCRD: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "foos.sigs.k8s.io", Annotations: map[string]string{v1beta1.KubeAPIApprovedAnnotation: "valid"}, ResourceVersion: "1"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
							Dependencies: apiextensions.JSONSchemaDependencies{
								"test": apiextensions.JSONSchemaPropsOrStringArray{
									Schema: &apiextensions.JSONSchemaProps{
										Type: "object",
										XValidations: apiextensions.ValidationRules{
											{
												Rule:    "size(self) > 0",
												Message: "size of scoped field should be greater than 0.",
											},
										},
									},
								},
							},
						},
					},
				},
			},
			expectedCRD: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "foos.sigs.k8s.io", Annotations: map[string]string{v1beta1.KubeAPIApprovedAnnotation: "valid"}, ResourceVersion: "1"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
							XValidations: apiextensions.ValidationRules{
								{
									Rule:    "size(self) > 0",
									Message: "openAPIV3Schema should contain more than 0 element.",
								},
							},
							Dependencies: apiextensions.JSONSchemaDependencies{
								"test": apiextensions.JSONSchemaPropsOrStringArray{
									Schema: &apiextensions.JSONSchemaProps{
										Type: "object",
										XValidations: apiextensions.ValidationRules{
											{
												Rule:    "size(self) > 0",
												Message: "size of scoped field should be greater than 0.",
											},
										},
									},
								},
							},
							Properties: map[string]apiextensions.JSONSchemaProps{
								"subRule": {
									Type: "object",
									XValidations: apiextensions.ValidationRules{
										{
											Rule:    "isTest == true",
											Message: "isTest should be true.",
										},
									},
									Properties: map[string]apiextensions.JSONSchemaProps{
										"isTest": {
											Type: "boolean",
										},
									},
								},
							},
						},
					},
				},
			},
		},
		{
			name:               "For update, FG disabled, oldCRD has no XValidations, drop XValidations",
			enableXValidations: false,
			crd: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "foos.sigs.k8s.io", Annotations: map[string]string{v1beta1.KubeAPIApprovedAnnotation: "valid"}, ResourceVersion: "1"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
							XValidations: apiextensions.ValidationRules{
								{
									Rule:    "size(self) > 0",
									Message: "openAPIV3Schema should contain more than 0 element.",
								},
							},
						},
					},
				},
			},
			oldCRD: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "foos.sigs.k8s.io", Annotations: map[string]string{v1beta1.KubeAPIApprovedAnnotation: "valid"}, ResourceVersion: "1"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
						},
					},
				},
			},
			expectedCRD: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "foos.sigs.k8s.io", Annotations: map[string]string{v1beta1.KubeAPIApprovedAnnotation: "valid"}, ResourceVersion: "1"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
						},
					},
				},
			},
		},
		{
			name:               "For update, FG enabled, oldCRD has XValidations, updated to newCRD",
			enableXValidations: true,
			crd: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "foos.sigs.k8s.io", Annotations: map[string]string{v1beta1.KubeAPIApprovedAnnotation: "valid"}, ResourceVersion: "1"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
							XValidations: apiextensions.ValidationRules{
								{
									Rule:    "size(self) > 0",
									Message: "openAPIV3Schema should contain more than 0 element.",
								},
							},
						},
					},
				},
			},
			oldCRD: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "foos.sigs.k8s.io", Annotations: map[string]string{v1beta1.KubeAPIApprovedAnnotation: "valid"}, ResourceVersion: "1"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
							XValidations: apiextensions.ValidationRules{
								{
									Rule:    "old data",
									Message: "old data",
								},
							},
						},
					},
				},
			},
			expectedCRD: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "foos.sigs.k8s.io", Annotations: map[string]string{v1beta1.KubeAPIApprovedAnnotation: "valid"}, ResourceVersion: "1"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
							XValidations: apiextensions.ValidationRules{
								{
									Rule:    "size(self) > 0",
									Message: "openAPIV3Schema should contain more than 0 element.",
								},
							},
						},
					},
				},
			},
		},
		{
			name:               "For update, FG enabled, oldCRD has no XValidations, updated to newCRD",
			enableXValidations: true,
			crd: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "foos.sigs.k8s.io", Annotations: map[string]string{v1beta1.KubeAPIApprovedAnnotation: "valid"}, ResourceVersion: "1"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
							XValidations: apiextensions.ValidationRules{
								{
									Rule:    "size(self) > 0",
									Message: "openAPIV3Schema should contain more than 0 element.",
								},
							},
						},
					},
				},
			},
			oldCRD: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "foos.sigs.k8s.io", Annotations: map[string]string{v1beta1.KubeAPIApprovedAnnotation: "valid"}, ResourceVersion: "1"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
						},
					},
				},
			},
			expectedCRD: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "foos.sigs.k8s.io", Annotations: map[string]string{v1beta1.KubeAPIApprovedAnnotation: "valid"}, ResourceVersion: "1"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
							XValidations: apiextensions.ValidationRules{
								{
									Rule:    "size(self) > 0",
									Message: "openAPIV3Schema should contain more than 0 element.",
								},
							},
						},
					},
				},
			},
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CustomResourceValidationExpressions, tc.enableXValidations)()
			old := tc.oldCRD.DeepCopy()

			dropDisabledFields(tc.crd, tc.oldCRD)

			// old crd should never be changed
			if diff := cmp.Diff(tc.oldCRD, old); diff != "" {
				t.Fatalf("old crd changed from %v to %v", tc.oldCRD, old)
			}

			if diff := cmp.Diff(tc.expectedCRD, tc.crd); diff != "" {
				t.Fatalf("unexpected crd: %v", tc.crd)
			}
		})
	}
}
