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
	"context"
	"testing"

	"github.com/google/go-cmp/cmp"

	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/validation"
	apiextensionsfeatures "k8s.io/apiextensions-apiserver/pkg/features"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/utils/pointer"
	"k8s.io/utils/ptr"
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
			ctx := context.TODO()
			if oldCRD == nil {
				actual = validation.ValidateCustomResourceDefinition(ctx, crd)
			} else {
				actual = validation.ValidateCustomResourceDefinitionUpdate(ctx, crd, oldCRD)
			}
			test.validateError(t, actual)
		})
	}
}

// TestDropDisabledFields tests if the drop functionality is working fine or not with feature gate switch
func TestDropDisabledFields(t *testing.T) {
	testCases := []struct {
		name                   string
		enableRatcheting       bool
		enableSelectableFields bool
		crd                    *apiextensions.CustomResourceDefinition
		oldCRD                 *apiextensions.CustomResourceDefinition
		expectedCRD            *apiextensions.CustomResourceDefinition
	}{
		{
			name:             "Ratcheting, For creation, FG disabled, no OptionalOldSelf, no field drop",
			enableRatcheting: false,
			crd:              &apiextensions.CustomResourceDefinition{},
			oldCRD:           nil,
			expectedCRD:      &apiextensions.CustomResourceDefinition{},
		},
		{
			name:             "Ratcheting, For creation, FG disabled, set OptionalOldSelf, drop OptionalOldSelf",
			enableRatcheting: false,
			crd: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "foos.sigs.k8s.io", Annotations: map[string]string{v1beta1.KubeAPIApprovedAnnotation: "valid"}, ResourceVersion: "1"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
							XValidations: apiextensions.ValidationRules{
								{
									Rule:            "size(self) > 0",
									Message:         "openAPIV3Schema should contain more than 0 element.",
									OptionalOldSelf: ptr.To(true),
								},
							},
							Properties: map[string]apiextensions.JSONSchemaProps{
								"subRule": {
									Type: "object",
									XValidations: apiextensions.ValidationRules{
										{
											Rule:            "isTest == true",
											Message:         "isTest should be true.",
											OptionalOldSelf: ptr.To(true),
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
			name:             "Ratcheting, For creation, FG enabled, set OptionalOldSelf, update with OptionalOldSelf",
			enableRatcheting: true,
			crd: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "foos.sigs.k8s.io", Annotations: map[string]string{v1beta1.KubeAPIApprovedAnnotation: "valid"}, ResourceVersion: "1"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
							XValidations: apiextensions.ValidationRules{
								{
									Rule:            "size(self) > 0",
									Message:         "openAPIV3Schema should contain more than 0 element.",
									OptionalOldSelf: ptr.To(true),
								},
							},
							Properties: map[string]apiextensions.JSONSchemaProps{
								"subRule": {
									Type: "object",
									XValidations: apiextensions.ValidationRules{
										{
											Rule:            "isTest == true",
											Message:         "isTest should be true.",
											OptionalOldSelf: ptr.To(true),
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
									Rule:            "size(self) > 0",
									Message:         "openAPIV3Schema should contain more than 0 element.",
									OptionalOldSelf: ptr.To(true),
								},
							},
							Properties: map[string]apiextensions.JSONSchemaProps{
								"subRule": {
									Type: "object",
									XValidations: apiextensions.ValidationRules{
										{
											Rule:            "isTest == true",
											Message:         "isTest should be true.",
											OptionalOldSelf: ptr.To(true),
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
			name:             "Ratcheting, For update, FG disabled, oldCRD OptionalOldSelf in use, don't drop OptionalOldSelfs",
			enableRatcheting: false,
			crd: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "foos.sigs.k8s.io", Annotations: map[string]string{v1beta1.KubeAPIApprovedAnnotation: "valid"}, ResourceVersion: "1"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
							XValidations: apiextensions.ValidationRules{
								{
									Rule:            "size(self) > 0",
									Message:         "openAPIV3Schema should contain more than 0 element.",
									OptionalOldSelf: ptr.To(true),
								},
							},
							Properties: map[string]apiextensions.JSONSchemaProps{
								"subRule": {
									Type: "object",
									XValidations: apiextensions.ValidationRules{
										{
											Rule:            "isTest == true",
											Message:         "isTest should be true.",
											OptionalOldSelf: ptr.To(true),
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
							Properties: map[string]apiextensions.JSONSchemaProps{
								"otherRule": {
									Type: "object",
									XValidations: apiextensions.ValidationRules{
										{
											Rule:            "self.isTest == true",
											Message:         "isTest should be true.",
											OptionalOldSelf: ptr.To(true),
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
									Rule:            "size(self) > 0",
									Message:         "openAPIV3Schema should contain more than 0 element.",
									OptionalOldSelf: ptr.To(true),
								},
							},
							Properties: map[string]apiextensions.JSONSchemaProps{
								"subRule": {
									Type: "object",
									XValidations: apiextensions.ValidationRules{
										{
											Rule:            "isTest == true",
											Message:         "isTest should be true.",
											OptionalOldSelf: ptr.To(true),
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
			name:             "Ratcheting, For update, FG disabled, oldCRD OptionalOldSelf in use, but different from new, don't drop OptionalOldSelfs",
			enableRatcheting: false,
			crd: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "foos.sigs.k8s.io", Annotations: map[string]string{v1beta1.KubeAPIApprovedAnnotation: "valid"}, ResourceVersion: "1"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
							XValidations: apiextensions.ValidationRules{
								{
									Rule:            "size(self) > 0",
									Message:         "openAPIV3Schema should contain more than 0 element.",
									OptionalOldSelf: ptr.To(true),
								},
							},
							Properties: map[string]apiextensions.JSONSchemaProps{
								"subRule": {
									Type: "object",
									XValidations: apiextensions.ValidationRules{
										{
											Rule:            "isTest == true",
											Message:         "isTest should be true.",
											OptionalOldSelf: ptr.To(true),
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
							Properties: map[string]apiextensions.JSONSchemaProps{
								"subRule": {
									Type: "object",
									XValidations: apiextensions.ValidationRules{
										{
											Rule:            "isTest == true",
											Message:         "isTest should be true.",
											OptionalOldSelf: ptr.To(true),
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
									Rule:            "size(self) > 0",
									Message:         "openAPIV3Schema should contain more than 0 element.",
									OptionalOldSelf: ptr.To(true),
								},
							},
							Properties: map[string]apiextensions.JSONSchemaProps{
								"subRule": {
									Type: "object",
									XValidations: apiextensions.ValidationRules{
										{
											Rule:            "isTest == true",
											Message:         "isTest should be true.",
											OptionalOldSelf: ptr.To(true),
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
			name:             "Ratcheting, For update, FG disabled, oldCRD has no OptionalOldSelf, drop OptionalOldSelf",
			enableRatcheting: false,
			crd: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "foos.sigs.k8s.io", Annotations: map[string]string{v1beta1.KubeAPIApprovedAnnotation: "valid"}, ResourceVersion: "1"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
							XValidations: apiextensions.ValidationRules{
								{
									Rule:            "size(self) > 0",
									Message:         "openAPIV3Schema should contain more than 0 element.",
									OptionalOldSelf: ptr.To(true),
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
		{
			name:             "Ratcheting, For update, FG enabled, oldCRD has optionalOldSelf, updated to newCRD",
			enableRatcheting: true,
			crd: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "foos.sigs.k8s.io", Annotations: map[string]string{v1beta1.KubeAPIApprovedAnnotation: "valid"}, ResourceVersion: "1"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
							XValidations: apiextensions.ValidationRules{
								{
									Rule:            "size(self) > 0",
									Message:         "openAPIV3Schema should contain more than 0 element.",
									OptionalOldSelf: ptr.To(true),
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
									Rule:            "old data",
									Message:         "old data",
									OptionalOldSelf: ptr.To(true),
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
									Rule:            "size(self) > 0",
									Message:         "openAPIV3Schema should contain more than 0 element.",
									OptionalOldSelf: ptr.To(true),
								},
							},
						},
					},
				},
			},
		},
		{
			name:             "Ratcheting, For update, FG enabled, oldCRD has no OptionalOldSelf, updated to newCRD",
			enableRatcheting: true,
			crd: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "foos.sigs.k8s.io", Annotations: map[string]string{v1beta1.KubeAPIApprovedAnnotation: "valid"}, ResourceVersion: "1"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
							XValidations: apiextensions.ValidationRules{
								{
									Rule:            "size(self) > 0",
									Message:         "openAPIV3Schema should contain more than 0 element.",
									OptionalOldSelf: ptr.To(true),
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
									Rule:            "size(self) > 0",
									Message:         "openAPIV3Schema should contain more than 0 element.",
									OptionalOldSelf: ptr.To(true),
								},
							},
						},
					},
				},
			},
		},
		// SelectableFields
		{
			name:                   "SelectableFields, For create, FG disabled, SelectableFields in update, dropped",
			enableSelectableFields: false,
			crd: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "foos.sigs.k8s.io", Annotations: map[string]string{v1beta1.KubeAPIApprovedAnnotation: "valid"}, ResourceVersion: "1"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
							Properties: map[string]apiextensions.JSONSchemaProps{
								"field": {
									Type: "string",
								},
							},
						},
					},
					SelectableFields: []apiextensions.SelectableField{
						{
							JSONPath: ".field",
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
							Properties: map[string]apiextensions.JSONSchemaProps{
								"field": {
									Type: "string",
								},
							},
						},
					},
				},
			},
		},
		{
			name:                   "SelectableFields, For create, FG enabled, no SelectableFields in update, no drop",
			enableSelectableFields: true,
			crd: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "foos.sigs.k8s.io", Annotations: map[string]string{v1beta1.KubeAPIApprovedAnnotation: "valid"}, ResourceVersion: "1"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
							Properties: map[string]apiextensions.JSONSchemaProps{
								"field": {
									Type: "string",
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
							Properties: map[string]apiextensions.JSONSchemaProps{
								"field": {
									Type: "string",
								},
							},
						},
					},
				},
			},
		},
		{
			name:                   "SelectableFields, For create, FG enabled, SelectableFields in update, no drop",
			enableSelectableFields: true,
			crd: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "foos.sigs.k8s.io", Annotations: map[string]string{v1beta1.KubeAPIApprovedAnnotation: "valid"}, ResourceVersion: "1"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
							Properties: map[string]apiextensions.JSONSchemaProps{
								"field": {
									Type: "string",
								},
							},
						},
					},
					SelectableFields: []apiextensions.SelectableField{
						{
							JSONPath: ".field",
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
							Properties: map[string]apiextensions.JSONSchemaProps{
								"field": {
									Type: "string",
								},
							},
						},
					},
					SelectableFields: []apiextensions.SelectableField{
						{
							JSONPath: ".field",
						},
					},
				},
			},
		},
		{
			name:                   "SelectableFields, For update, FG disabled, oldCRD has SelectableFields, SelectableFields in update, no drop",
			enableSelectableFields: false,
			crd: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "foos.sigs.k8s.io", Annotations: map[string]string{v1beta1.KubeAPIApprovedAnnotation: "valid"}, ResourceVersion: "1"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
							Properties: map[string]apiextensions.JSONSchemaProps{
								"field1": {
									Type: "string",
								},
								"field2": {
									Type: "string",
								},
							},
						},
					},
					SelectableFields: []apiextensions.SelectableField{
						{
							JSONPath: ".field1",
						},
						{
							JSONPath: ".field2",
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
							Properties: map[string]apiextensions.JSONSchemaProps{
								"field1": {
									Type: "string",
								},
							},
						},
					},
					SelectableFields: []apiextensions.SelectableField{
						{
							JSONPath: ".field1",
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
							Properties: map[string]apiextensions.JSONSchemaProps{
								"field1": {
									Type: "string",
								},
								"field2": {
									Type: "string",
								},
							},
						},
					},
					SelectableFields: []apiextensions.SelectableField{
						{
							JSONPath: ".field1",
						},
						{
							JSONPath: ".field2",
						},
					},
				},
			},
		},
		{
			name:                   "SelectableFields, For update, FG disabled, oldCRD does not have SelectableFields, no SelectableFields in update, no drop",
			enableSelectableFields: false,
			crd: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "foos.sigs.k8s.io", Annotations: map[string]string{v1beta1.KubeAPIApprovedAnnotation: "valid"}, ResourceVersion: "1"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
							Properties: map[string]apiextensions.JSONSchemaProps{
								"field1": {
									Type: "string",
								},
								"field2": {
									Type: "string",
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
							Properties: map[string]apiextensions.JSONSchemaProps{
								"field1": {
									Type: "string",
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
							Properties: map[string]apiextensions.JSONSchemaProps{
								"field1": {
									Type: "string",
								},
								"field2": {
									Type: "string",
								},
							},
						},
					},
				},
			},
		},
		{
			name:                   "SelectableFields, For update, FG disabled, oldCRD does not have SelectableFields, SelectableFields in update, dropped",
			enableSelectableFields: false,
			crd: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "foos.sigs.k8s.io", Annotations: map[string]string{v1beta1.KubeAPIApprovedAnnotation: "valid"}, ResourceVersion: "1"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
							Properties: map[string]apiextensions.JSONSchemaProps{
								"field1": {
									Type: "string",
								},
								"field2": {
									Type: "string",
								},
							},
						},
					},
					SelectableFields: []apiextensions.SelectableField{
						{
							JSONPath: ".field1",
						},
						{
							JSONPath: ".field2",
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
							Properties: map[string]apiextensions.JSONSchemaProps{
								"field1": {
									Type: "string",
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
							Properties: map[string]apiextensions.JSONSchemaProps{
								"field1": {
									Type: "string",
								},
								"field2": {
									Type: "string",
								},
							},
						},
					},
				},
			},
		},
		{
			name:                   "SelectableFields, For update, FG enabled, oldCRD has SelectableFields, SelectableFields in update, no drop",
			enableSelectableFields: true,
			crd: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "foos.sigs.k8s.io", Annotations: map[string]string{v1beta1.KubeAPIApprovedAnnotation: "valid"}, ResourceVersion: "1"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
							Properties: map[string]apiextensions.JSONSchemaProps{
								"field1": {
									Type: "string",
								},
								"field2": {
									Type: "string",
								},
							},
						},
					},
					SelectableFields: []apiextensions.SelectableField{
						{
							JSONPath: ".field1",
						},
						{
							JSONPath: ".field2",
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
							Properties: map[string]apiextensions.JSONSchemaProps{
								"field1": {
									Type: "string",
								},
							},
						},
					},
					SelectableFields: []apiextensions.SelectableField{
						{
							JSONPath: ".field1",
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
							Properties: map[string]apiextensions.JSONSchemaProps{
								"field1": {
									Type: "string",
								},
								"field2": {
									Type: "string",
								},
							},
						},
					},
					SelectableFields: []apiextensions.SelectableField{
						{
							JSONPath: ".field1",
						},
						{
							JSONPath: ".field2",
						},
					},
				},
			},
		},
		{
			name:                   "SelectableFields, For update, FG enabled, oldCRD does not have SelectableFields, SelectableFields in update, no drop",
			enableSelectableFields: true,
			crd: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "foos.sigs.k8s.io", Annotations: map[string]string{v1beta1.KubeAPIApprovedAnnotation: "valid"}, ResourceVersion: "1"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
							Properties: map[string]apiextensions.JSONSchemaProps{
								"field1": {
									Type: "string",
								},
								"field2": {
									Type: "string",
								},
							},
						},
					},
					SelectableFields: []apiextensions.SelectableField{
						{
							JSONPath: ".field1",
						},
						{
							JSONPath: ".field2",
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
							Properties: map[string]apiextensions.JSONSchemaProps{
								"field1": {
									Type: "string",
								},
							},
						},
					},
					SelectableFields: []apiextensions.SelectableField{
						{
							JSONPath: ".field1",
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
							Properties: map[string]apiextensions.JSONSchemaProps{
								"field1": {
									Type: "string",
								},
								"field2": {
									Type: "string",
								},
							},
						},
					},
					SelectableFields: []apiextensions.SelectableField{
						{
							JSONPath: ".field1",
						},
						{
							JSONPath: ".field2",
						},
					},
				},
			},
		},
		{
			name:                   "pre-version SelectableFields, For update, FG disabled, oldCRD does not have SelectableFields, SelectableFields in update, dropped",
			enableSelectableFields: false,
			crd: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "foos.sigs.k8s.io", Annotations: map[string]string{v1beta1.KubeAPIApprovedAnnotation: "valid"}, ResourceVersion: "1"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{
							Name: "v1",
							Schema: &apiextensions.CustomResourceValidation{
								OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
									Type: "object",
									Properties: map[string]apiextensions.JSONSchemaProps{
										"field1": {
											Type: "string",
										},
										"field2": {
											Type: "string",
										},
									},
								},
							},
							SelectableFields: []apiextensions.SelectableField{
								{
									JSONPath: ".field1",
								},
								{
									JSONPath: ".field2",
								},
							},
						},
						{
							Name: "v2",
							Schema: &apiextensions.CustomResourceValidation{
								OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
									Type: "object",
									Properties: map[string]apiextensions.JSONSchemaProps{
										"field3": {
											Type: "string",
										},
										"field4": {
											Type: "string",
										},
									},
								},
							},
							SelectableFields: []apiextensions.SelectableField{
								{
									JSONPath: ".field3",
								},
								{
									JSONPath: ".field4",
								},
							},
						},
					},
				},
			},
			oldCRD: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "foos.sigs.k8s.io", Annotations: map[string]string{v1beta1.KubeAPIApprovedAnnotation: "valid"}, ResourceVersion: "1"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{
							Name: "v1",
							Schema: &apiextensions.CustomResourceValidation{
								OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
									Type: "object",
									Properties: map[string]apiextensions.JSONSchemaProps{
										"field1": {
											Type: "string",
										},
										"field2": {
											Type: "string",
										},
									},
								},
							},
						},
						{
							Name: "v2",
							Schema: &apiextensions.CustomResourceValidation{
								OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
									Type: "object",
									Properties: map[string]apiextensions.JSONSchemaProps{
										"field3": {
											Type: "string",
										},
										"field4": {
											Type: "string",
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
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{
							Name: "v1",
							Schema: &apiextensions.CustomResourceValidation{
								OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
									Type: "object",
									Properties: map[string]apiextensions.JSONSchemaProps{
										"field1": {
											Type: "string",
										},
										"field2": {
											Type: "string",
										},
									},
								},
							},
						},
						{
							Name: "v2",
							Schema: &apiextensions.CustomResourceValidation{
								OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
									Type: "object",
									Properties: map[string]apiextensions.JSONSchemaProps{
										"field3": {
											Type: "string",
										},
										"field4": {
											Type: "string",
										},
									},
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
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, apiextensionsfeatures.CRDValidationRatcheting, tc.enableRatcheting)()
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, apiextensionsfeatures.CustomResourceFieldSelectors, tc.enableSelectableFields)()
			old := tc.oldCRD.DeepCopy()

			dropDisabledFields(tc.crd, tc.oldCRD)

			// old crd should never be changed
			if diff := cmp.Diff(tc.oldCRD, old); diff != "" {
				t.Fatalf("old crd changed from %v to %v\n%v", tc.oldCRD, old, diff)
			}

			if diff := cmp.Diff(tc.expectedCRD, tc.crd); diff != "" {
				t.Fatalf("unexpected crd: %v\n%v", tc.crd, diff)
			}
		})
	}
}
