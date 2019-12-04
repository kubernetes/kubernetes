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
	"fmt"
	"reflect"
	"testing"

	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/validation"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/utils/pointer"
)

func TestDropDisableFieldsCustomResourceDefinition(t *testing.T) {
	t.Log("testing unversioned validation..")
	crdWithUnversionedValidation := func() *apiextensions.CustomResourceDefinition {
		// crd with non-versioned validation
		return &apiextensions.CustomResourceDefinition{
			Spec: apiextensions.CustomResourceDefinitionSpec{
				Validation: &apiextensions.CustomResourceValidation{
					OpenAPIV3Schema: &apiextensions.JSONSchemaProps{},
				},
			},
		}
	}
	crdWithoutUnversionedValidation := func() *apiextensions.CustomResourceDefinition {
		// crd with non-versioned validation
		return &apiextensions.CustomResourceDefinition{
			Spec: apiextensions.CustomResourceDefinitionSpec{},
		}
	}
	crdInfos := []struct {
		name string
		crd  func() *apiextensions.CustomResourceDefinition
	}{
		{
			name: "has unversioned validation",
			crd:  crdWithUnversionedValidation,
		},
		{
			name: "doesn't have unversioned validation",
			crd:  crdWithoutUnversionedValidation,
		},
		{
			name: "nil",
			crd:  func() *apiextensions.CustomResourceDefinition { return nil },
		},
	}
	for _, oldCRDInfo := range crdInfos {
		for _, newCRDInfo := range crdInfos {
			oldCRD := oldCRDInfo.crd()
			newCRD := newCRDInfo.crd()
			if newCRD == nil {
				continue
			}
			t.Run(fmt.Sprintf("old CRD %v, new CRD %v", oldCRDInfo.name, newCRDInfo.name),
				func(t *testing.T) {
					var oldCRDSpec *apiextensions.CustomResourceDefinitionSpec
					if oldCRD != nil {
						oldCRDSpec = &oldCRD.Spec
					}
					dropDisabledFields(&newCRD.Spec, oldCRDSpec)
					// old CRD should never be changed
					if !reflect.DeepEqual(oldCRD, oldCRDInfo.crd()) {
						t.Errorf("old crd changed: %v", diff.ObjectReflectDiff(oldCRD, oldCRDInfo.crd()))
					}
					if !reflect.DeepEqual(newCRD, newCRDInfo.crd()) {
						t.Errorf("new crd changed: %v", diff.ObjectReflectDiff(newCRD, newCRDInfo.crd()))
					}
				},
			)
		}
	}

	t.Log("testing unversioned subresources...")
	crdWithUnversionedSubresources := func() *apiextensions.CustomResourceDefinition {
		// crd with unversioned subresources
		return &apiextensions.CustomResourceDefinition{
			Spec: apiextensions.CustomResourceDefinitionSpec{
				Subresources: &apiextensions.CustomResourceSubresources{},
			},
		}
	}
	crdWithoutUnversionedSubresources := func() *apiextensions.CustomResourceDefinition {
		// crd without unversioned subresources
		return &apiextensions.CustomResourceDefinition{
			Spec: apiextensions.CustomResourceDefinitionSpec{},
		}
	}
	crdInfos = []struct {
		name string
		crd  func() *apiextensions.CustomResourceDefinition
	}{
		{
			name: "has unversioned subresources",
			crd:  crdWithUnversionedSubresources,
		},
		{
			name: "doesn't have unversioned subresources",
			crd:  crdWithoutUnversionedSubresources,
		},
		{
			name: "nil",
			crd:  func() *apiextensions.CustomResourceDefinition { return nil },
		},
	}
	for _, oldCRDInfo := range crdInfos {
		for _, newCRDInfo := range crdInfos {
			oldCRD := oldCRDInfo.crd()
			newCRD := newCRDInfo.crd()
			if newCRD == nil {
				continue
			}
			t.Run(fmt.Sprintf("old CRD %v, new CRD %v", oldCRDInfo.name, newCRDInfo.name),
				func(t *testing.T) {
					var oldCRDSpec *apiextensions.CustomResourceDefinitionSpec
					if oldCRD != nil {
						oldCRDSpec = &oldCRD.Spec
					}
					dropDisabledFields(&newCRD.Spec, oldCRDSpec)
					// old CRD should never be changed
					if !reflect.DeepEqual(oldCRD, oldCRDInfo.crd()) {
						t.Errorf("old crd changed: %v", diff.ObjectReflectDiff(oldCRD, oldCRDInfo.crd()))
					}
					if !reflect.DeepEqual(newCRD, newCRDInfo.crd()) {
						t.Errorf("new crd changed: %v", diff.ObjectReflectDiff(newCRD, newCRDInfo.crd()))
					}
				},
			)
		}
	}

	t.Log("testing versioned validation..")
	crdWithVersionedValidation := func() *apiextensions.CustomResourceDefinition {
		// crd with versioned validation
		return &apiextensions.CustomResourceDefinition{
			Spec: apiextensions.CustomResourceDefinitionSpec{
				Versions: []apiextensions.CustomResourceDefinitionVersion{
					{
						Name: "v1",
						Schema: &apiextensions.CustomResourceValidation{
							OpenAPIV3Schema: &apiextensions.JSONSchemaProps{},
						},
					},
				},
			},
		}
	}
	crdWithoutVersionedValidation := func() *apiextensions.CustomResourceDefinition {
		// crd with versioned validation
		return &apiextensions.CustomResourceDefinition{
			Spec: apiextensions.CustomResourceDefinitionSpec{
				Versions: []apiextensions.CustomResourceDefinitionVersion{
					{
						Name: "v1",
					},
				},
			},
		}
	}
	crdInfos = []struct {
		name string
		crd  func() *apiextensions.CustomResourceDefinition
	}{
		{
			name: "has versioned validation",
			crd:  crdWithVersionedValidation,
		},
		{
			name: "doesn't have versioned validation",
			crd:  crdWithoutVersionedValidation,
		},
		{
			name: "nil",
			crd:  func() *apiextensions.CustomResourceDefinition { return nil },
		},
	}
	for _, oldCRDInfo := range crdInfos {
		for _, newCRDInfo := range crdInfos {
			oldCRD := oldCRDInfo.crd()
			newCRD := newCRDInfo.crd()
			if newCRD == nil {
				continue
			}
			t.Run(fmt.Sprintf("old CRD %v, new CRD %v", oldCRDInfo.name, newCRDInfo.name),
				func(t *testing.T) {
					var oldCRDSpec *apiextensions.CustomResourceDefinitionSpec
					if oldCRD != nil {
						oldCRDSpec = &oldCRD.Spec
					}
					dropDisabledFields(&newCRD.Spec, oldCRDSpec)
					// old CRD should never be changed
					if !reflect.DeepEqual(oldCRD, oldCRDInfo.crd()) {
						t.Errorf("old crd changed: %v", diff.ObjectReflectDiff(oldCRD, oldCRDInfo.crd()))
					}
					if !reflect.DeepEqual(newCRD, newCRDInfo.crd()) {
						t.Errorf("new crd changed: %v", diff.ObjectReflectDiff(newCRD, newCRDInfo.crd()))
					}
				},
			)
		}
	}

	t.Log("testing versioned subresources w/ conversion enabled..")
	crdWithVersionedSubresources := func() *apiextensions.CustomResourceDefinition {
		// crd with versioned subresources
		return &apiextensions.CustomResourceDefinition{
			Spec: apiextensions.CustomResourceDefinitionSpec{
				Versions: []apiextensions.CustomResourceDefinitionVersion{
					{
						Name:         "v1",
						Subresources: &apiextensions.CustomResourceSubresources{},
					},
				},
			},
		}
	}
	crdWithoutVersionedSubresources := func() *apiextensions.CustomResourceDefinition {
		// crd without versioned subresources
		return &apiextensions.CustomResourceDefinition{
			Spec: apiextensions.CustomResourceDefinitionSpec{
				Versions: []apiextensions.CustomResourceDefinitionVersion{
					{
						Name: "v1",
					},
				},
			},
		}
	}
	crdInfos = []struct {
		name string
		crd  func() *apiextensions.CustomResourceDefinition
	}{
		{
			name: "has versioned subresources",
			crd:  crdWithVersionedSubresources,
		},
		{
			name: "doesn't have versioned subresources",
			crd:  crdWithoutVersionedSubresources,
		},
		{
			name: "nil",
			crd:  func() *apiextensions.CustomResourceDefinition { return nil },
		},
	}
	for _, oldCRDInfo := range crdInfos {
		for _, newCRDInfo := range crdInfos {
			oldCRD := oldCRDInfo.crd()
			newCRD := newCRDInfo.crd()
			if newCRD == nil {
				continue
			}
			t.Run(fmt.Sprintf("old CRD %v, new CRD %v", oldCRDInfo.name, newCRDInfo.name),
				func(t *testing.T) {
					var oldCRDSpec *apiextensions.CustomResourceDefinitionSpec
					if oldCRD != nil {
						oldCRDSpec = &oldCRD.Spec
					}
					dropDisabledFields(&newCRD.Spec, oldCRDSpec)
					// old CRD should never be changed
					if !reflect.DeepEqual(oldCRD, oldCRDInfo.crd()) {
						t.Errorf("old crd changed: %v", diff.ObjectReflectDiff(oldCRD, oldCRDInfo.crd()))
					}
					if !reflect.DeepEqual(newCRD, newCRDInfo.crd()) {
						t.Errorf("new crd changed: %v", diff.ObjectReflectDiff(newCRD, newCRDInfo.crd()))
					}
				},
			)
		}
	}

	t.Log("testing conversion webhook..")
	crdWithUnversionedConversionWebhook := func() *apiextensions.CustomResourceDefinition {
		// crd with conversion webhook
		return &apiextensions.CustomResourceDefinition{
			Spec: apiextensions.CustomResourceDefinitionSpec{
				Conversion: &apiextensions.CustomResourceConversion{
					WebhookClientConfig: &apiextensions.WebhookClientConfig{},
				},
			},
		}
	}
	crdWithoutUnversionedConversionWebhook := func() *apiextensions.CustomResourceDefinition {
		// crd with conversion webhook
		return &apiextensions.CustomResourceDefinition{
			Spec: apiextensions.CustomResourceDefinitionSpec{
				Conversion: &apiextensions.CustomResourceConversion{},
			},
		}
	}
	crdInfos = []struct {
		name string
		crd  func() *apiextensions.CustomResourceDefinition
	}{
		{
			name: "has conversion webhook",
			crd:  crdWithUnversionedConversionWebhook,
		},
		{
			name: "doesn't have conversion webhook",
			crd:  crdWithoutUnversionedConversionWebhook,
		},
		{
			name: "nil",
			crd:  func() *apiextensions.CustomResourceDefinition { return nil },
		},
	}
	for _, oldCRDInfo := range crdInfos {
		for _, newCRDInfo := range crdInfos {
			oldCRD := oldCRDInfo.crd()
			newCRD := newCRDInfo.crd()
			if newCRD == nil {
				continue
			}
			t.Run(fmt.Sprintf("old CRD %v, new CRD %v", oldCRDInfo.name, newCRDInfo.name),
				func(t *testing.T) {
					var oldCRDSpec *apiextensions.CustomResourceDefinitionSpec
					if oldCRD != nil {
						oldCRDSpec = &oldCRD.Spec
					}
					dropDisabledFields(&newCRD.Spec, oldCRDSpec)
					// old CRD should never be changed
					if !reflect.DeepEqual(oldCRD, oldCRDInfo.crd()) {
						t.Errorf("old crd changed: %v", diff.ObjectReflectDiff(oldCRD, oldCRDInfo.crd()))
					}
					if !reflect.DeepEqual(newCRD, newCRDInfo.crd()) {
						t.Errorf("new crd changed: %v", diff.ObjectReflectDiff(newCRD, newCRDInfo.crd()))
					}
				},
			)
		}
	}
}

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

		version            string
		group              string
		annotationValue    string
		oldAnnotationValue *string
		validateError      func(t *testing.T, errors field.ErrorList)
	}{
		{
			name:            "ignore v1beta1",
			version:         "v1beta1",
			group:           "sigs.k8s.io",
			annotationValue: "invalid",
			validateError:   okFn,
		},
		{
			name:            "ignore non-k8s group",
			version:         "v1",
			group:           "other.io",
			annotationValue: "invalid",
			validateError:   okFn,
		},
		{
			name:            "invalid annotation create",
			version:         "v1",
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
			version:            "v1",
			group:              "sigs.k8s.io",
			annotationValue:    "invalid",
			oldAnnotationValue: strPtr("invalid"),
			validateError:      okFn,
		},
		{
			name:               "invalid annotation to missing",
			version:            "v1",
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
			version:            "v1",
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
			version:         "v1",
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
			version:            "v1",
			group:              "sigs.k8s.io",
			annotationValue:    "",
			oldAnnotationValue: strPtr(""),
			validateError:      okFn,
		},
		{
			name:            "url",
			version:         "v1",
			group:           "sigs.k8s.io",
			annotationValue: "https://github.com/kubernetes/kubernetes/pull/79724",
			validateError:   okFn,
		},
		{
			name:            "unapproved",
			version:         "v1",
			group:           "sigs.k8s.io",
			annotationValue: "unapproved, other reason",
			validateError:   okFn,
		},
		{
			name:            "next version validates",
			version:         "v2",
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
				actual = validation.ValidateCustomResourceDefinition(crd, schema.GroupVersion{Group: "apiextensions.k8s.io", Version: test.version})
			} else {
				actual = validation.ValidateCustomResourceDefinitionUpdate(crd, oldCRD, schema.GroupVersion{Group: "apiextensions.k8s.io", Version: test.version})
			}
			test.validateError(t, actual)
		})
	}
}
