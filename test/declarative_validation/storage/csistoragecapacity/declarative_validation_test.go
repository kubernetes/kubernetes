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

package csistoragecapacity

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	storage "k8s.io/kubernetes/pkg/apis/storage"
	registry "k8s.io/kubernetes/pkg/registry/storage/csistoragecapacity"
	"k8s.io/kubernetes/test/declarative_validation/meta"
)

func TestDeclarativeValidate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			testDeclarativeValidate(t, apiVersion)
		})
	}
}

func TestDeclarativeValidateUpdate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			testDeclarativeValidateUpdate(t, apiVersion)
		})
	}
}

func testDeclarativeValidate(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithNamespace(genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIPrefix:         "apis",
		APIGroup:          "storage.k8s.io",
		APIVersion:        apiVersion,
		Resource:          "csistoragecapacities",
		IsResourceRequest: true,
		Verb:              "create",
	}), metav1.NamespaceDefault)

	testCases := map[string]struct {
		input        storage.CSIStorageCapacity
		expectedErrs field.ErrorList
	}{
		"valid": {
			input: mkCSIStorageCapacity(),
		},
		"storage class name is required": {
			input: mkCSIStorageCapacity(func(obj *storage.CSIStorageCapacity) {
				obj.StorageClassName = ""
			}),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("storageClassName"), "").MarkAlpha(),
			},
		},
		// TODO: Add more test cases
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			apitesting.VerifyValidationEquivalence(t, ctx, &tc.input, registry.Strategy, tc.expectedErrs)
		})
	}

	obj := mkCSIStorageCapacity()
	meta.RunObjectMetaTestCases(t, ctx, &obj, registry.Strategy, meta.WithStringentFinalizerValidation())
}

func testDeclarativeValidateUpdate(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithNamespace(genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIPrefix:         "apis",
		APIGroup:          "storage.k8s.io",
		APIVersion:        apiVersion,
		Resource:          "csistoragecapacities",
		Name:              "valid-obj",
		IsResourceRequest: true,
		Verb:              "update",
	}), metav1.NamespaceDefault)

	testCases := map[string]struct {
		old, update  storage.CSIStorageCapacity
		expectedErrs field.ErrorList
	}{
		"valid": {
			old:    mkCSIStorageCapacity(),
			update: mkCSIStorageCapacity(),
		},
		"storage class name is changed": {
			old: mkCSIStorageCapacity(),
			update: mkCSIStorageCapacity(func(obj *storage.CSIStorageCapacity) {
				obj.StorageClassName = "other.example.com"
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("storageClassName"), "other.example.com", "").WithOrigin("immutable").MarkAlpha(),
			},
		},
		"storage class name is unset": {
			old: mkCSIStorageCapacity(),
			update: mkCSIStorageCapacity(func(obj *storage.CSIStorageCapacity) {
				obj.StorageClassName = ""
			}),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("storageClassName"), "").MarkAlpha(),
				field.Invalid(field.NewPath("storageClassName"), "", "").WithOrigin("immutable").MarkAlpha(),
			},
		},
		"storage class name is set from unset": {
			old: mkCSIStorageCapacity(func(obj *storage.CSIStorageCapacity) {
				obj.StorageClassName = ""
			}),
			update: mkCSIStorageCapacity(),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("storageClassName"), "example.com", "").WithOrigin("immutable").MarkAlpha(),
			},
		},
		// TODO: Add more test cases
	}

	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			tc.old.ResourceVersion = "1"
			tc.update.ResourceVersion = "2"
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.update, &tc.old, registry.Strategy, tc.expectedErrs)
		})
	}

	updateObj := mkCSIStorageCapacity()
	meta.RunObjectMetaUpdateTestCases(t, ctx, &updateObj, registry.Strategy, meta.WithStringentFinalizerValidation())
}

func mkCSIStorageCapacity(tweaks ...func(capacity *storage.CSIStorageCapacity)) storage.CSIStorageCapacity {
	capacity := storage.CSIStorageCapacity{
		Name:             "valid-obj",
		Namespace:        metav1.NamespaceDefault,
		StorageClassName: "example.com",
	}
	for _, tweak := range tweaks {
		tweak(&capacity)
	}
	return capacity
}
