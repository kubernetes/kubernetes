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

package volumeattributesclass

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	storage "k8s.io/kubernetes/pkg/apis/storage"
	registry "k8s.io/kubernetes/pkg/registry/storage/volumeattributesclass"
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
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIPrefix:         "apis",
		APIGroup:          "storage.k8s.io",
		APIVersion:        apiVersion,
		Resource:          "volumeattributesclasses",
		IsResourceRequest: true,
		Verb:              "create",
	})
	testCases := map[string]struct {
		input        storage.VolumeAttributesClass
		expectedErrs field.ErrorList
	}{
		"valid": {
			input: mkVolumeAttributesClass(),
		},
		"Driver name cannot be empty": {
			input: mkVolumeAttributesClass(func(obj *storage.VolumeAttributesClass) {
				obj.DriverName = ""
			}),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("driverName"), "").MarkAlpha(),
			},
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			apitesting.VerifyValidationEquivalence(t, ctx, &tc.input, registry.Strategy, tc.expectedErrs)
		})
	}
	obj := mkVolumeAttributesClass()
	meta.RunObjectMetaTestCases(t, ctx, &obj, registry.Strategy, meta.WithStringentFinalizerValidation())
}

func testDeclarativeValidateUpdate(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIPrefix:         "apis",
		APIGroup:          "storage.k8s.io",
		APIVersion:        apiVersion,
		Resource:          "volumeattributesclasses",
		Name:              "valid-obj",
		IsResourceRequest: true,
		Verb:              "update",
	})

	testCases := map[string]struct {
		oldInput     storage.VolumeAttributesClass
		newInput     storage.VolumeAttributesClass
		expectedErrs field.ErrorList
	}{
		"valid": {
			oldInput: mkVolumeAttributesClass(),
			newInput: mkVolumeAttributesClass(),
		},
		"driver name cannot be updated": {
			oldInput: mkVolumeAttributesClass(),
			newInput: mkVolumeAttributesClass(func(obj *storage.VolumeAttributesClass) {
				obj.DriverName = "ebs.update.csi.aws.com"
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("driverName"), nil, "field is immutable").WithOrigin("immutable").MarkAlpha(),
			},
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			tc.oldInput.ObjectMeta.ResourceVersion = "1"
			tc.newInput.ObjectMeta.ResourceVersion = "2"
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.newInput, &tc.oldInput, registry.Strategy, tc.expectedErrs)
		})
	}

	updateObj := mkVolumeAttributesClass()
	meta.RunObjectMetaUpdateTestCases(t, ctx, &updateObj, registry.Strategy, meta.WithStringentFinalizerValidation())
}

func mkVolumeAttributesClass(tweaks ...func(class *storage.VolumeAttributesClass)) storage.VolumeAttributesClass {
	class := storage.VolumeAttributesClass{
		Name:       "valid-obj",
		DriverName: "ebs.csi.aws.com",
		Parameters: map[string]string{"foo": "bar"},
	}
	for _, tweak := range tweaks {
		tweak(&class)
	}
	return class
}
