/*
Copyright 2025 The Kubernetes Authors.

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

package volumeattachment

import (
	"strings"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	storage "k8s.io/kubernetes/pkg/apis/storage"
)

func TestDeclarativeValidate(t *testing.T) {
	apiVersions := []string{"v1", "v1alpha1", "v1beta1"}
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			testDeclarativeValidate(t, apiVersion)
		})
	}
}

func TestDeclarativeValidateUpdate(t *testing.T) {
	apiVersions := []string{"v1", "v1alpha1", "v1beta1"}
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			testDeclarativeValidateUpdate(t, apiVersion)
		})
	}
}

func testDeclarativeValidate(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:          "storage.k8s.io",
		APIVersion:        apiVersion,
		Resource:          "volumeattachments",
		IsResourceRequest: true,
		Verb:              "create",
	})

	testCases := map[string]struct {
		input        storage.VolumeAttachment
		expectedErrs field.ErrorList
	}{
		"valid": {
			input: mkValidVolumeAttachment(),
		},
		"attacher with upper characters": {
			input: mkValidVolumeAttachment(TweakAttacher("AAAAAAAA")),
		},
		"attacher with 63 characters": {
			input: mkValidVolumeAttachment(TweakAttacher(strings.Repeat("a", 63))),
		},
		"invalid attacher (required)": {
			input: mkValidVolumeAttachment(TweakAttacher("")),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("spec", "attacher"), ""),
			},
		},
		"attacher with special characters": {
			input: mkValidVolumeAttachment(TweakAttacher("asdadasd&@!")),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "attacher"), "", "").WithOrigin("format=k8s-long-name-caseless"),
			},
		},
		"attacher with number of characters exceeds 63": {
			input: mkValidVolumeAttachment(TweakAttacher(strings.Repeat("a", 64))),
			expectedErrs: field.ErrorList{
				field.TooLong(field.NewPath("spec", "attacher"), strings.Repeat("a", 64), 63).WithOrigin("maxLength"),
			},
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			apitesting.VerifyValidationEquivalence(t, ctx, &tc.input, Strategy.Validate, tc.expectedErrs)
		})
	}
}

func testDeclarativeValidateUpdate(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:          "storage.k8s.io",
		APIVersion:        apiVersion,
		Resource:          "volumeattachments",
		IsResourceRequest: true,
		Verb:              "update",
	})

	testCases := map[string]struct {
		oldInput     storage.VolumeAttachment
		newInput     storage.VolumeAttachment
		expectedErrs field.ErrorList
	}{
		"valid update": {
			oldInput: mkValidVolumeAttachment(),
			newInput: mkValidVolumeAttachment(),
		},
		"immutable spec": {
			oldInput: mkValidVolumeAttachment(),
			newInput: mkValidVolumeAttachment(TweakAttacher("different.com")),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec"), nil, "field is immutable").WithOrigin("immutable"),
			},
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.newInput, &tc.oldInput, Strategy.ValidateUpdate, tc.expectedErrs)
		})
	}
}

func TweakAttacher(attacher string) func(obj *storage.VolumeAttachment) {
	return func(obj *storage.VolumeAttachment) {
		obj.Spec.Attacher = attacher
	}
}

func mkValidVolumeAttachment(tweaks ...func(obj *storage.VolumeAttachment)) storage.VolumeAttachment {
	pvName := "pv-001"
	obj := storage.VolumeAttachment{
		ObjectMeta: metav1.ObjectMeta{
			Name: "valid-volume-attachment",
		},
		Spec: storage.VolumeAttachmentSpec{
			Attacher: "example.com",
			Source: storage.VolumeAttachmentSource{
				PersistentVolumeName: &pvName,
			},
			NodeName: "node-1",
		},
	}
	for _, tweak := range tweaks {
		tweak(&obj)
	}
	return obj
}