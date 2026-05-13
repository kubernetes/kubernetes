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
	"context"
	"strings"
	"testing"

	corev1 "k8s.io/api/core/v1"
	apistoragev1 "k8s.io/api/storage/v1"
	apistoragev1alpha1 "k8s.io/api/storage/v1alpha1"
	apistoragev1beta1 "k8s.io/api/storage/v1beta1"
	"k8s.io/apimachinery/pkg/api/operation"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/test/coverage"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/rest"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	storage "k8s.io/kubernetes/pkg/apis/storage"
	validationstoragev1 "k8s.io/kubernetes/pkg/apis/storage/v1"
	validationstoragev1alpha1 "k8s.io/kubernetes/pkg/apis/storage/v1alpha1"
	validationstoragev1beta1 "k8s.io/kubernetes/pkg/apis/storage/v1beta1"
	registry "k8s.io/kubernetes/pkg/registry/storage/volumeattachment"
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

func TestDeclarativeValidateInlineVolumeSpecVolumeMode(t *testing.T) {
	blockMode := corev1.PersistentVolumeBlock
	filesystemMode := corev1.PersistentVolumeFilesystem
	ctx := rest.WithAllDeclarativeEnforcedForTest(context.Background())
	op := operation.Operation{Type: operation.Update}
	fldPath := field.NewPath("spec", "source")

	t.Run("v1", func(t *testing.T) {
		oldObj := &apistoragev1.VolumeAttachmentSource{InlineVolumeSpec: mkVersionedInlineVolumeSpec(&filesystemMode)}
		newObj := &apistoragev1.VolumeAttachmentSource{InlineVolumeSpec: mkVersionedInlineVolumeSpec(&blockMode)}
		assertInlineVolumeSpecVolumeMode(t, "v1", validationstoragev1.Validate_VolumeAttachmentSource(ctx, op, fldPath, newObj, oldObj))
	})
	t.Run("v1alpha1", func(t *testing.T) {
		oldObj := &apistoragev1alpha1.VolumeAttachmentSource{InlineVolumeSpec: mkVersionedInlineVolumeSpec(&filesystemMode)}
		newObj := &apistoragev1alpha1.VolumeAttachmentSource{InlineVolumeSpec: mkVersionedInlineVolumeSpec(&blockMode)}
		assertInlineVolumeSpecVolumeMode(t, "v1alpha1", validationstoragev1alpha1.Validate_VolumeAttachmentSource(ctx, op, fldPath, newObj, oldObj))
	})
	t.Run("v1beta1", func(t *testing.T) {
		oldObj := &apistoragev1beta1.VolumeAttachmentSource{InlineVolumeSpec: mkVersionedInlineVolumeSpec(&filesystemMode)}
		newObj := &apistoragev1beta1.VolumeAttachmentSource{InlineVolumeSpec: mkVersionedInlineVolumeSpec(&blockMode)}
		assertInlineVolumeSpecVolumeMode(t, "v1beta1", validationstoragev1beta1.Validate_VolumeAttachmentSource(ctx, op, fldPath, newObj, oldObj))
	})
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
				field.Required(field.NewPath("spec", "attacher"), "").MarkAlpha(),
			},
		},
		"attacher with special characters": {
			input: mkValidVolumeAttachment(TweakAttacher("asdadasd&@!")),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "attacher"), "", "").WithOrigin("format=k8s-long-name-caseless").MarkAlpha(),
			},
		},
		"attacher with number of characters exceeds 63": {
			input: mkValidVolumeAttachment(TweakAttacher(strings.Repeat("a", 64))),
			expectedErrs: field.ErrorList{
				field.TooLong(field.NewPath("spec", "attacher"), strings.Repeat("a", 64), 63).WithOrigin("maxLength").MarkAlpha(),
			},
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			apitesting.VerifyValidationEquivalence(t, ctx, &tc.input, registry.Strategy, tc.expectedErrs)
		})
	}

	obj := mkValidVolumeAttachment()
	meta.RunObjectMetaTestCases(t, ctx, &obj, registry.Strategy, meta.WithStringentFinalizerValidation())
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
		"immutable spec.attacher": {
			oldInput: mkValidVolumeAttachment(),
			newInput: mkValidVolumeAttachment(TweakAttacher("different.com")),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec"), nil, "field is immutable").WithOrigin("immutable").MarkAlpha(),
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
	updateObj := mkValidVolumeAttachment()
	meta.RunObjectMetaUpdateTestCases(t, ctx, &updateObj, registry.Strategy, meta.WithStringentFinalizerValidation())
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

func mkVersionedInlineVolumeSpec(volumeMode *corev1.PersistentVolumeMode) *corev1.PersistentVolumeSpec {
	return &corev1.PersistentVolumeSpec{
		AccessModes: []corev1.PersistentVolumeAccessMode{corev1.ReadWriteOnce},
		PersistentVolumeSource: corev1.PersistentVolumeSource{
			CSI: &corev1.CSIPersistentVolumeSource{
				Driver:       "com.test.foo",
				VolumeHandle: "valid-volume",
			},
		},
		VolumeMode: volumeMode,
	}
}

func assertInlineVolumeSpecVolumeMode(t *testing.T, apiVersion string, errs field.ErrorList) {
	t.Helper()
	expectedErrs := field.ErrorList{
		field.Invalid(field.NewPath("spec", "source", "inlineVolumeSpec", "volumeMode"), nil, "").WithOrigin("immutable").MarkAlpha(),
	}
	field.ErrorMatcher{}.ByType().ByOrigin().ByField().ByValidationStabilityLevel().BySource().Test(t, expectedErrs, errs)
	coverage.RecordObservedRules(schema.GroupVersionKind{
		Group:   "storage.k8s.io",
		Version: apiVersion,
		Kind:    "VolumeAttachment",
	}, errs)
}
