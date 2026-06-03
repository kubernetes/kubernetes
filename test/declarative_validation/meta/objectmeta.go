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

package meta

import (
	"context"
	"testing"

	apimeta "k8s.io/apimachinery/pkg/api/meta"
	apivalidation "k8s.io/apimachinery/pkg/api/validation"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/registry/rest"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
)

func RunObjectMetaTestCases[T runtime.Object](t *testing.T, ctx context.Context, baseObj T, strategy rest.RESTCreateStrategy, options ...apitesting.ValidationTestConfig) {
	t.Helper()
	fldPath := field.NewPath("metadata")

	testCases := []struct {
		Name         string
		Modify       func(metav1.Object)
		ExpectedErrs field.ErrorList
	}{
		{
			Name: "annotations: invalid key",
			Modify: func(meta metav1.Object) {
				meta.SetAnnotations(map[string]string{
					"invalid/key/format": "value",
				})
			},
			ExpectedErrs: field.ErrorList{
				field.Invalid(fldPath.Child("annotations"), "", "").WithOrigin("format=k8s-label-key").MarkFromImperative(),
			},
		},
		{
			Name: "annotations: too long",
			Modify: func(meta metav1.Object) {
				meta.SetAnnotations(map[string]string{
					"a": string(make([]byte, apivalidation.TotalAnnotationSizeLimitB)),
				})
			},
			ExpectedErrs: field.ErrorList{
				field.TooLong(fldPath.Child("annotations"), "", apivalidation.TotalAnnotationSizeLimitB).MarkFromImperative(),
			},
		},
	}

	for _, tc := range testCases {
		t.Run("objectmeta: "+tc.Name, func(t *testing.T) {
			obj := baseObj.DeepCopyObject().(T)
			if accessor, err := apimeta.Accessor(obj); err == nil {
				tc.Modify(accessor)
			} else {
				t.Fatalf("failed to get accessor: %v", err)
			}
			apitesting.VerifyValidationEquivalence(t, ctx, obj, strategy, tc.ExpectedErrs, options...)
		})
	}
}

func RunObjectMetaUpdateTestCases[T runtime.Object](t *testing.T, ctx context.Context, baseObj T, strategy rest.RESTUpdateStrategy, options ...apitesting.ValidationTestConfig) {
	t.Helper()
	fldPath := field.NewPath("metadata")

	testCases := []struct {
		Name         string
		Modify       func(old, new metav1.Object)
		ExpectedErrs field.ErrorList
	}{
		{
			Name: "update: annotations: invalid key",
			Modify: func(old, new metav1.Object) {
				old.SetResourceVersion("1")
				new.SetResourceVersion("2")
				new.SetAnnotations(map[string]string{
					"invalid/key/format": "value",
				})
			},
			ExpectedErrs: field.ErrorList{
				field.Invalid(fldPath.Child("annotations"), "", "").WithOrigin("format=k8s-label-key").MarkFromImperative(),
			},
		},
		{
			Name: "update: annotations: too long",
			Modify: func(old, new metav1.Object) {
				old.SetResourceVersion("1")
				new.SetResourceVersion("2")
				new.SetAnnotations(map[string]string{
					"a": string(make([]byte, apivalidation.TotalAnnotationSizeLimitB)),
				})
			},
			ExpectedErrs: field.ErrorList{
				field.TooLong(fldPath.Child("annotations"), "", 0).MarkFromImperative(),
			},
		},
	}

	for _, tc := range testCases {

		t.Run("objectmeta: "+tc.Name, func(t *testing.T) {
			currOld := baseObj.DeepCopyObject().(T)
			currNew := baseObj.DeepCopyObject().(T)
			newAcc, _ := apimeta.Accessor(currNew)
			oldAcc, _ := apimeta.Accessor(currOld)
			tc.Modify(oldAcc, newAcc)
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, currNew, currOld, strategy, tc.ExpectedErrs, options...)
		})
	}
}
