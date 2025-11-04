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

package storageclass

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/storage"
)

func TestDeclarativeValidate(t *testing.T) {
	apiVersions := []string{"v1beta1", "v1"} // StorageClass.Provisioner not in v1alpha1
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			testDeclarativeValidate(t, apiVersion)
		})
	}
}

func testDeclarativeValidate(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:          "storage.k8s.io",
		APIVersion:        apiVersion,
		Resource:          "storageclasses",
		IsResourceRequest: true,
		Verb:              "create",
	})

	testCases := map[string]struct {
		input        storage.StorageClass
		expectedErrs field.ErrorList
	}{
		"valid": {
			input: mkValidStorageClass(),
		},
		"invalid provisioner": {
			input: mkValidStorageClass(func(obj *storage.StorageClass) {
				obj.Provisioner = ""
			}),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("provisioner"), ""),
			},
		},
		// TODO: Add more test cases
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			apitesting.VerifyValidationEquivalence(t, ctx, &tc.input, Strategy.Validate, tc.expectedErrs)
		})
	}
}

func TestDeclarativeValidateUpdate(t *testing.T) {
	apiVersions := []string{"v1beta1", "v1"} // StorageClass.Provisioner not in v1alpha1
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			testDeclarativeValidateUpdate(t, apiVersion)
		})
	}
}

func testDeclarativeValidateUpdate(t *testing.T, apiVersion string) {
	testCases := map[string]struct {
		oldObj       storage.StorageClass
		updateObj    storage.StorageClass
		expectedErrs field.ErrorList
	}{
		"valid update": {
			oldObj:    mkValidStorageClass(func(obj *storage.StorageClass) { obj.ResourceVersion = "1" }),
			updateObj: mkValidStorageClass(func(obj *storage.StorageClass) { obj.ResourceVersion = "1" }),
		},
		"invalid update provisioner": {
			oldObj: mkValidStorageClass(func(obj *storage.StorageClass) { obj.ResourceVersion = "1" }),
			updateObj: mkValidStorageClass(func(obj *storage.StorageClass) {
				obj.ResourceVersion = "1"
				obj.Provisioner = ""
			}),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("provisioner"), ""),
				field.Forbidden(field.NewPath("provisioner"), "updates to provisioner are forbidden."),
			},
		},
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
				APIPrefix:         "apis",
				APIGroup:          "storage.k8s.io",
				APIVersion:        apiVersion,
				Resource:          "storageclasses",
				Name:              "valid-storage-class",
				IsResourceRequest: true,
				Verb:              "update",
			})
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.updateObj, &tc.oldObj, Strategy.ValidateUpdate, tc.expectedErrs)
		})
	}
}

func mkValidStorageClass(tweaks ...func(obj *storage.StorageClass)) storage.StorageClass {
	reclaimPolicy := api.PersistentVolumeReclaimDelete
	volumeBindingMode := storage.VolumeBindingImmediate
	obj := storage.StorageClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: "valid-storage-class",
		},
		Provisioner:       "kubernetes.io/gce-pd",
		ReclaimPolicy:     &reclaimPolicy,
		VolumeBindingMode: &volumeBindingMode,
	}
	for _, tweak := range tweaks {
		tweak(&obj)
	}
	return obj
}
