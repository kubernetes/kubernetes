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

var apiVersions = []string{"v1beta1", "v1"} // StorageClass.Provisioner not in v1alpha1

func TestDeclarativeValidate(t *testing.T) {
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
			oldObj:    mkValidStorageClass(),
			updateObj: mkValidStorageClass(),
		},
		"invalid update provisioner changed": {
			oldObj:    mkValidStorageClass(),
			updateObj: mkValidStorageClass(TweakProvisioner("kubernetes.io/aws-ebs")),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("provisioner"), "kubernetes.io/aws-ebs", "field is immutable").WithOrigin("immutable"),
			},
		},
		"invalid update parameters changed": {
			oldObj:    mkValidStorageClass(),
			updateObj: mkValidStorageClass(TweakParameters(map[string]string{"new": "value"})),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("parameters"), map[string]string{"new": "value"}, "field is immutable").WithOrigin("immutable"),
			},
		},
		"invalid update reclaimPolicy changed": {
			oldObj:    mkValidStorageClass(),
			updateObj: mkValidStorageClass(TweakReclaimPolicy(api.PersistentVolumeReclaimRetain)),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("reclaimPolicy"), api.PersistentVolumeReclaimRetain, "field is immutable").WithOrigin("immutable"),
			},
		},
		"invalid update volumeBindingMode changed": {
			oldObj:    mkValidStorageClass(),
			updateObj: mkValidStorageClass(TweakVolumeBindingMode(storage.VolumeBindingWaitForFirstConsumer)),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("volumeBindingMode"), storage.VolumeBindingWaitForFirstConsumer, "field is immutable").WithOrigin("immutable"),
			},
		},
		"invalid update provisioner unset to set": {
			oldObj:    mkValidStorageClass(TweakProvisioner("")),
			updateObj: mkValidStorageClass(),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("provisioner"), "kubernetes.io/gce-pd", "field is immutable").WithOrigin("immutable"),
			},
		},
		"invalid update provisioner set to unset": {
			oldObj:    mkValidStorageClass(),
			updateObj: mkValidStorageClass(TweakProvisioner("")),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("provisioner"), "", "field is immutable").WithOrigin("immutable"),
				field.Required(field.NewPath("provisioner"), ""),
			},
		},
		"invalid update parameters unset to set": {
			oldObj:    mkValidStorageClass(),
			updateObj: mkValidStorageClass(TweakParameters(map[string]string{"foo": "bar"})),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("parameters"), map[string]string{"foo": "bar"}, "field is immutable").WithOrigin("immutable"),
			},
		},
		"invalid update parameters set to unset": {
			oldObj:    mkValidStorageClass(TweakParameters(map[string]string{"foo": "bar"})),
			updateObj: mkValidStorageClass(),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("parameters"), nil, "field is immutable").WithOrigin("immutable"),
			},
		},
		"invalid update reclaimPolicy unset to set": {
			oldObj:    mkValidStorageClass(TweakReclaimPolicyNil()),
			updateObj: mkValidStorageClass(),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("reclaimPolicy"), api.PersistentVolumeReclaimDelete, "field is immutable").WithOrigin("immutable"),
			},
		},
		"invalid update reclaimPolicy set to unset": {
			oldObj:    mkValidStorageClass(),
			updateObj: mkValidStorageClass(TweakReclaimPolicyNil()),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("reclaimPolicy"), nil, "field is immutable").WithOrigin("immutable"),
			},
		},
		"invalid update volumeBindingMode unset to set": {
			oldObj:    mkValidStorageClass(TweakVolumeBindingModeNil()),
			updateObj: mkValidStorageClass(),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("volumeBindingMode"), storage.VolumeBindingImmediate, "field is immutable").WithOrigin("immutable"),
			},
		},
		"invalid update volumeBindingMode set to unset": {
			oldObj:    mkValidStorageClass(),
			updateObj: mkValidStorageClass(TweakVolumeBindingModeNil()),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("volumeBindingMode"), nil, "field is immutable").WithOrigin("immutable"),
				field.Required(field.NewPath("volumeBindingMode"), ""),
			},
		},
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			tc.oldObj.ResourceVersion = "1"
			tc.updateObj.ResourceVersion = "2"
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

func TweakProvisioner(provisioner string) func(obj *storage.StorageClass) {
	return func(obj *storage.StorageClass) {
		obj.Provisioner = provisioner
	}
}

func TweakParameters(parameters map[string]string) func(obj *storage.StorageClass) {
	return func(obj *storage.StorageClass) {
		obj.Parameters = parameters
	}
}

func TweakReclaimPolicy(reclaimPolicy api.PersistentVolumeReclaimPolicy) func(obj *storage.StorageClass) {
	return func(obj *storage.StorageClass) {
		obj.ReclaimPolicy = &reclaimPolicy
	}
}

func TweakReclaimPolicyNil() func(obj *storage.StorageClass) {
	return func(obj *storage.StorageClass) {
		obj.ReclaimPolicy = nil
	}
}

func TweakVolumeBindingMode(volumeBindingMode storage.VolumeBindingMode) func(obj *storage.StorageClass) {
	return func(obj *storage.StorageClass) {
		obj.VolumeBindingMode = &volumeBindingMode
	}
}

func TweakVolumeBindingModeNil() func(obj *storage.StorageClass) {
	return func(obj *storage.StorageClass) {
		obj.VolumeBindingMode = nil
	}
}
