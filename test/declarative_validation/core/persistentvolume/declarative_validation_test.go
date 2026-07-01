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

package persistentvolume

import (
	"testing"

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	core "k8s.io/kubernetes/pkg/apis/core"
	registry "k8s.io/kubernetes/pkg/registry/core/persistentvolume"
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
		APIPrefix:         "api",
		APIGroup:          "",
		APIVersion:        apiVersion,
		Resource:          "persistentvolumes",
		IsResourceRequest: true,
		Verb:              "create",
	})

	obj := mkValidPersistentVolume()
	apitesting.VerifyValidationEquivalence(t, ctx, &obj, registry.Strategy, nil)
	meta.RunObjectMetaTestCases(t, ctx, &obj, registry.Strategy, meta.WithStringentFinalizerValidation())
}

func testDeclarativeValidateUpdate(t *testing.T, apiVersion string) {
	blockMode := core.PersistentVolumeBlock
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIPrefix:         "api",
		APIGroup:          "",
		APIVersion:        apiVersion,
		Resource:          "persistentvolumes",
		Name:              "valid-pv",
		IsResourceRequest: true,
		Verb:              "update",
	})

	testCases := map[string]struct {
		oldObj       core.PersistentVolume
		updateObj    core.PersistentVolume
		expectedErrs field.ErrorList
	}{
		"valid update": {
			oldObj:    mkValidPersistentVolume(),
			updateObj: mkValidPersistentVolume(),
		},
		"invalid update volumeMode changed": {
			oldObj:    mkValidPersistentVolume(),
			updateObj: mkValidPersistentVolume(tweakVolumeMode(&blockMode)),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "volumeMode"), nil, "").WithOrigin("immutable").MarkAlpha(),
			},
		},
		"invalid update volumeMode unset from set": {
			oldObj:    mkValidPersistentVolume(),
			updateObj: mkValidPersistentVolume(tweakVolumeMode(nil)),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "volumeMode"), nil, "").WithOrigin("immutable").MarkAlpha(),
			},
		},
		"invalid update volumeMode set from unset": {
			oldObj:    mkValidPersistentVolume(tweakVolumeMode(nil)),
			updateObj: mkValidPersistentVolume(),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "volumeMode"), nil, "").WithOrigin("immutable").MarkAlpha(),
			},
		},
	}
	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			tc.oldObj.ResourceVersion = "1"
			tc.updateObj.ResourceVersion = "2"
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.updateObj, &tc.oldObj, registry.Strategy, tc.expectedErrs)
		})
	}

	updateObj := mkValidPersistentVolume()
	meta.RunObjectMetaUpdateTestCases(t, ctx, &updateObj, registry.Strategy, meta.WithStringentFinalizerValidation())
}

func mkValidPersistentVolume(tweaks ...func(obj *core.PersistentVolume)) core.PersistentVolume {
	mode := core.PersistentVolumeFilesystem
	obj := core.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			Name: "valid-pv",
		},
		Spec: core.PersistentVolumeSpec{
			Capacity: core.ResourceList{
				core.ResourceStorage: resource.MustParse("10Gi"),
			},
			AccessModes: []core.PersistentVolumeAccessMode{core.ReadWriteOnce},
			PersistentVolumeSource: core.PersistentVolumeSource{
				HostPath: &core.HostPathVolumeSource{
					Path: "/tmp/data",
				},
			},
			VolumeMode: &mode,
		},
	}
	for _, tweak := range tweaks {
		tweak(&obj)
	}
	return obj
}

func tweakVolumeMode(mode *core.PersistentVolumeMode) func(obj *core.PersistentVolume) {
	return func(obj *core.PersistentVolume) {
		obj.Spec.VolumeMode = mode
	}
}
