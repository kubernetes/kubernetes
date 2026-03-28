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

package persistentvolume

import (
	"testing"

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	api "k8s.io/kubernetes/pkg/apis/core"

	// ensure types are installed
	_ "k8s.io/kubernetes/pkg/apis/core/install"
)

var apiVersions = []string{"v1"}

func TestDeclarativeValidate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			testDeclarativeValidate(t, apiVersion)
		})
	}
}

func testDeclarativeValidate(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:          "",
		APIVersion:        apiVersion,
		Resource:          "persistentvolumes",
		IsResourceRequest: true,
		Verb:              "create",
	})

	testCases := map[string]struct {
		input        api.PersistentVolume
		expectedErrs field.ErrorList
	}{
		"valid": {
			input: mkValidPersistentVolume(),
		},
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
		oldObj       api.PersistentVolume
		updateObj    api.PersistentVolume
		expectedErrs field.ErrorList
	}{
		"valid update": {
			oldObj:    mkValidPersistentVolume(),
			updateObj: mkValidPersistentVolume(),
		},
		"invalid update volumeMode changed": {
			oldObj: mkValidPersistentVolume(),
			updateObj: mkValidPersistentVolume(func(obj *api.PersistentVolume) {
				mode := api.PersistentVolumeBlock
				obj.Spec.VolumeMode = &mode
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "volumeMode"), nil, "").WithOrigin("immutable").MarkAlpha(),
			},
		},
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			tc.oldObj.ResourceVersion = "1"
			tc.updateObj.ResourceVersion = "2"
			ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
				APIGroup:          "",
				APIVersion:        apiVersion,
				Resource:          "persistentvolumes",
				Name:              "valid-pv",
				IsResourceRequest: true,
				Verb:              "update",
			})
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.updateObj, &tc.oldObj, Strategy.ValidateUpdate, tc.expectedErrs)
		})
	}
}

func mkValidPersistentVolume(tweaks ...func(obj *api.PersistentVolume)) api.PersistentVolume {
	mode := api.PersistentVolumeFilesystem
	obj := api.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			Name: "valid-pv",
		},
		Spec: api.PersistentVolumeSpec{
			Capacity: api.ResourceList{
				api.ResourceStorage: resource.MustParse("10Gi"),
			},
			AccessModes: []api.PersistentVolumeAccessMode{api.ReadWriteOnce},
			PersistentVolumeSource: api.PersistentVolumeSource{
				HostPath: &api.HostPathVolumeSource{
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
