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

package persistentvolumeclaim

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
		Resource:          "persistentvolumeclaims",
		IsResourceRequest: true,
		Verb:              "create",
	})

	testCases := map[string]struct {
		input        api.PersistentVolumeClaim
		expectedErrs field.ErrorList
	}{
		"valid": {
			input: mkValidPersistentVolumeClaim(),
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
		oldObj       api.PersistentVolumeClaim
		updateObj    api.PersistentVolumeClaim
		expectedErrs field.ErrorList
	}{
		"valid update": {
			oldObj:    mkValidPersistentVolumeClaim(),
			updateObj: mkValidPersistentVolumeClaim(),
		},
		"invalid update volumeMode changed": {
			oldObj: mkValidPersistentVolumeClaim(),
			updateObj: mkValidPersistentVolumeClaim(func(obj *api.PersistentVolumeClaim) {
				mode := api.PersistentVolumeBlock
				obj.Spec.VolumeMode = &mode
			}),
			expectedErrs: field.ErrorList{
				field.Forbidden(field.NewPath("spec"), "").MarkFromImperative(),
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
				Resource:          "persistentvolumeclaims",
				Name:              "valid-pvc",
				IsResourceRequest: true,
				Verb:              "update",
			})
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.updateObj, &tc.oldObj, Strategy.ValidateUpdate, tc.expectedErrs)
		})
	}
}

func mkValidPersistentVolumeClaim(tweaks ...func(obj *api.PersistentVolumeClaim)) api.PersistentVolumeClaim {
	mode := api.PersistentVolumeFilesystem
	obj := api.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "valid-pvc",
			Namespace: metav1.NamespaceDefault,
		},
		Spec: api.PersistentVolumeClaimSpec{
			AccessModes: []api.PersistentVolumeAccessMode{api.ReadWriteOnce},
			Resources: api.VolumeResourceRequirements{
				Requests: api.ResourceList{
					api.ResourceStorage: resource.MustParse("10Gi"),
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
