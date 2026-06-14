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

package csinode

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/apis/storage"
)

func TestDeclarativeValidate(t *testing.T) {
	// CSINode had v1beta1 → v1, keep both to catch skew
	apiVersions := []string{"v1beta1", "v1"}
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			testDeclarativeValidate(t, apiVersion)
		})
	}
}

func TestDeclarativeValidateUpdate(t *testing.T) {
	// CSINode had v1beta1 → v1, keep both to catch skew
	apiVersions := []string{"v1beta1", "v1"}
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			testDeclarativeValidateUpdate(t, apiVersion)
		})
	}
}

func testDeclarativeValidate(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithRequestInfo(
		genericapirequest.NewDefaultContext(),
		&genericapirequest.RequestInfo{
			APIPrefix:         "apis",
			APIGroup:          "storage.k8s.io",
			APIVersion:        apiVersion,
			Resource:          "csinodes",
			IsResourceRequest: true,
			Verb:              "create",
		},
	)

	testCases := map[string]struct {
		input        storage.CSINode
		expectedErrs field.ErrorList
	}{
		//"valid": {
		//	input: mkValidCSINodeDriverNode(),
		//},
		"missing name": {
			input: mkValidCSINodeDriverNode(func(driver *storage.CSINodeDriver) {
				driver.Name = ""
			}),
			expectedErrs: field.ErrorList{
				field.Required(
					field.NewPath("spec").Child("drivers").Index(0).Child("name"),
					"",
				),
			},
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			apitesting.VerifyValidationEquivalence(
				t,
				ctx,
				&tc.input,
				Strategy.Validate,
				tc.expectedErrs,
			)
		})
	}
}

func testDeclarativeValidateUpdate(t *testing.T, apiVersion string) {
	testCases := map[string]struct {
		oldObj       storage.CSINode
		updateObj    storage.CSINode
		expectedErrs field.ErrorList
	}{
		//"valid update": {
		//	oldObj:    mkValidCSINodeDriverNode(func(driver *storage.CSINodeDriver) {}),
		//	updateObj: mkValidCSINodeDriverNode(func(driver *storage.CSINodeDriver) {}),
		//},
		"invalid update name": {
			oldObj: mkValidCSINodeDriverNode(func(driver *storage.CSINodeDriver) {}),
			updateObj: mkValidCSINodeDriverNode(func(driver *storage.CSINodeDriver) {
				driver.Name = "io.kubernetes.storage.csi.driver-1"
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec").Child("drivers").Index(0).Child("name"), "",
					"").WithOrigin("immutable"),
			},
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			ctx := genericapirequest.WithRequestInfo(
				genericapirequest.NewDefaultContext(),
				&genericapirequest.RequestInfo{
					APIPrefix:         "apis",
					APIGroup:          "storage.k8s.io",
					APIVersion:        apiVersion,
					Resource:          "csinodes",
					Name:              "node-1",
					IsResourceRequest: true,
					Verb:              "update",
				},
			)
			apitesting.VerifyUpdateValidationEquivalence(
				t,
				ctx,
				&tc.updateObj,
				&tc.oldObj,
				Strategy.ValidateUpdate,
				tc.expectedErrs,
			)
		})
	}
}

func mkValidCSINodeDriverNode(tweaks ...func(*storage.CSINodeDriver)) storage.CSINode {
	driver := storage.CSINodeDriver{
		Name:   "io.kubernetes.storage.csi.driver-1",
		NodeID: "node-1",
	}
	for _, tweak := range tweaks {
		tweak(&driver)
	}

	return storage.CSINode{
		ObjectMeta: metav1.ObjectMeta{
			Name: "node-1",
		},
		Spec: storage.CSINodeSpec{
			Drivers: []storage.CSINodeDriver{driver},
		},
	}
}
