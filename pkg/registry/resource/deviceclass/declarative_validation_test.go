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

package deviceclass

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/apis/resource"
	_ "k8s.io/kubernetes/pkg/apis/resource/install"
)

var apiVersions = []string{"v1", "v1beta1", "v1beta2"}

func TestDeclarativeValidate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
				APIGroup:   "resource.k8s.io",
				APIVersion: apiVersion,
				Resource:   "deviceclasses",
			})

			strategy := Strategy

			testCases := map[string]struct {
				input        resource.DeviceClass
				expectedErrs field.ErrorList
			}{
				"valid": {
					input: mkDeviceClass(),
				},
				// TODO: Add more test cases
			}

			for k, tc := range testCases {
				t.Run(k, func(t *testing.T) {
					apitesting.VerifyValidationEquivalence(t, ctx, &tc.input, strategy.Validate, tc.expectedErrs)
				})
			}
		})
	}
}

func TestDeclarativeValidateUpdate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
				APIGroup:   "resource.k8s.io",
				APIVersion: apiVersion,
				Resource:   "deviceclasses",
			})

			strategy := Strategy

			testCases := map[string]struct {
				old          resource.DeviceClass
				update       resource.DeviceClass
				expectedErrs field.ErrorList
			}{
				"valid no changes": {
					old:    mkDeviceClass(),
					update: mkDeviceClass(),
				},
				// TODO: Add more test cases
			}

			for k, tc := range testCases {
				t.Run(k, func(t *testing.T) {
					tc.old.ResourceVersion = "1"
					tc.update.ResourceVersion = "1"
					apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.update, &tc.old, strategy.ValidateUpdate, tc.expectedErrs)
				})
			}
		})
	}
}

// Helper function to create a DeviceClass with default values and optional mutators
func mkDeviceClass(mutators ...func(*resource.DeviceClass)) resource.DeviceClass {
	dc := resource.DeviceClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-class",
		},
		Spec: resource.DeviceClassSpec{
			Selectors: []resource.DeviceSelector{
				{
					CEL: &resource.CELDeviceSelector{
						Expression: "device.driver == \"test.driver.io\"",
					},
				},
			},
			Config: []resource.DeviceClassConfiguration{
				{
					DeviceConfiguration: resource.DeviceConfiguration{
						Opaque: &resource.OpaqueDeviceConfiguration{
							Driver: "test.driver.io",
							Parameters: runtime.RawExtension{
								Raw: []byte(`{"key":"value"}`),
							},
						},
					},
				},
			},
		},
	}
	for _, mutate := range mutators {
		mutate(&dc)
	}
	return dc
}
