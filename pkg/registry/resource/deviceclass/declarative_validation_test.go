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
	"fmt"
	"strings"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/apis/resource"
	_ "k8s.io/kubernetes/pkg/apis/resource/install"
	"k8s.io/utils/ptr"
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
				// spec.selectors.
				"valid: at limit selectors": {
					input: mkDeviceClass(tweakSelectors(32)),
				},
				"too many selectors": {
					input: mkDeviceClass(tweakSelectors(33)),
					expectedErrs: field.ErrorList{
						field.TooMany(field.NewPath("spec", "selectors"), 33, 32).WithOrigin("maxItems"),
					},
				},
				// spec.config
				"too many configs": {
					input: mkDeviceClass(tweakConfig(33)),
					expectedErrs: field.ErrorList{
						field.TooMany(field.NewPath("spec", "config"), 33, 32).WithOrigin("maxItems"),
					},
				},
				"valid: at limit configs": {
					input: mkDeviceClass(tweakConfig(32)),
				},

				// ExtendedResourceName
				"valid extended resource name": {
					input: mkDeviceClass(tweakExtendedResourceName("example.com/my-resource")),
				},
				"invalid extended resource name": {
					input: mkDeviceClass(tweakExtendedResourceName("invalid_name")),
					expectedErrs: field.ErrorList{
						field.Invalid(field.NewPath("spec", "extendedResourceName"), "invalid_name", "").WithOrigin("format=k8s-extended-resource-name"),
					},
				},
				"invalid extended resource name, no slash": {
					input: mkDeviceClass(tweakExtendedResourceName("noslash")),
					expectedErrs: field.ErrorList{
						field.Invalid(field.NewPath("spec", "extendedResourceName"), "noslash", "").WithOrigin("format=k8s-extended-resource-name"),
					},
				},
				"invalid extended resource name, kubernetes.io domain": {
					input: mkDeviceClass(tweakExtendedResourceName("kubernetes.io/foo")),
					expectedErrs: field.ErrorList{
						field.Invalid(field.NewPath("spec", "extendedResourceName"), "kubernetes.io/foo", "").WithOrigin("format=k8s-extended-resource-name"),
					},
				},
				"invalid extended resource name, requests. prefix": {
					input: mkDeviceClass(tweakExtendedResourceName("requests.example.com/foo")),
					expectedErrs: field.ErrorList{
						field.Invalid(field.NewPath("spec", "extendedResourceName"), "requests.example.com/foo", "").WithOrigin("format=k8s-extended-resource-name"),
					},
				},
				"invalid extended resource name, too long": {
					input: mkDeviceClass(tweakExtendedResourceName("example.com/" + strings.Repeat("a", 64))),
					expectedErrs: field.ErrorList{
						field.Invalid(field.NewPath("spec", "extendedResourceName"), "example.com/"+strings.Repeat("a", 64), "").WithOrigin("format=k8s-extended-resource-name"),
					},
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
				"valid update: at limit selectors": {
					old:    mkDeviceClass(),
					update: mkDeviceClass(tweakSelectors(32)),
				},
				"update with too many selectors": {
					old:    mkDeviceClass(),
					update: mkDeviceClass(tweakSelectors(33)),
					expectedErrs: field.ErrorList{
						field.TooMany(field.NewPath("spec", "selectors"), 33, 32).WithOrigin("maxItems"),
					},
				},
				"valid update: at limit configs": {
					old:    mkDeviceClass(),
					update: mkDeviceClass(tweakConfig(32)),
				},
				"update with too many configs": {
					old:    mkDeviceClass(),
					update: mkDeviceClass(tweakConfig(33)),
					expectedErrs: field.ErrorList{
						field.TooMany(field.NewPath("spec", "config"), 33, 32).WithOrigin("maxItems"),
					},
				},

				// spec.ExtendedResourceName
				"valid extended resource name update": {
					old:    mkDeviceClass(),
					update: mkDeviceClass(tweakExtendedResourceName("example.com/my-resource")),
				},
				"invalid extended resource name update": {
					old:    mkDeviceClass(),
					update: mkDeviceClass(tweakExtendedResourceName("invalid_name")),
					expectedErrs: field.ErrorList{
						field.Invalid(field.NewPath("spec", "extendedResourceName"), "invalid_name", "").WithOrigin("format=k8s-extended-resource-name"),
					},
				},
				"invalid extended resource name update, no slash": {
					old:    mkDeviceClass(),
					update: mkDeviceClass(tweakExtendedResourceName("noslash")),
					expectedErrs: field.ErrorList{
						field.Invalid(field.NewPath("spec", "extendedResourceName"), "noslash", "").WithOrigin("format=k8s-extended-resource-name"),
					},
				},
				"invalid extended resource name update, kubernetes.io domain": {
					old:    mkDeviceClass(),
					update: mkDeviceClass(tweakExtendedResourceName("kubernetes.io/foo")),
					expectedErrs: field.ErrorList{
						field.Invalid(field.NewPath("spec", "extendedResourceName"), "kubernetes.io/foo", "").WithOrigin("format=k8s-extended-resource-name"),
					},
				},
				"invalid extended resource name update, requests. prefix": {
					old:    mkDeviceClass(),
					update: mkDeviceClass(tweakExtendedResourceName("requests.example.com/foo")),
					expectedErrs: field.ErrorList{
						field.Invalid(field.NewPath("spec", "extendedResourceName"), "requests.example.com/foo", "").WithOrigin("format=k8s-extended-resource-name"),
					},
				},
				"invalid extended resource name update, too long": {
					old:    mkDeviceClass(),
					update: mkDeviceClass(tweakExtendedResourceName("example.com/" + strings.Repeat("a", 64))),
					expectedErrs: field.ErrorList{
						field.Invalid(field.NewPath("spec", "extendedResourceName"), "example.com/"+strings.Repeat("a", 64), "").WithOrigin("format=k8s-extended-resource-name"),
					},
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
			ExtendedResourceName: ptr.To("example.com/my-resource"),
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

func tweakSelectors(count int) func(*resource.DeviceClass) {
	return func(dc *resource.DeviceClass) {
		dc.Spec.Selectors = []resource.DeviceSelector{}

		for i := 0; i < count; i++ {
			dc.Spec.Selectors = append(dc.Spec.Selectors, resource.DeviceSelector{
				CEL: &resource.CELDeviceSelector{
					Expression: fmt.Sprintf("device.driver == \"test.driver.io%d\"", i),
				},
			})
		}
	}
}

func tweakConfig(count int) func(*resource.DeviceClass) {
	return func(dc *resource.DeviceClass) {
		dc.Spec.Config = []resource.DeviceClassConfiguration{}
		for i := 0; i < count; i++ {
			dc.Spec.Config = append(dc.Spec.Config, resource.DeviceClassConfiguration{
				DeviceConfiguration: resource.DeviceConfiguration{
					Opaque: &resource.OpaqueDeviceConfiguration{
						Driver: "test.driver.io",
						Parameters: runtime.RawExtension{
							Raw: []byte(fmt.Sprintf(`{"key":"value%d"}`, i)),
						},
					},
				},
			})
		}
	}
}

func tweakExtendedResourceName(name string) func(*resource.DeviceClass) {
	return func(dc *resource.DeviceClass) {
		nameCopy := name
		dc.Spec.ExtendedResourceName = &nameCopy
	}
}
