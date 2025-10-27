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

package resourceslice

import (
	"fmt"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/apis/resource"
	_ "k8s.io/kubernetes/pkg/apis/resource/install"
	"k8s.io/kubernetes/pkg/apis/resource/validation"
	"k8s.io/utils/ptr"
)

var apiVersions = []string{"v1", "v1beta1", "v1beta2"}

func TestDeclarativeValidate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
				APIGroup:   "resource.k8s.io",
				APIVersion: apiVersion,
				Resource:   "ResourceSlice",
			})

			strategy := Strategy

			testCases := map[string]struct {
				input        resource.ResourceSlice
				expectedErrs field.ErrorList
			}{
				"valid": {
					input: mkResourceSlice(),
				},
				// spec.devices[%d].bindingConditions
				"valid: one binding condition": {
					input: mkResourceSlice(tweakBindingConditions(1)),
				},
				"valid: at limit binding conditions": {
					input: mkResourceSlice(tweakBindingConditions(resource.BindingConditionsMaxSize)),
				},
				"invalid: too many binding conditions": {
					input: mkResourceSlice(tweakBindingConditions(resource.BindingConditionsMaxSize + 1)),
					expectedErrs: field.ErrorList{
						field.TooMany(field.NewPath("spec", "devices").Index(0).Child("bindingConditions"), resource.BindingConditionsMaxSize+1, resource.BindingConditionsMaxSize).WithOrigin("maxItems"),
					},
				},
				// spec.devices[%d].bindingFailureConditions
				"valid: one binding failure conditions": {
					input: mkResourceSlice(tweakBindingFailureConditions(1)),
				},
				"valid: at limit binding failure conditions": {
					input: mkResourceSlice(tweakBindingFailureConditions(resource.BindingFailureConditionsMaxSize)),
				},
				"invalid: too many binding failure conditions": {
					input: mkResourceSlice(tweakBindingFailureConditions(resource.BindingFailureConditionsMaxSize + 1)),
					expectedErrs: field.ErrorList{
						field.TooMany(field.NewPath("spec", "devices").Index(0).Child("bindingFailureConditions"), resource.BindingFailureConditionsMaxSize+1, resource.BindingFailureConditionsMaxSize).WithOrigin("maxItems"),
					},
				},
				// spec.Devices[%d].Taints[%d].Effect
				"valid: taint NoSchedule": {
					input: mkResourceSlice(tweakDeviceTaintEffect(string(resource.DeviceTaintEffectNoSchedule))),
				},
				"valid: taint NoExecute": {
					input: mkResourceSlice(tweakDeviceTaintEffect(string(resource.DeviceTaintEffectNoExecute))),
				},
				"invalid: taint Invalid": {
					input: mkResourceSlice(tweakDeviceTaintEffect("Invalid")),
					expectedErrs: field.ErrorList{
						field.NotSupported(
							field.NewPath("spec", "devices").Index(0).Child("taints").Index(0).Child("effect"),
							resource.DeviceTaintEffect("Invalid"), []string{}),
					},
				},
				"invalid: taint empty": {
					input: mkResourceSlice(tweakDeviceTaintEffect("")),
					expectedErrs: field.ErrorList{
						field.Required(field.NewPath("spec", "devices").Index(0).Child("taints").Index(0).Child("effect"), ""),
					},
				},
				// spec.Devices[%].attribute
				"valid: device attribute int": {
					input: mkResourceSlice(tweakDeviceAttribute("test.io/int", resource.DeviceAttribute{IntValue: ptr.To[int64](123)})),
				},
				"valid: device attribute bool": {
					input: mkResourceSlice(tweakDeviceAttribute("test.io/bool", resource.DeviceAttribute{BoolValue: ptr.To(true)})),
				},
				"valid: device attribute string": {
					input: mkResourceSlice(tweakDeviceAttribute("test.io/string", resource.DeviceAttribute{StringValue: ptr.To("value")})),
				},
				"valid: device attribute version": {
					input: mkResourceSlice(tweakDeviceAttribute("test.io/version", resource.DeviceAttribute{VersionValue: ptr.To("1.2.3")})),
				},
				"invalid: device attribute with multiple values": {
					input: mkResourceSlice(tweakDeviceAttribute("test.io/multiple", resource.DeviceAttribute{IntValue: ptr.To[int64](123), BoolValue: ptr.To(true)})),
					expectedErrs: field.ErrorList{
						field.Invalid(field.NewPath("spec", "devices").Index(0).Child("attributes").Key("test.io/multiple"), "", ""),
					},
				},
				"invalid: device attribute no value": {
					input: mkResourceSlice(tweakDeviceAttribute("test.io/multiple", resource.DeviceAttribute{})),
					expectedErrs: field.ErrorList{
						field.Invalid(field.NewPath("spec", "devices").Index(0).Child("attributes").Key("test.io/multiple"), "", ""),
					},
				},
				// TODO: Add more test cases
			}

			for k, tc := range testCases {
				t.Run(k, func(t *testing.T) {
					apitesting.VerifyValidationEquivalence(t, ctx, &tc.input, strategy.Validate, tc.expectedErrs, apitesting.WithNormalizationRules(validation.ResourceNormalizationRules...))
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
				Resource:   "ResourceSlice",
			})

			strategy := Strategy

			testCases := map[string]struct {
				old          resource.ResourceSlice
				update       resource.ResourceSlice
				expectedErrs field.ErrorList
			}{
				"valid no changes": {
					old:    mkResourceSlice(),
					update: mkResourceSlice(),
				},
				// spec.devices[%d].bindingConditions
				"valid update: at limit binding conditions": {
					old:    mkResourceSlice(),
					update: mkResourceSlice(tweakBindingConditions(resource.BindingConditionsMaxSize)),
				},
				"invalid update:update with too many binding conditions": {
					old:    mkResourceSlice(),
					update: mkResourceSlice(tweakBindingConditions(resource.BindingConditionsMaxSize + 1)),
					expectedErrs: field.ErrorList{
						field.TooMany(field.NewPath("spec", "devices").Index(0).Child("bindingConditions"), resource.BindingConditionsMaxSize+1, resource.BindingConditionsMaxSize).WithOrigin("maxItems"),
					},
				},
				// spec.devices[%d].bindingFailureConditions
				"valid update: at limit binding failure conditions": {
					old:    mkResourceSlice(),
					update: mkResourceSlice(tweakBindingFailureConditions(resource.BindingFailureConditionsMaxSize)),
				},
				"update with too many binding failure conditions": {
					old:    mkResourceSlice(),
					update: mkResourceSlice(tweakBindingFailureConditions(resource.BindingFailureConditionsMaxSize + 1)),
					expectedErrs: field.ErrorList{
						field.TooMany(field.NewPath("spec", "devices").Index(0).Child("bindingFailureConditions"), resource.BindingFailureConditionsMaxSize+1, resource.BindingFailureConditionsMaxSize).WithOrigin("maxItems"),
					},
				},
				// spec.devices.taints.effect
				"valid update: NoSchedule taint effect": {
					old:    mkResourceSlice(),
					update: mkResourceSlice(tweakDeviceTaintEffect(string(resource.DeviceTaintEffectNoSchedule))),
				},
				"valid update: NoExecute taint effect": {
					old:    mkResourceSlice(),
					update: mkResourceSlice(tweakDeviceTaintEffect(string(resource.DeviceTaintEffectNoExecute))),
				},
				"invalid update: unsupported taint effect": {
					old:    mkResourceSlice(),
					update: mkResourceSlice(tweakDeviceTaintEffect("InvalidEffect")),
					expectedErrs: field.ErrorList{
						field.NotSupported(field.NewPath("spec", "devices").Index(0).Child("taints").Index(0).Child("effect"), "InvalidEffect", []string{string(resource.DeviceTaintEffectNoSchedule), string(resource.DeviceTaintEffectNoExecute)}),
					},
				},
				"invalid update: empty taint effect": {
					old:    mkResourceSlice(),
					update: mkResourceSlice(tweakDeviceTaintEffect("")),
					expectedErrs: field.ErrorList{
						field.Required(field.NewPath("spec", "devices").Index(0).Child("taints").Index(0).Child("effect"), ""),
					},
				},
				"valid update: device attribute": {
					old:    mkResourceSlice(tweakDeviceAttribute("test.io/int", resource.DeviceAttribute{IntValue: ptr.To[int64](123)})),
					update: mkResourceSlice(tweakDeviceAttribute("test.io/int", resource.DeviceAttribute{BoolValue: ptr.To(true)})),
				},
				"invalid update: device attribute with multiple values": {
					old:    mkResourceSlice(),
					update: mkResourceSlice(tweakDeviceAttribute("test.io/multiple", resource.DeviceAttribute{IntValue: ptr.To[int64](123), BoolValue: ptr.To(true)})),
					expectedErrs: field.ErrorList{
						field.Invalid(field.NewPath("spec", "devices").Index(0).Child("attributes").Key("test.io/multiple"), "", "may have only one of the following fields set: bool, int, string, version"),
					},
				},
			}
			for k, tc := range testCases {
				t.Run(k, func(t *testing.T) {
					tc.old.ResourceVersion = "1"
					tc.update.ResourceVersion = "1"
					apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.update, &tc.old, strategy.ValidateUpdate, tc.expectedErrs, apitesting.WithNormalizationRules(validation.ResourceNormalizationRules...))
				})
			}
		})
	}
}

func mkResourceSlice(mutators ...func(*resource.ResourceSlice)) resource.ResourceSlice {
	rs := resource.ResourceSlice{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-slice",
		},
		Spec: resource.ResourceSliceSpec{
			Driver:   "test.driver.io",
			NodeName: ptr.To("test-node"),
			Pool: resource.ResourcePool{
				Name:               "test-pool",
				ResourceSliceCount: 5,
			},
			Devices: []resource.Device{
				{
					Name: "device-1",
					Taints: []resource.DeviceTaint{
						{
							Key:    "key1",
							Value:  "value1",
							Effect: resource.DeviceTaintEffectNoSchedule,
						},
					},
				},
			},
		},
	}
	for _, mutate := range mutators {
		mutate(&rs)
	}
	return rs
}

func tweakBindingFailureConditions(count int) func(*resource.ResourceSlice) {
	return func(rs *resource.ResourceSlice) {
		if rs.Spec.Devices[0].BindingConditions == nil {
			rs.Spec.Devices[0].BindingConditions = []string{"conditions"}
		}
		rs.Spec.Devices[0].BindingFailureConditions = []string{}
		for i := 0; i < count; i++ {
			rs.Spec.Devices[0].BindingFailureConditions = append(rs.Spec.Devices[0].BindingFailureConditions, fmt.Sprintf("failure-condition-%d", i))
		}
	}
}

func tweakBindingConditions(count int) func(*resource.ResourceSlice) {
	return func(rs *resource.ResourceSlice) {
		if rs.Spec.Devices[0].BindingFailureConditions == nil {
			rs.Spec.Devices[0].BindingFailureConditions = []string{"failure-conditions"}
		}
		rs.Spec.Devices[0].BindingConditions = []string{}
		for i := 0; i < count; i++ {
			rs.Spec.Devices[0].BindingConditions = append(rs.Spec.Devices[0].BindingConditions, fmt.Sprintf("condition-%d", i))
		}
	}
}

func tweakDeviceTaintEffect(effect string) func(*resource.ResourceSlice) {
	return func(rs *resource.ResourceSlice) {
		rs.Spec.Devices[0].Taints[0].Effect = resource.DeviceTaintEffect(effect)
	}
}

func tweakDeviceAttribute(name resource.QualifiedName, value resource.DeviceAttribute) func(*resource.ResourceSlice) {
	return func(rs *resource.ResourceSlice) {
		if rs.Spec.Devices[0].Attributes == nil {
			rs.Spec.Devices[0].Attributes = make(map[resource.QualifiedName]resource.DeviceAttribute)
		}
		rs.Spec.Devices[0].Attributes[name] = value
	}
}
