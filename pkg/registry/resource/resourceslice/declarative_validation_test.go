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
					input: mkResourceSliceWithDevices(),
				},
				// spec.devices[%d].bindingConditions
				"valid: one binding condition": {
					input: mkResourceSliceWithDevices(tweakBindingConditions(1)),
				},
				"valid: at limit binding conditions": {
					input: mkResourceSliceWithDevices(tweakBindingConditions(resource.BindingConditionsMaxSize)),
				},
				"invalid: too many binding conditions": {
					input: mkResourceSliceWithDevices(tweakBindingConditions(resource.BindingConditionsMaxSize + 1)),
					expectedErrs: field.ErrorList{
						field.TooMany(field.NewPath("spec", "devices").Index(0).Child("bindingConditions"), resource.BindingConditionsMaxSize+1, resource.BindingConditionsMaxSize).WithOrigin("maxItems"),
					},
				},
				// spec.devices[%d].bindingFailureConditions
				"valid: one binding failure conditions": {
					input: mkResourceSliceWithDevices(tweakBindingFailureConditions(1)),
				},
				"valid: at limit binding failure conditions": {
					input: mkResourceSliceWithDevices(tweakBindingFailureConditions(resource.BindingFailureConditionsMaxSize)),
				},
				"invalid: too many binding failure conditions": {
					input: mkResourceSliceWithDevices(tweakBindingFailureConditions(resource.BindingFailureConditionsMaxSize + 1)),
					expectedErrs: field.ErrorList{
						field.TooMany(field.NewPath("spec", "devices").Index(0).Child("bindingFailureConditions"), resource.BindingFailureConditionsMaxSize+1, resource.BindingFailureConditionsMaxSize).WithOrigin("maxItems"),
					},
				},
				// spec.Devices[%d].Taints[%d].Effect
				"valid: taint NoSchedule": {
					input: mkResourceSliceWithDevices(tweakDeviceTaintEffect(string(resource.DeviceTaintEffectNoSchedule))),
				},
				"valid: taint NoExecute": {
					input: mkResourceSliceWithDevices(tweakDeviceTaintEffect(string(resource.DeviceTaintEffectNoExecute))),
				},
				"invalid: taint Invalid": {
					input: mkResourceSliceWithDevices(tweakDeviceTaintEffect("Invalid")),
					expectedErrs: field.ErrorList{
						field.NotSupported(
							field.NewPath("spec", "devices").Index(0).Child("taints").Index(0).Child("effect"),
							resource.DeviceTaintEffect("Invalid"), []string{}),
					},
				},
				"invalid: taint empty": {
					input: mkResourceSliceWithDevices(tweakDeviceTaintEffect("")),
					expectedErrs: field.ErrorList{
						field.Required(field.NewPath("spec", "devices").Index(0).Child("taints").Index(0).Child("effect"), ""),
					},
				},
				// spec.Devices[%].attribute
				"valid: device attribute int": {
					input: mkResourceSliceWithDevices(tweakDeviceAttribute("test.io/int", resource.DeviceAttribute{IntValue: ptr.To[int64](123)})),
				},
				"valid: device attribute bool": {
					input: mkResourceSliceWithDevices(tweakDeviceAttribute("test.io/bool", resource.DeviceAttribute{BoolValue: ptr.To(true)})),
				},
				"valid: device attribute string": {
					input: mkResourceSliceWithDevices(tweakDeviceAttribute("test.io/string", resource.DeviceAttribute{StringValue: ptr.To("value")})),
				},
				"valid: device attribute version": {
					input: mkResourceSliceWithDevices(tweakDeviceAttribute("test.io/version", resource.DeviceAttribute{VersionValue: ptr.To("1.2.3")})),
				},
				"invalid: device attribute with multiple values": {
					input: mkResourceSliceWithDevices(tweakDeviceAttribute("test.io/multiple", resource.DeviceAttribute{IntValue: ptr.To[int64](123), BoolValue: ptr.To(true)})),
					expectedErrs: field.ErrorList{
						field.Invalid(field.NewPath("spec", "devices").Index(0).Child("attributes").Key("test.io/multiple"), "", "").WithOrigin("union"),
					},
				},
				"invalid: device attribute no value": {
					input: mkResourceSliceWithDevices(tweakDeviceAttribute("test.io/multiple", resource.DeviceAttribute{})),
					expectedErrs: field.ErrorList{
						field.Invalid(field.NewPath("spec", "devices").Index(0).Child("attributes").Key("test.io/multiple"), "", "").WithOrigin("union"),
					},
				},
				// spec.sharedCounters
				"valid: at limit shared counters": {
					input: mkResourceSliceWithSharedCounters(tweakSharedCounters(resource.ResourceSliceMaxCounterSets)),
				},
				"invalid: too many shared counters": {
					input: mkResourceSliceWithSharedCounters(tweakSharedCounters(resource.ResourceSliceMaxCounterSets + 1)),
					expectedErrs: field.ErrorList{
						field.TooMany(field.NewPath("spec").Child("sharedCounters"), resource.ResourceSliceMaxCounterSets+1, resource.ResourceSliceMaxCounterSets).WithOrigin("maxItems"),
					},
				},
				// spec.devices.consumesCounters
				"valid: at limit device consumes counters": {
					input: mkResourceSliceWithDevices(tweakDeviceConsumesCounters(resource.ResourceSliceMaxDeviceCounterConsumptionsPerDevice)),
				},
				"invalid: too many device consumes counters": {
					input: mkResourceSliceWithDevices(tweakDeviceConsumesCounters(resource.ResourceSliceMaxDeviceCounterConsumptionsPerDevice + 1)),
					expectedErrs: field.ErrorList{
						field.TooMany(field.NewPath("spec", "devices").Index(0).Child("consumesCounters"), resource.ResourceSliceMaxDeviceCounterConsumptionsPerDevice+1, resource.ResourceSliceMaxDeviceCounterConsumptionsPerDevice).WithOrigin("maxItems"),
					},
				},
				// spec.sharedCounters.name
				"valid: counter set name": {
					input: mkResourceSliceWithSharedCounters(tweakSharedCountersName("valid-key")),
				},
				"invalid: counter set name": {
					input: mkResourceSliceWithSharedCounters(tweakSharedCountersName("InvalidKey")),
					expectedErrs: field.ErrorList{
						field.Invalid(field.NewPath("spec", "sharedCounters").Index(0).Child("name"), "InvalidKey", "").WithOrigin("format=k8s-short-name"),
					},
				},
				"invalid: counter set name not set": {
					input: mkResourceSliceWithSharedCounters(tweakSharedCountersName("")),
					expectedErrs: field.ErrorList{
						field.Required(field.NewPath("spec", "sharedCounters").Index(0).Child("name"), ""),
					},
				},
				// spec.devices.consumesCounters.counterSet
				"valid: device consumes counters counter set name": {
					input: mkResourceSliceWithDevices(tweakDeviceConsumesCountersCounterSetName("valid-key")),
				},
				"invalid: device consumes counters counter set name": {
					input: mkResourceSliceWithDevices(tweakDeviceConsumesCountersCounterSetName("InvalidKey")),
					expectedErrs: field.ErrorList{
						field.Invalid(field.NewPath("spec", "devices").Index(0).Child("consumesCounters").Index(0).Child("counterSet"), "InvalidKey", "").WithOrigin("format=k8s-short-name"),
					},
				},
				"invalid: device consumes counters counter set name not set": {
					input: mkResourceSliceWithDevices(tweakDeviceConsumesCountersCounterSetName("")),
					expectedErrs: field.ErrorList{
						field.Required(field.NewPath("spec", "devices").Index(0).Child("consumesCounters").Index(0).Child("counterSet"), ""),
					},
				},
				// spec.sharedCounters
				"valid: distinct names for shared counters": {
					input: mkResourceSliceWithSharedCounters(tweakSharedCountersName("valid-key-1", "valid-key-2")),
				},
				"invalid: duplicate names for shared counters": {
					input: mkResourceSliceWithSharedCounters(tweakSharedCountersName("duplicate-key", "duplicate-key")),
					expectedErrs: field.ErrorList{
						field.Duplicate(field.NewPath("spec").Child("sharedCounters").Index(1), "duplicate-key"),
					},
				},
				// spec.devices.consumesCounters
				"valid: distinct names for counter set in device counter consumption": {
					input: mkResourceSliceWithDevices(tweakDeviceConsumesCountersCounterSetName("valid-key-1", "valid-key-2")),
				},
				"invalid: duplicate names for counter set in device counter consumption": {
					input: mkResourceSliceWithDevices(tweakDeviceConsumesCountersCounterSetName("duplicate-key", "duplicate-key")),
					expectedErrs: field.ErrorList{
						field.Duplicate(field.NewPath("spec").Child("devices").Index(0).Child("consumesCounters").Index(1), "duplicate-key"),
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
					old:    mkResourceSliceWithDevices(),
					update: mkResourceSliceWithDevices(),
				},
				// spec.devices[%d].bindingConditions
				"valid update: at limit binding conditions": {
					old:    mkResourceSliceWithDevices(),
					update: mkResourceSliceWithDevices(tweakBindingConditions(resource.BindingConditionsMaxSize)),
				},
				"invalid update:update with too many binding conditions": {
					old:    mkResourceSliceWithDevices(),
					update: mkResourceSliceWithDevices(tweakBindingConditions(resource.BindingConditionsMaxSize + 1)),
					expectedErrs: field.ErrorList{
						field.TooMany(field.NewPath("spec", "devices").Index(0).Child("bindingConditions"), resource.BindingConditionsMaxSize+1, resource.BindingConditionsMaxSize).WithOrigin("maxItems"),
					},
				},
				// spec.devices[%d].bindingFailureConditions
				"valid update: at limit binding failure conditions": {
					old:    mkResourceSliceWithDevices(),
					update: mkResourceSliceWithDevices(tweakBindingFailureConditions(resource.BindingFailureConditionsMaxSize)),
				},
				"update with too many binding failure conditions": {
					old:    mkResourceSliceWithDevices(),
					update: mkResourceSliceWithDevices(tweakBindingFailureConditions(resource.BindingFailureConditionsMaxSize + 1)),
					expectedErrs: field.ErrorList{
						field.TooMany(field.NewPath("spec", "devices").Index(0).Child("bindingFailureConditions"), resource.BindingFailureConditionsMaxSize+1, resource.BindingFailureConditionsMaxSize).WithOrigin("maxItems"),
					},
				},
				// spec.devices.taints.effect
				"valid update: NoSchedule taint effect": {
					old:    mkResourceSliceWithDevices(),
					update: mkResourceSliceWithDevices(tweakDeviceTaintEffect(string(resource.DeviceTaintEffectNoSchedule))),
				},
				"valid update: NoExecute taint effect": {
					old:    mkResourceSliceWithDevices(),
					update: mkResourceSliceWithDevices(tweakDeviceTaintEffect(string(resource.DeviceTaintEffectNoExecute))),
				},
				"invalid update: unsupported taint effect": {
					old:    mkResourceSliceWithDevices(),
					update: mkResourceSliceWithDevices(tweakDeviceTaintEffect("InvalidEffect")),
					expectedErrs: field.ErrorList{
						field.NotSupported(field.NewPath("spec", "devices").Index(0).Child("taints").Index(0).Child("effect"), "InvalidEffect", []string{string(resource.DeviceTaintEffectNoSchedule), string(resource.DeviceTaintEffectNoExecute)}),
					},
				},
				"invalid update: empty taint effect": {
					old:    mkResourceSliceWithDevices(),
					update: mkResourceSliceWithDevices(tweakDeviceTaintEffect("")),
					expectedErrs: field.ErrorList{
						field.Required(field.NewPath("spec", "devices").Index(0).Child("taints").Index(0).Child("effect"), ""),
					},
				},
				"valid update: device attribute": {
					old:    mkResourceSliceWithDevices(tweakDeviceAttribute("test.io/int", resource.DeviceAttribute{IntValue: ptr.To[int64](123)})),
					update: mkResourceSliceWithDevices(tweakDeviceAttribute("test.io/int", resource.DeviceAttribute{BoolValue: ptr.To(true)})),
				},
				"invalid update: device attribute with multiple values": {
					old:    mkResourceSliceWithDevices(),
					update: mkResourceSliceWithDevices(tweakDeviceAttribute("test.io/multiple", resource.DeviceAttribute{IntValue: ptr.To[int64](123), BoolValue: ptr.To(true)})),
					expectedErrs: field.ErrorList{
						field.Invalid(field.NewPath("spec", "devices").Index(0).Child("attributes").Key("test.io/multiple"), "", "").WithOrigin("union"),
					},
				},
				// spec.sharedCounters
				"valid update: at limit shared counters": {
					old:    mkResourceSliceWithSharedCounters(),
					update: mkResourceSliceWithSharedCounters(tweakSharedCounters(resource.ResourceSliceMaxCounterSets)),
				},
				"invalid update: too many shared counters": {
					old:    mkResourceSliceWithSharedCounters(),
					update: mkResourceSliceWithSharedCounters(tweakSharedCounters(resource.ResourceSliceMaxCounterSets + 1)),
					expectedErrs: field.ErrorList{
						field.TooMany(field.NewPath("spec").Child("sharedCounters"), resource.ResourceSliceMaxCounterSets+1, resource.ResourceSliceMaxCounterSets).WithOrigin("maxItems"),
					},
				},
				// spec.devices.consumesCounters
				"valid update: at limit device consumes counters": {
					old:    mkResourceSliceWithDevices(),
					update: mkResourceSliceWithDevices(tweakDeviceConsumesCounters(resource.ResourceSliceMaxDeviceCounterConsumptionsPerDevice)),
				},
				"invalid update: too many device consumes counters": {
					old:    mkResourceSliceWithDevices(),
					update: mkResourceSliceWithDevices(tweakDeviceConsumesCounters(resource.ResourceSliceMaxDeviceCounterConsumptionsPerDevice + 1)),
					expectedErrs: field.ErrorList{
						field.TooMany(field.NewPath("spec", "devices").Index(0).Child("consumesCounters"), resource.ResourceSliceMaxDeviceCounterConsumptionsPerDevice+1, resource.ResourceSliceMaxDeviceCounterConsumptionsPerDevice).WithOrigin("maxItems"),
					},
				},
				// spec.sharedCounters.name
				"valid update: counter set name": {
					old:    mkResourceSliceWithSharedCounters(),
					update: mkResourceSliceWithSharedCounters(tweakSharedCountersName("valid-key")),
				},
				"invalid update: counter set name": {
					old:    mkResourceSliceWithSharedCounters(),
					update: mkResourceSliceWithSharedCounters(tweakSharedCountersName("InvalidKey")),
					expectedErrs: field.ErrorList{
						field.Invalid(field.NewPath("spec", "sharedCounters").Index(0).Child("name"), "InvalidKey", "").WithOrigin("format=k8s-short-name"),
					},
				},
				"invalid update: counter set name not set": {
					old:    mkResourceSliceWithSharedCounters(),
					update: mkResourceSliceWithSharedCounters(tweakSharedCountersName("")),
					expectedErrs: field.ErrorList{
						field.Required(field.NewPath("spec", "sharedCounters").Index(0).Child("name"), ""),
					},
				},
				// spec.devices.consumesCounters.counterSet
				"valid update: device consumes counters counter set name": {
					old:    mkResourceSliceWithDevices(),
					update: mkResourceSliceWithDevices(tweakDeviceConsumesCountersCounterSetName("valid-key")),
				},
				"invalid update: device consumes counters counter set name": {
					old:    mkResourceSliceWithDevices(),
					update: mkResourceSliceWithDevices(tweakDeviceConsumesCountersCounterSetName("InvalidKey")),
					expectedErrs: field.ErrorList{
						field.Invalid(field.NewPath("spec", "devices").Index(0).Child("consumesCounters").Index(0).Child("counterSet"), "InvalidKey", "").WithOrigin("format=k8s-short-name"),
					},
				},
				"invalidupdate: device consumes counters counter set name not set": {
					old:    mkResourceSliceWithDevices(),
					update: mkResourceSliceWithDevices(tweakDeviceConsumesCountersCounterSetName("")),
					expectedErrs: field.ErrorList{
						field.Required(field.NewPath("spec", "devices").Index(0).Child("consumesCounters").Index(0).Child("counterSet"), ""),
					},
				},
				// spec.sharedCounters
				"valid update: distinct names for shared counters": {
					old:    mkResourceSliceWithSharedCounters(),
					update: mkResourceSliceWithSharedCounters(tweakSharedCountersName("valid-key-1", "valid-key-2")),
				},
				"invalid update: duplicate names for shared counters": {
					old:    mkResourceSliceWithSharedCounters(),
					update: mkResourceSliceWithSharedCounters(tweakSharedCountersName("duplicate-key", "duplicate-key")),
					expectedErrs: field.ErrorList{
						field.Duplicate(field.NewPath("spec").Child("sharedCounters").Index(1), "duplicate-key"),
					},
				},
				// spec.devices.consumesCounters
				"valid update: distinct names for counter set in device counter consumption": {
					old:    mkResourceSliceWithDevices(),
					update: mkResourceSliceWithDevices(tweakDeviceConsumesCountersCounterSetName("valid-key-1", "valid-key-2")),
				},
				"invalid update: duplicate names for counter set in device counter consumption": {
					old:    mkResourceSliceWithDevices(),
					update: mkResourceSliceWithDevices(tweakDeviceConsumesCountersCounterSetName("duplicate-key", "duplicate-key")),
					expectedErrs: field.ErrorList{
						field.Duplicate(field.NewPath("spec").Child("devices").Index(0).Child("consumesCounters").Index(1), "duplicate-key"),
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

func mkResourceSliceWithDevices(mutators ...func(*resource.ResourceSlice)) resource.ResourceSlice {
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

func mkResourceSliceWithSharedCounters(mutators ...func(*resource.ResourceSlice)) resource.ResourceSlice {
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
			SharedCounters: []resource.CounterSet{
				{
					Name: "shared-counter-set",
					Counters: map[string]resource.Counter{
						"valid-key": {},
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

func tweakSharedCountersName(names ...string) func(*resource.ResourceSlice) {
	return func(rs *resource.ResourceSlice) {
		var sharedCounters []resource.CounterSet
		for _, name := range names {
			sharedCounters = append(sharedCounters, resource.CounterSet{
				Name: name,
				Counters: map[string]resource.Counter{
					"valid-key": {},
				},
			})
		}
		rs.Spec.SharedCounters = sharedCounters
	}
}

func tweakSharedCounters(count int) func(*resource.ResourceSlice) {
	return func(rs *resource.ResourceSlice) {
		var counterSets []resource.CounterSet
		for i := 0; i < count; i++ {
			counterSets = append(counterSets, resource.CounterSet{
				Name: fmt.Sprintf("shared-counter-set-%d", i),
				Counters: map[string]resource.Counter{
					"valid-key": {},
				},
			})
		}
		rs.Spec.SharedCounters = counterSets
	}
}

func tweakDeviceConsumesCounters(count int) func(*resource.ResourceSlice) {
	return func(rs *resource.ResourceSlice) {
		var consumesCounters []resource.DeviceCounterConsumption
		for i := 0; i < count; i++ {
			consumesCounters = append(consumesCounters, resource.DeviceCounterConsumption{
				CounterSet: fmt.Sprintf("shared-counter-set-%d", i),
				Counters: map[string]resource.Counter{
					"valid-key": {},
				},
			})
		}
		rs.Spec.Devices[0].ConsumesCounters = consumesCounters
	}
}

func tweakDeviceConsumesCountersCounterSetName(counterSets ...string) func(*resource.ResourceSlice) {
	return func(rs *resource.ResourceSlice) {
		var consumesCounters []resource.DeviceCounterConsumption
		for _, counterSet := range counterSets {
			consumesCounters = append(consumesCounters, resource.DeviceCounterConsumption{
				CounterSet: counterSet,
				Counters: map[string]resource.Counter{
					"valid-key": {},
				},
			})
		}
		rs.Spec.Devices[0].ConsumesCounters = consumesCounters
	}
}
