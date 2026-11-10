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

package dra

import (
	"fmt"
	"math"
	"strings"
	"time"

	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/api/validate/content"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/utils/ptr"
)

// Object size budget for ResourceSlice
//
// All limits in the resource.k8s.io API together have to fit into 1.5 MiB, the
// default maximum object size in etcd.
//
// Why the device limit is conditional and attribute entries and values have
// separate limits:
//
//   - The device limit is halved as soon as one device uses a feature which adds
//     a significant number of bytes, instead of giving each feature a limit of its
//     own. A table of limits per feature combination would be more precise, but
//     harder to reason about.
//
//   - The combined limit on attribute and capacity entries bounds the space used
//     by map keys and entry overhead. A separate per-device limit counts all
//     attribute values, including each scalar value and each list item, to bound
//     the payload size even when lists are used. The value limit was chosen to
//     leave room in the object size budget for future device features.
//
// Only the largest valid slices are built here, not every feature combination:
// what has to fit into the budget is the maximum.
//
// Sizes measured with the command at the end of this comment:
//
//	variant            devices    spec size    total size    % of budget
//	basic                  128      729,946       744,798             47
//	advanced                64    1,356,250     1,371,101             87
//	shared-counters          0       20,856        35,700              2
//
// Total size minus spec size is almost entirely ManagedFields. The total varies
// by a byte or two between runs because the encoded timestamps differ in length.
//
// These fields are not at their maximum:
//
//   - capacity: its entries share the limit of 32 with the attributes, and
//     attributes encode to more bytes, so the maximum uses attributes only.
//   - nodeSelector: devices set nodeName instead. A node selector term has no
//     size limit of its own, so a slice using per-device node selectors can be
//     larger than anything measured here.
//
// To re-measure after changing a limit or adding a field:
//
//	go test -v -run='TestDRA/all/MaxResourceSlice' ./test/integration/dra/all -args -v=0

// maxObjectSize is the 1.5 MiB budget described above. testMaxResourceSlice
// asserts it for each of the objects built here.
const maxObjectSize = 1536 * 1024

// NewMaxResourceSlices creates slices that are as large as possible given the current validation constraints.
func NewMaxResourceSlices() map[string]*resourceapi.ResourceSlice {
	slices := map[string]*resourceapi.ResourceSlice{
		"basic": newBasicResourceSlice(resourceapi.ResourceSliceMaxDevices),
		// advanced combines all device-level features (taints, consumes counters
		// with compatibility groups, list values, and binding conditions) into the
		// largest possible device-based slice.
		// Compatibility groups grow this maximal slice by ~16,640 B (1,190,621 B -> ~1,207,261 B, +1.4%),
		// binding conditions by another ~163,840 B (~1,207,262 B -> ~1,371,101 B, +13.6%),
		// which stays within the object size budget.
		"advanced": newAdvancedResourceSlice(),
		// shared-counters is a distinct slice kind: SharedCounters and Devices are
		// mutually exclusive per slice, so it cannot be merged into "advanced".
		"shared-counters": newSharedCountersResourceSlice(),
	}
	return slices
}

func newResourceSliceWithTaintsAndConsumesCounters() *resourceapi.ResourceSlice {
	slice := newBasicResourceSlice(resourceapi.ResourceSliceMaxDevicesWithAdvancedFeatures)
	for i := range slice.Spec.Devices {
		for range resourceapi.DeviceTaintsMaxLength {
			slice.Spec.Devices[i].Taints = append(slice.Spec.Devices[i].Taints,
				resourceapi.DeviceTaint{
					Key:       maxLabelName(i),
					Value:     maxLabelValue(i),
					Effect:    resourceapi.DeviceTaintEffectNoSchedule,
					TimeAdded: &metav1.Time{Time: time.Now().Truncate(time.Second)},
				},
			)
		}
		slice.Spec.Devices[i].ConsumesCounters = func() []resourceapi.DeviceCounterConsumption {
			var consumesCounters []resourceapi.DeviceCounterConsumption
			for i := range resourceapi.ResourceSliceMaxDeviceCounterConsumptionsPerDevice {
				consumesCounters = append(consumesCounters, resourceapi.DeviceCounterConsumption{
					CounterSet: maxDNSLabel(i),
					Counters: func() map[string]resourceapi.Counter {
						counters := make(map[string]resourceapi.Counter)
						for i := range resourceapi.ResourceSliceMaxCountersPerDeviceCounterConsumption {
							counters[maxDNSLabel(i)] = resourceapi.Counter{
								Value: resource.MustParse("80Gi"),
							}
						}
						return counters
					}(),
				})
			}
			return consumesCounters
		}()
	}
	return slice
}

func newBasicResourceSlice(numDevices int) *resourceapi.ResourceSlice {
	slice := commonResourceSlice()
	slice.Spec.PerDeviceNodeSelection = ptr.To(true)
	var devices []resourceapi.Device
	for i := range numDevices {
		devices = append(devices, resourceapi.Device{
			Name: maxDNSLabel(i),
			// Use attributes rather than capacity since it is more expensive.
			Attributes: func() map[resourceapi.QualifiedName]resourceapi.DeviceAttribute {
				attributes := make(map[resourceapi.QualifiedName]resourceapi.DeviceAttribute)
				for i := range resourceapi.ResourceSliceMaxAttributesAndCapacitiesPerDevice {
					attributes[maxResourceQualifiedName(i)] = resourceapi.DeviceAttribute{
						StringValue: ptr.To(maxDNSLabel(i)),
					}
				}
				return attributes
			}(),
			NodeName: ptr.To(maxSubDomain(0)),
		})
	}
	slice.Spec.Devices = devices
	return slice
}

func newSharedCountersResourceSlice() *resourceapi.ResourceSlice {
	slice := commonResourceSlice()
	slice.Spec.NodeName = ptr.To(maxSubDomain(0))
	var counterSets []resourceapi.CounterSet
	for i := range resourceapi.ResourceSliceMaxCounterSets {
		counterSets = append(counterSets, resourceapi.CounterSet{
			Name: maxDNSLabel(i),
			Counters: func() map[string]resourceapi.Counter {
				counters := make(map[string]resourceapi.Counter)
				for i := range resourceapi.ResourceSliceMaxCountersPerCounterSet {
					counters[maxDNSLabel(i)] = resourceapi.Counter{
						Value: resource.MustParse("80Gi"),
					}
				}
				return counters
			}(),
		})
	}
	slice.Spec.SharedCounters = counterSets
	return slice
}

// newAdvancedResourceSlice builds the maximal device-feature slice: taints,
// consumes counters, compatibility groups, and list values combined.
func newAdvancedResourceSlice() *resourceapi.ResourceSlice {
	slice := newResourceSliceWithTaintsAndConsumesCounters()
	addListValues(slice)
	addCompatibilityGroups(slice)
	addBindingConditions(slice)
	return slice
}

// addCompatibilityGroups declares the maximum number of compatibility groups
// on each device counter consumption.
func addCompatibilityGroups(slice *resourceapi.ResourceSlice) {
	for i := range slice.Spec.Devices {
		for j := range slice.Spec.Devices[i].ConsumesCounters {
			groups := make([]string, 0, resourceapi.DeviceCompatibilityGroupsMaxSize)
			for k := range resourceapi.DeviceCompatibilityGroupsMaxSize {
				groups = append(groups, maxDNSLabel(k))
			}
			slice.Spec.Devices[i].ConsumesCounters[j].CompatibilityGroups = groups
		}
	}
}

// addBindingConditions declares the maximum number of binding conditions on
// each device. Both lists have to be set together and must not overlap.
func addBindingConditions(slice *resourceapi.ResourceSlice) {
	conditions := make([]string, 0, resourceapi.BindingConditionsMaxSize)
	for i := range resourceapi.BindingConditionsMaxSize {
		conditions = append(conditions, maxQualifiedName(i))
	}
	failureConditions := make([]string, 0, resourceapi.BindingFailureConditionsMaxSize)
	for i := range resourceapi.BindingFailureConditionsMaxSize {
		failureConditions = append(failureConditions, maxQualifiedName(resourceapi.BindingConditionsMaxSize+i))
	}
	for i := range slice.Spec.Devices {
		slice.Spec.Devices[i].BindingConditions = conditions
		slice.Spec.Devices[i].BindingFailureConditions = failureConditions
	}
}

func addListValues(slice *resourceapi.ResourceSlice) {
	for i, device := range slice.Spec.Devices {
		// Make each attribute a list of strings (adds some encoding overhead).
		// The first list gets as many additional strings as allowed by the overall value limit.
		numAdditionalValues := resourceapi.ResourceSliceMaxAttributeValuesPerDevice - resourceapi.ResourceSliceMaxAttributesAndCapacitiesPerDevice
		for k, v := range device.Attributes {
			v.StringValues = []string{*v.StringValue}
			v.StringValue = nil
			for range numAdditionalValues {
				v.StringValues = append(v.StringValues, v.StringValues[0])
			}
			numAdditionalValues = 0
			device.Attributes[k] = v
		}
		slice.Spec.Devices[i] = device
	}
}

func commonResourceSlice() *resourceapi.ResourceSlice {
	return &resourceapi.ResourceSlice{
		ObjectMeta: metav1.ObjectMeta{
			Name: maxSubDomain(1),
			// Number of labels is not restricted.
			Labels: maxKeyValueMap(10),
			// Total size of annotations is limited to TotalAnnotationSizeLimitB = 256 KB.
			// Let's be a bit more realistic.
			Annotations: maxKeyValueMap(10),
		},

		Spec: resourceapi.ResourceSliceSpec{
			Driver: strings.Repeat("x", resourceapi.DriverNameMaxLength),
			Pool: resourceapi.ResourcePool{
				Name:               strings.Repeat("x", resourceapi.PoolNameMaxLength),
				Generation:         math.MaxInt64,
				ResourceSliceCount: math.MaxInt64,
			},
		},
	}
}

// maxKeyValueMap produces a map for labels or annotations.
func maxKeyValueMap(n int) map[string]string {
	m := make(map[string]string)
	for i := range n {
		m[maxQualifiedName(i)] = maxLabelValue(0)
	}
	return m
}

func maxLabelName(i int) string {
	// A "label" is a qualified name.
	return maxQualifiedName(i)
}

func maxResourceQualifiedName(i int) resourceapi.QualifiedName {
	return resourceapi.QualifiedName(maxString(i, resourceapi.DeviceMaxDomainLength) + "/" + maxString(i, resourceapi.DeviceMaxIDLength))
}

func maxQualifiedName(i int) string {
	return maxString(0, validation.DNS1123SubdomainMaxLength-4) + ".com/" + maxString(i, 63 /* qualifiedNameMaxLength */)
}

func maxLabelValue(i int) string {
	return maxString(i, content.LabelValueMaxLength)
}

func maxSubDomain(i int) string {
	return maxString(i, validation.DNS1123SubdomainMaxLength)
}

func maxDNSLabel(i int) string {
	return maxString(i, validation.DNS1123LabelMaxLength)
}

func maxString(i, l int) string {
	return strings.Repeat("x", l-4) + fmt.Sprintf("%04d", i)
}
