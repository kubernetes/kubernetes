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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/utils/ptr"
)

// NewMaxResourceSlices creates slices that are as large as possible given the current validation constraints.
func NewMaxResourceSlices() map[string]*resourceapi.ResourceSlice {
	slices := map[string]*resourceapi.ResourceSlice{
		"basic":                  newBasicResourceSlice(resourceapi.ResourceSliceMaxDevices),
		"with-consumed-counters": newResourceSliceWithConsumedCounters(resourceapi.ResourceSliceMaxDevicesWithConsumesCounters),
		"with-shared-counters":   newSharedCountersResourceSlice(),
	}
	return slices
}

// NewMaxResourceSlice creates a slice that is as large as possible given the current validation constraints.
func newBasicResourceSlice(numDevices int) *resourceapi.ResourceSlice {
	slice := commonResourceSlice()
	slice.Spec.PerDeviceNodeSelection = ptr.To(true)
	var devices []resourceapi.Device
	for i := 0; i < numDevices; i++ {
		devices = append(devices, resourceapi.Device{
			Name: maxDNSLabel(i),
			// Use attributes rather than capacity since it is more expensive.
			Attributes: func() map[resourceapi.QualifiedName]resourceapi.DeviceAttribute {
				attributes := make(map[resourceapi.QualifiedName]resourceapi.DeviceAttribute)
				for i := 0; i < resourceapi.ResourceSliceMaxAttributesAndCapacitiesPerDevice; i++ {
					attributes[maxResourceQualifiedName(i)] = resourceapi.DeviceAttribute{
						StringValue: ptr.To(maxDNSLabel(i)),
					}
				}
				return attributes
			}(),
			NodeName: ptr.To(maxSubDomain(0)),
			Taints: func() []resourceapi.DeviceTaint {
				var taints []resourceapi.DeviceTaint
				for i := 0; i < resourceapi.DeviceTaintsMaxLength; i++ {
					taints = append(taints, resourceapi.DeviceTaint{
						Key:       maxLabelName(i),
						Value:     maxLabelValue(i),
						Effect:    resourceapi.DeviceTaintEffectNoSchedule,
						TimeAdded: &metav1.Time{Time: time.Now().Truncate(time.Second)},
					})
				}
				return taints
			}(),
		})
	}
	slice.Spec.Devices = devices
	return slice
}

func newResourceSliceWithConsumedCounters(numDevices int) *resourceapi.ResourceSlice {
	slice := commonResourceSlice()
	slice.Spec.PerDeviceNodeSelection = ptr.To(true)
	var devices []resourceapi.Device
	for i := 0; i < numDevices; i++ {
		devices = append(devices, resourceapi.Device{
			Name: maxDNSLabel(i),
			// Use attributes rather than capacity since it is more expensive.
			Attributes: func() map[resourceapi.QualifiedName]resourceapi.DeviceAttribute {
				attributes := make(map[resourceapi.QualifiedName]resourceapi.DeviceAttribute)
				for i := 0; i < resourceapi.ResourceSliceMaxAttributesAndCapacitiesPerDevice; i++ {
					attributes[maxResourceQualifiedName(i)] = resourceapi.DeviceAttribute{
						StringValue: ptr.To(maxDNSLabel(i)),
					}
				}
				return attributes
			}(),
			ConsumesCounters: func() []resourceapi.DeviceCounterConsumption {
				var consumesCounters []resourceapi.DeviceCounterConsumption
				for i := 0; i < resourceapi.ResourceSliceMaxDeviceCounterConsumptionsPerDevice; i++ {
					consumesCounters = append(consumesCounters, resourceapi.DeviceCounterConsumption{
						CounterSet: maxDNSLabel(i),
						Counters: func() map[string]resourceapi.Counter {
							counters := make(map[string]resourceapi.Counter)
							for i := 0; i < resourceapi.ResourceSliceMaxCountersPerDeviceCounterConsumption; i++ {
								counters[maxDNSLabel(i)] = resourceapi.Counter{
									Value: resource.MustParse("80Gi"),
								}
							}
							return counters
						}(),
					})
				}
				return consumesCounters
			}(),
			NodeName: ptr.To(maxSubDomain(0)),
			Taints: func() []resourceapi.DeviceTaint {
				var taints []resourceapi.DeviceTaint
				for i := 0; i < resourceapi.DeviceTaintsMaxLength; i++ {
					taints = append(taints, resourceapi.DeviceTaint{
						Key:       maxLabelName(i),
						Value:     maxLabelValue(i),
						Effect:    resourceapi.DeviceTaintEffectNoSchedule,
						TimeAdded: &metav1.Time{Time: time.Now().Truncate(time.Second)},
					})
				}
				return taints
			}(),
		})
	}
	slice.Spec.Devices = devices
	return slice
}

func newSharedCountersResourceSlice() *resourceapi.ResourceSlice {
	slice := commonResourceSlice()
	slice.Spec.NodeName = ptr.To(maxSubDomain(0))
	var counterSets []resourceapi.CounterSet
	for i := 0; i < resourceapi.ResourceSliceMaxCounterSets; i++ {
		counterSets = append(counterSets, resourceapi.CounterSet{
			Name: maxDNSLabel(i),
			Counters: func() map[string]resourceapi.Counter {
				counters := make(map[string]resourceapi.Counter)
				for i := 0; i < resourceapi.ResourceSliceMaxCountersPerCounterSet; i++ {
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
	for i := 0; i < n; i++ {
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
	return maxString(0, validation.LabelValueMaxLength)
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
