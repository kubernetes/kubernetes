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

	resourceapi "k8s.io/api/resource/v1beta2"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/utils/ptr"

	_ "k8s.io/kubernetes/pkg/apis/resource/install"
)

// NewMaxResourceSlice creates a slice that is as large as possible given the current validation constraints.
func NewMaxResourceSlice() *resourceapi.ResourceSlice {
	slice := &resourceapi.ResourceSlice{
		ObjectMeta: metav1.ObjectMeta{
			Name: maxSubDomain(0),
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
			// use PerDeviceNodeSelection as it requires setting the node selection on
			// every device and therefore will be the most expensive option in terms of
			// object size.
			PerDeviceNodeSelection: ptr.To(true),
			// The validation caps the total number of counters across all CounterSets. So
			// the most expensive option is to have a single counter per CounterSet.
			SharedCounters: func() []resourceapi.CounterSet {
				var counterSets []resourceapi.CounterSet
				for i := 0; i < resourceapi.ResourceSliceMaxSharedCounters; i++ {
					counterSets = append(counterSets, resourceapi.CounterSet{
						Name: maxDNSLabel(i),
						Counters: map[string]resourceapi.Counter{
							maxDNSLabel(0): {
								Value: resource.MustParse("80Gi"),
							},
						},
					})
				}
				return counterSets
			}(),
			Devices: func() []resourceapi.Device {
				var devices []resourceapi.Device
				for i := 0; i < resourceapi.ResourceSliceMaxDevices; i++ {
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
							for i := 0; i < resourceapi.ResourceSliceMaxDeviceCountersPerSlice/resourceapi.ResourceSliceMaxDevices; i++ {
								consumesCounters = append(consumesCounters, resourceapi.DeviceCounterConsumption{
									CounterSet: maxDNSLabel(i),
									Counters: map[string]resourceapi.Counter{
										maxDNSLabel(0): {
											Value: resource.MustParse("80Gi"),
										},
									},
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
				return devices
			}(),
		},
	}
	return slice
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
