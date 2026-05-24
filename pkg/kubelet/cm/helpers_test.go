/*
Copyright 2026 The Kubernetes Authors.

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

package cm

import (
	"reflect"
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	evictionapi "k8s.io/kubernetes/pkg/kubelet/eviction/api"
)

func TestHardEvictionReservation(t *testing.T) {
	capacity := v1.ResourceList{
		v1.ResourceMemory:           resource.MustParse("8Gi"),
		v1.ResourceEphemeralStorage: resource.MustParse("100Gi"),
	}

	tests := []struct {
		name       string
		thresholds []evictionapi.Threshold
		expected   v1.ResourceList
	}{
		{
			name:       "empty thresholds returns nil",
			thresholds: nil,
			expected:   nil,
		},
		{
			name: "memory available threshold is reserved",
			thresholds: []evictionapi.Threshold{
				{
					Signal:   evictionapi.SignalMemoryAvailable,
					Operator: evictionapi.OpLessThan,
					Value: evictionapi.ThresholdValue{
						Quantity: resource.NewQuantity(500*1024*1024, resource.BinarySI),
					},
				},
			},
			expected: v1.ResourceList{
				v1.ResourceMemory: *resource.NewQuantity(500*1024*1024, resource.BinarySI),
			},
		},
		{
			name: "node filesystem available threshold is reserved",
			thresholds: []evictionapi.Threshold{
				{
					Signal:   evictionapi.SignalNodeFsAvailable,
					Operator: evictionapi.OpLessThan,
					Value: evictionapi.ThresholdValue{
						Quantity: resource.NewQuantity(10*1024*1024*1024, resource.BinarySI),
					},
				},
			},
			expected: v1.ResourceList{
				v1.ResourceEphemeralStorage: *resource.NewQuantity(10*1024*1024*1024, resource.BinarySI),
			},
		},
		{
			name: "non-less-than thresholds are ignored",
			thresholds: []evictionapi.Threshold{
				{
					Signal:   evictionapi.SignalMemoryAvailable,
					Operator: evictionapi.OpLessThan,
					Value: evictionapi.ThresholdValue{
						Quantity: resource.NewQuantity(500*1024*1024, resource.BinarySI),
					},
				},
				{
					Signal:   evictionapi.SignalMemoryAvailable,
					Operator: "GreaterThan",
					Value: evictionapi.ThresholdValue{
						Quantity: resource.NewQuantity(100*1024*1024, resource.BinarySI),
					},
				},
			},
			expected: v1.ResourceList{
				v1.ResourceMemory: *resource.NewQuantity(500*1024*1024, resource.BinarySI),
			},
		},
		{
			name: "unsupported signals are ignored",
			thresholds: []evictionapi.Threshold{
				{
					Signal:   evictionapi.SignalPIDAvailable,
					Operator: evictionapi.OpLessThan,
					Value: evictionapi.ThresholdValue{
						Quantity: resource.NewQuantity(100, resource.DecimalSI),
					},
				},
			},
			expected: v1.ResourceList{},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got := hardEvictionReservation(test.thresholds, capacity)
			if !reflect.DeepEqual(got, test.expected) {
				t.Errorf("hardEvictionReservation() = %#v, want %#v", got, test.expected)
			}
		})
	}
}
