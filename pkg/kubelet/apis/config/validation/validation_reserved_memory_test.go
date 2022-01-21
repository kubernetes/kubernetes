/*
Copyright 2020 The Kubernetes Authors.

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

package validation

import (
	"fmt"
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
)

func TestValidateReservedMemoryConfiguration(t *testing.T) {
	testCases := []struct {
		description          string
		kubeletConfiguration *kubeletconfig.KubeletConfiguration
		expectedError        error
	}{
		{
			description:          "The kubelet configuration does not have reserved memory parameter",
			kubeletConfiguration: &kubeletconfig.KubeletConfiguration{},
			expectedError:        nil,
		},
		{
			description: "The kubelet configuration has valid reserved memory parameter",
			kubeletConfiguration: &kubeletconfig.KubeletConfiguration{
				ReservedMemory: []kubeletconfig.MemoryReservation{
					{
						NumaNode: 0,
						Limits: v1.ResourceList{
							v1.ResourceMemory: *resource.NewQuantity(128, resource.DecimalSI),
						},
					},
				},
			},
			expectedError: nil,
		},
		{
			description: "The reserved memory has duplications for the NUMA node and limit type",
			kubeletConfiguration: &kubeletconfig.KubeletConfiguration{
				ReservedMemory: []kubeletconfig.MemoryReservation{
					{
						NumaNode: 0,
						Limits: v1.ResourceList{
							v1.ResourceMemory: *resource.NewQuantity(128, resource.DecimalSI),
						},
					},
					{
						NumaNode: 0,
						Limits: v1.ResourceList{
							v1.ResourceMemory: *resource.NewQuantity(64, resource.DecimalSI),
						},
					},
				},
			},
			expectedError: fmt.Errorf("invalid configuration: the reserved memory has a duplicate value for NUMA node %d and resource %q", 0, v1.ResourceMemory),
		},
		{
			description: "The reserved memory has unsupported limit type",
			kubeletConfiguration: &kubeletconfig.KubeletConfiguration{
				ReservedMemory: []kubeletconfig.MemoryReservation{
					{
						NumaNode: 0,
						Limits: v1.ResourceList{
							"blabla": *resource.NewQuantity(128, resource.DecimalSI),
						},
					},
				},
			},
			expectedError: fmt.Errorf("invalid configuration: the limit type %q for NUMA node %d is not supported, only [memory hugepages-<HugePageSize>] is accepted", "blabla", 0),
		},
		{
			description: "The reserved memory has limit type with zero value",
			kubeletConfiguration: &kubeletconfig.KubeletConfiguration{
				ReservedMemory: []kubeletconfig.MemoryReservation{
					{
						NumaNode: 0,
						Limits: v1.ResourceList{
							v1.ResourceMemory: *resource.NewQuantity(0, resource.DecimalSI),
						},
					},
				},
			},
			expectedError: fmt.Errorf("invalid configuration: reserved memory may not be zero for NUMA node %d and resource %q", 0, v1.ResourceMemory),
		},
	}

	for _, testCase := range testCases {
		errors := validateReservedMemoryConfiguration(testCase.kubeletConfiguration)

		if len(errors) != 0 && testCase.expectedError == nil {
			t.Errorf("expected errors %v, got %v", errors, testCase.expectedError)
		}

		if testCase.expectedError != nil {
			if len(errors) == 0 {
				t.Errorf("expected error %v, got %v", testCase.expectedError, errors)
			}

			if errors[0].Error() != testCase.expectedError.Error() {
				t.Errorf("expected error %v, got %v", testCase.expectedError, errors[0])
			}
		}
	}
}
