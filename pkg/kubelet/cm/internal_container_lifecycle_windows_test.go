//go:build windows

/*
Copyright 2024 The Kubernetes Authors.

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
	"testing"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/pkg/kubelet/winstats"
	"k8s.io/utils/cpuset"
)

func TestComputeCPUSet(t *testing.T) {
	affinities := []winstats.GroupAffinity{
		{Mask: 0b1010, Group: 0}, // CPUs 1 and 3 in Group 0
		{Mask: 0b1001, Group: 1}, // CPUs 0 and 3 in Group 1
	}

	expected := map[int]struct{}{
		1:  {}, // Group 0, CPU 1
		3:  {}, // Group 0, CPU 3
		64: {}, // Group 1, CPU 0
		67: {}, // Group 1, CPU 3
	}

	result := computeCPUSet(affinities)
	if len(result) != len(expected) {
		t.Errorf("expected length %v, but got length %v", len(expected), len(result))
	}
	for key := range expected {
		if _, exists := result[key]; !exists {
			t.Errorf("expected key %v to be in result", key)
		}
	}
}

func TestGroupMasks(t *testing.T) {
	tests := []struct {
		cpuSet   sets.Set[int]
		expected map[int]uint64
	}{
		{
			cpuSet: sets.New[int](0, 1, 2, 3, 64, 65, 66, 67),
			expected: map[int]uint64{
				0: 0b1111,
				1: 0b1111,
			},
		},
		{
			cpuSet: sets.New[int](0, 2, 64, 66),
			expected: map[int]uint64{
				0: 0b0101,
				1: 0b0101,
			},
		},
		{
			cpuSet: sets.New[int](1, 65),
			expected: map[int]uint64{
				0: 0b0010,
				1: 0b0010,
			},
		},
		{
			cpuSet:   sets.New[int](),
			expected: map[int]uint64{},
		},
	}

	for _, test := range tests {
		result := groupMasks(test.cpuSet)
		if len(result) != len(test.expected) {
			t.Errorf("expected length %v, but got length %v", len(test.expected), len(result))
		}
		for group, mask := range test.expected {
			if result[group] != mask {
				t.Errorf("expected group %v to have mask %v, but got mask %v", group, mask, result[group])
			}
		}
	}
}

func TestComputeFinalCpuSet(t *testing.T) {
	tests := []struct {
		name            string
		allocatedCPUs   cpuset.CPUSet
		allNumaNodeCPUs []winstats.GroupAffinity
		expectedCPUSet  sets.Set[int]
	}{
		{
			name:          "Both managers enabled, CPU manager selects more CPUs",
			allocatedCPUs: cpuset.New(0, 1, 2, 3),
			allNumaNodeCPUs: []winstats.GroupAffinity{
				{Mask: 0b0011, Group: 0}, // CPUs 0 and 1 in Group 0
			},
			expectedCPUSet: sets.New[int](0, 1, 2, 3),
		},
		{
			name:          "Both managers enabled, CPU manager selects fewer CPUs within NUMA nodes",
			allocatedCPUs: cpuset.New(0, 1),
			allNumaNodeCPUs: []winstats.GroupAffinity{
				{Mask: 0b1111, Group: 0}, // CPUs 0, 1, 2, 3 in Group 0
			},
			expectedCPUSet: sets.New[int](0, 1),
		},
		{
			name:          "Both managers enabled, CPU manager selects fewer CPUs outside NUMA nodes",
			allocatedCPUs: cpuset.New(0, 1),
			allNumaNodeCPUs: []winstats.GroupAffinity{
				{Mask: 0b1100, Group: 0}, // CPUs 2 and 3 in Group 0
			},
			expectedCPUSet: sets.New[int](0, 1, 2, 3),
		},
		{
			name:            "Only CPU manager enabled",
			allocatedCPUs:   cpuset.New(0, 1),
			allNumaNodeCPUs: nil,
			expectedCPUSet:  sets.New[int](0, 1),
		},
		{
			name:          "Only memory manager enabled",
			allocatedCPUs: cpuset.New(),
			allNumaNodeCPUs: []winstats.GroupAffinity{
				{Mask: 0b1100, Group: 0}, // CPUs 2 and 3 in Group 0
			},
			expectedCPUSet: sets.New[int](2, 3),
		},
		{
			name:            "Neither manager enabled",
			allocatedCPUs:   cpuset.New(),
			allNumaNodeCPUs: nil,
			expectedCPUSet:  nil,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			result := computeFinalCpuSet(test.allocatedCPUs, test.allNumaNodeCPUs)
			if !result.Equal(test.expectedCPUSet) {
				t.Errorf("expected %v, but got %v", test.expectedCPUSet, result)
			}
		})
	}
}
