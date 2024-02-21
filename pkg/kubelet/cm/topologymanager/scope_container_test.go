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

package topologymanager

import (
	"reflect"
	"testing"

	v1 "k8s.io/api/core/v1"
)

func TestContainerCalculateAffinity(t *testing.T) {
	tcases := []struct {
		name     string
		prov     []ResourceAllocator
		expected []map[string][]TopologyHint
	}{
		{
			name:     "No hint providers",
			prov:     []ResourceAllocator{},
			expected: ([]map[string][]TopologyHint)(nil),
		},
		{
			name: "HintProvider returns empty non-nil map[string][]TopologyHint",
			prov: []ResourceAllocator{
				&fakeResourceAllocator{
					map[string][]TopologyHint{},
				},
			},
			expected: []map[string][]TopologyHint{
				{},
			},
		},
		{
			name: "HintProvider returns -nil map[string][]TopologyHint from provider",
			prov: []ResourceAllocator{
				&fakeResourceAllocator{
					map[string][]TopologyHint{
						"resource": nil,
					},
				},
			},
			expected: []map[string][]TopologyHint{
				{
					"resource": nil,
				},
			},
		},
		{
			name: "Assorted HintProviders",
			prov: []ResourceAllocator{
				&fakeResourceAllocator{
					map[string][]TopologyHint{
						"resource-1/A": {
							{NUMANodeAffinity: NewTestBitMask(0), Preferred: true},
							{NUMANodeAffinity: NewTestBitMask(0, 1), Preferred: false},
						},
						"resource-1/B": {
							{NUMANodeAffinity: NewTestBitMask(1), Preferred: true},
							{NUMANodeAffinity: NewTestBitMask(1, 2), Preferred: false},
						},
					},
				},
				&fakeResourceAllocator{
					map[string][]TopologyHint{
						"resource-2/A": {
							{NUMANodeAffinity: NewTestBitMask(2), Preferred: true},
							{NUMANodeAffinity: NewTestBitMask(3, 4), Preferred: false},
						},
						"resource-2/B": {
							{NUMANodeAffinity: NewTestBitMask(2), Preferred: true},
							{NUMANodeAffinity: NewTestBitMask(3, 4), Preferred: false},
						},
					},
				},
				&fakeResourceAllocator{
					map[string][]TopologyHint{
						"resource-3": nil,
					},
				},
			},
			expected: []map[string][]TopologyHint{
				{
					"resource-1/A": {
						{NUMANodeAffinity: NewTestBitMask(0), Preferred: true},
						{NUMANodeAffinity: NewTestBitMask(0, 1), Preferred: false},
					},
					"resource-1/B": {
						{NUMANodeAffinity: NewTestBitMask(1), Preferred: true},
						{NUMANodeAffinity: NewTestBitMask(1, 2), Preferred: false},
					},
				},
				{
					"resource-2/A": {
						{NUMANodeAffinity: NewTestBitMask(2), Preferred: true},
						{NUMANodeAffinity: NewTestBitMask(3, 4), Preferred: false},
					},
					"resource-2/B": {
						{NUMANodeAffinity: NewTestBitMask(2), Preferred: true},
						{NUMANodeAffinity: NewTestBitMask(3, 4), Preferred: false},
					},
				},
				{
					"resource-3": nil,
				},
			},
		},
	}

	for _, tc := range tcases {
		ctnScope := &containerScope{
			scope{
				providers: tc.prov,
				policy:    &mockPolicy{},
				name:      podTopologyScope,
			},
		}

		ctnScope.calculateAffinity(&v1.Pod{}, &v1.Container{})
		actual := ctnScope.policy.(*mockPolicy).ph
		if !reflect.DeepEqual(tc.expected, actual) {
			t.Errorf("Test Case: %s", tc.name)
			t.Errorf("Expected result to be %v, got %v", tc.expected, actual)
		}
	}
}

func TestContainerAccumulateProvidersHints(t *testing.T) {
	tcases := []struct {
		name     string
		prov     []ResourceAllocator
		expected []map[string][]TopologyHint
	}{
		{
			name:     "TopologyHint not set",
			prov:     []ResourceAllocator{},
			expected: nil,
		},
		{
			name: "HintProvider returns empty non-nil map[string][]TopologyHint",
			prov: []ResourceAllocator{
				&fakeResourceAllocator{
					map[string][]TopologyHint{},
				},
			},
			expected: []map[string][]TopologyHint{
				{},
			},
		},
		{
			name: "HintProvider returns - nil map[string][]TopologyHint from provider",
			prov: []ResourceAllocator{
				&fakeResourceAllocator{
					map[string][]TopologyHint{
						"resource": nil,
					},
				},
			},
			expected: []map[string][]TopologyHint{
				{
					"resource": nil,
				},
			},
		},
		{
			name: "2 HintProviders with 1 resource returns hints",
			prov: []ResourceAllocator{
				&fakeResourceAllocator{
					map[string][]TopologyHint{
						"resource1": {TopologyHint{}},
					},
				},
				&fakeResourceAllocator{
					map[string][]TopologyHint{
						"resource2": {TopologyHint{}},
					},
				},
			},
			expected: []map[string][]TopologyHint{
				{
					"resource1": {TopologyHint{}},
				},
				{
					"resource2": {TopologyHint{}},
				},
			},
		},
		{
			name: "2 HintProviders 1 with 1 resource 1 with nil hints",
			prov: []ResourceAllocator{
				&fakeResourceAllocator{
					map[string][]TopologyHint{
						"resource1": {TopologyHint{}},
					},
				},
				&fakeResourceAllocator{nil},
			},
			expected: []map[string][]TopologyHint{
				{
					"resource1": {TopologyHint{}},
				},
				nil,
			},
		},
		{
			name: "2 HintProviders 1 with 1 resource 1 empty hints",
			prov: []ResourceAllocator{
				&fakeResourceAllocator{
					map[string][]TopologyHint{
						"resource1": {TopologyHint{}},
					},
				},
				&fakeResourceAllocator{
					map[string][]TopologyHint{},
				},
			},
			expected: []map[string][]TopologyHint{
				{
					"resource1": {TopologyHint{}},
				},
				{},
			},
		},
		{
			name: "HintProvider with 2 resources returns hints",
			prov: []ResourceAllocator{
				&fakeResourceAllocator{
					map[string][]TopologyHint{
						"resource1": {TopologyHint{}},
						"resource2": {TopologyHint{}},
					},
				},
			},
			expected: []map[string][]TopologyHint{
				{
					"resource1": {TopologyHint{}},
					"resource2": {TopologyHint{}},
				},
			},
		},
	}

	for _, tc := range tcases {
		ctnScope := containerScope{
			scope{
				providers: tc.prov,
			},
		}
		actual := ctnScope.accumulateProvidersHints(&v1.Pod{}, &v1.Container{})
		if !reflect.DeepEqual(actual, tc.expected) {
			t.Errorf("Test Case %s: Expected NUMANodeAffinity in result to be %v, got %v", tc.name, tc.expected, actual)
		}
	}
}
