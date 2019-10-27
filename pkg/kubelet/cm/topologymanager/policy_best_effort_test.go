/*
Copyright 2019 The Kubernetes Authors.

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
	"testing"
)

func TestPolicyBestEffortName(t *testing.T) {
	tcases := []struct {
		name     string
		expected string
	}{
		{
			name:     "New Best Effort Policy",
			expected: "best-effort",
		},
	}
	for _, tc := range tcases {
		policy := NewBestEffortPolicy()
		if policy.Name() != tc.expected {
			t.Errorf("Expected Policy Name to be %s, got %s", tc.expected, policy.Name())
		}
	}
}

func TestPolicyBestEffortCanAdmitPodResult(t *testing.T) {
	tcases := []struct {
		name     string
		hint     TopologyHint
		expected bool
	}{
		{
			name:     "Preferred is set to false in topology hints",
			hint:     TopologyHint{nil, false},
			expected: true,
		},
		{
			name:     "Preferred is set to true in topology hints",
			hint:     TopologyHint{nil, true},
			expected: true,
		},
	}

	for _, tc := range tcases {
		policy := NewBestEffortPolicy()
		result := policy.CanAdmitPodResult(&tc.hint)

		if result.Admit != tc.expected {
			t.Errorf("Expected Admit field in result to be %t, got %t", tc.expected, result.Admit)
		}
	}
}

func TestPolicyBestEffortMerge(t *testing.T) {
	testOptimalMerge(t, NewBestEffortPolicy())
}

func testOptimalMerge(t *testing.T, policy Policy) {
	numaNodes := []int{0, 1}

	tcases := []struct {
		name           string
		providersHints []map[string][]TopologyHint
		expected       TopologyHint
	}{
		{
			name:           "Empty providers Hints",
			providersHints: []map[string][]TopologyHint{},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestBitMask(numaNodes...),
				Preferred:        true,
			},
		},
		{
			name:           "providers Hints with nil map",
			providersHints: []map[string][]TopologyHint{nil},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestBitMask(numaNodes...),
				Preferred:        true,
			},
		},
		{
			name: "providersHints with resource and nil []TopologyHint",
			providersHints: []map[string][]TopologyHint{
				{
					"resource": nil,
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestBitMask(numaNodes...),
				Preferred:        true,
			},
		},
		{
			name: "providersHints with empty non-nil []TopologyHint",
			providersHints: []map[string][]TopologyHint{
				{
					"resource": {},
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestBitMask(numaNodes...),
				Preferred:        false,
			},
		},
		{
			name: "Single TopologyHint with Preferred as true and NUMANodeAffinity as nil",
			providersHints: []map[string][]TopologyHint{
				{
					"resource": {
						{
							NUMANodeAffinity: nil,
							Preferred:        true,
						},
					},
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestBitMask(numaNodes...),
				Preferred:        true,
			},
		},
		{
			name: "Single TopologyHint with Preferred as false and NUMANodeAffinity as nil",
			providersHints: []map[string][]TopologyHint{
				{
					"resource": {
						{
							NUMANodeAffinity: nil,
							Preferred:        false,
						},
					},
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestBitMask(numaNodes...),
				Preferred:        true,
			},
		},
		{
			name: "Two resources, 1 hint each, same mask, both preferred 1/2",
			providersHints: []map[string][]TopologyHint{
				{
					"resource1": {
						{
							NUMANodeAffinity: NewTestBitMask(0),
							Preferred:        true},
					},
					"resource2": {
						{
							NUMANodeAffinity: NewTestBitMask(0),
							Preferred:        true,
						},
					},
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestBitMask(0),
				Preferred:        true,
			},
		},
		{
			name: "Two resources, 1 hint each, same mask, both preferred 2/2",
			providersHints: []map[string][]TopologyHint{
				{
					"resource1": {
						{
							NUMANodeAffinity: NewTestBitMask(1),
							Preferred:        true,
						},
					},
					"resource2": {
						{
							NUMANodeAffinity: NewTestBitMask(1),
							Preferred:        true,
						},
					},
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestBitMask(1),
				Preferred:        true,
			},
		},
		{
			name: "Two resources, 1 hint each, 1 wider mask, both preferred 1/2",
			providersHints: []map[string][]TopologyHint{
				{
					"resource1": {
						{
							NUMANodeAffinity: NewTestBitMask(0),
							Preferred:        true,
						},
					},
				},
				{
					"resource2": {
						{
							NUMANodeAffinity: NewTestBitMask(0, 1),
							Preferred:        true,
						},
					},
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestBitMask(0),
				Preferred:        true,
			},
		},
		{
			name: "Two resources, 1 hint each, 1 wider mask, both preferred 2/2",
			providersHints: []map[string][]TopologyHint{
				{
					"resource1": {
						{
							NUMANodeAffinity: NewTestBitMask(1),
							Preferred:        true,
						},
					},
					"resource2": {
						{
							NUMANodeAffinity: NewTestBitMask(0, 1),
							Preferred:        true,
						},
					},
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestBitMask(1),
				Preferred:        true,
			},
		},
		{
			name: "Two resources, 1 hint each, no common mask",
			providersHints: []map[string][]TopologyHint{
				{
					"resource1": {
						{
							NUMANodeAffinity: NewTestBitMask(0),
							Preferred:        true,
						},
					},
					"resource2": {
						{
							NUMANodeAffinity: NewTestBitMask(1),
							Preferred:        true,
						},
					},
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestBitMask(numaNodes...),
				Preferred:        false,
			},
		},
		{
			name: "Two resources, 1 hint each, same mask, 1 preferred, 1 not 1/2",
			providersHints: []map[string][]TopologyHint{
				{
					"resource1": {
						{
							NUMANodeAffinity: NewTestBitMask(0),
							Preferred:        true,
						},
					},
					"resource2": {
						{
							NUMANodeAffinity: NewTestBitMask(0),
							Preferred:        false,
						},
					},
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestBitMask(0),
				Preferred:        false,
			},
		},
		{
			name: "Two resources, 1 hint each, same mask, 1 preferred, 1 not 2/2",
			providersHints: []map[string][]TopologyHint{
				{
					"resource1": {
						{
							NUMANodeAffinity: NewTestBitMask(1),
							Preferred:        true,
						},
					},
					"resource2": {
						{
							NUMANodeAffinity: NewTestBitMask(1),
							Preferred:        false,
						},
					},
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestBitMask(1),
				Preferred:        false,
			},
		},
		{
			name: "Two resources, 1 with 2 hints, 1 with single hint matching 1/2",
			providersHints: []map[string][]TopologyHint{
				{
					"resource1": {
						{
							NUMANodeAffinity: NewTestBitMask(0),
							Preferred:        true,
						},
						{
							NUMANodeAffinity: NewTestBitMask(1),
							Preferred:        true,
						},
					},
					"resource2": {
						{
							NUMANodeAffinity: NewTestBitMask(0),
							Preferred:        true,
						},
					},
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestBitMask(0),
				Preferred:        true,
			},
		},
		{
			name: "Two resources, 1 with 2 hints, 1 with single hint matching 2/2",
			providersHints: []map[string][]TopologyHint{
				{
					"resource1": {
						{
							NUMANodeAffinity: NewTestBitMask(0),
							Preferred:        true,
						},
						{
							NUMANodeAffinity: NewTestBitMask(1),
							Preferred:        true,
						},
					},
					"resource2": {
						{
							NUMANodeAffinity: NewTestBitMask(1),
							Preferred:        true,
						},
					},
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestBitMask(1),
				Preferred:        true,
			},
		},
		{
			name: "Two resources, 1 with 2 hints, 1 with single non-preferred hint matching",
			providersHints: []map[string][]TopologyHint{
				{
					"resource1": {
						{
							NUMANodeAffinity: NewTestBitMask(0),
							Preferred:        true,
						},
						{
							NUMANodeAffinity: NewTestBitMask(1),
							Preferred:        true,
						},
					},
					"resource2": {
						{
							NUMANodeAffinity: NewTestBitMask(0, 1),
							Preferred:        false,
						},
					},
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestBitMask(0),
				Preferred:        false,
			},
		},
		{
			name: "Two resources, both with 2 hints, matching narrower preferred hint from both",
			providersHints: []map[string][]TopologyHint{
				{
					"resource1": {
						{
							NUMANodeAffinity: NewTestBitMask(0),
							Preferred:        true,
						},
						{
							NUMANodeAffinity: NewTestBitMask(1),
							Preferred:        true,
						},
					},
					"resource2": {
						{
							NUMANodeAffinity: NewTestBitMask(0),
							Preferred:        true,
						},
						{
							NUMANodeAffinity: NewTestBitMask(0, 1),
							Preferred:        false,
						},
					},
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestBitMask(0),
				Preferred:        true,
			},
		},
		{
			name: "Ensure less narrow preferred hints are chosen over narrower non-preferred hints",
			providersHints: []map[string][]TopologyHint{
				{
					"resource1": {
						{
							NUMANodeAffinity: NewTestBitMask(1),
							Preferred:        true,
						},
						{
							NUMANodeAffinity: NewTestBitMask(0, 1),
							Preferred:        false,
						},
					},
					"resource2": {
						{
							NUMANodeAffinity: NewTestBitMask(0),
							Preferred:        true,
						},
						{
							NUMANodeAffinity: NewTestBitMask(1),
							Preferred:        true,
						},
						{
							NUMANodeAffinity: NewTestBitMask(0, 1),
							Preferred:        false,
						},
					},
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestBitMask(1),
				Preferred:        true,
			},
		},
	}

	for _, tc := range tcases {
		result := policy.Merge(tc.providersHints, numaNodes)
		if !result.IsEqual(tc.expected) {
			t.Errorf("Test Case: %s: Expected merge hint to be %v, got %v", tc.name, tc.expected.String(), result.String())
		}
	}
}
