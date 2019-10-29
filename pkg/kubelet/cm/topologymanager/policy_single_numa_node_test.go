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
	"reflect"
	"testing"
)

func TestPolicySingleNumaNodeCanAdmitPodResult(t *testing.T) {
	tcases := []struct {
		name     string
		hint     TopologyHint
		expected bool
	}{
		{
			name:     "Preferred is set to false in topology hints",
			hint:     TopologyHint{nil, false},
			expected: false,
		},
		{
			name:     "NUMANodeAffinity has multiple NUMA Nodes masked in topology hints",
			hint:     TopologyHint{NewTestBitMask(0, 1), true},
			expected: false,
		},
		{
			name:     "NUMANodeAffinity has one NUMA Node masked in topology hints",
			hint:     TopologyHint{NewTestBitMask(0), true},
			expected: true,
		},
	}

	numaNodes := []int{0, 1}
	for _, tc := range tcases {
		policy := NewSingleNumaNodePolicy(numaNodes)
		hint := tc.hint
		result := policy.(*singleNumaNodePolicy).canAdmitPodResult(&hint)

		if result.Admit != tc.expected {
			t.Errorf("Expected Admit field in result to be %t, got %t", tc.expected, result.Admit)
		}

		if tc.expected == false {
			if len(result.Reason) == 0 {
				t.Errorf("Expected Reason field to be not empty")
			}
			if len(result.Message) == 0 {
				t.Errorf("Expected Message field to be not empty")
			}
		}
	}
}

func TestPolicySingleNumaNodeFilterHints(t *testing.T) {
	tcases := []struct {
		name              string
		allResources      [][]TopologyHint
		expectedResources [][]TopologyHint
	}{
		{
			name:              "filter empty resources",
			allResources:      [][]TopologyHint{},
			expectedResources: [][]TopologyHint(nil),
		},
		{
			name: "filter hints with nil socket mask",
			allResources: [][]TopologyHint{
				{
					{NUMANodeAffinity: nil, Preferred: false},
				},
				{
					{NUMANodeAffinity: nil, Preferred: true},
				},
			},
			expectedResources: [][]TopologyHint{
				[]TopologyHint(nil),
				[]TopologyHint(nil),
			},
		},
		{
			name: "filter hints with nil socket mask",
			allResources: [][]TopologyHint{
				{
					{NUMANodeAffinity: NewTestBitMask(0), Preferred: true},
					{NUMANodeAffinity: nil, Preferred: false},
				},
				{
					{NUMANodeAffinity: NewTestBitMask(1), Preferred: true},
					{NUMANodeAffinity: nil, Preferred: true},
				},
			},
			expectedResources: [][]TopologyHint{
				{
					{NUMANodeAffinity: NewTestBitMask(0), Preferred: true},
				},
				{
					{NUMANodeAffinity: NewTestBitMask(1), Preferred: true},
				},
			},
		},
		{
			name: "filter hints with empty resource socket mask",
			allResources: [][]TopologyHint{
				{
					{NUMANodeAffinity: NewTestBitMask(1), Preferred: true},
					{NUMANodeAffinity: NewTestBitMask(0), Preferred: true},
					{NUMANodeAffinity: nil, Preferred: false},
				},
				{},
			},
			expectedResources: [][]TopologyHint{
				{
					{NUMANodeAffinity: NewTestBitMask(1), Preferred: true},
					{NUMANodeAffinity: NewTestBitMask(0), Preferred: true},
				},
				[]TopologyHint(nil),
			},
		},
		{
			name: "filter hints with wide sockemask",
			allResources: [][]TopologyHint{
				{
					{NUMANodeAffinity: NewTestBitMask(0), Preferred: true},
					{NUMANodeAffinity: NewTestBitMask(1), Preferred: true},
					{NUMANodeAffinity: NewTestBitMask(1, 2), Preferred: false},
					{NUMANodeAffinity: NewTestBitMask(0, 1, 2), Preferred: false},
					{NUMANodeAffinity: nil, Preferred: false},
				},
				{
					{NUMANodeAffinity: NewTestBitMask(1, 2), Preferred: false},
					{NUMANodeAffinity: NewTestBitMask(0, 1, 2), Preferred: false},
					{NUMANodeAffinity: NewTestBitMask(0, 2), Preferred: false},
					{NUMANodeAffinity: NewTestBitMask(3), Preferred: false},
				},
				{
					{NUMANodeAffinity: NewTestBitMask(1, 2), Preferred: false},
					{NUMANodeAffinity: NewTestBitMask(0, 1, 2), Preferred: false},
					{NUMANodeAffinity: NewTestBitMask(0, 2), Preferred: false},
				},
			},
			expectedResources: [][]TopologyHint{
				{
					{NUMANodeAffinity: NewTestBitMask(0), Preferred: true},
					{NUMANodeAffinity: NewTestBitMask(1), Preferred: true},
				},
				{
					{NUMANodeAffinity: NewTestBitMask(3), Preferred: false},
				},
				[]TopologyHint(nil),
			},
		},
	}

	numaNodes := []int{0, 1, 2, 3}
	for _, tc := range tcases {
		policy := NewSingleNumaNodePolicy(numaNodes)
		actual := policy.(*singleNumaNodePolicy).filterHints(tc.allResources)
		if !reflect.DeepEqual(tc.expectedResources, actual) {
			t.Errorf("Test Case: %s", tc.name)
			t.Errorf("Expected result to be %v, got %v", tc.expectedResources, actual)
		}
	}
}

func TestPolicySingleNumaNodeGetHintMatch(t *testing.T) {
	tcases := []struct {
		name          string
		resources     [][]TopologyHint
		expectedFound bool
		expectedMatch TopologyHint
	}{
		{
			name: "match single resource single hint",
			resources: [][]TopologyHint{
				{
					{NUMANodeAffinity: NewTestBitMask(3), Preferred: true},
				},
			},
			expectedFound: true,
			expectedMatch: TopologyHint{
				NUMANodeAffinity: NewTestBitMask(3),
				Preferred:        true,
			},
		},
		{
			name: "match single resource multiple hints (Selected hint preferred is true) 1/2",
			resources: [][]TopologyHint{
				{
					{NUMANodeAffinity: NewTestBitMask(3), Preferred: true},
					{NUMANodeAffinity: NewTestBitMask(5), Preferred: false},
					{NUMANodeAffinity: NewTestBitMask(2), Preferred: true},
					{NUMANodeAffinity: NewTestBitMask(1), Preferred: true},
				},
			},
			expectedFound: true,
			expectedMatch: TopologyHint{
				NUMANodeAffinity: NewTestBitMask(1),
				Preferred:        true,
			},
		},
		{
			name: "match single resource multiple hints (Selected hint preferred is false) 2/2",
			resources: [][]TopologyHint{
				{
					{NUMANodeAffinity: NewTestBitMask(3), Preferred: true},
					{NUMANodeAffinity: NewTestBitMask(5), Preferred: false},
					{NUMANodeAffinity: NewTestBitMask(2), Preferred: true},
					{NUMANodeAffinity: NewTestBitMask(1), Preferred: false},
				},
			},
			expectedFound: true,
			expectedMatch: TopologyHint{
				NUMANodeAffinity: NewTestBitMask(1),
				Preferred:        true,
			},
		},
		{
			name: "match multiple resource single hint",
			resources: [][]TopologyHint{
				{
					{NUMANodeAffinity: NewTestBitMask(2), Preferred: true},
				},
				{
					{NUMANodeAffinity: NewTestBitMask(2), Preferred: true},
				},
				{
					{NUMANodeAffinity: NewTestBitMask(2), Preferred: false},
				},
			},
			expectedFound: true,
			expectedMatch: TopologyHint{
				NUMANodeAffinity: NewTestBitMask(2),
				Preferred:        true,
			},
		},
		{
			name: "match multiple resource single hint no match",
			resources: [][]TopologyHint{
				{
					{NUMANodeAffinity: NewTestBitMask(0), Preferred: true},
				},
				{
					{NUMANodeAffinity: NewTestBitMask(1), Preferred: true},
				},
				{
					{NUMANodeAffinity: NewTestBitMask(2), Preferred: false},
				},
			},
			expectedFound: false,
			expectedMatch: TopologyHint{},
		},
		{
			name: "multiple resources no match",
			resources: [][]TopologyHint{
				{
					{NUMANodeAffinity: NewTestBitMask(3), Preferred: true},
					{NUMANodeAffinity: NewTestBitMask(4), Preferred: false},
					{NUMANodeAffinity: NewTestBitMask(2), Preferred: true},
					{NUMANodeAffinity: NewTestBitMask(1), Preferred: false},
				},
				{
					{NUMANodeAffinity: NewTestBitMask(3), Preferred: true},
					{NUMANodeAffinity: NewTestBitMask(5), Preferred: false},
					{NUMANodeAffinity: NewTestBitMask(0), Preferred: true},
					{NUMANodeAffinity: NewTestBitMask(1), Preferred: false},
				},
				{
					{NUMANodeAffinity: NewTestBitMask(0), Preferred: true},
					{NUMANodeAffinity: NewTestBitMask(5), Preferred: true},
					{NUMANodeAffinity: NewTestBitMask(4), Preferred: true},
				},
			},
			expectedFound: false,
			expectedMatch: TopologyHint{},
		},
		{
			name: "multiple resources with match",
			resources: [][]TopologyHint{
				{
					{NUMANodeAffinity: NewTestBitMask(3), Preferred: false},
					{NUMANodeAffinity: NewTestBitMask(4), Preferred: false},
					{NUMANodeAffinity: NewTestBitMask(2), Preferred: true},
					{NUMANodeAffinity: NewTestBitMask(1), Preferred: true},
				},
				{
					{NUMANodeAffinity: NewTestBitMask(3), Preferred: true},
					{NUMANodeAffinity: NewTestBitMask(5), Preferred: false},
					{NUMANodeAffinity: NewTestBitMask(0), Preferred: true},
					{NUMANodeAffinity: NewTestBitMask(1), Preferred: false},
				},
				{
					{NUMANodeAffinity: NewTestBitMask(0), Preferred: true},
					{NUMANodeAffinity: NewTestBitMask(5), Preferred: true},
					{NUMANodeAffinity: NewTestBitMask(4), Preferred: true},
					{NUMANodeAffinity: NewTestBitMask(3), Preferred: true},
				},
			},
			expectedFound: true,
			expectedMatch: TopologyHint{
				NUMANodeAffinity: NewTestBitMask(3),
				Preferred:        true,
			},
		},
	}

	numaNodes := []int{0, 1, 2, 3, 4, 5}
	for _, tc := range tcases {
		policy := NewSingleNumaNodePolicy(numaNodes)
		found, match := policy.(*singleNumaNodePolicy).getHintMatch(tc.resources)
		if found != tc.expectedFound {
			t.Errorf("Test Case: %s", tc.name)
			t.Errorf("Expected found to be %v, got %v", tc.expectedFound, found)
		}
		if found {
			if !match.IsEqual(tc.expectedMatch) {
				t.Errorf("Test Case: %s", tc.name)
				t.Errorf("Expected match to be %v, got %v", tc.expectedMatch, match)
			}
		}
	}
}

func TestPolicySingleNumaNodeMerge(t *testing.T) {
	numaNodes := []int{0, 1}

	tcases := []struct {
		name           string
		providersHints []map[string][]TopologyHint
		expected       TopologyHint
	}{
		{
			name:           "No provider hints",
			providersHints: []map[string][]TopologyHint{},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestBitMask(numaNodes...),
				Preferred:        true,
			},
		},
		{
			name:           "One nil provider hints",
			providersHints: []map[string][]TopologyHint{nil},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestBitMask(numaNodes...),
				Preferred:        true,
			},
		},
		{
			name: "One provider hints with no resources",
			providersHints: []map[string][]TopologyHint{
				{},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestBitMask(numaNodes...),
				Preferred:        true,
			},
		},
		{
			name: "One provider hints with one resource and nil hints",
			providersHints: []map[string][]TopologyHint{
				{
					"resource/A": nil,
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestBitMask(numaNodes...),
				Preferred:        true,
			},
		},
		{
			name: "One provider hints with one resource no hints",
			providersHints: []map[string][]TopologyHint{
				{
					"resource/A": {},
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestBitMask(numaNodes...),
				Preferred:        false,
			},
		},
		{
			name: "Single Provider hint with single resource and single TopologyHint with Preferred as true and " +
				"NUMANodeAffinity as nil",
			providersHints: []map[string][]TopologyHint{
				{
					"resource/A": {
						{NUMANodeAffinity: nil, Preferred: true},
					},
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestBitMask(numaNodes...),
				Preferred:        false,
			},
		},
		{
			name: "Single Provider hint with single resource and single TopologyHint with Preferred as false and " +
				"NUMANodeAffinity as nil",
			providersHints: []map[string][]TopologyHint{
				{
					"resource/A": {
						{NUMANodeAffinity: nil, Preferred: false},
					},
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestBitMask(numaNodes...),
				Preferred:        false,
			},
		},
		{
			name: "Two providers, 1 resource with 1 hint each, same mask, both preferred 1/2",
			providersHints: []map[string][]TopologyHint{
				{
					"resource/A": {
						{NUMANodeAffinity: NewTestBitMask(0), Preferred: true},
					},
				},
				{
					"resource/B": {
						{NUMANodeAffinity: NewTestBitMask(0), Preferred: true},
					},
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestBitMask(0),
				Preferred:        true,
			},
		},
		{
			name: "Two providers, 1 resource with 1 hint each, same mask, both preferred 2/2",
			providersHints: []map[string][]TopologyHint{
				{
					"resource/A": {
						{NUMANodeAffinity: NewTestBitMask(1), Preferred: true},
					},
				},
				{
					"resource/B": {
						{NUMANodeAffinity: NewTestBitMask(1), Preferred: true},
					},
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestBitMask(1),
				Preferred:        true,
			},
		},
		{
			name: "Two providers, 1 resource with 1 hint each, 1 wider mask, both preferred 1/2",
			providersHints: []map[string][]TopologyHint{
				{
					"resource/A": {
						{NUMANodeAffinity: NewTestBitMask(0), Preferred: true},
					},
				},
				{
					"resource/B": {
						{NUMANodeAffinity: NewTestBitMask(0, 1), Preferred: true},
					},
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestBitMask(numaNodes...),
				Preferred:        false,
			},
		},
		{
			name: "Two providers, 1 resource with 1 hint each, 1 wider mask, both preferred 2/2",
			providersHints: []map[string][]TopologyHint{
				{
					"resource/A": {
						{NUMANodeAffinity: NewTestBitMask(1), Preferred: true},
					},
				},
				{
					"resource/B": {
						{NUMANodeAffinity: NewTestBitMask(0, 1), Preferred: true},
					},
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestBitMask(numaNodes...),
				Preferred:        false,
			},
		},
		{
			name: "Two providers, 1 resource with 1 hint each, no common mask",
			providersHints: []map[string][]TopologyHint{
				{
					"resource/A": {
						{NUMANodeAffinity: NewTestBitMask(0), Preferred: true},
					},
				},
				{
					"resource/B": {
						{NUMANodeAffinity: NewTestBitMask(1), Preferred: true},
					},
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestBitMask(numaNodes...),
				Preferred:        false,
			},
		},
		{
			name: "Two providers, 1 resource with 1 hint each, same mask, 1 preferred, 1 not 1/2",
			providersHints: []map[string][]TopologyHint{
				{
					"resource/A": {
						{NUMANodeAffinity: NewTestBitMask(0), Preferred: true},
					},
				},
				{
					"resource/B": {
						{NUMANodeAffinity: NewTestBitMask(0), Preferred: false},
					},
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestBitMask(0),
				Preferred:        true,
			},
		},
		{
			name: "Two providers, 1 resource with 1 hint each, same mask, 1 preferred, 1 not 2/2",
			providersHints: []map[string][]TopologyHint{
				{
					"resource/A": {
						{NUMANodeAffinity: NewTestBitMask(1), Preferred: false},
					},
				},
				{
					"resource/B": {
						{NUMANodeAffinity: NewTestBitMask(1), Preferred: true},
					},
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestBitMask(1),
				Preferred:        true,
			},
		},
		{
			name: "Two providers, 1 no resources, 1 single hint preferred 1/3",
			providersHints: []map[string][]TopologyHint{
				{},
				{
					"resource/B": {
						{NUMANodeAffinity: NewTestBitMask(0), Preferred: true},
					},
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestBitMask(0),
				Preferred:        true,
			},
		},
		{
			name: "Two providers, 1 with a resource and nil hints, 1 single resource single hint preferred 2/3",
			providersHints: []map[string][]TopologyHint{
				{
					"resource/A": nil,
				},
				{
					"resource/B": {
						{NUMANodeAffinity: NewTestBitMask(1), Preferred: true},
					},
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestBitMask(1),
				Preferred:        true,
			},
		},
		{
			name: "Two providers, 1 with a resource and no hints, 1 single resource single hint preferred 3/3",
			providersHints: []map[string][]TopologyHint{
				{
					"resource/A": {},
				},
				{
					"resource/B": {
						{NUMANodeAffinity: NewTestBitMask(1), Preferred: true},
					},
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestBitMask(numaNodes...),
				Preferred:        false,
			},
		},
		{
			name: "Two providers, 1 resource with 2 hints, 1 resource with single hint matching 1/2",
			providersHints: []map[string][]TopologyHint{
				{
					"resource/A": {
						{NUMANodeAffinity: NewTestBitMask(0), Preferred: true},
						{NUMANodeAffinity: NewTestBitMask(1), Preferred: true},
					},
				},
				{
					"resource/B": {
						{NUMANodeAffinity: NewTestBitMask(0), Preferred: true},
					},
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestBitMask(0),
				Preferred:        true,
			},
		},
		{
			name: "Two providers, 1 resource with 2 hints, 1 resource with single hint matching 2/2",
			providersHints: []map[string][]TopologyHint{
				{
					"resource/A": {
						{NUMANodeAffinity: NewTestBitMask(0), Preferred: true},
						{NUMANodeAffinity: NewTestBitMask(1), Preferred: true},
					},
				},
				{
					"resource/B": {
						{NUMANodeAffinity: NewTestBitMask(1), Preferred: true},
					},
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestBitMask(1),
				Preferred:        true,
			},
		},
		{
			name: "Two providers, 1 resource with 2 hints, 1 resource with single non-preferred hint matching",
			providersHints: []map[string][]TopologyHint{
				{
					"resource/A": {
						{NUMANodeAffinity: NewTestBitMask(0), Preferred: true},
						{NUMANodeAffinity: NewTestBitMask(1), Preferred: true},
					},
				},
				{
					"resource/B": {
						{NUMANodeAffinity: NewTestBitMask(0, 1), Preferred: false},
					},
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestBitMask(numaNodes...),
				Preferred:        false,
			},
		},
		{
			name: "Two providers, one resource each, both with 2 hints, matching narrower preferred hint from both",
			providersHints: []map[string][]TopologyHint{
				{
					"resource/A": {
						{NUMANodeAffinity: NewTestBitMask(1), Preferred: true},
						{NUMANodeAffinity: NewTestBitMask(0), Preferred: true},
					},
				},
				{
					"resource/B": {
						{NUMANodeAffinity: NewTestBitMask(1), Preferred: true},
						{NUMANodeAffinity: NewTestBitMask(0), Preferred: true},
					},
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestBitMask(0),
				Preferred:        true,
			},
		},
		{
			name: "Multiple resources, same provider",
			providersHints: []map[string][]TopologyHint{
				{
					"resource/A": {
						{NUMANodeAffinity: NewTestBitMask(1), Preferred: true},
						{NUMANodeAffinity: NewTestBitMask(0, 1), Preferred: true},
					},
					"resource/B": {
						{NUMANodeAffinity: NewTestBitMask(0), Preferred: true},
						{NUMANodeAffinity: NewTestBitMask(1), Preferred: true},
						{NUMANodeAffinity: NewTestBitMask(0, 1), Preferred: false},
					},
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestBitMask(1),
				Preferred:        true,
			},
		},
		{
			name: "2 providers, Multiple resources, all hints on 2 sockets",
			providersHints: []map[string][]TopologyHint{
				{
					"resource/A": {
						{NUMANodeAffinity: NewTestBitMask(0, 1), Preferred: true},
					},
					"resource/B": {
						{NUMANodeAffinity: NewTestBitMask(0, 1), Preferred: false},
					},
				},
				{
					"resource/C": {
						{NUMANodeAffinity: NewTestBitMask(0, 1), Preferred: true},
					},
					"resource/D": {
						{NUMANodeAffinity: NewTestBitMask(0, 1), Preferred: false},
					},
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestBitMask(numaNodes...),
				Preferred:        false,
			},
		},
	}

	for _, tc := range tcases {
		policy := NewSingleNumaNodePolicy(numaNodes)
		actual := policy.(*singleNumaNodePolicy).merge(tc.providersHints)
		if !actual.IsEqual(tc.expected) {
			t.Errorf("Test Case: %s", tc.name)
			t.Errorf("Expected result to be %v, got %v", tc.expected, actual)
		}
	}
}
