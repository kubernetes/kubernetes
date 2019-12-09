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
	}

	for _, tc := range tcases {
		numaNodes := []int{0, 1}
		policy := NewSingleNumaNodePolicy(numaNodes)
		result := policy.(*singleNumaNodePolicy).canAdmitPodResult(&tc.hint)

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
		expectedExists    bool
	}{
		{
			name:              "filter empty resources",
			allResources:      [][]TopologyHint{},
			expectedResources: [][]TopologyHint(nil),
			expectedExists:    false,
		},
		{
			name: "filter hints with nil socket mask, preferred true",
			allResources: [][]TopologyHint{
				{
					{NUMANodeAffinity: nil, Preferred: true},
				},
			},
			expectedResources: [][]TopologyHint{
				[]TopologyHint(nil),
			},
			expectedExists: true,
		},

		{
			name: "filter hints with nil socket mask, preferred false",
			allResources: [][]TopologyHint{
				{
					{NUMANodeAffinity: nil, Preferred: false},
				},
			},
			expectedResources: [][]TopologyHint{
				[]TopologyHint(nil),
			},
			expectedExists: false,
		},
		{
			name: "filter hints with nil socket mask, preferred both true",
			allResources: [][]TopologyHint{
				{
					{NUMANodeAffinity: nil, Preferred: true},
				},
				{
					{NUMANodeAffinity: nil, Preferred: true},
				},
			},
			expectedResources: [][]TopologyHint{
				[]TopologyHint(nil),
				[]TopologyHint(nil),
			},
			expectedExists: true,
		},
		{
			name: "filter hints with nil socket mask, preferred both false",
			allResources: [][]TopologyHint{
				{
					{NUMANodeAffinity: nil, Preferred: false},
				},
				{
					{NUMANodeAffinity: nil, Preferred: false},
				},
			},
			expectedResources: [][]TopologyHint{
				[]TopologyHint(nil),
				[]TopologyHint(nil),
			},
			expectedExists: false,
		},

		{
			name: "filter hints with nil socket mask, preferred true and false",
			allResources: [][]TopologyHint{
				{
					{NUMANodeAffinity: nil, Preferred: true},
				},
				{
					{NUMANodeAffinity: nil, Preferred: false},
				},
			},
			expectedResources: [][]TopologyHint{
				[]TopologyHint(nil),
				[]TopologyHint(nil),
			},
			expectedExists: false,
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
			expectedExists: false,
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
			expectedExists: false,
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
				[]TopologyHint(nil),
				[]TopologyHint(nil),
			},
			expectedExists: false,
		},
	}

	numaNodes := []int{0, 1, 2, 3}
	for _, tc := range tcases {
		policy := NewSingleNumaNodePolicy(numaNodes)
		actual, exists := policy.(*singleNumaNodePolicy).filterHints(tc.allResources)
		if !reflect.DeepEqual(tc.expectedResources, actual) {
			t.Errorf("Test Case: %s", tc.name)
			t.Errorf("Expected result to be %v, got %v", tc.expectedResources, actual)
		}
		if !reflect.DeepEqual(tc.expectedResources, actual) {
			t.Errorf("Test Case: %s", tc.name)
			t.Errorf("Expected result to be %v, got %v", tc.expectedExists, exists)
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
	policy := NewSingleNumaNodePolicy(numaNodes)

	tcases := commonPolicyMergeTestCases(numaNodes)
	tcases = append(tcases, policy.(*singleNumaNodePolicy).mergeTestCases(numaNodes)...)

	testPolicyMerge(policy, tcases, t)
}
