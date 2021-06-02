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

func TestPolicyRestrictedName(t *testing.T) {
	tcases := []struct {
		name     string
		expected string
	}{
		{
			name:     "New Restricted Policy",
			expected: "restricted",
		},
	}
	for _, tc := range tcases {
		policy := NewRestrictedPolicy([]int{0, 1})
		if policy.Name() != tc.expected {
			t.Errorf("Expected Policy Name to be %s, got %s", tc.expected, policy.Name())
		}
	}
}

func TestPolicyRestrictedCanAdmitPodResult(t *testing.T) {
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
			name:     "Preferred is set to true in topology hints",
			hint:     TopologyHint{nil, true},
			expected: true,
		},
	}

	for _, tc := range tcases {
		numaNodes := []int{0, 1}
		policy := NewRestrictedPolicy(numaNodes)
		result := policy.(*restrictedPolicy).canAdmitPodResult(&tc.hint)

		if result != tc.expected {
			t.Errorf("Expected result to be %t, got %t", tc.expected, result)
		}
	}
}

func TestPolicyRestrictedMerge(t *testing.T) {
	numaNodes := []int{0, 1}
	policy := NewRestrictedPolicy(numaNodes)

	tcases := commonPolicyMergeTestCases(numaNodes)
	tcases = append(tcases, policy.(*restrictedPolicy).mergeTestCases(numaNodes)...)

	testPolicyMerge(policy, tcases, t)
}
