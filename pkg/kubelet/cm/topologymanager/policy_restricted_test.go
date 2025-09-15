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
	numaInfo := commonNUMAInfoTwoNodes()
	for _, tc := range tcases {
		policy := &restrictedPolicy{bestEffortPolicy{numaInfo: numaInfo, opts: PolicyOptions{}}}
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
		numaInfo := commonNUMAInfoTwoNodes()
		policy := &restrictedPolicy{bestEffortPolicy{numaInfo: numaInfo}}
		result := policy.canAdmitPodResult(&tc.hint)

		if result != tc.expected {
			t.Errorf("Expected result to be %t, got %t", tc.expected, result)
		}
	}
}

func TestPolicyRestrictedMerge(t *testing.T) {
	numaInfo := commonNUMAInfoFourNodes()
	policy := &restrictedPolicy{bestEffortPolicy{numaInfo: numaInfo}}

	tcases := commonPolicyMergeTestCases(numaInfo.Nodes)
	tcases = append(tcases, policy.mergeTestCases(numaInfo.Nodes)...)
	tcases = append(tcases, policy.mergeTestCasesNoPolicies(numaInfo.Nodes)...)

	testPolicyMerge(policy, tcases, t)
}

func TestPolicyRestrictedMergeClosestNUMA(t *testing.T) {
	numaInfo := commonNUMAInfoEightNodes()
	policy := &restrictedPolicy{bestEffortPolicy{numaInfo: numaInfo, opts: PolicyOptions{PreferClosestNUMA: true}}}

	tcases := commonPolicyMergeTestCases(numaInfo.Nodes)
	tcases = append(tcases, policy.mergeTestCases(numaInfo.Nodes)...)
	tcases = append(tcases, policy.mergeTestCasesClosestNUMA(numaInfo.Nodes)...)

	testPolicyMerge(policy, tcases, t)
}
