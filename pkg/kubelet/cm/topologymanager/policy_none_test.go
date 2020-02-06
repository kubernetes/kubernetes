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
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
	"testing"
)

func TestPolicyNoneName(t *testing.T) {
	tcases := []struct {
		name     string
		expected string
	}{
		{
			name:     "New None Policy",
			expected: "none",
		},
	}
	for _, tc := range tcases {
		policy := NewNonePolicy()
		if policy.Name() != tc.expected {
			t.Errorf("Expected Policy Name to be %s, got %s", tc.expected, policy.Name())
		}
	}
}

func TestPolicyNoneCanAdmitPodResult(t *testing.T) {
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
		policy := NewNonePolicy()
		result := policy.(*nonePolicy).canAdmitPodResult(&tc.hint)

		if result.Admit != tc.expected {
			t.Errorf("Expected Admit field in result to be %t, got %t", tc.expected, result.Admit)
		}
	}
}

func TestPolicyNoneMerge(t *testing.T) {
	tcases := []struct {
		name           string
		providersHints []map[string][]TopologyHint
		expectedHint   TopologyHint
		expectedAdmit  lifecycle.PodAdmitResult
	}{
		{
			name:           "merged empty providers hints",
			providersHints: []map[string][]TopologyHint{},
			expectedHint:   TopologyHint{},
			expectedAdmit:  lifecycle.PodAdmitResult{Admit: true},
		},
		{
			name: "merge with a single provider with a single preferred resource",
			providersHints: []map[string][]TopologyHint{
				{
					"resource": {{NUMANodeAffinity: NewTestBitMask(0, 1), Preferred: true}},
				},
			},
			expectedHint:  TopologyHint{},
			expectedAdmit: lifecycle.PodAdmitResult{Admit: true},
		},
		{
			name: "merge with a single provider with a single non-preferred resource",
			providersHints: []map[string][]TopologyHint{
				{
					"resource": {{NUMANodeAffinity: NewTestBitMask(0, 1), Preferred: false}},
				},
			},
			expectedHint:  TopologyHint{},
			expectedAdmit: lifecycle.PodAdmitResult{Admit: true},
		},
	}

	for _, tc := range tcases {
		policy := NewNonePolicy()
		result, admit := policy.Merge(tc.providersHints)
		if !result.IsEqual(tc.expectedHint) || admit.Admit != tc.expectedAdmit.Admit {
			t.Errorf("Test Case: %s: Expected merge hint to be %v, got %v", tc.name, tc.expectedHint, result)
		}
	}
}
