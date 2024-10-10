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

	"k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager/bitmask"
)

type policyMergeTestCase struct {
	name     string
	hp       []HintProvider
	expected TopologyHint
}

func commonPolicyMergeTestCases(numaNodes []int) []policyMergeTestCase {
	return []policyMergeTestCase{
		{
			name: "Two providers, 1 hint each, same mask, both preferred 1/2",
			hp: []HintProvider{
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource1": {
							{
								NUMANodeAffinity: NewTestBitMask(0),
								Preferred:        true,
							},
						},
					},
					nil,
				},
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource2": {
							{
								NUMANodeAffinity: NewTestBitMask(0),
								Preferred:        true,
							},
						},
					},
					nil,
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestBitMask(0),
				Preferred:        true,
			},
		},
		{
			name: "Two providers, 1 hint each, same mask, both preferred 2/2",
			hp: []HintProvider{
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource1": {
							{
								NUMANodeAffinity: NewTestBitMask(1),
								Preferred:        true,
							},
						},
					},
					nil,
				},
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource2": {
							{
								NUMANodeAffinity: NewTestBitMask(1),
								Preferred:        true,
							},
						},
					},
					nil,
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestBitMask(1),
				Preferred:        true,
			},
		},
		{
			name: "Two providers, 1 no hints, 1 single hint preferred 1/2",
			hp: []HintProvider{
				&mockHintProvider{},
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource": {
							{
								NUMANodeAffinity: NewTestBitMask(0),
								Preferred:        true,
							},
						},
					},
					nil,
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestBitMask(0),
				Preferred:        true,
			},
		},
		{
			name: "Two providers, 1 no hints, 1 single hint preferred 2/2",
			hp: []HintProvider{
				&mockHintProvider{},
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource": {
							{
								NUMANodeAffinity: NewTestBitMask(1),
								Preferred:        true,
							},
						},
					},
					nil,
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestBitMask(1),
				Preferred:        true,
			},
		},
		{
			name: "Two providers, 1 with 2 hints, 1 with single hint matching 1/2",
			hp: []HintProvider{
				&mockHintProvider{
					map[string][]TopologyHint{
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
					},
					nil,
				},
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource2": {
							{
								NUMANodeAffinity: NewTestBitMask(0),
								Preferred:        true,
							},
						},
					},
					nil,
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestBitMask(0),
				Preferred:        true,
			},
		},
		{
			name: "Two providers, 1 with 2 hints, 1 with single hint matching 2/2",
			hp: []HintProvider{
				&mockHintProvider{
					map[string][]TopologyHint{
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
					},
					nil,
				},
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource2": {
							{
								NUMANodeAffinity: NewTestBitMask(1),
								Preferred:        true,
							},
						},
					},
					nil,
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestBitMask(1),
				Preferred:        true,
			},
		},
		{
			name: "Two providers, both with 2 hints, matching narrower preferred hint from both",
			hp: []HintProvider{
				&mockHintProvider{
					map[string][]TopologyHint{
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
					},
					nil,
				},
				&mockHintProvider{
					map[string][]TopologyHint{
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
					nil,
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestBitMask(0),
				Preferred:        true,
			},
		},
		{
			name: "Ensure less narrow preferred hints are chosen over narrower non-preferred hints",
			hp: []HintProvider{
				&mockHintProvider{
					map[string][]TopologyHint{
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
					},
					nil,
				},
				&mockHintProvider{
					map[string][]TopologyHint{
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
					nil,
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestBitMask(1),
				Preferred:        true,
			},
		},
		{
			name: "Multiple resources, same provider",
			hp: []HintProvider{
				&mockHintProvider{
					map[string][]TopologyHint{
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
					nil,
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestBitMask(1),
				Preferred:        true,
			},
		},
	}
}

func (p *bestEffortPolicy) mergeTestCases(numaNodes []int) []policyMergeTestCase {
	return []policyMergeTestCase{
		{
			name: "Two providers, 2 hints each, same mask (some with different bits), same preferred",
			hp: []HintProvider{
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource1": {
							{
								NUMANodeAffinity: NewTestBitMask(0, 1),
								Preferred:        true,
							},
							{
								NUMANodeAffinity: NewTestBitMask(0, 2),
								Preferred:        true,
							},
						},
					},
					nil,
				},
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource2": {
							{
								NUMANodeAffinity: NewTestBitMask(0, 1),
								Preferred:        true,
							},
							{
								NUMANodeAffinity: NewTestBitMask(0, 2),
								Preferred:        true,
							},
						},
					},
					nil,
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestBitMask(0, 1),
				Preferred:        true,
			},
		},
		{
			name: "TopologyHint not set",
			hp:   []HintProvider{},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestBitMask(numaNodes...),
				Preferred:        true,
			},
		},
		{
			name: "HintProvider returns empty non-nil map[string][]TopologyHint",
			hp: []HintProvider{
				&mockHintProvider{
					map[string][]TopologyHint{},
					nil,
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestBitMask(numaNodes...),
				Preferred:        true,
			},
		},
		{
			name: "HintProvider returns -nil map[string][]TopologyHint from provider",
			hp: []HintProvider{
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource": nil,
					},
					nil,
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestBitMask(numaNodes...),
				Preferred:        true,
			},
		},
		{
			name: "HintProvider returns empty non-nil map[string][]TopologyHint from provider", hp: []HintProvider{
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource": {},
					},
					nil,
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestBitMask(numaNodes...),
				Preferred:        false,
			},
		},
		{
			name: "Single TopologyHint with Preferred as true and NUMANodeAffinity as nil",
			hp: []HintProvider{
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource": {
							{
								NUMANodeAffinity: nil,
								Preferred:        true,
							},
						},
					},
					nil,
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestBitMask(numaNodes...),
				Preferred:        true,
			},
		},
		{
			name: "Single TopologyHint with Preferred as false and NUMANodeAffinity as nil",
			hp: []HintProvider{
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource": {
							{
								NUMANodeAffinity: nil,
								Preferred:        false,
							},
						},
					},
					nil,
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestBitMask(numaNodes...),
				Preferred:        false,
			},
		},
		{
			name: "Two providers, 1 hint each, no common mask",
			hp: []HintProvider{
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource1": {
							{
								NUMANodeAffinity: NewTestBitMask(0),
								Preferred:        true,
							},
						},
					},
					nil,
				},
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource2": {
							{
								NUMANodeAffinity: NewTestBitMask(1),
								Preferred:        true,
							},
						},
					},
					nil,
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestBitMask(numaNodes...),
				Preferred:        false,
			},
		},
		{
			name: "Two providers, 1 hint each, same mask, 1 preferred, 1 not 1/2",
			hp: []HintProvider{
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource1": {
							{
								NUMANodeAffinity: NewTestBitMask(0),
								Preferred:        true,
							},
						},
					},
					nil,
				},
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource2": {
							{
								NUMANodeAffinity: NewTestBitMask(0),
								Preferred:        false,
							},
						},
					},
					nil,
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestBitMask(0),
				Preferred:        false,
			},
		},
		{
			name: "Two providers, 1 hint each, same mask, 1 preferred, 1 not 2/2",
			hp: []HintProvider{
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource1": {
							{
								NUMANodeAffinity: NewTestBitMask(1),
								Preferred:        true,
							},
						},
					},
					nil,
				},
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource2": {
							{
								NUMANodeAffinity: NewTestBitMask(1),
								Preferred:        false,
							},
						},
					},
					nil,
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestBitMask(1),
				Preferred:        false,
			},
		},
		{
			name: "Two providers, 1 hint each, 1 wider mask, both preferred 1/2",
			hp: []HintProvider{
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource1": {
							{
								NUMANodeAffinity: NewTestBitMask(0),
								Preferred:        true,
							},
						},
					},
					nil,
				},
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource2": {
							{
								NUMANodeAffinity: NewTestBitMask(0, 1),
								Preferred:        true,
							},
						},
					},
					nil,
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestBitMask(0),
				Preferred:        false,
			},
		},
		{
			name: "Two providers, 1 with 2 hints, 1 with single non-preferred hint matching",
			hp: []HintProvider{
				&mockHintProvider{
					map[string][]TopologyHint{
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
					},
					nil,
				},
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource2": {
							{
								NUMANodeAffinity: NewTestBitMask(0, 1),
								Preferred:        false,
							},
						},
					},
					nil,
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestBitMask(0),
				Preferred:        false,
			},
		},
		{
			name: "Two providers, 1 hint each, 1 wider mask, both preferred 2/2",
			hp: []HintProvider{
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource1": {
							{
								NUMANodeAffinity: NewTestBitMask(1),
								Preferred:        true,
							},
						},
					},
					nil,
				},
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource2": {
							{
								NUMANodeAffinity: NewTestBitMask(0, 1),
								Preferred:        true,
							},
						},
					},
					nil,
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestBitMask(1),
				Preferred:        false,
			},
		},
		{
			name: "bestNonPreferredAffinityCount (1)",
			hp: []HintProvider{
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource1": {
							{
								NUMANodeAffinity: NewTestBitMask(0, 1, 2, 3),
								Preferred:        false,
							},
							{
								NUMANodeAffinity: NewTestBitMask(0, 1),
								Preferred:        false,
							},
						},
					},
					nil,
				},
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource2": {
							{
								NUMANodeAffinity: NewTestBitMask(0, 1),
								Preferred:        false,
							},
						},
					},
					nil,
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestBitMask(0, 1),
				Preferred:        false,
			},
		},
		{
			name: "bestNonPreferredAffinityCount (2)",
			hp: []HintProvider{
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource1": {
							{
								NUMANodeAffinity: NewTestBitMask(0, 1, 2, 3),
								Preferred:        false,
							},
							{
								NUMANodeAffinity: NewTestBitMask(0, 1),
								Preferred:        false,
							},
						},
					},
					nil,
				},
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource2": {
							{
								NUMANodeAffinity: NewTestBitMask(0, 3),
								Preferred:        false,
							},
						},
					},
					nil,
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestBitMask(0, 3),
				Preferred:        false,
			},
		},
		{
			name: "bestNonPreferredAffinityCount (3)",
			hp: []HintProvider{
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource1": {
							{
								NUMANodeAffinity: NewTestBitMask(0, 1, 2, 3),
								Preferred:        false,
							},
							{
								NUMANodeAffinity: NewTestBitMask(0, 1),
								Preferred:        false,
							},
						},
					},
					nil,
				},
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource2": {
							{
								NUMANodeAffinity: NewTestBitMask(1, 2),
								Preferred:        false,
							},
						},
					},
					nil,
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestBitMask(1, 2),
				Preferred:        false,
			},
		},
		{
			name: "bestNonPreferredAffinityCount (4)",
			hp: []HintProvider{
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource1": {
							{
								NUMANodeAffinity: NewTestBitMask(0, 1, 2, 3),
								Preferred:        false,
							},
							{
								NUMANodeAffinity: NewTestBitMask(0, 1),
								Preferred:        false,
							},
						},
					},
					nil,
				},
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource2": {
							{
								NUMANodeAffinity: NewTestBitMask(2, 3),
								Preferred:        false,
							},
						},
					},
					nil,
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestBitMask(2, 3),
				Preferred:        false,
			},
		},
	}
}

func (p *bestEffortPolicy) mergeTestCasesNoPolicies(numaNodes []int) []policyMergeTestCase {
	return []policyMergeTestCase{
		{
			name: "bestNonPreferredAffinityCount (5)",
			hp: []HintProvider{
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource1": {
							{
								NUMANodeAffinity: NewTestBitMask(0, 1, 2, 3),
								Preferred:        false,
							},
							{
								NUMANodeAffinity: NewTestBitMask(0, 1),
								Preferred:        false,
							},
						},
					},
					nil,
				},
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource2": {
							{
								NUMANodeAffinity: NewTestBitMask(1, 2),
								Preferred:        false,
							},
							{
								NUMANodeAffinity: NewTestBitMask(2, 3),
								Preferred:        false,
							},
						},
					},
					nil,
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestBitMask(1, 2),
				Preferred:        false,
			},
		},
		{
			name: "bestNonPreferredAffinityCount (6)",
			hp: []HintProvider{
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource1": {
							{
								NUMANodeAffinity: NewTestBitMask(0, 1, 2, 3),
								Preferred:        false,
							},
							{
								NUMANodeAffinity: NewTestBitMask(0, 1),
								Preferred:        false,
							},
						},
					},
					nil,
				},
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource2": {
							{
								NUMANodeAffinity: NewTestBitMask(1, 2, 3),
								Preferred:        false,
							},
							{
								NUMANodeAffinity: NewTestBitMask(1, 2),
								Preferred:        false,
							},
							{
								NUMANodeAffinity: NewTestBitMask(1, 3),
								Preferred:        false,
							},
							{
								NUMANodeAffinity: NewTestBitMask(2, 3),
								Preferred:        false,
							},
						},
					},
					nil,
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestBitMask(1, 2),
				Preferred:        false,
			},
		},
	}
}

func (p *bestEffortPolicy) mergeTestCasesClosestNUMA(numaNodes []int) []policyMergeTestCase {
	return []policyMergeTestCase{
		{
			name: "Two providers, 2 hints each, same mask (some with different bits), same preferred",
			hp: []HintProvider{
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource1": {
							{
								NUMANodeAffinity: NewTestBitMask(0, 4),
								Preferred:        true,
							},
							{
								NUMANodeAffinity: NewTestBitMask(0, 2),
								Preferred:        true,
							},
						},
					},
					nil,
				},
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource2": {
							{
								NUMANodeAffinity: NewTestBitMask(0, 4),
								Preferred:        true,
							},
							{
								NUMANodeAffinity: NewTestBitMask(0, 2),
								Preferred:        true,
							},
						},
					},
					nil,
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestBitMask(0, 2),
				Preferred:        true,
			},
		},
		{
			name: "Two providers, 2 hints each, different mask",
			hp: []HintProvider{
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource1": {
							{
								NUMANodeAffinity: NewTestBitMask(4),
								Preferred:        true,
							},
							{
								NUMANodeAffinity: NewTestBitMask(0, 2),
								Preferred:        true,
							},
						},
					},
					nil,
				},
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource2": {
							{
								NUMANodeAffinity: NewTestBitMask(4),
								Preferred:        true,
							},
							{
								NUMANodeAffinity: NewTestBitMask(0, 2),
								Preferred:        true,
							},
						},
					},
					nil,
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestBitMask(4),
				Preferred:        true,
			},
		},
		{
			name: "bestNonPreferredAffinityCount (5)",
			hp: []HintProvider{
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource1": {
							{
								NUMANodeAffinity: NewTestBitMask(0, 1, 2, 3),
								Preferred:        false,
							},
							{
								NUMANodeAffinity: NewTestBitMask(0, 1),
								Preferred:        false,
							},
						},
					},
					nil,
				},
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource2": {
							{
								NUMANodeAffinity: NewTestBitMask(1, 2),
								Preferred:        false,
							},
							{
								NUMANodeAffinity: NewTestBitMask(2, 3),
								Preferred:        false,
							},
						},
					},
					nil,
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestBitMask(2, 3),
				Preferred:        false,
			},
		},
		{
			name: "bestNonPreferredAffinityCount (6)",
			hp: []HintProvider{
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource1": {
							{
								NUMANodeAffinity: NewTestBitMask(0, 1, 2, 3),
								Preferred:        false,
							},
							{
								NUMANodeAffinity: NewTestBitMask(0, 1),
								Preferred:        false,
							},
						},
					},
					nil,
				},
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource2": {
							{
								NUMANodeAffinity: NewTestBitMask(1, 2, 3),
								Preferred:        false,
							},
							{
								NUMANodeAffinity: NewTestBitMask(1, 2),
								Preferred:        false,
							},
							{
								NUMANodeAffinity: NewTestBitMask(1, 3),
								Preferred:        false,
							},
							{
								NUMANodeAffinity: NewTestBitMask(2, 3),
								Preferred:        false,
							},
						},
					},
					nil,
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestBitMask(2, 3),
				Preferred:        false,
			},
		},
	}
}

func (p *singleNumaNodePolicy) mergeTestCases(numaNodes []int) []policyMergeTestCase {
	return []policyMergeTestCase{
		{
			name: "TopologyHint not set",
			hp:   []HintProvider{},
			expected: TopologyHint{
				NUMANodeAffinity: nil,
				Preferred:        true,
			},
		},
		{
			name: "HintProvider returns empty non-nil map[string][]TopologyHint",
			hp: []HintProvider{
				&mockHintProvider{
					map[string][]TopologyHint{},
					nil,
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: nil,
				Preferred:        true,
			},
		},
		{
			name: "HintProvider returns -nil map[string][]TopologyHint from provider",
			hp: []HintProvider{
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource": nil,
					},
					nil,
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: nil,
				Preferred:        true,
			},
		},
		{
			name: "HintProvider returns empty non-nil map[string][]TopologyHint from provider", hp: []HintProvider{
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource": {},
					},
					nil,
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: nil,
				Preferred:        false,
			},
		},
		{
			name: "Single TopologyHint with Preferred as true and NUMANodeAffinity as nil",
			hp: []HintProvider{
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource": {
							{
								NUMANodeAffinity: nil,
								Preferred:        true,
							},
						},
					},
					nil,
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: nil,
				Preferred:        true,
			},
		},
		{
			name: "Single TopologyHint with Preferred as false and NUMANodeAffinity as nil",
			hp: []HintProvider{
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource": {
							{
								NUMANodeAffinity: nil,
								Preferred:        false,
							},
						},
					},
					nil,
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: nil,
				Preferred:        false,
			},
		},
		{
			name: "Two providers, 1 hint each, no common mask",
			hp: []HintProvider{
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource1": {
							{
								NUMANodeAffinity: NewTestBitMask(0),
								Preferred:        true,
							},
						},
					},
					nil,
				},
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource2": {
							{
								NUMANodeAffinity: NewTestBitMask(1),
								Preferred:        true,
							},
						},
					},
					nil,
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: nil,
				Preferred:        false,
			},
		},
		{
			name: "Two providers, 1 hint each, same mask, 1 preferred, 1 not 1/2",
			hp: []HintProvider{
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource1": {
							{
								NUMANodeAffinity: NewTestBitMask(0),
								Preferred:        true,
							},
						},
					},
					nil,
				},
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource2": {
							{
								NUMANodeAffinity: NewTestBitMask(0),
								Preferred:        false,
							},
						},
					},
					nil,
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: nil,
				Preferred:        false,
			},
		},
		{
			name: "Two providers, 1 hint each, same mask, 1 preferred, 1 not 2/2",
			hp: []HintProvider{
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource1": {
							{
								NUMANodeAffinity: NewTestBitMask(1),
								Preferred:        true,
							},
						},
					},
					nil,
				},
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource2": {
							{
								NUMANodeAffinity: NewTestBitMask(1),
								Preferred:        false,
							},
						},
					},
					nil,
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: nil,
				Preferred:        false,
			},
		},
		{
			name: "Two providers, 1 with 2 hints, 1 with single non-preferred hint matching",
			hp: []HintProvider{
				&mockHintProvider{
					map[string][]TopologyHint{
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
					},
					nil,
				},
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource2": {
							{
								NUMANodeAffinity: NewTestBitMask(0, 1),
								Preferred:        false,
							},
						},
					},
					nil,
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: nil,
				Preferred:        false,
			},
		},
		{
			name: "Single NUMA hint generation",
			hp: []HintProvider{
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource1": {
							{
								NUMANodeAffinity: NewTestBitMask(0, 1),
								Preferred:        true,
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
					nil,
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: nil,
				Preferred:        false,
			},
		},
		{
			name: "One no-preference provider",
			hp: []HintProvider{
				&mockHintProvider{
					map[string][]TopologyHint{
						"resource1": {
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
					nil,
				},
				&mockHintProvider{
					nil,
					nil,
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestBitMask(0),
				Preferred:        true,
			},
		},
	}
}

func testPolicyMerge(policy Policy, tcases []policyMergeTestCase, t *testing.T) {
	for _, tc := range tcases {
		var providersHints []map[string][]TopologyHint
		for _, provider := range tc.hp {
			hints := provider.GetTopologyHints(&v1.Pod{}, &v1.Container{})
			providersHints = append(providersHints, hints)
		}

		actual, _ := policy.Merge(providersHints)
		if !reflect.DeepEqual(actual, tc.expected) {
			t.Errorf("%v: Expected Topology Hint to be %v, got %v:", tc.name, tc.expected, actual)
		}
	}
}

func TestMaxOfMinAffinityCounts(t *testing.T) {
	tcases := []struct {
		hints    [][]TopologyHint
		expected int
	}{
		{
			[][]TopologyHint{},
			0,
		},
		{
			[][]TopologyHint{
				{
					TopologyHint{NewTestBitMask(), true},
				},
			},
			0,
		},
		{
			[][]TopologyHint{
				{
					TopologyHint{NewTestBitMask(0), true},
				},
			},
			1,
		},
		{
			[][]TopologyHint{
				{
					TopologyHint{NewTestBitMask(0, 1), true},
				},
			},
			2,
		},
		{
			[][]TopologyHint{
				{
					TopologyHint{NewTestBitMask(0, 1), true},
					TopologyHint{NewTestBitMask(0, 1, 2), true},
				},
			},
			2,
		},
		{
			[][]TopologyHint{
				{
					TopologyHint{NewTestBitMask(0, 1), true},
					TopologyHint{NewTestBitMask(0, 1, 2), true},
				},
				{
					TopologyHint{NewTestBitMask(0, 1, 2), true},
				},
			},
			3,
		},
		{
			[][]TopologyHint{
				{
					TopologyHint{NewTestBitMask(0, 1), true},
					TopologyHint{NewTestBitMask(0, 1, 2), true},
				},
				{
					TopologyHint{NewTestBitMask(0, 1, 2), true},
					TopologyHint{NewTestBitMask(0, 1, 2, 3), true},
				},
			},
			3,
		},
	}

	for _, tc := range tcases {
		t.Run("", func(t *testing.T) {
			result := maxOfMinAffinityCounts(tc.hints)
			if result != tc.expected {
				t.Errorf("Expected result to be %v, got %v", tc.expected, result)
			}
		})
	}
}

func TestCompareHintsNarrowest(t *testing.T) {
	tcases := []struct {
		description                   string
		bestNonPreferredAffinityCount int
		current                       *TopologyHint
		candidate                     *TopologyHint
		expected                      string
	}{
		{
			"candidate.NUMANodeAffinity.Count() == 0 (1)",
			-1,
			nil,
			&TopologyHint{bitmask.NewEmptyBitMask(), false},
			"current",
		},
		{
			"candidate.NUMANodeAffinity.Count() == 0 (2)",
			-1,
			&TopologyHint{NewTestBitMask(), true},
			&TopologyHint{NewTestBitMask(), false},
			"current",
		},
		{
			"current == nil (1)",
			-1,
			nil,
			&TopologyHint{NewTestBitMask(0), true},
			"candidate",
		},
		{
			"current == nil (2)",
			-1,
			nil,
			&TopologyHint{NewTestBitMask(0), false},
			"candidate",
		},
		{
			"!current.Preferred && candidate.Preferred",
			-1,
			&TopologyHint{NewTestBitMask(0), false},
			&TopologyHint{NewTestBitMask(0), true},
			"candidate",
		},
		{
			"current.Preferred && !candidate.Preferred",
			-1,
			&TopologyHint{NewTestBitMask(0), true},
			&TopologyHint{NewTestBitMask(0), false},
			"current",
		},
		{
			"current.Preferred && candidate.Preferred (1)",
			-1,
			&TopologyHint{NewTestBitMask(0), true},
			&TopologyHint{NewTestBitMask(0), true},
			"current",
		},
		{
			"current.Preferred && candidate.Preferred (2)",
			-1,
			&TopologyHint{NewTestBitMask(0, 1), true},
			&TopologyHint{NewTestBitMask(0), true},
			"candidate",
		},
		{
			"current.Preferred && candidate.Preferred (3)",
			-1,
			&TopologyHint{NewTestBitMask(0), true},
			&TopologyHint{NewTestBitMask(0, 1), true},
			"current",
		},
		{
			"!current.Preferred && !candidate.Preferred (1.1)",
			1,
			&TopologyHint{NewTestBitMask(0, 1), false},
			&TopologyHint{NewTestBitMask(0, 1), false},
			"current",
		},
		{
			"!current.Preferred && !candidate.Preferred (1.2)",
			1,
			&TopologyHint{NewTestBitMask(1, 2), false},
			&TopologyHint{NewTestBitMask(0, 1), false},
			"candidate",
		},
		{
			"!current.Preferred && !candidate.Preferred (1.3)",
			1,
			&TopologyHint{NewTestBitMask(0, 1), false},
			&TopologyHint{NewTestBitMask(1, 2), false},
			"current",
		},
		{
			"!current.Preferred && !candidate.Preferred (2.1)",
			2,
			&TopologyHint{NewTestBitMask(0, 1), false},
			&TopologyHint{NewTestBitMask(0), false},
			"current",
		},
		{
			"!current.Preferred && !candidate.Preferred (2.2)",
			2,
			&TopologyHint{NewTestBitMask(0, 1), false},
			&TopologyHint{NewTestBitMask(0, 1), false},
			"current",
		},
		{
			"!current.Preferred && !candidate.Preferred (2.3)",
			2,
			&TopologyHint{NewTestBitMask(1, 2), false},
			&TopologyHint{NewTestBitMask(0, 1), false},
			"candidate",
		},
		{
			"!current.Preferred && !candidate.Preferred (2.4)",
			2,
			&TopologyHint{NewTestBitMask(0, 1), false},
			&TopologyHint{NewTestBitMask(1, 2), false},
			"current",
		},
		{
			"!current.Preferred && !candidate.Preferred (3a)",
			2,
			&TopologyHint{NewTestBitMask(0), false},
			&TopologyHint{NewTestBitMask(0, 1, 2), false},
			"current",
		},
		{
			"!current.Preferred && !candidate.Preferred (3b)",
			2,
			&TopologyHint{NewTestBitMask(0), false},
			&TopologyHint{NewTestBitMask(0, 1), false},
			"candidate",
		},
		{
			"!current.Preferred && !candidate.Preferred (3ca.1)",
			3,
			&TopologyHint{NewTestBitMask(0), false},
			&TopologyHint{NewTestBitMask(0, 1), false},
			"candidate",
		},
		{
			"!current.Preferred && !candidate.Preferred (3ca.2)",
			3,
			&TopologyHint{NewTestBitMask(0), false},
			&TopologyHint{NewTestBitMask(1, 2), false},
			"candidate",
		},
		{
			"!current.Preferred && !candidate.Preferred (3ca.3)",
			4,
			&TopologyHint{NewTestBitMask(0, 1), false},
			&TopologyHint{NewTestBitMask(1, 2, 3), false},
			"candidate",
		},
		{
			"!current.Preferred && !candidate.Preferred (3cb)",
			4,
			&TopologyHint{NewTestBitMask(1, 2, 3), false},
			&TopologyHint{NewTestBitMask(0, 1), false},
			"current",
		},
		{
			"!current.Preferred && !candidate.Preferred (3cc.1)",
			4,
			&TopologyHint{NewTestBitMask(0, 1, 2), false},
			&TopologyHint{NewTestBitMask(0, 1, 2), false},
			"current",
		},
		{
			"!current.Preferred && !candidate.Preferred (3cc.2)",
			4,
			&TopologyHint{NewTestBitMask(0, 1, 2), false},
			&TopologyHint{NewTestBitMask(1, 2, 3), false},
			"current",
		},
		{
			"!current.Preferred && !candidate.Preferred (3cc.3)",
			4,
			&TopologyHint{NewTestBitMask(1, 2, 3), false},
			&TopologyHint{NewTestBitMask(0, 1, 2), false},
			"candidate",
		},
	}

	for _, tc := range tcases {
		t.Run(tc.description, func(t *testing.T) {
			numaInfo := &NUMAInfo{}
			merger := NewHintMerger(numaInfo, [][]TopologyHint{}, PolicyBestEffort, PolicyOptions{})
			merger.BestNonPreferredAffinityCount = tc.bestNonPreferredAffinityCount

			result := merger.compare(tc.current, tc.candidate)
			if result != tc.current && result != tc.candidate {
				t.Errorf("Expected result to be either 'current' or 'candidate' hint")
			}
			if tc.expected == "current" && result != tc.current {
				t.Errorf("Expected result to be %v, got %v", tc.current, result)
			}
			if tc.expected == "candidate" && result != tc.candidate {
				t.Errorf("Expected result to be %v, got %v", tc.candidate, result)
			}
		})
	}
}

func commonNUMAInfoTwoNodes() *NUMAInfo {
	return &NUMAInfo{
		Nodes: []int{0, 1},
		NUMADistances: NUMADistances{
			0: {10, 11},
			1: {11, 10},
		},
	}
}

func commonNUMAInfoFourNodes() *NUMAInfo {
	return &NUMAInfo{
		Nodes: []int{0, 1, 2, 3},
		NUMADistances: NUMADistances{
			0: {10, 11, 12, 12},
			1: {11, 10, 12, 12},
			2: {12, 12, 10, 11},
			3: {12, 12, 11, 10},
		},
	}
}

func commonNUMAInfoEightNodes() *NUMAInfo {
	return &NUMAInfo{
		Nodes: []int{0, 1, 2, 3, 4, 5, 6, 7},
		NUMADistances: NUMADistances{
			0: {10, 11, 12, 12, 30, 30, 30, 30},
			1: {11, 10, 12, 12, 30, 30, 30, 30},
			2: {12, 12, 10, 11, 30, 30, 30, 30},
			3: {12, 12, 11, 10, 30, 30, 30, 30},
			4: {30, 30, 30, 30, 10, 11, 12, 12},
			5: {30, 30, 30, 30, 11, 10, 12, 12},
			6: {30, 30, 30, 30, 12, 12, 10, 11},
			7: {30, 30, 30, 30, 12, 12, 13, 10},
		},
	}
}
