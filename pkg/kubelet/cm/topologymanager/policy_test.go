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
	"fmt"
	"reflect"
	"testing"

	"k8s.io/api/core/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	pkgfeatures "k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager/bitmask"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
	"k8s.io/kubernetes/test/utils/ktesting"
)

type policyMergeTestCase struct {
	name     string
	hp       []HintProvider
	expected TopologyHint
}

func commonPolicyMergeTestCases(_ []int) []policyMergeTestCase {
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
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestBitMask(2, 3),
				Preferred:        false,
			},
		},
	}
}

func (p *bestEffortPolicy) mergeTestCasesNoPolicies(_ []int) []policyMergeTestCase {
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
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestBitMask(1, 2),
				Preferred:        false,
			},
		},
	}
}

func (p *bestEffortPolicy) mergeTestCasesClosestNUMA(_ []int) []policyMergeTestCase {
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
				},
			},
			expected: TopologyHint{
				NUMANodeAffinity: NewTestBitMask(2, 3),
				Preferred:        false,
			},
		},
	}
}

func (p *singleNumaNodePolicy) mergeTestCases(_ []int) []policyMergeTestCase {
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
				},
				&mockHintProvider{
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
	logger, _ := ktesting.NewTestContext(t)

	for _, tc := range tcases {
		var providersHints []map[string][]TopologyHint
		for _, provider := range tc.hp {
			hints := provider.GetTopologyHints(logger, &v1.Pod{}, &v1.Container{}, lifecycle.AddOperation)
			providersHints = append(providersHints, hints)
		}

		actual, _ := policy.Merge(logger, providersHints)
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
					TopologyHint{NUMANodeAffinity: NewTestBitMask(), Preferred: true},
				},
			},
			0,
		},
		{
			[][]TopologyHint{
				{
					TopologyHint{NUMANodeAffinity: NewTestBitMask(0), Preferred: true},
				},
			},
			1,
		},
		{
			[][]TopologyHint{
				{
					TopologyHint{NUMANodeAffinity: NewTestBitMask(0, 1), Preferred: true},
				},
			},
			2,
		},
		{
			[][]TopologyHint{
				{
					TopologyHint{NUMANodeAffinity: NewTestBitMask(0, 1), Preferred: true},
					TopologyHint{NUMANodeAffinity: NewTestBitMask(0, 1, 2), Preferred: true},
				},
			},
			2,
		},
		{
			[][]TopologyHint{
				{
					TopologyHint{NUMANodeAffinity: NewTestBitMask(0, 1), Preferred: true},
					TopologyHint{NUMANodeAffinity: NewTestBitMask(0, 1, 2), Preferred: true},
				},
				{
					TopologyHint{NUMANodeAffinity: NewTestBitMask(0, 1, 2), Preferred: true},
				},
			},
			3,
		},
		{
			[][]TopologyHint{
				{
					TopologyHint{NUMANodeAffinity: NewTestBitMask(0, 1), Preferred: true},
					TopologyHint{NUMANodeAffinity: NewTestBitMask(0, 1, 2), Preferred: true},
				},
				{
					TopologyHint{NUMANodeAffinity: NewTestBitMask(0, 1, 2), Preferred: true},
					TopologyHint{NUMANodeAffinity: NewTestBitMask(0, 1, 2, 3), Preferred: true},
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
			&TopologyHint{NUMANodeAffinity: bitmask.NewEmptyBitMask(), Preferred: false},
			"current",
		},
		{
			"candidate.NUMANodeAffinity.Count() == 0 (2)",
			-1,
			&TopologyHint{NUMANodeAffinity: NewTestBitMask(), Preferred: true},
			&TopologyHint{NUMANodeAffinity: NewTestBitMask(), Preferred: false},
			"current",
		},
		{
			"current == nil (1)",
			-1,
			nil,
			&TopologyHint{NUMANodeAffinity: NewTestBitMask(0), Preferred: true},
			"candidate",
		},
		{
			"current == nil (2)",
			-1,
			nil,
			&TopologyHint{NUMANodeAffinity: NewTestBitMask(0), Preferred: false},
			"candidate",
		},
		{
			"!current.Preferred && candidate.Preferred",
			-1,
			&TopologyHint{NUMANodeAffinity: NewTestBitMask(0), Preferred: false},
			&TopologyHint{NUMANodeAffinity: NewTestBitMask(0), Preferred: true},
			"candidate",
		},
		{
			"current.Preferred && !candidate.Preferred",
			-1,
			&TopologyHint{NUMANodeAffinity: NewTestBitMask(0), Preferred: true},
			&TopologyHint{NUMANodeAffinity: NewTestBitMask(0), Preferred: false},
			"current",
		},
		{
			"current.Preferred && candidate.Preferred (1)",
			-1,
			&TopologyHint{NUMANodeAffinity: NewTestBitMask(0), Preferred: true},
			&TopologyHint{NUMANodeAffinity: NewTestBitMask(0), Preferred: true},
			"current",
		},
		{
			"current.Preferred && candidate.Preferred (2)",
			-1,
			&TopologyHint{NUMANodeAffinity: NewTestBitMask(0, 1), Preferred: true},
			&TopologyHint{NUMANodeAffinity: NewTestBitMask(0), Preferred: true},
			"candidate",
		},
		{
			"current.Preferred && candidate.Preferred (3)",
			-1,
			&TopologyHint{NUMANodeAffinity: NewTestBitMask(0), Preferred: true},
			&TopologyHint{NUMANodeAffinity: NewTestBitMask(0, 1), Preferred: true},
			"current",
		},
		{
			"!current.Preferred && !candidate.Preferred (1.1)",
			1,
			&TopologyHint{NUMANodeAffinity: NewTestBitMask(0, 1), Preferred: false},
			&TopologyHint{NUMANodeAffinity: NewTestBitMask(0, 1), Preferred: false},
			"current",
		},
		{
			"!current.Preferred && !candidate.Preferred (1.2)",
			1,
			&TopologyHint{NUMANodeAffinity: NewTestBitMask(1, 2), Preferred: false},
			&TopologyHint{NUMANodeAffinity: NewTestBitMask(0, 1), Preferred: false},
			"candidate",
		},
		{
			"!current.Preferred && !candidate.Preferred (1.3)",
			1,
			&TopologyHint{NUMANodeAffinity: NewTestBitMask(0, 1), Preferred: false},
			&TopologyHint{NUMANodeAffinity: NewTestBitMask(1, 2), Preferred: false},
			"current",
		},
		{
			"!current.Preferred && !candidate.Preferred (2.1)",
			2,
			&TopologyHint{NUMANodeAffinity: NewTestBitMask(0, 1), Preferred: false},
			&TopologyHint{NUMANodeAffinity: NewTestBitMask(0), Preferred: false},
			"current",
		},
		{
			"!current.Preferred && !candidate.Preferred (2.2)",
			2,
			&TopologyHint{NUMANodeAffinity: NewTestBitMask(0, 1), Preferred: false},
			&TopologyHint{NUMANodeAffinity: NewTestBitMask(0, 1), Preferred: false},
			"current",
		},
		{
			"!current.Preferred && !candidate.Preferred (2.3)",
			2,
			&TopologyHint{NUMANodeAffinity: NewTestBitMask(1, 2), Preferred: false},
			&TopologyHint{NUMANodeAffinity: NewTestBitMask(0, 1), Preferred: false},
			"candidate",
		},
		{
			"!current.Preferred && !candidate.Preferred (2.4)",
			2,
			&TopologyHint{NUMANodeAffinity: NewTestBitMask(0, 1), Preferred: false},
			&TopologyHint{NUMANodeAffinity: NewTestBitMask(1, 2), Preferred: false},
			"current",
		},
		{
			"!current.Preferred && !candidate.Preferred (3a)",
			2,
			&TopologyHint{NUMANodeAffinity: NewTestBitMask(0), Preferred: false},
			&TopologyHint{NUMANodeAffinity: NewTestBitMask(0, 1, 2), Preferred: false},
			"current",
		},
		{
			"!current.Preferred && !candidate.Preferred (3b)",
			2,
			&TopologyHint{NUMANodeAffinity: NewTestBitMask(0), Preferred: false},
			&TopologyHint{NUMANodeAffinity: NewTestBitMask(0, 1), Preferred: false},
			"candidate",
		},
		{
			"!current.Preferred && !candidate.Preferred (3ca.1)",
			3,
			&TopologyHint{NUMANodeAffinity: NewTestBitMask(0), Preferred: false},
			&TopologyHint{NUMANodeAffinity: NewTestBitMask(0, 1), Preferred: false},
			"candidate",
		},
		{
			"!current.Preferred && !candidate.Preferred (3ca.2)",
			3,
			&TopologyHint{NUMANodeAffinity: NewTestBitMask(0), Preferred: false},
			&TopologyHint{NUMANodeAffinity: NewTestBitMask(1, 2), Preferred: false},
			"candidate",
		},
		{
			"!current.Preferred && !candidate.Preferred (3ca.3)",
			4,
			&TopologyHint{NUMANodeAffinity: NewTestBitMask(0, 1), Preferred: false},
			&TopologyHint{NUMANodeAffinity: NewTestBitMask(1, 2, 3), Preferred: false},
			"candidate",
		},
		{
			"!current.Preferred && !candidate.Preferred (3cb)",
			4,
			&TopologyHint{NUMANodeAffinity: NewTestBitMask(1, 2, 3), Preferred: false},
			&TopologyHint{NUMANodeAffinity: NewTestBitMask(0, 1), Preferred: false},
			"current",
		},
		{
			"!current.Preferred && !candidate.Preferred (3cc.1)",
			4,
			&TopologyHint{NUMANodeAffinity: NewTestBitMask(0, 1, 2), Preferred: false},
			&TopologyHint{NUMANodeAffinity: NewTestBitMask(0, 1, 2), Preferred: false},
			"current",
		},
		{
			"!current.Preferred && !candidate.Preferred (3cc.2)",
			4,
			&TopologyHint{NUMANodeAffinity: NewTestBitMask(0, 1, 2), Preferred: false},
			&TopologyHint{NUMANodeAffinity: NewTestBitMask(1, 2, 3), Preferred: false},
			"current",
		},
		{
			"!current.Preferred && !candidate.Preferred (3cc.3)",
			4,
			&TopologyHint{NUMANodeAffinity: NewTestBitMask(1, 2, 3), Preferred: false},
			&TopologyHint{NUMANodeAffinity: NewTestBitMask(0, 1, 2), Preferred: false},
			"candidate",
		},
	}

	for _, tc := range tcases {
		t.Run(tc.description, func(t *testing.T) {
			numaInfo := &NUMAInfo{}
			merger := NewHintMerger(numaInfo, [][]TopologyHint{}, nil, PolicyBestEffort, PolicyOptions{})
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

// TestFilterProvidersHints checks the positional correspondence between the
// two slices filterProvidersHints returns: the hints at index i must belong to
// the resource named at index i. Providers are iterated as maps, so the order
// of the entries is not fixed and the expectations are keyed by resource name
// rather than by position.
func TestFilterProvidersHints(t *testing.T) {
	tcases := []struct {
		name          string
		providersHint []map[string][]TopologyHint
		expected      map[string][][]TopologyHint
	}{
		{
			name:          "no providers",
			providersHint: nil,
			expected:      map[string][][]TopologyHint{},
		},
		{
			name:          "provider with no preference for any resource",
			providersHint: []map[string][]TopologyHint{nil},
			expected: map[string][][]TopologyHint{
				"": {{{NUMANodeAffinity: nil, Preferred: true}}},
			},
		},
		{
			name: "provider with no preference for a resource",
			providersHint: []map[string][]TopologyHint{
				{"cpu": nil},
			},
			expected: map[string][][]TopologyHint{
				"cpu": {{{NUMANodeAffinity: nil, Preferred: true}}},
			},
		},
		{
			name: "provider with no possible affinity for a resource",
			providersHint: []map[string][]TopologyHint{
				{"cpu": {}},
			},
			expected: map[string][][]TopologyHint{
				"cpu": {{{NUMANodeAffinity: nil, Preferred: false}}},
			},
		},
		{
			name: "several resources from several providers",
			providersHint: []map[string][]TopologyHint{
				{
					"cpu": {
						{NUMANodeAffinity: NewTestBitMask(0), Preferred: true, Score: 20},
						{NUMANodeAffinity: NewTestBitMask(1), Preferred: true, Score: 40},
					},
				},
				{
					"memory": {
						{NUMANodeAffinity: NewTestBitMask(1), Preferred: true, Score: 60},
					},
					"nvidia.com/gpu": {
						{NUMANodeAffinity: NewTestBitMask(0), Preferred: false, Score: 80},
					},
				},
				nil,
			},
			expected: map[string][][]TopologyHint{
				"cpu": {{
					{NUMANodeAffinity: NewTestBitMask(0), Preferred: true, Score: 20},
					{NUMANodeAffinity: NewTestBitMask(1), Preferred: true, Score: 40},
				}},
				"memory": {{
					{NUMANodeAffinity: NewTestBitMask(1), Preferred: true, Score: 60},
				}},
				"nvidia.com/gpu": {{
					{NUMANodeAffinity: NewTestBitMask(0), Preferred: false, Score: 80},
				}},
				"": {{{NUMANodeAffinity: nil, Preferred: true}}},
			},
		},
	}

	logger, _ := ktesting.NewTestContext(t)

	for _, tc := range tcases {
		t.Run(tc.name, func(t *testing.T) {
			hints, resourceNames := filterProvidersHints(logger, tc.providersHint)

			if len(hints) != len(resourceNames) {
				t.Fatalf("Expected as many resource names as hint lists, got %v names for %v hint lists", len(resourceNames), len(hints))
			}

			got := map[string][][]TopologyHint{}
			for i := range hints {
				got[resourceNames[i]] = append(got[resourceNames[i]], hints[i])
			}
			if !reflect.DeepEqual(got, tc.expected) {
				t.Errorf("Expected hints per resource to be %v, got %v", tc.expected, got)
			}
		})
	}
}

func TestAggregateHintScores(t *testing.T) {
	tcases := []struct {
		name          string
		permutation   []TopologyHint
		expectedScore int64
		expectedOk    bool
	}{
		{
			name: "all scores zero",
			permutation: []TopologyHint{
				{NUMANodeAffinity: NewTestBitMask(0), Preferred: true, Score: 0},
				{NUMANodeAffinity: NewTestBitMask(0), Preferred: true, Score: 0},
			},
			expectedScore: 0,
			expectedOk:    false,
		},
		{
			name: "single scored contributor",
			permutation: []TopologyHint{
				{NUMANodeAffinity: NewTestBitMask(0), Preferred: true, Score: 60},
			},
			expectedScore: 60,
			expectedOk:    true,
		},
		{
			name: "multiple scored contributors averaged",
			permutation: []TopologyHint{
				{NUMANodeAffinity: NewTestBitMask(0), Preferred: true, Score: 40},
				{NUMANodeAffinity: NewTestBitMask(0), Preferred: true, Score: 80},
			},
			expectedScore: 60,
			expectedOk:    true,
		},
		{
			name: "nil affinity contributor ignored",
			permutation: []TopologyHint{
				{NUMANodeAffinity: NewTestBitMask(0), Preferred: true, Score: 80},
				{NUMANodeAffinity: nil, Preferred: true, Score: 40},
			},
			expectedScore: 80,
			expectedOk:    true,
		},
		{
			name: "mixed scored and unscored contributors",
			permutation: []TopologyHint{
				{NUMANodeAffinity: NewTestBitMask(0), Preferred: true, Score: 90},
				{NUMANodeAffinity: NewTestBitMask(0), Preferred: true, Score: 0},
			},
			expectedScore: 90,
			expectedOk:    true,
		},
		{
			name:          "empty permutation",
			permutation:   []TopologyHint{},
			expectedScore: 0,
			expectedOk:    false,
		},
	}

	for _, tc := range tcases {
		t.Run(tc.name, func(t *testing.T) {
			score, ok := aggregateHintScores(tc.permutation, nil, nil)
			if score != tc.expectedScore {
				t.Errorf("expected score %d, got %d", tc.expectedScore, score)
			}
			if ok != tc.expectedOk {
				t.Errorf("expected ok %v, got %v", tc.expectedOk, ok)
			}
		})
	}
}

func TestMergePermutationCarriesScore(t *testing.T) {
	defaultAffinity := NewTestBitMask(0, 1)

	tcases := []struct {
		name          string
		permutation   []TopologyHint
		expectedScore int64
	}{
		{
			name: "merged hint carries aggregated score",
			permutation: []TopologyHint{
				{NUMANodeAffinity: NewTestBitMask(0), Preferred: true, Score: 40},
				{NUMANodeAffinity: NewTestBitMask(0), Preferred: true, Score: 80},
			},
			expectedScore: 60,
		},
		{
			name: "merged hint score is zero when no contributors have scores",
			permutation: []TopologyHint{
				{NUMANodeAffinity: NewTestBitMask(0), Preferred: true, Score: 0},
			},
			expectedScore: 0,
		},
	}

	for _, tc := range tcases {
		t.Run(tc.name, func(t *testing.T) {
			logger, _ := ktesting.NewTestContext(t)
			merged := mergePermutation(logger, defaultAffinity, tc.permutation, nil, nil)
			if merged.Score != tc.expectedScore {
				t.Errorf("expected merged Score %d, got %d", tc.expectedScore, merged.Score)
			}
		})
	}
}

func TestCompareWinnerUnchangedByScore(t *testing.T) {
	numaInfo := commonNUMAInfoTwoNodes()
	hints := [][]TopologyHint{}
	merger := NewHintMerger(numaInfo, hints, nil, PolicyBestEffort, PolicyOptions{})

	narrower := &TopologyHint{NUMANodeAffinity: NewTestBitMask(0), Preferred: true, Score: 10}
	wider := &TopologyHint{NUMANodeAffinity: NewTestBitMask(0, 1), Preferred: true, Score: 90}

	result := merger.compare(narrower, wider)
	if !result.NUMANodeAffinity.IsEqual(narrower.NUMANodeAffinity) {
		t.Errorf("expected narrower hint to win regardless of score, got %v", result.NUMANodeAffinity)
	}
}

func TestCompareNUMAAffinityMasksWithAllocationStrategy(t *testing.T) {
	numaInfo := commonNUMAInfoTwoNodes()

	// Both masks are a single node, so the structural comparison rates them
	// equally and the allocation strategy decides. Narrowest resolves such a
	// tie in favour of the mask with the lower-numbered bits set, which is
	// what the strategy has to override to be observable here.
	current := &TopologyHint{NUMANodeAffinity: NewTestBitMask(0), Preferred: true, Score: 20}
	candidate := &TopologyHint{NUMANodeAffinity: NewTestBitMask(1), Preferred: true, Score: 80}

	tcases := []struct {
		description string
		strategy    string
		gateEnabled bool
		expected    string
	}{
		{
			description: "most-allocated packs onto the higher scoring hint",
			strategy:    NUMAAllocationStrategyMostAllocated,
			gateEnabled: true,
			expected:    "candidate",
		},
		{
			description: "the allocation strategy is ignored while the alpha options are disabled",
			strategy:    NUMAAllocationStrategyMostAllocated,
			gateEnabled: false,
			expected:    "current",
		},
		{
			description: "the none strategy leaves the structural comparison alone",
			strategy:    NUMAAllocationStrategyNone,
			gateEnabled: true,
			expected:    "current",
		},
		{
			description: "an unset strategy leaves the structural comparison alone",
			gateEnabled: true,
			expected:    "current",
		},
	}

	for _, tc := range tcases {
		t.Run(tc.description, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.TopologyManagerPolicyAlphaOptions, tc.gateEnabled)

			opts := PolicyOptions{NUMAAllocationStrategy: tc.strategy}
			merger := NewHintMerger(numaInfo, [][]TopologyHint{}, nil, PolicySingleNumaNode, opts)

			result := merger.CompareNUMAAffinityMasks(current, candidate)
			if tc.expected == "current" && result != current {
				t.Errorf("Expected result to be %v, got %v", current, result)
			}
			if tc.expected == "candidate" && result != candidate {
				t.Errorf("Expected result to be %v, got %v", candidate, result)
			}
		})
	}
}

func TestCompareHintScores(t *testing.T) {
	tcases := []struct {
		description string
		strategy    string
		current     int64
		candidate   int64
		expected    string
	}{
		{
			description: "most-allocated prefers the higher score",
			strategy:    NUMAAllocationStrategyMostAllocated,
			current:     40,
			candidate:   80,
			expected:    "candidate",
		},
		{
			description: "most-allocated keeps the higher scoring incumbent",
			strategy:    NUMAAllocationStrategyMostAllocated,
			current:     80,
			candidate:   40,
			expected:    "current",
		},
		{
			description: "least-allocated prefers the lower score",
			strategy:    NUMAAllocationStrategyLeastAllocated,
			current:     80,
			candidate:   40,
			expected:    "candidate",
		},
		{
			description: "least-allocated keeps the lower scoring incumbent",
			strategy:    NUMAAllocationStrategyLeastAllocated,
			current:     40,
			candidate:   80,
			expected:    "current",
		},
		{
			description: "equal scores express no preference",
			strategy:    NUMAAllocationStrategyMostAllocated,
			current:     50,
			candidate:   50,
			expected:    "none",
		},
		{
			description: "an unscored candidate expresses no preference",
			strategy:    NUMAAllocationStrategyLeastAllocated,
			current:     50,
			candidate:   0,
			expected:    "none",
		},
		{
			description: "an unscored current expresses no preference",
			strategy:    NUMAAllocationStrategyLeastAllocated,
			current:     0,
			candidate:   50,
			expected:    "none",
		},
		{
			description: "the none strategy expresses no preference",
			strategy:    NUMAAllocationStrategyNone,
			current:     40,
			candidate:   80,
			expected:    "none",
		},
	}

	for _, tc := range tcases {
		t.Run(tc.description, func(t *testing.T) {
			current := &TopologyHint{NUMANodeAffinity: NewTestBitMask(0), Preferred: true, Score: tc.current}
			candidate := &TopologyHint{NUMANodeAffinity: NewTestBitMask(1), Preferred: true, Score: tc.candidate}

			result := compareHintScores(tc.strategy, current, candidate)
			switch tc.expected {
			case "current":
				if result != current {
					t.Errorf("Expected result to be %v, got %v", current, result)
				}
			case "candidate":
				if result != candidate {
					t.Errorf("Expected result to be %v, got %v", candidate, result)
				}
			case "none":
				if result != nil {
					t.Errorf("Expected no preference, got %v", result)
				}
			}
		})
	}
}

func TestCompareNUMAAffinityMasksStrategyDoesNotBeatStructure(t *testing.T) {
	numaInfo := commonNUMAInfoTwoNodes()

	// The wider mask scores higher, but most-allocated must not widen the
	// affinity: the structural comparison keeps the last word.
	current := &TopologyHint{NUMANodeAffinity: NewTestBitMask(0), Preferred: true, Score: 20}
	candidate := &TopologyHint{NUMANodeAffinity: NewTestBitMask(0, 1), Preferred: true, Score: 90}

	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.TopologyManagerPolicyAlphaOptions, true)

	opts := PolicyOptions{NUMAAllocationStrategy: NUMAAllocationStrategyMostAllocated}
	merger := NewHintMerger(numaInfo, [][]TopologyHint{}, nil, PolicyBestEffort, opts)

	if result := merger.CompareNUMAAffinityMasks(current, candidate); result != current {
		t.Errorf("Expected the narrowest hint %v to win, got %v", current, result)
	}
}

// TestHintMergerWithAllocationStrategy exercises the allocation strategy
// through a whole merge rather than through CompareNUMAAffinityMasks directly,
// so that what drives the selection is the aggregated score of a permutation
// rather than the score of an individual provider hint.
//
// Each fixture below deliberately makes the strategy pull against the
// structural tiebreak. Both Narrowest and Closest resolve equally wide,
// equally distant masks in favour of the one with the lower-numbered bits set,
// so a case whose expected winner is also the lower-numbered mask would come
// out the same with and without the strategy applied and would prove nothing.
func TestHintMergerWithAllocationStrategy(t *testing.T) {
	tcases := []struct {
		description string
		// numaInfo defaults to commonNUMAInfoTwoNodes, policyName to
		// PolicySingleNumaNode.
		numaInfo         *NUMAInfo
		policyName       string
		preferClosest    bool
		strategy         string
		gateEnabled      bool
		hints            [][]TopologyHint
		expectedAffinity bitmask.BitMask
	}{
		{
			description: "most-allocated packs onto the higher scoring node",
			strategy:    NUMAAllocationStrategyMostAllocated,
			gateEnabled: true,
			hints: [][]TopologyHint{
				{
					{NUMANodeAffinity: NewTestBitMask(0), Preferred: true, Score: 20},
					{NUMANodeAffinity: NewTestBitMask(1), Preferred: true, Score: 80},
				},
			},
			expectedAffinity: NewTestBitMask(1),
		},
		{
			description: "least-allocated spreads onto the lower scoring node",
			strategy:    NUMAAllocationStrategyLeastAllocated,
			gateEnabled: true,
			hints: [][]TopologyHint{
				{
					{NUMANodeAffinity: NewTestBitMask(0), Preferred: true, Score: 80},
					{NUMANodeAffinity: NewTestBitMask(1), Preferred: true, Score: 20},
				},
			},
			expectedAffinity: NewTestBitMask(1),
		},
		{
			description: "the aggregated score of the permutation decides, not a single provider's",
			strategy:    NUMAAllocationStrategyMostAllocated,
			gateEnabled: true,
			// The first provider rates node0 above node1, but the average over
			// both providers rates node1 higher, and that is what has to win.
			hints: [][]TopologyHint{
				{
					{NUMANodeAffinity: NewTestBitMask(0), Preferred: true, Score: 50},
					{NUMANodeAffinity: NewTestBitMask(1), Preferred: true, Score: 90},
				},
				{
					{NUMANodeAffinity: NewTestBitMask(0), Preferred: true, Score: 10},
					{NUMANodeAffinity: NewTestBitMask(1), Preferred: true, Score: 70},
				},
			},
			expectedAffinity: NewTestBitMask(1),
		},
		{
			description: "equal scores leave the structural tiebreak in charge",
			strategy:    NUMAAllocationStrategyMostAllocated,
			gateEnabled: true,
			hints: [][]TopologyHint{
				{
					{NUMANodeAffinity: NewTestBitMask(0), Preferred: true, Score: 50},
					{NUMANodeAffinity: NewTestBitMask(1), Preferred: true, Score: 50},
				},
			},
			expectedAffinity: NewTestBitMask(0),
		},
		{
			description: "an unscored hint does not win least-allocated by default",
			strategy:    NUMAAllocationStrategyLeastAllocated,
			gateEnabled: true,
			// Score 0 means unscored, not empty. Were it read as a value, node1
			// would look like the emptiest node and take every least-allocated
			// comparison.
			hints: [][]TopologyHint{
				{
					{NUMANodeAffinity: NewTestBitMask(0), Preferred: true, Score: 50},
					{NUMANodeAffinity: NewTestBitMask(1), Preferred: true, Score: 0},
				},
			},
			expectedAffinity: NewTestBitMask(0),
		},
		{
			description: "the strategy never widens the affinity mask",
			strategy:    NUMAAllocationStrategyMostAllocated,
			gateEnabled: true,
			hints: [][]TopologyHint{
				{
					{NUMANodeAffinity: NewTestBitMask(0), Preferred: true, Score: 20},
					{NUMANodeAffinity: NewTestBitMask(0, 1), Preferred: true, Score: 90},
				},
			},
			expectedAffinity: NewTestBitMask(0),
		},
		{
			description: "a preferred hint beats a higher scoring non-preferred one",
			strategy:    NUMAAllocationStrategyMostAllocated,
			gateEnabled: true,
			hints: [][]TopologyHint{
				{
					{NUMANodeAffinity: NewTestBitMask(0), Preferred: true, Score: 20},
					{NUMANodeAffinity: NewTestBitMask(1), Preferred: false, Score: 90},
				},
			},
			expectedAffinity: NewTestBitMask(0),
		},
		{
			description: "the strategy is inert while the alpha options are disabled",
			strategy:    NUMAAllocationStrategyMostAllocated,
			gateEnabled: false,
			hints: [][]TopologyHint{
				{
					{NUMANodeAffinity: NewTestBitMask(0), Preferred: true, Score: 20},
					{NUMANodeAffinity: NewTestBitMask(1), Preferred: true, Score: 80},
				},
			},
			expectedAffinity: NewTestBitMask(0),
		},
		{
			description: "the none strategy is inert",
			strategy:    NUMAAllocationStrategyNone,
			gateEnabled: true,
			hints: [][]TopologyHint{
				{
					{NUMANodeAffinity: NewTestBitMask(0), Preferred: true, Score: 20},
					{NUMANodeAffinity: NewTestBitMask(1), Preferred: true, Score: 80},
				},
			},
			expectedAffinity: NewTestBitMask(0),
		},
		{
			description: "an unset strategy is inert",
			gateEnabled: true,
			hints: [][]TopologyHint{
				{
					{NUMANodeAffinity: NewTestBitMask(0), Preferred: true, Score: 20},
					{NUMANodeAffinity: NewTestBitMask(1), Preferred: true, Score: 80},
				},
			},
			expectedAffinity: NewTestBitMask(0),
		},
		{
			description:   "most-allocated decides between equally distant masks under prefer-closest",
			numaInfo:      commonNUMAInfoFourNodes(),
			policyName:    PolicyBestEffort,
			preferClosest: true,
			strategy:      NUMAAllocationStrategyMostAllocated,
			gateEnabled:   true,
			// Both masks pair two nodes at distance 11, so Closest rates them
			// equally and falls back to the lower-numbered mask.
			hints: [][]TopologyHint{
				{
					{NUMANodeAffinity: NewTestBitMask(0, 1), Preferred: true, Score: 20},
					{NUMANodeAffinity: NewTestBitMask(2, 3), Preferred: true, Score: 80},
				},
			},
			expectedAffinity: NewTestBitMask(2, 3),
		},
		{
			description:   "prefer-closest keeps the last word over the strategy",
			numaInfo:      commonNUMAInfoFourNodes(),
			policyName:    PolicyBestEffort,
			preferClosest: true,
			strategy:      NUMAAllocationStrategyMostAllocated,
			gateEnabled:   true,
			// Nodes 1 and 2 sit further apart (average 11) than nodes 0 and 1
			// (average 10.5), so Closest separates the two masks on its own and
			// the higher score must not override it.
			hints: [][]TopologyHint{
				{
					{NUMANodeAffinity: NewTestBitMask(0, 1), Preferred: true, Score: 20},
					{NUMANodeAffinity: NewTestBitMask(1, 2), Preferred: true, Score: 80},
				},
			},
			expectedAffinity: NewTestBitMask(0, 1),
		},
	}

	for _, tc := range tcases {
		t.Run(tc.description, func(t *testing.T) {
			logger, _ := ktesting.NewTestContext(t)
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.TopologyManagerPolicyAlphaOptions, tc.gateEnabled)

			numaInfo := tc.numaInfo
			if numaInfo == nil {
				numaInfo = commonNUMAInfoTwoNodes()
			}
			policyName := tc.policyName
			if policyName == "" {
				policyName = PolicySingleNumaNode
			}

			opts := PolicyOptions{
				NUMAAllocationStrategy: tc.strategy,
				PreferClosestNUMA:      tc.preferClosest,
			}
			merger := NewHintMerger(numaInfo, tc.hints, nil, policyName, opts)

			result := merger.Merge(logger)
			if !result.NUMANodeAffinity.IsEqual(tc.expectedAffinity) {
				t.Errorf("Expected affinity %v, got %v", tc.expectedAffinity, result.NUMANodeAffinity)
			}
		})
	}
}

func TestAggregateHintScoresWithWeights(t *testing.T) {
	maskNode0 := NewTestBitMask(0)

	tcases := []struct {
		description   string
		permutation   []TopologyHint
		resourceNames []string
		weights       map[string]int
		expectedScore int64
		expectedOk    bool
	}{
		{
			description: "nil weights give the equal-weight average",
			permutation: []TopologyHint{
				{NUMANodeAffinity: maskNode0, Preferred: true, Score: 80},
				{NUMANodeAffinity: maskNode0, Preferred: true, Score: 60},
			},
			resourceNames: []string{"cpu", "memory"},
			weights:       nil,
			expectedScore: 70,
			expectedOk:    true,
		},
		{
			description: "empty weights give the equal-weight average",
			permutation: []TopologyHint{
				{NUMANodeAffinity: maskNode0, Preferred: true, Score: 80},
				{NUMANodeAffinity: maskNode0, Preferred: true, Score: 60},
			},
			resourceNames: []string{"cpu", "memory"},
			weights:       map[string]int{},
			expectedScore: 70,
			expectedOk:    true,
		},
		{
			description: "explicit weights tilt the average towards the heavier resource",
			permutation: []TopologyHint{
				{NUMANodeAffinity: maskNode0, Preferred: true, Score: 80},
				{NUMANodeAffinity: maskNode0, Preferred: true, Score: 60},
			},
			resourceNames: []string{"cpu", "memory"},
			weights:       map[string]int{"cpu": 7, "memory": 3},
			// (80*7 + 60*3) / 10
			expectedScore: 74,
			expectedOk:    true,
		},
		{
			description: "only the ratios between weights matter",
			permutation: []TopologyHint{
				{NUMANodeAffinity: maskNode0, Preferred: true, Score: 80},
				{NUMANodeAffinity: maskNode0, Preferred: true, Score: 60},
			},
			resourceNames: []string{"cpu", "memory"},
			// Ten times the weights of the previous case, for the same result.
			weights:       map[string]int{"cpu": 70, "memory": 30},
			expectedScore: 74,
			expectedOk:    true,
		},
		{
			description: "a single amplified resource dominates the rest",
			permutation: []TopologyHint{
				{NUMANodeAffinity: maskNode0, Preferred: true, Score: 60},
				{NUMANodeAffinity: maskNode0, Preferred: true, Score: 40},
				{NUMANodeAffinity: maskNode0, Preferred: true, Score: 90},
			},
			resourceNames: []string{"cpu", "memory", "nvidia.com/gpu"},
			weights:       map[string]int{"cpu": 2, "memory": 2, "nvidia.com/gpu": 6},
			// (60*2 + 40*2 + 90*6) / 10
			expectedScore: 74,
			expectedOk:    true,
		},
		{
			description: "a resource the weights do not name sits at the baseline weight",
			permutation: []TopologyHint{
				{NUMANodeAffinity: maskNode0, Preferred: true, Score: 80},
				{NUMANodeAffinity: maskNode0, Preferred: true, Score: 40},
			},
			resourceNames: []string{"cpu", "intel.com/nic"},
			weights:       map[string]int{"cpu": 8},
			// (80*8 + 40*1) / 9, truncated
			expectedScore: 75,
			expectedOk:    true,
		},
		{
			description: "naming only resources the pod does not request changes nothing",
			permutation: []TopologyHint{
				{NUMANodeAffinity: maskNode0, Preferred: true, Score: 80},
				{NUMANodeAffinity: maskNode0, Preferred: true, Score: 60},
				{NUMANodeAffinity: maskNode0, Preferred: true, Score: 40},
			},
			resourceNames: []string{"cpu", "memory", "intel.com/nic"},
			weights:       map[string]int{"nvidia.com/gpu": 10},
			// All three contributors stay at the baseline: (80+60+40) / 3
			expectedScore: 60,
			expectedOk:    true,
		},
		{
			description: "a weight of 0 drops the resource from the average",
			permutation: []TopologyHint{
				{NUMANodeAffinity: maskNode0, Preferred: true, Score: 80},
				{NUMANodeAffinity: maskNode0, Preferred: true, Score: 40},
			},
			resourceNames: []string{"cpu", "memory"},
			weights:       map[string]int{"cpu": 10, "memory": 0},
			expectedScore: 80,
			expectedOk:    true,
		},
		{
			description: "the hint is unscored once every contributor is weighted 0",
			permutation: []TopologyHint{
				{NUMANodeAffinity: maskNode0, Preferred: true, Score: 80},
				{NUMANodeAffinity: maskNode0, Preferred: true, Score: 40},
			},
			resourceNames: []string{"cpu", "memory"},
			weights:       map[string]int{"cpu": 0, "memory": 0},
			expectedScore: 0,
			expectedOk:    false,
		},
		{
			description: "an unscored contributor takes no part, whatever its weight",
			permutation: []TopologyHint{
				{NUMANodeAffinity: maskNode0, Preferred: true, Score: 80},
				{NUMANodeAffinity: maskNode0, Preferred: true, Score: 0},
			},
			resourceNames: []string{"cpu", "memory"},
			// Were the unscored contributor weighted in, its 0 would drag the
			// average down to (80*5) / 10 = 40.
			weights:       map[string]int{"cpu": 5, "memory": 5},
			expectedScore: 80,
			expectedOk:    true,
		},
		{
			description: "a contributor with no affinity takes no part, whatever its weight",
			permutation: []TopologyHint{
				{NUMANodeAffinity: maskNode0, Preferred: true, Score: 80},
				{NUMANodeAffinity: nil, Preferred: true, Score: 40},
			},
			resourceNames: []string{"cpu", "memory"},
			weights:       map[string]int{"cpu": 5, "memory": 5},
			expectedScore: 80,
			expectedOk:    true,
		},
		{
			description: "weights are looked up by name, not by position",
			permutation: []TopologyHint{
				{NUMANodeAffinity: maskNode0, Preferred: true, Score: 50},
				{NUMANodeAffinity: maskNode0, Preferred: true, Score: 90},
				{NUMANodeAffinity: maskNode0, Preferred: true, Score: 30},
			},
			// The names are deliberately out of the order the weights are
			// written in, so weighting by index would pick the wrong ones.
			resourceNames: []string{"memory", "nvidia.com/gpu", "cpu"},
			weights:       map[string]int{"cpu": 2, "memory": 2, "nvidia.com/gpu": 6},
			// (50*2 + 90*6 + 30*2) / 10
			expectedScore: 70,
			expectedOk:    true,
		},
		{
			description: "a contributor with no name sits at the baseline weight",
			permutation: []TopologyHint{
				{NUMANodeAffinity: maskNode0, Preferred: true, Score: 80},
				{NUMANodeAffinity: maskNode0, Preferred: true, Score: 40},
			},
			// A hint provider which expressed no preference at all has no
			// resource to name, and callers which do not weight anything pass
			// no names whatsoever.
			resourceNames: []string{"cpu"},
			weights:       map[string]int{"cpu": 3},
			// (80*3 + 40*1) / 4
			expectedScore: 70,
			expectedOk:    true,
		},
		{
			description:   "an empty permutation leaves the hint unscored",
			permutation:   []TopologyHint{},
			resourceNames: []string{},
			weights:       map[string]int{"cpu": 5},
			expectedScore: 0,
			expectedOk:    false,
		},
	}

	for _, tc := range tcases {
		t.Run(tc.description, func(t *testing.T) {
			score, ok := aggregateHintScores(tc.permutation, tc.resourceNames, tc.weights)
			if score != tc.expectedScore {
				t.Errorf("expected score %d, got %d", tc.expectedScore, score)
			}
			if ok != tc.expectedOk {
				t.Errorf("expected ok %v, got %v", tc.expectedOk, ok)
			}
		})
	}
}

// bestScoringHint folds compareHintScores over a set of hints the way the merger
// folds it over the permutations of a pod's hints, so that a test can assert
// which NUMA node a strategy ends up selecting rather than eyeballing the
// aggregated scores. Hints the strategy cannot separate leave the incumbent in
// place.
func bestScoringHint(strategy string, hints []TopologyHint) *TopologyHint {
	var best *TopologyHint
	for i := range hints {
		if best == nil {
			best = &hints[i]
			continue
		}
		if winner := compareHintScores(strategy, best, &hints[i]); winner != nil {
			best = winner
		}
	}
	return best
}

// TestAggregateHintScoresKEPWorkedExamples pins the worked examples the KEP
// publishes as the user-facing documentation of numa-score-weights. The weight
// strings are the literal ones from the KEP and go through the same parser the
// kubelet uses, so each example is verified end to end from the string an
// operator writes to the NUMA node it selects.
func TestAggregateHintScoresKEPWorkedExamples(t *testing.T) {
	t.Run("prioritizing GPU consolidation", func(t *testing.T) {
		// A two-NUMA-node machine, each node with 64 exclusively allocatable
		// CPUs, 128Gi of memory and 4 GPUs, in the allocation state
		//
		//   NUMA node | CPUs     | memory      | GPUs | cpu | memory | gpu
		//   numa0     | 48 / 64  |  64 / 128Gi | 1/4  |  75 |     50 |  25
		//   numa1     | 16 / 64  |  32 / 128Gi | 3/4  |  25 |     25 |  75
		//
		// A pod requests 8 exclusive CPUs, 16Gi of memory and 1 GPU. Both nodes
		// can satisfy it with a single-node affinity, so the two hints are
		// structurally identical and the aggregated score decides. The operator
		// runs most-allocated to consolidate GPU usage: filling numa1's last GPU
		// keeps a block of 3 free GPUs on numa0 for a later multi-GPU pod.
		resourceNames := []string{"cpu", "memory", "nvidia.com/gpu"}
		numa0 := []TopologyHint{
			{NUMANodeAffinity: NewTestBitMask(0), Preferred: true, Score: 75},
			{NUMANodeAffinity: NewTestBitMask(0), Preferred: true, Score: 50},
			{NUMANodeAffinity: NewTestBitMask(0), Preferred: true, Score: 25},
		}
		numa1 := []TopologyHint{
			{NUMANodeAffinity: NewTestBitMask(1), Preferred: true, Score: 25},
			{NUMANodeAffinity: NewTestBitMask(1), Preferred: true, Score: 25},
			{NUMANodeAffinity: NewTestBitMask(1), Preferred: true, Score: 75},
		}

		tcases := []struct {
			weights          string
			expectedNUMA0    int64
			expectedNUMA1    int64
			strategy         string
			expectedSelected bitmask.BitMask
		}{
			{
				// CPU and memory pressure outvote the GPU signal, and the free
				// GPU block is broken up.
				weights:          "",
				expectedNUMA0:    50,
				expectedNUMA1:    41,
				strategy:         NUMAAllocationStrategyMostAllocated,
				expectedSelected: NewTestBitMask(0),
			},
			{
				// GPU dominates, so the nearly-full GPU node wins.
				weights:          "nvidia.com/gpu=10",
				expectedNUMA0:    31,
				expectedNUMA1:    66,
				strategy:         NUMAAllocationStrategyMostAllocated,
				expectedSelected: NewTestBitMask(1),
			},
			{
				// GPU still leads, CPU moderates it.
				weights:          "cpu=3,memory=1,nvidia.com/gpu=6",
				expectedNUMA0:    42,
				expectedNUMA1:    55,
				strategy:         NUMAAllocationStrategyMostAllocated,
				expectedSelected: NewTestBitMask(1),
			},
			{
				// Ranking CPU as highly as the GPU hands the decision back to
				// CPU utilization, even though the GPU weight did not change.
				weights:          "cpu=5,nvidia.com/gpu=5",
				expectedNUMA0:    50,
				expectedNUMA1:    47,
				strategy:         NUMAAllocationStrategyMostAllocated,
				expectedSelected: NewTestBitMask(0),
			},
			{
				// CPU excluded outright, GPU and memory decide.
				weights:          "nvidia.com/gpu=10,cpu=0",
				expectedNUMA0:    27,
				expectedNUMA1:    70,
				strategy:         NUMAAllocationStrategyMostAllocated,
				expectedSelected: NewTestBitMask(1),
			},
			{
				// Uniform weights are equivalent to leaving the option unset.
				weights:          "cpu=100,memory=100,nvidia.com/gpu=100",
				expectedNUMA0:    50,
				expectedNUMA1:    41,
				strategy:         NUMAAllocationStrategyMostAllocated,
				expectedSelected: NewTestBitMask(0),
			},
			{
				// The weights decide which resource's utilization matters, the
				// strategy decides which direction to move along it: the same
				// weights under least-allocated select the node with the most
				// free GPUs instead.
				weights:          "nvidia.com/gpu=10",
				expectedNUMA0:    31,
				expectedNUMA1:    66,
				strategy:         NUMAAllocationStrategyLeastAllocated,
				expectedSelected: NewTestBitMask(0),
			},
		}

		for _, tc := range tcases {
			t.Run(fmt.Sprintf("%s/%q", tc.strategy, tc.weights), func(t *testing.T) {
				weights, err := parseNUMAScoreWeights(tc.weights)
				if err != nil {
					t.Fatalf("unexpected error parsing %q: %v", tc.weights, err)
				}

				score0, ok0 := aggregateHintScores(numa0, resourceNames, weights)
				score1, ok1 := aggregateHintScores(numa1, resourceNames, weights)
				if !ok0 || !ok1 {
					t.Fatalf("expected both nodes to be scored, got ok %v and %v", ok0, ok1)
				}
				if score0 != tc.expectedNUMA0 {
					t.Errorf("expected numa0 aggregate %d, got %d", tc.expectedNUMA0, score0)
				}
				if score1 != tc.expectedNUMA1 {
					t.Errorf("expected numa1 aggregate %d, got %d", tc.expectedNUMA1, score1)
				}

				selected := bestScoringHint(tc.strategy, []TopologyHint{
					{NUMANodeAffinity: NewTestBitMask(0), Preferred: true, Score: score0},
					{NUMANodeAffinity: NewTestBitMask(1), Preferred: true, Score: score1},
				})
				if !selected.NUMANodeAffinity.IsEqual(tc.expectedSelected) {
					t.Errorf("expected %v to be selected, got %v", tc.expectedSelected, selected.NUMANodeAffinity)
				}
			})
		}
	})

	t.Run("excluding a resource from scoring", func(t *testing.T) {
		// An NFV node with four NUMA nodes, each with 32 exclusively allocatable
		// CPUs, 128Gi of memory and 8 SR-IOV VFs. The operator runs
		// least-allocated and cares only about VF availability, because a NUMA
		// node with no free VF cannot host the pod at all while CPU and memory
		// pressure is already handled by the scheduler.
		//
		//   NUMA node | cpu | memory | VF | free VFs
		//   numa0     |  25 |     25 | 75 |        2
		//   numa1     |  75 |     75 | 25 |        6
		//   numa2     |  50 |     50 | 50 |        4
		//   numa3     |  50 |     50 | 50 |        4
		resourceNames := []string{"cpu", "memory", "intel.com/sriov-nic"}
		perNode := [][]int64{
			{25, 25, 75},
			{75, 75, 25},
			{50, 50, 50},
			{50, 50, 50},
		}

		tcases := []struct {
			weights           string
			expectedAggregate []int64
			expectedSelected  bitmask.BitMask
		}{
			{
				// numa0's low CPU and memory utilization pulls its average down,
				// so least-allocated keeps picking the node with the *fewest*
				// free VFs and exhausts it while numa1 keeps 6 idle.
				weights:           "",
				expectedAggregate: []int64{41, 58, 50, 50},
				expectedSelected:  NewTestBitMask(0),
			},
			{
				// Zeroing CPU and memory leaves the NIC as the only applicable
				// provider, at the default weight, so the aggregate equals the
				// VF score and the node with the most free VFs wins.
				weights:           "cpu=0,memory=0",
				expectedAggregate: []int64{75, 25, 50, 50},
				expectedSelected:  NewTestBitMask(1),
			},
		}

		for _, tc := range tcases {
			t.Run(fmt.Sprintf("%q", tc.weights), func(t *testing.T) {
				weights, err := parseNUMAScoreWeights(tc.weights)
				if err != nil {
					t.Fatalf("unexpected error parsing %q: %v", tc.weights, err)
				}

				var scored []TopologyHint
				for node, scores := range perNode {
					permutation := make([]TopologyHint, 0, len(scores))
					for _, score := range scores {
						permutation = append(permutation, TopologyHint{
							NUMANodeAffinity: NewTestBitMask(node),
							Preferred:        true,
							Score:            score,
						})
					}

					aggregate, ok := aggregateHintScores(permutation, resourceNames, weights)
					if !ok {
						t.Fatalf("expected numa%d to be scored", node)
					}
					if aggregate != tc.expectedAggregate[node] {
						t.Errorf("expected numa%d aggregate %d, got %d", node, tc.expectedAggregate[node], aggregate)
					}
					scored = append(scored, TopologyHint{
						NUMANodeAffinity: NewTestBitMask(node),
						Preferred:        true,
						Score:            aggregate,
					})
				}

				selected := bestScoringHint(NUMAAllocationStrategyLeastAllocated, scored)
				if !selected.NUMANodeAffinity.IsEqual(tc.expectedSelected) {
					t.Errorf("expected %v to be selected, got %v", tc.expectedSelected, selected.NUMANodeAffinity)
				}
			})
		}
	})
}

func TestNewHintMergerGatesScoreWeights(t *testing.T) {
	weights := map[string]int{"nvidia.com/gpu": 10}

	for _, gateEnabled := range []bool{true, false} {
		t.Run(fmt.Sprintf("alpha options enabled=%v", gateEnabled), func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.TopologyManagerPolicyAlphaOptions, gateEnabled)

			opts := PolicyOptions{NUMAScoreWeights: weights}
			merger := NewHintMerger(commonNUMAInfoTwoNodes(), [][]TopologyHint{}, nil, PolicySingleNumaNode, opts)

			// NewPolicyOptions already refuses the option while the alpha
			// options are disabled, so the merger only ever sees weights with
			// the gate on. Pin that a PolicyOptions value built by other means
			// cannot get them past it either.
			expected := weights
			if !gateEnabled {
				expected = nil
			}
			if !reflect.DeepEqual(merger.ScoreWeights, expected) {
				t.Errorf("expected ScoreWeights %v, got %v", expected, merger.ScoreWeights)
			}
		})
	}
}

// TestHintMergerWithScoreWeights drives the weights through merger.Merge, to
// cover what aggregateHintScores on its own cannot: that the weights reach the
// aggregation and that they compose with numa-allocation-strategy.
//
// Every case runs the same allocation state, the one from the KEP's GPU
// consolidation example: cpu and memory rate numa0 as the busier node, the GPU
// rates numa1 as the busier one. Which node wins therefore says which resource
// the weights put in charge, and the cases come in pairs which differ only in
// the weights so that neither half can pass by accident.
//
// Note that a case expecting numa0 proves less on its own than one expecting
// numa1: IsNarrowerThan tie-breaks equal-width masks with IsLessThan, so numa0
// is what the structural comparison falls back to anyway. Each such case is
// paired with the one that pulls the other way.
func TestHintMergerWithScoreWeights(t *testing.T) {
	resourceNames := []string{"cpu", "memory", "nvidia.com/gpu"}
	hints := [][]TopologyHint{
		{
			{NUMANodeAffinity: NewTestBitMask(0), Preferred: true, Score: 75},
			{NUMANodeAffinity: NewTestBitMask(1), Preferred: true, Score: 25},
		},
		{
			{NUMANodeAffinity: NewTestBitMask(0), Preferred: true, Score: 50},
			{NUMANodeAffinity: NewTestBitMask(1), Preferred: true, Score: 25},
		},
		{
			{NUMANodeAffinity: NewTestBitMask(0), Preferred: true, Score: 25},
			{NUMANodeAffinity: NewTestBitMask(1), Preferred: true, Score: 75},
		},
	}

	tcases := []struct {
		description      string
		strategy         string
		weights          map[string]int
		gateEnabled      bool
		expectedAffinity bitmask.BitMask
		expectedScore    int64
	}{
		{
			description:      "without weights most-allocated follows the CPU and memory majority",
			strategy:         NUMAAllocationStrategyMostAllocated,
			gateEnabled:      true,
			expectedAffinity: NewTestBitMask(0),
			expectedScore:    50,
		},
		{
			description:      "amplifying the GPU makes most-allocated follow it instead",
			strategy:         NUMAAllocationStrategyMostAllocated,
			weights:          map[string]int{"nvidia.com/gpu": 10},
			gateEnabled:      true,
			expectedAffinity: NewTestBitMask(1),
			expectedScore:    66,
		},
		{
			description:      "without weights least-allocated follows the CPU and memory majority",
			strategy:         NUMAAllocationStrategyLeastAllocated,
			gateEnabled:      true,
			expectedAffinity: NewTestBitMask(1),
			expectedScore:    41,
		},
		{
			description:      "amplifying the GPU makes least-allocated follow it instead",
			strategy:         NUMAAllocationStrategyLeastAllocated,
			weights:          map[string]int{"nvidia.com/gpu": 10},
			gateEnabled:      true,
			expectedAffinity: NewTestBitMask(0),
			expectedScore:    31,
		},
		{
			description: "excluding the CPU and memory leaves the GPU deciding on its own",
			strategy:    NUMAAllocationStrategyMostAllocated,
			weights:     map[string]int{"cpu": 0, "memory": 0},
			gateEnabled: true,
			// The GPU is the only resource left with a weight, so the merged
			// score is its own.
			expectedAffinity: NewTestBitMask(1),
			expectedScore:    75,
		},
		{
			description: "weights do nothing without an allocation strategy",
			strategy:    NUMAAllocationStrategyNone,
			weights:     map[string]int{"nvidia.com/gpu": 10},
			gateEnabled: true,
			// The merged hint still carries the weighted score, but with no
			// strategy to consult it the structural tiebreak decides.
			expectedAffinity: NewTestBitMask(0),
			expectedScore:    31,
		},
		{
			description: "the weights are inert while the alpha options are disabled",
			strategy:    NUMAAllocationStrategyMostAllocated,
			weights:     map[string]int{"nvidia.com/gpu": 10},
			gateEnabled: false,
			// The gate stops the strategy and the weights alike, so this is the
			// equal-weight average and the structural tiebreak, exactly as
			// before either option existed.
			expectedAffinity: NewTestBitMask(0),
			expectedScore:    50,
		},
	}

	for _, tc := range tcases {
		t.Run(tc.description, func(t *testing.T) {
			logger, _ := ktesting.NewTestContext(t)
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.TopologyManagerPolicyAlphaOptions, tc.gateEnabled)

			opts := PolicyOptions{
				NUMAAllocationStrategy: tc.strategy,
				NUMAScoreWeights:       tc.weights,
			}
			merger := NewHintMerger(commonNUMAInfoTwoNodes(), hints, resourceNames, PolicySingleNumaNode, opts)

			result := merger.Merge(logger)
			if !result.NUMANodeAffinity.IsEqual(tc.expectedAffinity) {
				t.Errorf("expected affinity %v, got %v", tc.expectedAffinity, result.NUMANodeAffinity)
			}
			if result.Score != tc.expectedScore {
				t.Errorf("expected merged Score %d, got %d", tc.expectedScore, result.Score)
			}
		})
	}
}

// TestPolicyMergeAppliesScoreWeights closes the loop from a provider's
// map[string][]TopologyHint to the weighted aggregate, which the merger-level
// tests cannot: they hand the resource names to NewHintMerger themselves,
// whereas here the names have to survive filterProvidersHints and stay lined up
// with the hints they came from.
//
// All three resources come from a single provider, so Go randomizes the order
// filterProvidersHints reads them in. Weighting by position rather than by name
// would therefore pick the wrong weights for most of the orderings, and show up
// as an intermittent failure rather than a green run.
func TestPolicyMergeAppliesScoreWeights(t *testing.T) {
	providersHints := []map[string][]TopologyHint{
		{
			"cpu": {
				{NUMANodeAffinity: NewTestBitMask(0), Preferred: true, Score: 75},
				{NUMANodeAffinity: NewTestBitMask(1), Preferred: true, Score: 25},
			},
			"memory": {
				{NUMANodeAffinity: NewTestBitMask(0), Preferred: true, Score: 50},
				{NUMANodeAffinity: NewTestBitMask(1), Preferred: true, Score: 25},
			},
			"nvidia.com/gpu": {
				{NUMANodeAffinity: NewTestBitMask(0), Preferred: true, Score: 25},
				{NUMANodeAffinity: NewTestBitMask(1), Preferred: true, Score: 75},
			},
		},
	}

	tcases := []struct {
		description      string
		weights          map[string]int
		expectedAffinity bitmask.BitMask
		expectedScore    int64
	}{
		{
			description:      "equal weights consolidate onto the node the CPU and memory rate busiest",
			expectedAffinity: NewTestBitMask(0),
			expectedScore:    50,
		},
		{
			description:      "amplifying the GPU consolidates onto the node it rates busiest",
			weights:          map[string]int{"nvidia.com/gpu": 10},
			expectedAffinity: NewTestBitMask(1),
			expectedScore:    66,
		},
	}

	for _, tc := range tcases {
		t.Run(tc.description, func(t *testing.T) {
			logger, _ := ktesting.NewTestContext(t)
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.TopologyManagerPolicyAlphaOptions, true)

			policy := NewBestEffortPolicy(commonNUMAInfoTwoNodes(), PolicyOptions{
				NUMAAllocationStrategy: NUMAAllocationStrategyMostAllocated,
				NUMAScoreWeights:       tc.weights,
			})

			result, admit := policy.Merge(logger, providersHints)
			if !admit {
				t.Fatal("expected the best-effort policy to admit")
			}
			if !result.NUMANodeAffinity.IsEqual(tc.expectedAffinity) {
				t.Errorf("expected affinity %v, got %v", tc.expectedAffinity, result.NUMANodeAffinity)
			}
			if result.Score != tc.expectedScore {
				t.Errorf("expected merged Score %d, got %d", tc.expectedScore, result.Score)
			}
		})
	}
}
