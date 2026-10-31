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
	"k8s.io/ktesting"
	pkgfeatures "k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager/bitmask"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
)

type policyMergeTestCase struct {
	name     string
	hp       []HintProvider
	expected TopologyHint
}

// unnamedHintLists turns bare hint lists into resourceHints with no resource
// names attached, for the tests which exercise the merge arithmetic and do not
// care which resource a hint came from.
func unnamedHintLists(hintLists [][]TopologyHint) []resourceHints {
	if hintLists == nil {
		return nil
	}
	all := make([]resourceHints, 0, len(hintLists))
	for _, hints := range hintLists {
		all = append(all, resourceHints{hints: hints})
	}
	return all
}

// namedHintLists pairs each hint list with the resource it came from.
// resources[i] names hintLists[i]; a name missing from resources is left empty,
// which is what a provider with no preference at all contributes.
func namedHintLists(resources []string, hintLists [][]TopologyHint) []resourceHints {
	all := unnamedHintLists(hintLists)
	for i := range all {
		if i < len(resources) {
			all[i].resource = resources[i]
		}
	}
	return all
}

// mergerForPermutation builds a HintMerger which can score the given
// permutation directly, without going through Merge. The permutation's i-th
// element is attributed to resources[i], mirroring what iterating over
// m.Hints would have produced; a name missing from resources is left empty,
// which is what a provider with no preference at all contributes.
func mergerForPermutation(resources []string, permutation []TopologyHint, weights map[string]int) HintMerger {
	hints := make([]resourceHints, len(permutation))
	for i := range permutation {
		if i < len(resources) {
			hints[i].resource = resources[i]
		}
		hints[i].hints = []TopologyHint{permutation[i]}
	}
	// Apply defaults if weights is nil, matching NewPolicyOptions behavior
	if weights == nil {
		weights = defaultNUMAScoreWeights()
	}
	return HintMerger{Hints: hints, ScoreWeights: weights}
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
			result := maxOfMinAffinityCounts(unnamedHintLists(tc.hints))
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
			merger := NewHintMerger(numaInfo, nil, PolicyBestEffort, PolicyOptions{})
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

// TestFilterProvidersHints checks that each entry filterProvidersHints returns
// carries the hints together with the name of the resource they came from.
// Providers are iterated as maps, so the order of the entries is not fixed and
// the expectations are keyed by resource name rather than by position. A name
// maps to a list of hint lists because the same name can occur more than once.
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
			// Every provider with nothing to say contributes an entry under
			// the empty name, so a pod which pins neither CPUs nor memory nor
			// devices produces several of them. Each is an independent axis of
			// the merge permutation and all of them have to survive, which is
			// why the result is a slice and not a map keyed by resource name.
			name:          "several providers with no preference for any resource",
			providersHint: []map[string][]TopologyHint{nil, nil, nil},
			expected: map[string][][]TopologyHint{
				"": {
					{{NUMANodeAffinity: nil, Preferred: true}},
					{{NUMANodeAffinity: nil, Preferred: true}},
					{{NUMANodeAffinity: nil, Preferred: true}},
				},
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
			filtered := filterProvidersHints(logger, tc.providersHint)

			got := map[string][][]TopologyHint{}
			for _, oneResource := range filtered {
				got[oneResource.resource] = append(got[oneResource.resource], oneResource.hints)
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
			// Unscored and affinity-less contributors are skipped rather than
			// averaged in, so the weakest possible signal still sets the
			// merged score instead of being diluted towards zero.
			name: "lone minimum score survives unscored and affinity-less contributors",
			permutation: []TopologyHint{
				{NUMANodeAffinity: NewTestBitMask(0), Preferred: true, Score: 1},
				{NUMANodeAffinity: NewTestBitMask(0), Preferred: true, Score: 0},
				{NUMANodeAffinity: nil, Preferred: true, Score: 0},
				{NUMANodeAffinity: nil, Preferred: true, Score: 100},
			},
			expectedScore: 1,
			expectedOk:    true,
		},
		{
			// 121 / 2 = 60.5, truncated to 60.
			name: "odd sum truncates towards zero",
			permutation: []TopologyHint{
				{NUMANodeAffinity: NewTestBitMask(0), Preferred: true, Score: 40},
				{NUMANodeAffinity: NewTestBitMask(0), Preferred: true, Score: 81},
			},
			expectedScore: 60,
			expectedOk:    true,
		},
		{
			// 101 / 2 = 50.5, truncated to 50: the extremes of the range
			// average no differently.
			name: "minimum and maximum score truncate towards zero",
			permutation: []TopologyHint{
				{NUMANodeAffinity: NewTestBitMask(0), Preferred: true, Score: 1},
				{NUMANodeAffinity: NewTestBitMask(0), Preferred: true, Score: 100},
			},
			expectedScore: 50,
			expectedOk:    true,
		},
		{
			// 5 / 3 = 1.67, truncated to 1. Since every counted score is at
			// least 1, truncation can never report 0 and so can never be
			// mistaken for "no contributor had a score".
			name: "low odd scores truncate towards zero",
			permutation: []TopologyHint{
				{NUMANodeAffinity: NewTestBitMask(0), Preferred: true, Score: 1},
				{NUMANodeAffinity: NewTestBitMask(0), Preferred: true, Score: 1},
				{NUMANodeAffinity: NewTestBitMask(0), Preferred: true, Score: 3},
			},
			expectedScore: 1,
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
			// Provide resource names matching the default weights so hints participate in scoring
			resources := make([]string, len(tc.permutation))
			for i := range resources {
				if i%2 == 0 {
					resources[i] = "cpu"
				} else {
					resources[i] = "memory"
				}
			}
			score, ok := mergerForPermutation(resources, tc.permutation, nil).aggregateHintScores(tc.permutation)
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
			name: "merged hint truncates a fractional average",
			permutation: []TopologyHint{
				{NUMANodeAffinity: NewTestBitMask(0), Preferred: true, Score: 40},
				{NUMANodeAffinity: NewTestBitMask(0), Preferred: true, Score: 81},
			},
			expectedScore: 60,
		},
		{
			name: "merged hint carries a lone minimum score",
			permutation: []TopologyHint{
				{NUMANodeAffinity: NewTestBitMask(0), Preferred: true, Score: 1},
				{NUMANodeAffinity: NewTestBitMask(0), Preferred: true, Score: 0},
				{NUMANodeAffinity: nil, Preferred: true, Score: 0},
			},
			expectedScore: 1,
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
			// Provide resource names matching the default weights
			resources := make([]string, len(tc.permutation))
			for i := range resources {
				if i%2 == 0 {
					resources[i] = "cpu"
				} else {
					resources[i] = "memory"
				}
			}
			merged := mergerForPermutation(resources, tc.permutation, nil).mergePermutation(logger, defaultAffinity, tc.permutation)
			if merged.Score != tc.expectedScore {
				t.Errorf("expected merged Score %d, got %d", tc.expectedScore, merged.Score)
			}
		})
	}
}

// TestCompareWinnerUnchangedByScore documents that Score is inert during hint
// selection: the winner is decided by Preferred and by the affinity mask alone,
// so even the weakest score cannot steer the decision towards itself.
func TestCompareWinnerUnchangedByScore(t *testing.T) {
	numaInfo := commonNUMAInfoTwoNodes()
	hints := [][]TopologyHint{}
	merger := NewHintMerger(numaInfo, unnamedHintLists(hints), PolicyBestEffort, PolicyOptions{})

	tcases := []struct {
		name          string
		current       *TopologyHint
		candidate     *TopologyHint
		expectCurrent bool
	}{
		{
			name:          "narrower hint wins over wider hint with a higher score",
			current:       &TopologyHint{NUMANodeAffinity: NewTestBitMask(0), Preferred: true, Score: 10},
			candidate:     &TopologyHint{NUMANodeAffinity: NewTestBitMask(0, 1), Preferred: true, Score: 90},
			expectCurrent: true,
		},
		{
			name:          "minimum score does not cost the narrower hint the win",
			current:       &TopologyHint{NUMANodeAffinity: NewTestBitMask(0), Preferred: true, Score: 1},
			candidate:     &TopologyHint{NUMANodeAffinity: NewTestBitMask(0, 1), Preferred: true, Score: 100},
			expectCurrent: true,
		},
		{
			// Same pair with the arguments swapped: the outcome follows the
			// mask, not the position or the score.
			name:          "minimum score does not cost the narrower candidate the win",
			current:       &TopologyHint{NUMANodeAffinity: NewTestBitMask(0, 1), Preferred: true, Score: 100},
			candidate:     &TopologyHint{NUMANodeAffinity: NewTestBitMask(0), Preferred: true, Score: 1},
			expectCurrent: false,
		},
		{
			name:          "unscored narrower hint wins over a fully scored wider hint",
			current:       &TopologyHint{NUMANodeAffinity: NewTestBitMask(0), Preferred: true, Score: 0},
			candidate:     &TopologyHint{NUMANodeAffinity: NewTestBitMask(0, 1), Preferred: true, Score: 100},
			expectCurrent: true,
		},
		{
			name:          "preferred hint with the minimum score wins over a non-preferred hint with the maximum score",
			current:       &TopologyHint{NUMANodeAffinity: NewTestBitMask(0, 1), Preferred: false, Score: 100},
			candidate:     &TopologyHint{NUMANodeAffinity: NewTestBitMask(0), Preferred: true, Score: 1},
			expectCurrent: false,
		},
	}

	for _, tc := range tcases {
		t.Run(tc.name, func(t *testing.T) {
			expected := tc.candidate
			if tc.expectCurrent {
				expected = tc.current
			}
			result := merger.compare(tc.current, tc.candidate)
			if result != expected {
				t.Errorf("expected hint %v (Score %d) to win, got %v (Score %d)",
					expected.NUMANodeAffinity, expected.Score, result.NUMANodeAffinity, result.Score)
			}
		})
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
			merger := NewHintMerger(numaInfo, nil, PolicySingleNumaNode, opts)

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
	merger := NewHintMerger(numaInfo, nil, PolicyBestEffort, opts)

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
			description: "the aggregate across all providers decides the merged hint",
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
			// Score 0 means unscored. Were it read as a value, node1 would
			// look like the emptiest node and take every least-allocated
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
				NUMAScoreWeights:       defaultNUMAScoreWeights(),
			}
			// Give each hint list a resource name from the defaults so the
			// opt-in pattern includes them. "cpu" for all is sufficient since
			// these tests verify the allocation strategy, not the weights.
			resources := make([]string, len(tc.hints))
			for i := range resources {
				resources[i] = "cpu"
			}
			merger := NewHintMerger(numaInfo, namedHintLists(resources, tc.hints), policyName, opts)

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
			description: "empty weights exclude all resources (opt-in pattern)",
			permutation: []TopologyHint{
				{NUMANodeAffinity: maskNode0, Preferred: true, Score: 80},
				{NUMANodeAffinity: maskNode0, Preferred: true, Score: 60},
			},
			resourceNames: []string{"cpu", "memory"},
			weights:       map[string]int{},
			expectedScore: 0,
			expectedOk:    false,
		},
		{
			// The lookup is case-sensitive, so a weight whose case does not
			// match the name the provider reported simply never applies. With
			// the opt-in pattern, this excludes all resources, with no error
			// anywhere: a name matching nothing is indistinguishable from one
			// the pod does not request.
			description: "a weight whose case does not match excludes the resource (opt-in)",
			permutation: []TopologyHint{
				{NUMANodeAffinity: maskNode0, Preferred: true, Score: 80},
				{NUMANodeAffinity: maskNode0, Preferred: true, Score: 60},
			},
			resourceNames: []string{"cpu", "memory"},
			weights:       map[string]int{"CPU": 100},
			expectedScore: 0,
			expectedOk:    false,
		},
		{
			// Conversely, a name that does match exactly is applied, including
			// the mixed case hugepages resources carry. With opt-in pattern, cpu
			// is excluded since it's not in the weights map.
			description: "a mixed-case resource name matches when the case is exact",
			permutation: []TopologyHint{
				{NUMANodeAffinity: maskNode0, Preferred: true, Score: 80},
				{NUMANodeAffinity: maskNode0, Preferred: true, Score: 60},
			},
			resourceNames: []string{"cpu", "hugepages-2Mi"},
			weights:       map[string]int{"hugepages-2Mi": 3},
			// cpu not in weights → excluded. Only hugepages: (60*3) / 3
			expectedScore: 60,
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
			description: "a resource the weights do not name is excluded (opt-in pattern)",
			permutation: []TopologyHint{
				{NUMANodeAffinity: maskNode0, Preferred: true, Score: 80},
				{NUMANodeAffinity: maskNode0, Preferred: true, Score: 40},
			},
			resourceNames: []string{"cpu", "intel.com/nic"},
			weights:       map[string]int{"cpu": 8},
			// nic not in weights → excluded. Only CPU: (80*8) / 8
			expectedScore: 80,
			expectedOk:    true,
		},
		{
			description: "naming only resources the pod does not request excludes all contributors",
			permutation: []TopologyHint{
				{NUMANodeAffinity: maskNode0, Preferred: true, Score: 80},
				{NUMANodeAffinity: maskNode0, Preferred: true, Score: 60},
				{NUMANodeAffinity: maskNode0, Preferred: true, Score: 40},
			},
			resourceNames: []string{"cpu", "memory", "intel.com/nic"},
			weights:       map[string]int{"nvidia.com/gpu": 10},
			// All three not in weights → all excluded → no score
			expectedScore: 0,
			expectedOk:    false,
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
			description: "weights follow the resource name when the order differs",
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
			description: "a contributor with no name is excluded (opt-in pattern)",
			permutation: []TopologyHint{
				{NUMANodeAffinity: maskNode0, Preferred: true, Score: 80},
				{NUMANodeAffinity: maskNode0, Preferred: true, Score: 40},
			},
			// A hint provider which expressed no preference at all has no
			// resource to name. With opt-in pattern, unnamed resources are
			// excluded from scoring.
			resourceNames: []string{"cpu"},
			weights:       map[string]int{"cpu": 3},
			// Second hint has no name → excluded. Only first: (80*3) / 3
			expectedScore: 80,
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
			score, ok := mergerForPermutation(tc.resourceNames, tc.permutation, tc.weights).aggregateHintScores(tc.permutation)
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
				// Empty weights → defaults {cpu:1, memory:1}, GPU excluded.
				// numa0: (75+50)/2=62, numa1: (25+25)/2=25
				weights:          "",
				expectedNUMA0:    62,
				expectedNUMA1:    25,
				strategy:         NUMAAllocationStrategyMostAllocated,
				expectedSelected: NewTestBitMask(0),
			},
			{
				// Only GPU participates: numa0=25, numa1=75
				weights:          "nvidia.com/gpu=10",
				expectedNUMA0:    25,
				expectedNUMA1:    75,
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
				// cpu and gpu (memory excluded): numa0=(75*5+25*5)/10=50, numa1=(25*5+75*5)/10=50
				// Equal scores, structural tiebreak picks numa0
				weights:          "cpu=5,nvidia.com/gpu=5",
				expectedNUMA0:    50,
				expectedNUMA1:    50,
				strategy:         NUMAAllocationStrategyMostAllocated,
				expectedSelected: NewTestBitMask(0),
			},
			{
				// cpu=0 excludes it, only gpu participates: numa0=25, numa1=75
				weights:          "nvidia.com/gpu=10,cpu=0",
				expectedNUMA0:    25,
				expectedNUMA1:    75,
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
				// Only gpu: numa0=25, numa1=75
				// least-allocated picks lower score (numa0)
				weights:          "nvidia.com/gpu=10",
				expectedNUMA0:    25,
				expectedNUMA1:    75,
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
				// Apply defaults if empty, matching NewPolicyOptions behavior
				if len(weights) == 0 {
					weights = defaultNUMAScoreWeights()
				}

				score0, ok0 := mergerForPermutation(resourceNames, numa0, weights).aggregateHintScores(numa0)
				score1, ok1 := mergerForPermutation(resourceNames, numa1, weights).aggregateHintScores(numa1)
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
				// Empty weights → defaults {cpu:1, memory:1}, NIC excluded.
				// numa0: (25+25)/2=25, numa1: (75+75)/2=75, numa2/3: 50
				// least-allocated picks lowest (numa0)
				weights:           "",
				expectedAggregate: []int64{25, 75, 50, 50},
				expectedSelected:  NewTestBitMask(0),
			},
			{
				// Zeroing CPU and memory, including NIC explicitly (opt-in pattern).
				// Only NIC participates: scores are 75, 25, 50, 50
				weights:           "cpu=0,memory=0,intel.com/sriov-nic=1",
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
				// Apply defaults if empty, matching NewPolicyOptions behavior
				if len(weights) == 0 {
					weights = defaultNUMAScoreWeights()
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

					aggregate, ok := mergerForPermutation(resourceNames, permutation, weights).aggregateHintScores(permutation)
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
			// weights=nil gets defaults {cpu:1, memory:1}. With opt-in, only
			// cpu and memory participate (gpu excluded).
			// numa0: (75+50)/2=62, numa1: (25+25)/2=25
			description:      "without weights most-allocated follows the CPU and memory majority",
			strategy:         NUMAAllocationStrategyMostAllocated,
			weights:          defaultNUMAScoreWeights(),
			gateEnabled:      true,
			expectedAffinity: NewTestBitMask(0),
			expectedScore:    62,
		},
		{
			// weights={gpu:10}: only gpu participates (cpu/memory excluded).
			// numa0: 25, numa1: 75
			description:      "amplifying the GPU makes most-allocated follow it instead",
			strategy:         NUMAAllocationStrategyMostAllocated,
			weights:          map[string]int{"nvidia.com/gpu": 10},
			gateEnabled:      true,
			expectedAffinity: NewTestBitMask(1),
			expectedScore:    75,
		},
		{
			// Defaults {cpu:1, memory:1}: numa0=62, numa1=25
			// least-allocated picks lower score
			description:      "without weights least-allocated follows the CPU and memory majority",
			strategy:         NUMAAllocationStrategyLeastAllocated,
			weights:          defaultNUMAScoreWeights(),
			gateEnabled:      true,
			expectedAffinity: NewTestBitMask(1),
			expectedScore:    25,
		},
		{
			description:      "amplifying the GPU makes least-allocated follow it instead",
			strategy:         NUMAAllocationStrategyLeastAllocated,
			weights:          map[string]int{"nvidia.com/gpu": 10},
			gateEnabled:      true,
			expectedAffinity: NewTestBitMask(0),
			expectedScore:    25,
		},
		{
			// cpu and memory excluded, gpu explicitly included (opt-in pattern)
			description: "excluding the CPU and memory leaves the GPU deciding on its own",
			strategy:    NUMAAllocationStrategyMostAllocated,
			weights:     map[string]int{"cpu": 0, "memory": 0, "nvidia.com/gpu": 1},
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
			expectedScore:    25,
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
			merger := NewHintMerger(commonNUMAInfoTwoNodes(), namedHintLists(resourceNames, hints), PolicySingleNumaNode, opts)

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
			// weights=nil gets defaults {cpu:1, memory:1}, only cpu/memory participate.
			// numa0: (75+50)/2=62, numa1: (25+25)/2=25
			description:      "equal weights consolidate onto the node the CPU and memory rate busiest",
			weights:          defaultNUMAScoreWeights(),
			expectedAffinity: NewTestBitMask(0),
			expectedScore:    62,
		},
		{
			description:      "amplifying the GPU consolidates onto the node it rates busiest",
			weights:          map[string]int{"nvidia.com/gpu": 10},
			expectedAffinity: NewTestBitMask(1),
			expectedScore:    75,
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
