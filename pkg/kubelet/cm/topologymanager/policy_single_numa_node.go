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
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager/bitmask"
)

type singleNumaNodePolicy struct {
	//List of NUMA Nodes available on the underlying machine
	numaNodes []int
}

var _ Policy = &singleNumaNodePolicy{}

// PolicySingleNumaNode policy name.
const PolicySingleNumaNode string = "single-numa-node"

// NewSingleNumaNodePolicy returns single-numa-node policy.
func NewSingleNumaNodePolicy(numaNodes []int) Policy {
	return &singleNumaNodePolicy{numaNodes: numaNodes}
}

func (p *singleNumaNodePolicy) Name() string {
	return PolicySingleNumaNode
}

func (p *singleNumaNodePolicy) canAdmitPodResult(hint *TopologyHint) bool {
	return hint.Preferred
}

// Return hints that have valid bitmasks with exactly one bit set.
func filterSingleNumaHints(allResourcesHints [][]TopologyHint) [][]TopologyHint {
	var filteredResourcesHints [][]TopologyHint
	for _, oneResourceHints := range allResourcesHints {
		var filtered []TopologyHint
		for _, hint := range oneResourceHints {
			if hint.NUMANodeAffinity == nil && hint.Preferred == true {
				filtered = append(filtered, hint)
			}
			if hint.NUMANodeAffinity != nil && hint.NUMANodeAffinity.Count() == 1 && hint.Preferred == true {
				filtered = append(filtered, hint)
			}
		}
		filteredResourcesHints = append(filteredResourcesHints, filtered)
	}
	return filteredResourcesHints
}

func (p *singleNumaNodePolicy) Merge(providersHints []map[string][]TopologyHint) (TopologyHint, bool) {
	filteredHints := filterProvidersHints(providersHints)
	// Filter to only include don't cares and hints with a single NUMA node.
	singleNumaHints := filterSingleNumaHints(filteredHints)
	bestHint := mergeFilteredHints(p.numaNodes, singleNumaHints)

	defaultAffinity, _ := bitmask.NewBitMask(p.numaNodes...)
	if bestHint.NUMANodeAffinity.IsEqual(defaultAffinity) {
		bestHint = TopologyHint{nil, bestHint.Preferred}
	}

	admit := p.canAdmitPodResult(&bestHint)
	return bestHint, admit
}
