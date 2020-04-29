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

type podLevelSingleNumaNodePolicy struct {
	//List of NUMA Nodes available on the underlying machine
	numaNodes []int
}

var _ Policy = &podLevelSingleNumaNodePolicy{}

// PolicySingleNumaNode policy name.
const PolicyPodLevelSingleNumaNode string = "pod-levelsingle-numa-node"

// NewPodLevelSingleNumaNodePolicy returns single-numa-node policy.
func NewPodLevelSingleNumaNodePolicy(numaNodes []int) Policy {
	return &podLevelSingleNumaNodePolicy{numaNodes: numaNodes}
}

func (p *podLevelSingleNumaNodePolicy) Name() string {
	return PolicyPodLevelSingleNumaNode
}

func (p *podLevelSingleNumaNodePolicy) canAdmitContainerResult(hint *TopologyHint) bool {
	return hint.Preferred
}


func (p *podLevelSingleNumaNodePolicy) Merge(providersHints []map[string][]TopologyHint) (TopologyHint, bool) {
	filteredHints := filterProvidersHints(providersHints)
	// Filter to only include don't cares and hints with a single NUMA node.
	singleNumaHints := filterSingleNumaHints(filteredHints)
	bestHint := mergeFilteredHints(p.numaNodes, singleNumaHints)

	defaultAffinity, _ := bitmask.NewBitMask(p.numaNodes...)
	if bestHint.NUMANodeAffinity.IsEqual(defaultAffinity) {
		bestHint = TopologyHint{nil, bestHint.Preferred}
	}

	admit := p.canAdmitContainerResult(&bestHint)
	return bestHint, admit
}

func (p *podLevelSingleNumaNodePolicy) CanAdmitPodResult(allContainersHints []TopologyHint) bool {
	var allContainerAffinities []bitmask.BitMask
	for _, hint := range allContainersHints {
		allContainerAffinities = append(allContainerAffinities, hint.NUMANodeAffinity)
	}

	mergedAffinity := bitmask.NewEmptyBitMask()
	mergedAffinity.And(allContainerAffinities...)

	return mergedAffinity.Count() == 1
}