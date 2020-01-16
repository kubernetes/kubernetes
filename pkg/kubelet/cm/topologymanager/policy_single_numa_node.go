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
	"k8s.io/klog"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager/bitmask"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
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

func (p *singleNumaNodePolicy) canAdmitPodResult(hint *TopologyHint) lifecycle.PodAdmitResult {
	if !hint.Preferred {
		return lifecycle.PodAdmitResult{
			Admit:   false,
			Reason:  "Topology Affinity Error",
			Message: "Resources cannot be allocated with Topology Locality",
		}
	}
	return lifecycle.PodAdmitResult{
		Admit: true,
	}
}

// Return hints that have valid bitmasks with exactly one bit set.
func (p *singleNumaNodePolicy) filterHints(allResourcesHints [][]TopologyHint) [][]TopologyHint {
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

func (p *singleNumaNodePolicy) mergeProvidersHints(providersHints []map[string][]TopologyHint) TopologyHint {
	// Set the default affinity as an any-numa affinity containing the list
	// of NUMA Nodes available on this machine.
	defaultAffinity, _ := bitmask.NewBitMask(p.numaNodes...)

	// Loop through all provider hints and save an accumulated list of the
	// hints returned by each hint provider. If no hints are provided, assume
	// that provider has no preference for topology-aware allocation.
	var allProviderHints [][]TopologyHint
	for _, hints := range providersHints {
		// If hints is nil, insert a single, preferred any-numa hint into allProviderHints.
		if len(hints) == 0 {
			klog.Infof("[topologymanager] Hint Provider has no preference for NUMA affinity with any resource")
			allProviderHints = append(allProviderHints, []TopologyHint{{nil, true}})
			continue
		}

		// Otherwise, accumulate the hints for each resource type into allProviderHints.
		for resource := range hints {
			if hints[resource] == nil {
				klog.Infof("[topologymanager] Hint Provider has no preference for NUMA affinity with resource '%s'", resource)
				allProviderHints = append(allProviderHints, []TopologyHint{{nil, true}})
				continue
			}

			if len(hints[resource]) == 0 {
				klog.Infof("[topologymanager] Hint Provider has no possible NUMA affinities for resource '%s'", resource)
				allProviderHints = append(allProviderHints, []TopologyHint{{nil, false}})
				continue
			}

			allProviderHints = append(allProviderHints, hints[resource])
		}
	}

	// Filter to only include don't cares and hints with a single NUMA node.
	allProviderHints = p.filterHints(allProviderHints)

	// Set the bestHint to return from this function as {nil false}.
	// This will only be returned if no better hint can be found when
	// merging hints from each hint provider.
	bestHint := TopologyHint{defaultAffinity, false}
	iterateAllProviderTopologyHints(allProviderHints, func(permutation []TopologyHint) {
		// Get the NUMANodeAffinity from each hint in the permutation and see if any
		// of them encode unpreferred allocations.
		mergedHint := mergePermutation(p.numaNodes, permutation)

		// Only consider mergedHints that result in a NUMANodeAffinity > 0 to
		// replace the current bestHint.
		if mergedHint.NUMANodeAffinity.Count() == 0 {
			return
		}

		// If the current bestHint is non-preferred and the new mergedHint is
		// preferred, always choose the preferred hint over the non-preferred one.
		if mergedHint.Preferred && !bestHint.Preferred {
			bestHint = mergedHint
			return
		}

		// If the current bestHint is preferred and the new mergedHint is
		// non-preferred, never update bestHint, regardless of mergedHint's
		// narowness.
		if !mergedHint.Preferred && bestHint.Preferred {
			return
		}

		// If mergedHint and bestHint has the same preference, only consider
		// mergedHints that have a narrower NUMANodeAffinity than the
		// NUMANodeAffinity in the current bestHint.
		if !mergedHint.NUMANodeAffinity.IsNarrowerThan(bestHint.NUMANodeAffinity) {
			return
		}

		// In all other cases, update bestHint to the current mergedHint
		bestHint = mergedHint
	})

	if bestHint.NUMANodeAffinity.IsEqual(defaultAffinity) {
		bestHint = TopologyHint{nil, bestHint.Preferred}
	}

	return bestHint
}

func (p *singleNumaNodePolicy) Merge(providersHints []map[string][]TopologyHint) (TopologyHint, lifecycle.PodAdmitResult) {
	hint := p.mergeProvidersHints(providersHints)
	admit := p.canAdmitPodResult(&hint)
	return hint, admit
}
