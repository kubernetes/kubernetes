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
			if hint.NUMANodeAffinity != nil && hint.NUMANodeAffinity.Count() == 1 && hint.Preferred == true {
				filtered = append(filtered, hint)
			}
		}

		filteredResourcesHints = append(filteredResourcesHints, filtered)
	}
	return filteredResourcesHints
}

func (p *singleNumaNodePolicy) mergeProvidersHints(providersHints []map[string][]TopologyHint) TopologyHint {
	// Loop through all provider hints and save an accumulated list of the
	// hints returned by each hint provider. If no hints are provided, assume
	// that provider has no preference for topology-aware allocation.
	var allResourcesHints [][]TopologyHint
	for _, hints := range providersHints {
		if len(hints) == 0 {
			klog.Infof("[topologymanager] Hint Provider has no preference for NUMA affinity with any resource, " +
				"skipping.")
			continue
		}

		// Otherwise, accumulate the hints for each resource type into allProviderHints.
		for resource := range hints {
			if hints[resource] == nil {
				klog.Infof("[topologymanager] Hint Provider has no preference for NUMA affinity with resource "+
					"'%s', skipping.", resource)
				continue
			}

			if len(hints[resource]) == 0 {
				klog.Infof("[topologymanager] Hint Provider has no possible NUMA affinities for resource '%s'",
					resource)
				// return defaultHint which will fail pod admission
				return TopologyHint{}
			}
			klog.Infof("[topologymanager] TopologyHints for resource '%v': %v", resource, hints[resource])
			allResourcesHints = append(allResourcesHints, hints[resource])
		}
	}
	// In case allProviderHints length is zero it means that we have no
	// preference for NUMA affinity. In that case update default hint preferred
	// to true to allow scheduling.
	if len(allResourcesHints) == 0 {
		klog.Infof("[topologymanager] No preference for NUMA affinity from all providers")
		return TopologyHint{nil, true}
	}

	allResourcesHints = p.filterHints(allResourcesHints)
	// If no hints, or there is a resource with empty hints after filtering, then policy
	// cannot be satisfied
	if len(allResourcesHints) == 0 {
		klog.Infof("[topologymanager] No hints that align to a single NUMA node.")
		return TopologyHint{}
	}
	for _, hints := range allResourcesHints {
		if len(hints) == 0 {
			klog.Infof("[topologymanager] No hints that align to a single NUMA node for resource.")
			return TopologyHint{}
		}
	}

	// Set the bestHint to return from this function as {nil false}.
	// This will only be returned if no better hint can be found when
	// merging hints from each hint provider.
	bestHint := TopologyHint{}
	iterateAllProviderTopologyHints(allResourcesHints, func(permutation []TopologyHint) {
		mergedHint := mergePermutation(p.numaNodes, permutation)
		// Only consider mergedHints that result in a NUMANodeAffinity == 1 to
		// replace the current defaultHint.
		if mergedHint.NUMANodeAffinity.Count() != 1 {
			return
		}

		// If the current bestHint NUMANodeAffinity is nil, update bestHint
		// to the current mergedHint.
		if bestHint.NUMANodeAffinity == nil {
			bestHint = mergedHint
			return
		}

		// Only consider mergedHints that have a narrower NUMANodeAffinity
		// than the NUMANodeAffinity in the current bestHint.
		if !mergedHint.NUMANodeAffinity.IsNarrowerThan(bestHint.NUMANodeAffinity) {
			return
		}
		// In all other cases, update bestHint to the current mergedHint
		bestHint = mergedHint
	})
	return bestHint
}

func (p *singleNumaNodePolicy) Merge(providersHints []map[string][]TopologyHint) (TopologyHint, lifecycle.PodAdmitResult) {
	hint := p.mergeProvidersHints(providersHints)
	admit := p.canAdmitPodResult(&hint)
	return hint, admit
}
