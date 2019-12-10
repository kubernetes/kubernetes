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

// Iterate over all permutations of hints in 'allProviderHints [][]TopologyHint'.
//
// This procedure is implemented as a recursive function over the set of hints
// in 'allproviderHints[i]'. It applies the function 'callback' to each
// permutation as it is found. It is the equivalent of:
//
// for i := 0; i < len(providerHints[0]); i++
//     for j := 0; j < len(providerHints[1]); j++
//         for k := 0; k < len(providerHints[2]); k++
//             ...
//             for z := 0; z < len(providerHints[-1]); z++
//                 permutation := []TopologyHint{
//                     providerHints[0][i],
//                     providerHints[1][j],
//                     providerHints[2][k],
//                     ...
//                     providerHints[-1][z]
//                 }
//                 callback(permutation)
func (p *singleNumaNodePolicy) iterateAllProviderTopologyHints(allProviderHints [][]TopologyHint, callback func([]TopologyHint)) {
	// Internal helper function to accumulate the permutation before calling the callback.
	var iterate func(i int, accum []TopologyHint)
	iterate = func(i int, accum []TopologyHint) {
		// Base case: we have looped through all providers and have a full permutation.
		if i == len(allProviderHints) {
			callback(accum)
			return
		}

		// Loop through all hints for provider 'i', and recurse to build the
		// the permutation of this hint with all hints from providers 'i++'.
		for j := range allProviderHints[i] {
			iterate(i+1, append(accum, allProviderHints[i][j]))
		}
	}
	iterate(0, []TopologyHint{})
}

// Merge a TopologyHints permutation to a single hint by performing a bitwise-AND
// of their affinity masks. At this point, all hints have single numa node affinity
// and preferred-true.
func (p *singleNumaNodePolicy) mergePermutation(permutation []TopologyHint) TopologyHint {
	preferred := true
	var numaAffinities []bitmask.BitMask
	for _, hint := range permutation {
		numaAffinities = append(numaAffinities, hint.NUMANodeAffinity)
	}

	// Merge the affinities using a bitwise-and operation.
	mergedAffinity, _ := bitmask.NewBitMask(p.numaNodes...)
	mergedAffinity.And(numaAffinities...)
	// Build a mergedHint from the merged affinity mask.
	return TopologyHint{mergedAffinity, preferred}
}

// Return hints that have valid bitmasks with exactly one bit set. Also return bool
// which indicates whether allResourceHints only consists of {nil true} hints.
func (p *singleNumaNodePolicy) filterHints(allResourcesHints [][]TopologyHint) ([][]TopologyHint, bool) {
	var filteredResourcesHints [][]TopologyHint
	var noAffinityPreferredHints int
	var totalHints int
	if len(allResourcesHints) > 0 {
		for _, oneResourceHints := range allResourcesHints {
			var filtered []TopologyHint
			if len(oneResourceHints) > 0 {
				for _, hint := range oneResourceHints {
					totalHints++
					if hint.NUMANodeAffinity != nil && hint.NUMANodeAffinity.Count() == 1 && hint.Preferred == true {
						filtered = append(filtered, hint)
					}
					if hint.NUMANodeAffinity == nil && hint.Preferred == true {
						noAffinityPreferredHints++
					}
				}
			}
			filteredResourcesHints = append(filteredResourcesHints, filtered)
		}
	}
	// Check if all resource hints only consist of nil-affinity/preferred hint: {nil true}.
	if noAffinityPreferredHints == totalHints {
		return filteredResourcesHints, true
	}
	return filteredResourcesHints, false
}

func (p *singleNumaNodePolicy) mergeProvidersHints(providersHints []map[string][]TopologyHint) TopologyHint {
	// Set the default hint to return from this function as an any-NUMANode
	// affinity with an unpreferred allocation. This will only be returned if
	// no better hint can be found when merging hints from each hint provider.
	defaultAffinity, _ := bitmask.NewBitMask(p.numaNodes...)
	defaultHint := TopologyHint{defaultAffinity, false}

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
				return defaultHint
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
		defaultHint.Preferred = true
		return defaultHint
	}

	var noAffinityPreferredOnly bool
	allResourcesHints, noAffinityPreferredOnly = p.filterHints(allResourcesHints)
	// If no hints, then policy cannot be satisfied
	if len(allResourcesHints) == 0 {
		klog.Infof("[topologymanager] No hints that align to a single NUMA node.")
		return defaultHint
	}
	// If there is a resource with empty hints after filtering, then policy cannot be satisfied.
	// In the event that the only hints that exist are {nil true} update default hint preferred
	// to allow scheduling.
	for _, hints := range allResourcesHints {
		if len(hints) == 0 {
			klog.Infof("[topologymanager] No hints that align to a single NUMA node for resource.")
			if !noAffinityPreferredOnly {
				return defaultHint
			} else if noAffinityPreferredOnly {
				defaultHint.Preferred = true
				return defaultHint
			}
		}
	}

	p.iterateAllProviderTopologyHints(allResourcesHints, func(permutation []TopologyHint) {
		mergedHint := p.mergePermutation(permutation)
		// Only consider mergedHints that result in a NUMANodeAffinity == 1 to
		// replace the current defaultHint.
		if mergedHint.NUMANodeAffinity.Count() != 1 {
			return
		}

		// If the current defaultHint is the same size as the new mergedHint,
		// do not update defaultHint
		if mergedHint.NUMANodeAffinity.Count() == defaultHint.NUMANodeAffinity.Count() {
			return
		}
		// In all other cases, update defaultHint to the current mergedHint
		defaultHint = mergedHint
	})
	return defaultHint
}

func (p *singleNumaNodePolicy) Merge(providersHints []map[string][]TopologyHint) (TopologyHint, lifecycle.PodAdmitResult) {
	hint := p.mergeProvidersHints(providersHints)
	admit := p.canAdmitPodResult(&hint)
	return hint, admit
}
