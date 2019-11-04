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

type bestEffortPolicy struct {
	//List of NUMA Nodes available on the underlying machine
	numaNodes []int
}

var _ Policy = &bestEffortPolicy{}

// PolicyBestEffort policy name.
const PolicyBestEffort string = "best-effort"

// NewBestEffortPolicy returns best-effort policy.
func NewBestEffortPolicy(numaNodes []int) Policy {
	return &bestEffortPolicy{numaNodes: numaNodes}
}

func (p *bestEffortPolicy) Name() string {
	return PolicyBestEffort
}

func (p *bestEffortPolicy) canAdmitPodResult(hint *TopologyHint) lifecycle.PodAdmitResult {
	return lifecycle.PodAdmitResult{
		Admit: true,
	}
}

func (p *bestEffortPolicy) Merge(providersHints []map[string][]TopologyHint) (TopologyHint, lifecycle.PodAdmitResult) {
	hint := mergeProvidersHints(p, p.numaNodes, providersHints)
	admit := p.canAdmitPodResult(&hint)
	return hint, admit
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
func iterateAllProviderTopologyHints(allProviderHints [][]TopologyHint, callback func([]TopologyHint)) {
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

// Merge the hints from all hint providers to find the best one.
func mergeProvidersHints(policy Policy, numaNodes []int, providersHints []map[string][]TopologyHint) TopologyHint {
	// Set the default affinity as an any-numa affinity containing the list
	// of NUMA Nodes available on this machine.
	defaultAffinity, _ := bitmask.NewBitMask(numaNodes...)

	// Loop through all hint providers and save an accumulated list of the
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

	// Iterate over all permutations of hints in 'allProviderHints'. Merge the
	// hints in each permutation by taking the bitwise-and of their affinity masks.
	// Return the hint with the narrowest NUMANodeAffinity of all merged
	// permutations that have at least one NUMA ID set. If no merged mask can be
	// found that has at least one NUMA ID set, return the 'defaultAffinity'.
	bestHint := TopologyHint{defaultAffinity, false}
	iterateAllProviderTopologyHints(allProviderHints, func(permutation []TopologyHint) {
		// Get the NUMANodeAffinity from each hint in the permutation and see if any
		// of them encode unpreferred allocations.
		preferred := true
		var numaAffinities []bitmask.BitMask
		for _, hint := range permutation {
			if hint.NUMANodeAffinity == nil {
				numaAffinities = append(numaAffinities, defaultAffinity)
			} else {
				numaAffinities = append(numaAffinities, hint.NUMANodeAffinity)
			}

			if !hint.Preferred {
				preferred = false
			}

			// Special case PolicySingleNumaNode to only prefer hints where
			// all providers have a single NUMA affinity set.
			if policy != nil && policy.Name() == PolicySingleNumaNode && hint.NUMANodeAffinity != nil && hint.NUMANodeAffinity.Count() > 1 {
				preferred = false
			}
		}

		// Merge the affinities using a bitwise-and operation.
		mergedAffinity, _ := bitmask.NewBitMask(numaNodes...)
		mergedAffinity.And(numaAffinities...)

		// Build a mergedHintfrom the merged affinity mask, indicating if an
		// preferred allocation was used to generate the affinity mask or not.
		mergedHint := TopologyHint{mergedAffinity, preferred}

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

	return bestHint
}
