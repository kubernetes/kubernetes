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
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
)

// Policy interface for Topology Manager Pod Admit Result
type Policy interface {
	// Returns Policy Name
	Name() string
	// Returns a merged TopologyHint based on input from hint providers
	// and a Pod Admit Handler Response based on hints and policy type
	Merge(providersHints []map[string][]TopologyHint) (TopologyHint, lifecycle.PodAdmitResult)
}

// Merge a TopologyHints permutation to a single hint by performing a bitwise-AND
// of their affinity masks. The hint shall be preferred if all hits in the permutation
// are preferred.
func mergePermutation(numaNodes []int, permutation []TopologyHint) TopologyHint {
	// Get the NUMANodeAffinity from each hint in the permutation and see if any
	// of them encode unpreferred allocations.
	preferred := true
	defaultAffinity, _ := bitmask.NewBitMask(numaNodes...)
	var numaAffinities []bitmask.BitMask
	for _, hint := range permutation {
		// Only consider hints that have an actual NUMANodeAffinity set.
		if hint.NUMANodeAffinity == nil {
			numaAffinities = append(numaAffinities, defaultAffinity)
		} else {
			numaAffinities = append(numaAffinities, hint.NUMANodeAffinity)
		}

		if !hint.Preferred {
			preferred = false
		}
	}

	// Merge the affinities using a bitwise-and operation.
	mergedAffinity := bitmask.And(defaultAffinity, numaAffinities...)
	// Build a mergedHint from the merged affinity mask, indicating if an
	// preferred allocation was used to generate the affinity mask or not.
	return TopologyHint{mergedAffinity, preferred}
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
