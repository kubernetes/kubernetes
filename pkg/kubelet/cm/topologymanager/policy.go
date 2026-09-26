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
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/klog/v2"
	kubefeatures "k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager/bitmask"
)

// Policy interface for Topology Manager Pod Admit Result
type Policy interface {
	// Returns Policy Name
	Name() string
	// Returns a merged TopologyHint based on input from hint providers
	// and a Pod Admit Handler Response based on hints and policy type
	Merge(logger klog.Logger, providersHints []map[string][]TopologyHint) (TopologyHint, bool)
}

// IsAlignmentGuaranteed return true if the given policy guarantees that either
// the compute resources will be allocated within a NUMA boundary, or the allocation will fail at all.
func IsAlignmentGuaranteed(p Policy) bool {
	// We are abusing the name, but atm this matches almost 1:1 the policy name
	// so we are not adding new fields for now.
	return p.Name() == PolicySingleNumaNode
}

// aggregateHintScores reduces the scores of the hints in a permutation to the
// single score carried by the merged hint, as the weighted average
//
//	sum(weight[r] * score[r]) / sum(weight[r])
//
// over the contributors which reported a score. Contributors with no affinity,
// or an unscored (zero) Score, take no part in it.
//
// resourceNames is parallel to the permutation: resourceNames[i] names the
// resource permutation[i] came from, which is what the weight lookup is keyed
// on. A resource the weights do not name carries defaultNUMAScoreWeight, so nil
// or empty weights reduce the formula to the equal-weight average sum/count.
// A resource named with a weight of 0 is dropped from the average entirely.
//
// Dividing by the total applicable weight keeps the result in the same [1,100]
// range as the individual scores, and makes the weights self-normalizing: only
// the ratios between them matter, so "cpu=30,memory=10" behaves exactly like
// "cpu=3,memory=1".
//
// The second return value is false when nothing contributed, that is when the
// total applicable weight is 0. The merged hint is then left unscored and hint
// selection falls back to the structural comparison alone.
func aggregateHintScores(permutation []TopologyHint, resourceNames []string, weights map[string]int) (int64, bool) {
	var weightedSum int64
	var totalWeight int64
	for i, hint := range permutation {
		if hint.NUMANodeAffinity == nil || hint.Score <= 0 {
			continue
		}

		weight := defaultNUMAScoreWeight
		// Guard the index rather than trusting the parallelism: a caller which
		// supplies fewer names than hints leaves the excess contributors
		// unnamed, and an unnamed contributor sits at the default weight. That
		// is the same treatment a named-but-unweighted resource gets, and it
		// keeps a mismatch out of the pod admission path.
		if i < len(resourceNames) {
			if w, ok := weights[resourceNames[i]]; ok {
				weight = w
			}
		}
		if weight == 0 {
			continue
		}

		weightedSum += hint.Score * int64(weight)
		totalWeight += int64(weight)
	}
	if totalWeight == 0 {
		return 0, false
	}
	return weightedSum / totalWeight, true
}

// Merge a TopologyHints permutation to a single hint by performing a bitwise-AND
// of their affinity masks. The hint shall be preferred if all hits in the permutation
// are preferred.
//
// resourceNames and weights are only consulted to aggregate the scores; see
// aggregateHintScores for the correspondence they are required to have with the
// permutation.
func mergePermutation(logger klog.Logger, defaultAffinity bitmask.BitMask, permutation []TopologyHint, resourceNames []string, weights map[string]int) TopologyHint {
	// Get the NUMANodeAffinity from each hint in the permutation and see if any
	// of them encode unpreferred allocations.
	preferred := true
	var numaAffinities []bitmask.BitMask
	for _, hint := range permutation {
		// Only consider hints that have an actual NUMANodeAffinity set.
		if hint.NUMANodeAffinity != nil {
			numaAffinities = append(numaAffinities, hint.NUMANodeAffinity)
			// Only mark preferred if all affinities are equal.
			if !hint.NUMANodeAffinity.IsEqual(numaAffinities[0]) {
				preferred = false
			}
		}
		// Only mark preferred if all affinities are preferred.
		if !hint.Preferred {
			preferred = false
		}
	}

	// Merge the affinities using a bitwise-and operation.
	mergedAffinity := bitmask.And(defaultAffinity, numaAffinities...)

	score, hasScores := aggregateHintScores(permutation, resourceNames, weights)
	if hasScores {
		logger.V(4).Info("Merged hint includes aggregated score", "score", score)
	}

	// Build a mergedHint from the merged affinity mask, setting preferred as
	// appropriate based on the logic above.
	return TopologyHint{NUMANodeAffinity: mergedAffinity, Preferred: preferred, Score: score}
}

// filterProvidersHints flattens the per-provider hint maps into a list of hint
// lists, one per resource, and returns the resource names alongside it.
//
// The two returned slices are parallel: allProviderHints[i] holds the hints for
// resourceNames[i]. A provider which expressed no preference at all has no
// resource to name, so it contributes an empty name. Callers rely on this
// correspondence to attribute a hint back to the resource it came from.
func filterProvidersHints(logger klog.Logger, providersHints []map[string][]TopologyHint) ([][]TopologyHint, []string) {
	// Loop through all hint providers and save an accumulated list of the
	// hints returned by each hint provider. If no hints are provided, assume
	// that provider has no preference for topology-aware allocation.
	var allProviderHints [][]TopologyHint
	var resourceNames []string
	for _, hints := range providersHints {
		// If hints is nil, insert a single, preferred any-numa hint into allProviderHints.
		if len(hints) == 0 {
			logger.Info("Hint Provider has no preference for NUMA affinity with any resource")
			allProviderHints = append(allProviderHints, []TopologyHint{{NUMANodeAffinity: nil, Preferred: true}})
			resourceNames = append(resourceNames, "")
			continue
		}

		// Otherwise, accumulate the hints for each resource type into allProviderHints.
		for resource := range hints {
			if hints[resource] == nil {
				logger.Info("Hint Provider has no preference for NUMA affinity with resource", "resource", resource)
				allProviderHints = append(allProviderHints, []TopologyHint{{NUMANodeAffinity: nil, Preferred: true}})
				resourceNames = append(resourceNames, resource)
				continue
			}

			if len(hints[resource]) == 0 {
				logger.Info("Hint Provider has no possible NUMA affinities for resource", "resource", resource)
				allProviderHints = append(allProviderHints, []TopologyHint{{NUMANodeAffinity: nil, Preferred: false}})
				resourceNames = append(resourceNames, resource)
				continue
			}

			allProviderHints = append(allProviderHints, hints[resource])
			resourceNames = append(resourceNames, resource)
		}
	}
	return allProviderHints, resourceNames
}

func narrowestHint(hints []TopologyHint) *TopologyHint {
	if len(hints) == 0 {
		return nil
	}
	var narrowestHint *TopologyHint
	for i := range hints {
		if hints[i].NUMANodeAffinity == nil {
			continue
		}
		if narrowestHint == nil {
			narrowestHint = &hints[i]
		}
		if hints[i].NUMANodeAffinity.IsNarrowerThan(narrowestHint.NUMANodeAffinity) {
			narrowestHint = &hints[i]
		}
	}
	return narrowestHint
}

func maxOfMinAffinityCounts(filteredHints [][]TopologyHint) int {
	maxOfMinCount := 0
	for _, resourceHints := range filteredHints {
		narrowestHint := narrowestHint(resourceHints)
		if narrowestHint == nil {
			continue
		}
		if narrowestHint.NUMANodeAffinity.Count() > maxOfMinCount {
			maxOfMinCount = narrowestHint.NUMANodeAffinity.Count()
		}
	}
	return maxOfMinCount
}

type HintMerger struct {
	NUMAInfo *NUMAInfo
	Hints    [][]TopologyHint
	// ResourceNames is parallel to Hints: ResourceNames[i] names the resource
	// whose hints are in Hints[i]. Since a permutation takes its i-th element
	// from Hints[i], this also names the resource each hint in a permutation
	// belongs to.
	ResourceNames []string
	// ScoreWeights maps a resource name to the weight its score carries when
	// the scores of a permutation are aggregated. It is nil unless the user
	// asked for weights; every resource then sits at the default weight and
	// the aggregation is the equal-weight average.
	ScoreWeights map[string]int
	// Set bestNonPreferredAffinityCount to help decide which affinity mask is
	// preferred amongst all non-preferred hints. We calculate this value as
	// the maximum of the minimum affinity counts supplied for any given hint
	// provider. In other words, prefer a hint that has an affinity mask that
	// includes all of the NUMA nodes from the provider that requires the most
	// NUMA nodes to satisfy its allocation.
	BestNonPreferredAffinityCount int
	CompareNUMAAffinityMasks      func(candidate *TopologyHint, current *TopologyHint) (best *TopologyHint)
}

// compareHintScores returns the hint preferred by the given NUMA allocation
// strategy, or nil if the strategy expresses no preference between the two
// hints. Callers are expected to fall back to the structural comparison in
// that case.
func compareHintScores(strategy string, current, candidate *TopologyHint) *TopologyHint {
	// A Score of 0 means unscored: no provider reported a utilization signal
	// for the hint. Scored hints sit in [1,100], so reading 0 as a value would
	// rate an unscored hint as a completely empty NUMA node and hand it every
	// least-allocated comparison. Leave those to the structural comparison.
	if current.Score == 0 || candidate.Score == 0 {
		return nil
	}

	if current.Score == candidate.Score {
		return nil
	}

	switch strategy {
	case NUMAAllocationStrategyMostAllocated:
		// Packing: the NUMA nodes which are already the busiest score the
		// highest, so the higher score wins.
		if candidate.Score > current.Score {
			return candidate
		}
		return current
	case NUMAAllocationStrategyLeastAllocated:
		// Spreading: the NUMA nodes which are the emptiest score the lowest,
		// so the lower score wins.
		if candidate.Score < current.Score {
			return candidate
		}
		return current
	}

	return nil
}

func NewHintMerger(numaInfo *NUMAInfo, hints [][]TopologyHint, resourceNames []string, policyName string, opts PolicyOptions) HintMerger {
	preferClosest := (policyName != PolicySingleNumaNode) && opts.PreferClosestNUMA

	// The allocation strategy and the score weights are alpha-level policy
	// options. NewPolicyOptions already refuses them while the alpha options are
	// disabled; check the gate here as well so PolicyOptions values built by
	// other means cannot turn the feature on behind the gate's back.
	allocationStrategy := NUMAAllocationStrategyNone
	var scoreWeights map[string]int
	if utilfeature.DefaultFeatureGate.Enabled(kubefeatures.TopologyManagerPolicyAlphaOptions) {
		if opts.NUMAAllocationStrategy != "" {
			allocationStrategy = opts.NUMAAllocationStrategy
		}
		scoreWeights = opts.NUMAScoreWeights
	}

	compareNumaAffinityMasks := func(current, candidate *TopologyHint) *TopologyHint {
		// If current and candidate bitmasks are the same, prefer current hint.
		if candidate.NUMANodeAffinity.IsEqual(current.NUMANodeAffinity) {
			return current
		}

		// The structural comparison below comes first: the allocation strategy
		// only gets to decide between masks which Narrowest, respectively
		// Closest, considers equally good and would otherwise separate with an
		// arbitrary fallback.
		if allocationStrategy != NUMAAllocationStrategyNone &&
			numaInfo.equallyFit(current.NUMANodeAffinity, candidate.NUMANodeAffinity, preferClosest) {
			if best := compareHintScores(allocationStrategy, current, candidate); best != nil {
				return best
			}
		}

		// Otherwise compare the hints, based on the policy options provided
		var best bitmask.BitMask
		if preferClosest {
			best = numaInfo.Closest(current.NUMANodeAffinity, candidate.NUMANodeAffinity)
		} else {
			best = numaInfo.Narrowest(current.NUMANodeAffinity, candidate.NUMANodeAffinity)
		}
		if best.IsEqual(current.NUMANodeAffinity) {
			return current
		}
		return candidate
	}

	merger := HintMerger{
		NUMAInfo:                      numaInfo,
		Hints:                         hints,
		ResourceNames:                 resourceNames,
		ScoreWeights:                  scoreWeights,
		BestNonPreferredAffinityCount: maxOfMinAffinityCounts(hints),
		CompareNUMAAffinityMasks:      compareNumaAffinityMasks,
	}

	return merger
}

func (m HintMerger) compare(current *TopologyHint, candidate *TopologyHint) *TopologyHint {
	// Only consider candidates that result in a NUMANodeAffinity > 0 to
	// replace the current bestHint.
	if candidate.NUMANodeAffinity.Count() == 0 {
		return current
	}

	// If no current bestHint is set, return the candidate as the bestHint.
	if current == nil {
		return candidate
	}

	// If the current bestHint is non-preferred and the candidate hint is
	// preferred, always choose the preferred hint over the non-preferred one.
	if !current.Preferred && candidate.Preferred {
		return candidate
	}

	// If the current bestHint is preferred and the candidate hint is
	// non-preferred, never update the bestHint, regardless of how
	// the candidate hint's affinity mask compares to the current
	// hint's affinity mask.
	if current.Preferred && !candidate.Preferred {
		return current
	}

	// If the current bestHint and the candidate hint are both preferred,
	// then only consider fitter NUMANodeAffinity
	if current.Preferred && candidate.Preferred {
		return m.CompareNUMAAffinityMasks(current, candidate)

	}

	// The only case left is if the current best bestHint and the candidate
	// hint are both non-preferred. In this case, try and find a hint whose
	// affinity count is as close to (but not higher than) the
	// bestNonPreferredAffinityCount as possible. To do this we need to
	// consider the following cases and react accordingly:
	//
	//   1. current.NUMANodeAffinity.Count() >  bestNonPreferredAffinityCount
	//   2. current.NUMANodeAffinity.Count() == bestNonPreferredAffinityCount
	//   3. current.NUMANodeAffinity.Count() <  bestNonPreferredAffinityCount
	//
	// For case (1), the current bestHint is larger than the
	// bestNonPreferredAffinityCount, so updating to fitter mergeHint
	// is preferred over staying where we are.
	//
	// For case (2), the current bestHint is equal to the
	// bestNonPreferredAffinityCount, so we would like to stick with what
	// we have *unless* the candidate hint is also equal to
	// bestNonPreferredAffinityCount and it is fitter.
	//
	// For case (3), the current bestHint is less than
	// bestNonPreferredAffinityCount, so we would like to creep back up to
	// bestNonPreferredAffinityCount as close as we can. There are three
	// cases to consider here:
	//
	//   3a. candidate.NUMANodeAffinity.Count() >  bestNonPreferredAffinityCount
	//   3b. candidate.NUMANodeAffinity.Count() == bestNonPreferredAffinityCount
	//   3c. candidate.NUMANodeAffinity.Count() <  bestNonPreferredAffinityCount
	//
	// For case (3a), we just want to stick with the current bestHint
	// because choosing a new hint that is greater than
	// bestNonPreferredAffinityCount would be counter-productive.
	//
	// For case (3b), we want to immediately update bestHint to the
	// candidate hint, making it now equal to bestNonPreferredAffinityCount.
	//
	// For case (3c), we know that *both* the current bestHint and the
	// candidate hint are less than bestNonPreferredAffinityCount, so we
	// want to choose one that brings us back up as close to
	// bestNonPreferredAffinityCount as possible. There are three cases to
	// consider here:
	//
	//   3ca. candidate.NUMANodeAffinity.Count() >  current.NUMANodeAffinity.Count()
	//   3cb. candidate.NUMANodeAffinity.Count() <  current.NUMANodeAffinity.Count()
	//   3cc. candidate.NUMANodeAffinity.Count() == current.NUMANodeAffinity.Count()
	//
	// For case (3ca), we want to immediately update bestHint to the
	// candidate hint because that will bring us closer to the (higher)
	// value of bestNonPreferredAffinityCount.
	//
	// For case (3cb), we want to stick with the current bestHint because
	// choosing the candidate hint would strictly move us further away from
	// the bestNonPreferredAffinityCount.
	//
	// Finally, for case (3cc), we know that the current bestHint and the
	// candidate hint are equal, so we simply choose the fitter of the 2.

	// Case 1
	if current.NUMANodeAffinity.Count() > m.BestNonPreferredAffinityCount {
		return m.CompareNUMAAffinityMasks(current, candidate)
	}
	// Case 2
	if current.NUMANodeAffinity.Count() == m.BestNonPreferredAffinityCount {
		if candidate.NUMANodeAffinity.Count() != m.BestNonPreferredAffinityCount {
			return current
		}
		return m.CompareNUMAAffinityMasks(current, candidate)
	}
	// Case 3a
	if candidate.NUMANodeAffinity.Count() > m.BestNonPreferredAffinityCount {
		return current
	}
	// Case 3b
	if candidate.NUMANodeAffinity.Count() == m.BestNonPreferredAffinityCount {
		return candidate
	}

	// Case 3ca
	if candidate.NUMANodeAffinity.Count() > current.NUMANodeAffinity.Count() {
		return candidate
	}
	// Case 3cb
	if candidate.NUMANodeAffinity.Count() < current.NUMANodeAffinity.Count() {
		return current
	}

	// Case 3cc
	return m.CompareNUMAAffinityMasks(current, candidate)

}

func (m HintMerger) Merge(logger klog.Logger) TopologyHint {
	defaultAffinity := m.NUMAInfo.DefaultAffinityMask()

	var bestHint *TopologyHint
	iterateAllProviderTopologyHints(m.Hints, func(permutation []TopologyHint) {
		// Get the NUMANodeAffinity from each hint in the permutation and see if any
		// of them encode unpreferred allocations.
		//
		// permutation[i] is drawn from m.Hints[i], which holds the hints for
		// m.ResourceNames[i], so the permutation and the resource names line up
		// and the aggregation can weight each hint by the resource it came from.
		mergedHint := mergePermutation(logger, defaultAffinity, permutation, m.ResourceNames, m.ScoreWeights)

		// Compare the current bestHint with the candidate mergedHint and
		// update bestHint if appropriate.
		bestHint = m.compare(bestHint, &mergedHint)
	})

	if bestHint == nil {
		bestHint = &TopologyHint{NUMANodeAffinity: defaultAffinity, Preferred: false}
	}

	return *bestHint
}

// Iterate over all permutations of hints in 'allProviderHints [][]TopologyHint'.
//
// This procedure is implemented as a recursive function over the set of hints
// in 'allproviderHints[i]'. It applies the function 'callback' to each
// permutation as it is found. It is the equivalent of:
//
// for i := 0; i < len(providerHints[0]); i++
//
//	for j := 0; j < len(providerHints[1]); j++
//	    for k := 0; k < len(providerHints[2]); k++
//	        ...
//	        for z := 0; z < len(providerHints[-1]); z++
//	            permutation := []TopologyHint{
//	                providerHints[0][i],
//	                providerHints[1][j],
//	                providerHints[2][k],
//	                ...
//	                providerHints[-1][z]
//	            }
//	            callback(permutation)
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
		// permutation of this hint with all hints from providers 'i++'.
		for j := range allProviderHints[i] {
			iterate(i+1, append(accum, allProviderHints[i][j]))
		}
	}
	iterate(0, []TopologyHint{})
}
