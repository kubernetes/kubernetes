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
	"math/rand"
	"testing"

	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager/bitmask"
)

func TestHintMergerMergePreferredFound(t *testing.T) {
	numaInfo := commonNUMAInfoTwoNodes()
	hints := [][]TopologyHint{
		{
			{NUMANodeAffinity: NewTestBitMask(0), Preferred: true},
		},
		{
			{NUMANodeAffinity: NewTestBitMask(1), Preferred: true},
			{NUMANodeAffinity: nil, Preferred: true},
			{NUMANodeAffinity: NewTestBitMask(0), Preferred: true},
		},
	}

	merger := NewHintMerger(numaInfo, hints, PolicyRestricted, PolicyOptions{})
	best, ok := merger.mergePreferred(klog.Background())
	if !ok {
		t.Fatalf("expected a preferred hint, got nil")
	}
	if !best.Preferred {
		t.Fatalf("expected Preferred=true, got %v", best)
	}
	if !best.NUMANodeAffinity.IsEqual(NewTestBitMask(0)) {
		t.Fatalf("expected affinity %v, got %v", NewTestBitMask(0), best.NUMANodeAffinity)
	}
}

func TestHintMergerMergePreferredNotFound(t *testing.T) {
	numaInfo := commonNUMAInfoTwoNodes()
	hints := [][]TopologyHint{
		{
			{NUMANodeAffinity: NewTestBitMask(0, 1), Preferred: true},
		},
		{
			{NUMANodeAffinity: NewTestBitMask(1), Preferred: true},
		},
	}

	merger := NewHintMerger(numaInfo, hints, PolicyRestricted, PolicyOptions{})
	_, ok := merger.mergePreferred(klog.Background())
	if ok {
		t.Fatalf("expected false, got ok=true")
	}
}

func TestHintMergerMergePreferredNoPreferredDimension(t *testing.T) {
	numaInfo := commonNUMAInfoTwoNodes()
	hints := [][]TopologyHint{
		{
			{NUMANodeAffinity: NewTestBitMask(0), Preferred: false},
		},
		{
			{NUMANodeAffinity: NewTestBitMask(0), Preferred: true},
		},
	}

	merger := NewHintMerger(numaInfo, hints, PolicyRestricted, PolicyOptions{})
	_, ok := merger.mergePreferred(klog.Background())
	if ok {
		t.Fatalf("expected false, got ok=true")
	}
}

func TestHintMergerMergePreferredEmptyMergedAffinity(t *testing.T) {
	numaInfo := commonNUMAInfoTwoNodes()
	hints := [][]TopologyHint{
		{
			{NUMANodeAffinity: NewTestBitMask(2), Preferred: true},
		},
		{
			{NUMANodeAffinity: nil, Preferred: true},
		},
	}

	merger := NewHintMerger(numaInfo, hints, PolicyRestricted, PolicyOptions{})
	_, ok := merger.mergePreferred(klog.Background())
	if ok {
		t.Fatalf("expected false, got ok=true")
	}
}

func TestHintMergerMergePreferredAllNilPreferred(t *testing.T) {
	numaInfo := commonNUMAInfoTwoNodes()
	hints := [][]TopologyHint{
		{
			{NUMANodeAffinity: nil, Preferred: true},
		},
		{
			{NUMANodeAffinity: nil, Preferred: true},
		},
	}

	merger := NewHintMerger(numaInfo, hints, PolicyRestricted, PolicyOptions{})
	best, ok := merger.mergePreferred(klog.Background())
	if !ok {
		t.Fatalf("expected preferred hint, got zero")
	}
	if !best.NUMANodeAffinity.IsEqual(NewTestBitMask(0, 1)) {
		t.Fatalf("expected affinity %v, got %v", NewTestBitMask(0, 1), best.NUMANodeAffinity)
	}
}

func TestHintMergerMergePreferredMixedNilAndMask(t *testing.T) {
	numaInfo := commonNUMAInfoTwoNodes()
	hints := [][]TopologyHint{
		{
			{NUMANodeAffinity: nil, Preferred: true},
		},
		{
			{NUMANodeAffinity: NewTestBitMask(0), Preferred: true},
		},
		{
			{NUMANodeAffinity: nil, Preferred: true},
		},
	}

	merger := NewHintMerger(numaInfo, hints, PolicyRestricted, PolicyOptions{})
	best, ok := merger.mergePreferred(klog.Background())
	if !ok {
		t.Fatalf("expected preferred hint, got zero")
	}
	if !best.NUMANodeAffinity.IsEqual(NewTestBitMask(0)) {
		t.Fatalf("expected affinity %v, got %v", NewTestBitMask(0), best.NUMANodeAffinity)
	}
}

func TestHintMergerMergePreferredComplexPreferredHints(t *testing.T) {
	numaInfo := commonNUMAInfoTwoNodes()
	hints := [][]TopologyHint{
		{
			{NUMANodeAffinity: NewTestBitMask(0), Preferred: true},
			{NUMANodeAffinity: NewTestBitMask(1), Preferred: true},
		},
		{
			{NUMANodeAffinity: NewTestBitMask(0), Preferred: true},
		},
		{
			{NUMANodeAffinity: nil, Preferred: true},
		},
	}

	merger := NewHintMerger(numaInfo, hints, PolicyRestricted, PolicyOptions{})
	best, ok := merger.mergePreferred(klog.Background())
	if !ok {
		t.Fatalf("expected preferred hint, got zero")
	}
	if !best.NUMANodeAffinity.IsEqual(NewTestBitMask(0)) {
		t.Fatalf("expected affinity %v, got %v", NewTestBitMask(0), best.NUMANodeAffinity)
	}
}

func TestHintMergerMergePreferredAggregatesScore(t *testing.T) {
	numaInfo := commonNUMAInfoTwoNodes()
	hints := [][]TopologyHint{
		{
			{NUMANodeAffinity: NewTestBitMask(0), Preferred: true, Score: 40},
		},
		{
			{NUMANodeAffinity: NewTestBitMask(0), Preferred: true, Score: 80},
		},
		{
			{NUMANodeAffinity: nil, Preferred: true, Score: 100},
		},
	}

	merger := NewHintMerger(numaInfo, hints, PolicyRestricted, PolicyOptions{})
	best, ok := merger.mergePreferred(klog.Background())
	if !ok {
		t.Fatalf("expected a preferred hint, got nil")
	}
	if best.Score != 60 {
		t.Fatalf("expected score 60, got %d", best.Score)
	}
}

// TestHintMergerMergePreferredEquivalentToMerge verifies that the optimized
// preferred-only search returns the same affinity, preference, and aggregated
// score as the exhaustive merge for both policies, with and without closest
// NUMA node preference, over deterministic inputs.
func TestHintMergerMergePreferredEquivalentToMerge(t *testing.T) {
	// Generate 500 reproducible cases per policy and option set. Each case has
	// one to five dimensions with one to six hints each, bounding exhaustive
	// Merge at 6^5 permutations. One third of hints have nil affinity; otherwise
	// each of eight NUMA nodes has a 25% chance of inclusion. Preferred is split
	// evenly and Score spans the inclusive range [0, 100].
	const (
		randomSeed           = 1   // Reproduce the same generated cases on every run.
		iterations           = 500 // Exercise varied tuples per policy without slowing the unit suite.
		maxDimensions        = 5   // Bound the number of providers or resources in a tuple.
		maxHintsPerDimension = 6   // Bound candidates per dimension so exhaustive Merge stays tractable.
		numaNodeCount        = 8   // Match commonNUMAInfoEightNodes.
		nilAffinityOneIn     = 3   // Generate a nil affinity for one third of hints.
		nodeSelectedOneIn    = 4   // Include each NUMA node with 25% probability.
		preferredOneIn       = 2   // Split preferred and non-preferred hints evenly.
		scoreValueCount      = 101 // Generate every integral score from 0 through 100.
	)

	numaInfo := commonNUMAInfoEightNodes()
	optionSets := []struct {
		name string
		opts PolicyOptions
	}{
		{name: "default"},
		{
			name: "prefer-closest-numa-nodes",
			opts: PolicyOptions{PreferClosestNUMA: true},
		},
	}
	for _, optionSet := range optionSets {
		for _, policyName := range []string{PolicyBestEffort, PolicyRestricted} {
			t.Run(policyName+"/"+optionSet.name, func(t *testing.T) {
				// Reset the generator so every policy and option set sees identical inputs.
				r := rand.New(rand.NewSource(randomSeed))

				for range iterations {
					numDimensions := r.Intn(maxDimensions) + 1
					hints := make([][]TopologyHint, 0, numDimensions)
					for range numDimensions {
						numHints := r.Intn(maxHintsPerDimension) + 1
						hintsForDim := make([]TopologyHint, 0, numHints)
						for range numHints {
							var affinity bitmask.BitMask
							if r.Intn(nilAffinityOneIn) != 0 {
								var bits []int
								for node := range numaNodeCount {
									if r.Intn(nodeSelectedOneIn) == 0 {
										bits = append(bits, node)
									}
								}
								affinity = NewTestBitMask(bits...)
							}
							hintsForDim = append(hintsForDim, TopologyHint{
								NUMANodeAffinity: affinity,
								Preferred:        r.Intn(preferredOneIn) == 0,
								Score:            int64(r.Intn(scoreValueCount)),
							})
						}
						hints = append(hints, hintsForDim)
					}

					merger := NewHintMerger(numaInfo, hints, policyName, optionSet.opts)
					preferred, ok := merger.mergePreferred(klog.Background())
					best := merger.Merge(klog.Background())
					if ok != best.Preferred {
						t.Fatalf("expected mergePreferred ok=%v to match Merge Preferred=%v; full=%v hints=%v", ok, best.Preferred, best, hints)
					}
					if !ok {
						continue
					}
					if !preferred.IsEqual(best) || preferred.Score != best.Score {
						t.Fatalf("expected mergePreferred() == Merge(); got preferred=%v full=%v hints=%v", preferred, best, hints)
					}
				}
			})
		}
	}
}

func TestHintMergerMergePreferredEmptyHints(t *testing.T) {
	numaInfo := commonNUMAInfoTwoNodes()
	hints := [][]TopologyHint{}

	merger := NewHintMerger(numaInfo, hints, PolicyBestEffort, PolicyOptions{})
	preferred, ok := merger.mergePreferred(klog.Background())
	if !ok {
		t.Fatalf("expected preferred hint, got zero")
	}
	best := merger.Merge(klog.Background())
	if !preferred.IsEqual(best) || preferred.Score != best.Score {
		t.Fatalf("expected mergePreferred() == Merge(); got preferred=%v full=%v", preferred, best)
	}
}
