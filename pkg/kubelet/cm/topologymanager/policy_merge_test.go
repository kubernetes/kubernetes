/*
Copyright The Kubernetes Authors.

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
	"math/rand"
	"testing"

	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager/bitmask"
)

// Keep the exhaustive traversal as an independent oracle for pruning changes.
func mergeExhaustive(merger HintMerger) TopologyHint {
	logger := klog.Background()
	defaultAffinity := merger.NUMAInfo.DefaultAffinityMask()
	var best *TopologyHint
	iterateAllProviderTopologyHints(merger.Hints, func(permutation []TopologyHint) {
		candidate := mergePermutation(logger, defaultAffinity, permutation)
		best = merger.compare(best, &candidate)
	})
	if best == nil {
		return TopologyHint{NUMANodeAffinity: defaultAffinity}
	}
	return *best
}

func TestHintMergerMergeEquivalentPrefixes(t *testing.T) {
	for _, tc := range []struct {
		name  string
		hints [][]TopologyHint
		want  TopologyHint
	}{
		{
			name: "nil and default affinity have different preferred continuations",
			hints: [][]TopologyHint{
				{
					{NUMANodeAffinity: NewTestBitMask(0, 1), Preferred: true, Score: 10},
					{Preferred: true, Score: 90},
				},
				{{NUMANodeAffinity: NewTestBitMask(0), Preferred: true, Score: 20}},
			},
			want: TopologyHint{NUMANodeAffinity: NewTestBitMask(0), Preferred: true, Score: 20},
		},
		{
			name: "first affinity is compared before clipping to NUMA nodes",
			hints: [][]TopologyHint{
				{
					{NUMANodeAffinity: NewTestBitMask(0, 2), Preferred: true, Score: 10},
					{NUMANodeAffinity: NewTestBitMask(0), Preferred: true, Score: 90},
				},
				{{NUMANodeAffinity: NewTestBitMask(0), Preferred: true, Score: 20}},
			},
			want: TopologyHint{NUMANodeAffinity: NewTestBitMask(0), Preferred: true, Score: 55},
		},
		{
			name: "equal non-preferred prefixes retain the first score",
			hints: [][]TopologyHint{
				{
					{NUMANodeAffinity: NewTestBitMask(0), Score: 10},
					{NUMANodeAffinity: NewTestBitMask(0), Score: 90},
				},
				{
					{NUMANodeAffinity: NewTestBitMask(0), Preferred: true, Score: 21},
					{NUMANodeAffinity: NewTestBitMask(0), Preferred: true, Score: 81},
				},
				{{Preferred: true, Score: 100}},
			},
			want: TopologyHint{NUMANodeAffinity: NewTestBitMask(0), Score: 15},
		},
		{
			name:  "empty dimension",
			hints: [][]TopologyHint{{{NUMANodeAffinity: NewTestBitMask(0), Preferred: true}}, {}},
			want:  TopologyHint{NUMANodeAffinity: NewTestBitMask(0, 1)},
		},
		{
			name: "empty intersection",
			hints: [][]TopologyHint{
				{{NUMANodeAffinity: NewTestBitMask(0), Preferred: true}},
				{{NUMANodeAffinity: NewTestBitMask(1), Preferred: true}},
			},
			want: TopologyHint{NUMANodeAffinity: NewTestBitMask(0, 1)},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			merger := NewHintMerger(commonNUMAInfoTwoNodes(), tc.hints, PolicyBestEffort, PolicyOptions{})
			got := merger.Merge(klog.Background())
			if !got.IsEqual(tc.want) || got.Score != tc.want.Score {
				t.Fatalf("expected %v, got %v", tc.want, got)
			}
		})
	}
}

func TestHintMergerMergeEquivalentToExhaustive(t *testing.T) {
	for _, policy := range []string{PolicyBestEffort, PolicyRestricted, PolicySingleNumaNode} {
		for _, preferClosest := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/closest=%t", policy, preferClosest), func(t *testing.T) {
				r := rand.New(rand.NewSource(137700))
				// Four dimensions of up to five hints keep exhaustive traversal
				// bounded at 625 permutations. Include empty dimensions and masks,
				// nil affinities, duplicate prefixes and nodes outside the machine.
				for iteration := range 500 {
					hints := make([][]TopologyHint, r.Intn(5))
					for dim := range hints {
						for range r.Intn(6) {
							var affinity bitmask.BitMask
							if r.Intn(3) != 0 {
								var nodes []int
								for node := range 9 {
									if r.Intn(2) == 0 {
										nodes = append(nodes, node)
									}
								}
								affinity = NewTestBitMask(nodes...)
							}
							hints[dim] = append(hints[dim], TopologyHint{
								NUMANodeAffinity: affinity, Preferred: r.Intn(2) == 0, Score: int64(r.Intn(101)),
							})
						}
					}
					merger := NewHintMerger(commonNUMAInfoEightNodes(), hints, policy, PolicyOptions{PreferClosestNUMA: preferClosest})
					got := merger.Merge(klog.Background())
					want := mergeExhaustive(merger)
					if !got.IsEqual(want) || got.Score != want.Score {
						t.Fatalf("iteration %d: expected %v, got %v; hints=%v", iteration, want, got, hints)
					}
				}
			})
		}
	}
}

func TestHintMergerMergeRepeatedAffinities(t *testing.T) {
	numaInfo := &NUMAInfo{Nodes: []int{0, 1, 2, 3}}
	var providerHints []TopologyHint
	bitmask.IterateBitMasks(numaInfo.Nodes, func(mask bitmask.BitMask) {
		providerHints = append(providerHints, TopologyHint{NUMANodeAffinity: mask})
	})
	// Every provider offers all non-empty masks, but no preferred hint. Many
	// permutations produce the same merged affinity and cannot change the result.
	hints := [][]TopologyHint{providerHints, providerHints, providerHints, providerHints}
	merger := NewHintMerger(numaInfo, hints, PolicyBestEffort, PolicyOptions{})
	compare := merger.CompareNUMAAffinityMasks
	comparisons := 0
	merger.CompareNUMAAffinityMasks = func(current, candidate *TopologyHint) *TopologyHint {
		comparisons++
		return compare(current, candidate)
	}
	got := merger.Merge(klog.Background())
	want := TopologyHint{NUMANodeAffinity: NewTestBitMask(0)}
	if !got.IsEqual(want) {
		t.Fatalf("expected %v, got %v", want, got)
	}
	if comparisons > len(providerHints) {
		t.Errorf("compared %d merged affinities, expected at most %d distinct masks", comparisons, len(providerHints))
	}
}

func BenchmarkHintMergerNoPreferred(b *testing.B) {
	for _, nodes := range [][]int{{0, 1, 2, 3}, {0, 1, 2, 3, 4, 5, 6, 7}} {
		name := "four-nodes"
		if len(nodes) == 8 {
			name = "eight-nodes"
		}
		b.Run(name, func(b *testing.B) {
			var providerHints []TopologyHint
			bitmask.IterateBitMasks(nodes, func(mask bitmask.BitMask) {
				providerHints = append(providerHints, TopologyHint{NUMANodeAffinity: mask})
			})
			merger := NewHintMerger(&NUMAInfo{Nodes: nodes}, [][]TopologyHint{
				providerHints, providerHints, providerHints, providerHints,
			}, PolicyBestEffort, PolicyOptions{})
			logger := klog.Background()
			b.ResetTimer()
			for b.Loop() {
				merger.Merge(logger)
			}
		})
	}
}
