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
	best, ok := merger.mergePreferred()
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
	_, ok := merger.mergePreferred()
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
	_, ok := merger.mergePreferred()
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
	_, ok := merger.mergePreferred()
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
	best, ok := merger.mergePreferred()
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
	best, ok := merger.mergePreferred()
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
	best, ok := merger.mergePreferred()
	if !ok {
		t.Fatalf("expected preferred hint, got zero")
	}
	if !best.NUMANodeAffinity.IsEqual(NewTestBitMask(0)) {
		t.Fatalf("expected affinity %v, got %v", NewTestBitMask(0), best.NUMANodeAffinity)
	}
}

func TestHintMergerMergePreferredEquivalentToMerge(t *testing.T) {
	numaInfo := commonNUMAInfoEightNodes()
	r := rand.New(rand.NewSource(1))

	for range 500 {
		dim := r.Intn(5) + 1
		hints := make([][]TopologyHint, 0, dim)
		for range dim {
			numHints := r.Intn(6) + 1
			oneDim := make([]TopologyHint, 0, numHints)
			for range numHints {
				var affinity bitmask.BitMask
				if r.Intn(3) != 0 {
					var bits []int
					for k := range 8 {
						if r.Intn(4) == 0 {
							bits = append(bits, k)
						}
					}
					affinity = NewTestBitMask(bits...)
				}
				oneDim = append(oneDim, TopologyHint{
					NUMANodeAffinity: affinity,
					Preferred:        r.Intn(2) == 0,
				})
			}
			hints = append(hints, oneDim)
		}

		merger := NewHintMerger(numaInfo, hints, PolicyBestEffort, PolicyOptions{})
		preferred, ok := merger.mergePreferred()
		if !ok {
			continue
		}
		best := merger.Merge()
		if !preferred.IsEqual(best) {
			t.Fatalf("expected mergePreferred() == Merge(); got preferred=%v full=%v hints=%v", preferred, best, hints)
		}
	}
}

func TestHintMergerMergePreferredEmptyHints(t *testing.T) {
	numaInfo := commonNUMAInfoTwoNodes()
	hints := [][]TopologyHint{}

	merger := NewHintMerger(numaInfo, hints, PolicyBestEffort, PolicyOptions{})
	preferred, ok := merger.mergePreferred()
	if !ok {
		t.Fatalf("expected preferred hint, got zero")
	}
	best := merger.Merge()
	if !preferred.IsEqual(best) {
		t.Fatalf("expected mergePreferred() == Merge(); got preferred=%v full=%v", preferred, best)
	}
}
