package topologymanager

import "testing"

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
	best := merger.mergePreferred()
	if best == nil {
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
	best := merger.mergePreferred()
	if best != nil {
		t.Fatalf("expected nil, got %v", best)
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
	best := merger.mergePreferred()
	if best != nil {
		t.Fatalf("expected nil, got %v", best)
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
	best := merger.mergePreferred()
	if best != nil {
		t.Fatalf("expected nil, got %v", best)
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
	best := merger.mergePreferred()
	if best == nil {
		t.Fatalf("expected preferred hint, got nil")
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
	best := merger.mergePreferred()
	if best == nil {
		t.Fatalf("expected preferred hint, got nil")
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
	best := merger.mergePreferred()
	if best == nil {
		t.Fatalf("expected preferred hint, got nil")
	}
	if !best.NUMANodeAffinity.IsEqual(NewTestBitMask(0)) {
		t.Fatalf("expected affinity %v, got %v", NewTestBitMask(0), best.NUMANodeAffinity)
	}
}
