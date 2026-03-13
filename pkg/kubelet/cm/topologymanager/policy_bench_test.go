package topologymanager

import (
	"testing"

	"k8s.io/klog/v2/ktesting"
)

func BenchmarkHintMergerPreferredExists(b *testing.B) {
	numaInfo := commonNUMAInfoEightNodes()
	hints := [][]TopologyHint{
		{
			{NUMANodeAffinity: NewTestBitMask(0), Preferred: true},
			{NUMANodeAffinity: NewTestBitMask(1), Preferred: true},
			{NUMANodeAffinity: NewTestBitMask(0, 1, 2, 3), Preferred: false},
		},
		{
			{NUMANodeAffinity: NewTestBitMask(0), Preferred: true},
			{NUMANodeAffinity: NewTestBitMask(0, 1, 2, 3), Preferred: false},
		},
		{
			{NUMANodeAffinity: nil, Preferred: true},
			{NUMANodeAffinity: NewTestBitMask(0, 1, 2, 3), Preferred: false},
		},
		{
			{NUMANodeAffinity: NewTestBitMask(0), Preferred: true},
			{NUMANodeAffinity: NewTestBitMask(0, 1, 2, 3), Preferred: false},
		},
		{
			{NUMANodeAffinity: NewTestBitMask(0), Preferred: true},
			{NUMANodeAffinity: NewTestBitMask(4, 5, 6, 7), Preferred: true},
			{NUMANodeAffinity: NewTestBitMask(0, 1, 2, 3, 4, 5, 6, 7), Preferred: false},
		},
	}
	merger := NewHintMerger(numaInfo, hints, PolicyBestEffort, PolicyOptions{})

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = merger.Merge()
	}
}

func BenchmarkHintMergerPreferredMissing(b *testing.B) {
	numaInfo := commonNUMAInfoEightNodes()
	hints := [][]TopologyHint{
		{
			{NUMANodeAffinity: NewTestBitMask(0, 1, 2, 3), Preferred: false},
		},
		{
			{NUMANodeAffinity: NewTestBitMask(0), Preferred: true},
			{NUMANodeAffinity: NewTestBitMask(1), Preferred: true},
		},
		{
			{NUMANodeAffinity: NewTestBitMask(2), Preferred: true},
			{NUMANodeAffinity: NewTestBitMask(3), Preferred: true},
		},
		{
			{NUMANodeAffinity: nil, Preferred: true},
			{NUMANodeAffinity: NewTestBitMask(0, 1, 2, 3, 4, 5, 6, 7), Preferred: false},
		},
	}
	merger := NewHintMerger(numaInfo, hints, PolicyBestEffort, PolicyOptions{})

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = merger.Merge()
	}
}

func BenchmarkHintMergerManyHintsPerProvider(b *testing.B) {
	numaInfo := commonNUMAInfoEightNodes()
	logger, _ := ktesting.NewTestContext(b)
	masks := make([][]int, 0, 255)
	for mask := 1; mask < 1<<8; mask++ {
		var bits []int
		for i := 0; i < 8; i++ {
			if mask&(1<<i) != 0 {
				bits = append(bits, i)
			}
		}
		masks = append(masks, bits)
	}

	oneProviderHints := make([]TopologyHint, 0, len(masks))
	for i := range masks {
		affinity := NewTestBitMask(masks[i]...)
		oneProviderHints = append(oneProviderHints, TopologyHint{
			NUMANodeAffinity: affinity,
			Preferred:        affinity.Count() == 1,
		})
	}

	hints := [][]TopologyHint{
		oneProviderHints,
		oneProviderHints,
		oneProviderHints,
		oneProviderHints,
	}
	providersHints := []map[string][]TopologyHint{
		{"cpu": hints[0]},
		{"memory": hints[1]},
		{"hugepages": hints[2]},
		{"gpu": hints[3]},
	}
	policy := &bestEffortPolicy{numaInfo: numaInfo}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = policy.Merge(logger, providersHints)
	}
}
