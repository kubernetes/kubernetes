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
	"testing"

	"k8s.io/klog/v2/ktesting"
)

func BenchmarkPolicyMergePreferredExists(b *testing.B) {
	numaInfo := commonNUMAInfoEightNodes()
	logger, _ := ktesting.NewTestContext(b)
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
	policy := &bestEffortPolicy{numaInfo: numaInfo}
	providersHints := []map[string][]TopologyHint{
		{"resource0": hints[0]},
		{"resource1": hints[1]},
		{"resource2": hints[2]},
		{"resource3": hints[3]},
		{"resource4": hints[4]},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = policy.Merge(logger, providersHints)
	}
}

func BenchmarkPolicyMergePreferredMissing(b *testing.B) {
	numaInfo := commonNUMAInfoEightNodes()
	logger, _ := ktesting.NewTestContext(b)
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
	policy := &bestEffortPolicy{numaInfo: numaInfo}
	providersHints := []map[string][]TopologyHint{
		{"resource0": hints[0]},
		{"resource1": hints[1]},
		{"resource2": hints[2]},
		{"resource3": hints[3]},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = policy.Merge(logger, providersHints)
	}
}

func BenchmarkHintMergerManyHintsPerProvider(b *testing.B) {
	numaInfo := commonNUMAInfoEightNodes()
	logger, _ := ktesting.NewTestContext(b)
	masks := make([][]int, 0, 255)
	for mask := 1; mask < 1<<8; mask++ {
		var bits []int
		for i := range 8 {
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
