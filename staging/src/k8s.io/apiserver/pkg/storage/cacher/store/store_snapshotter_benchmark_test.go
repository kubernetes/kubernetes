/*
Copyright 2025 The Kubernetes Authors.

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

package store

import (
	"math/rand"
	"testing"
)

func BenchmarkSnapshotterAdd(b *testing.B) {
	cache := NewSnapshotter()
	lister := fakeOrderedLister{rv: 1}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cache.Add(uint64(i), lister)
	}
}

func BenchmarkSnapshotterGetLessOrEqual(b *testing.B) {
	cache := NewSnapshotter()
	const size = 10000
	for i := uint64(0); i < size; i++ {
		cache.Add(i, fakeOrderedLister{rv: int(i)})
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cache.GetLessOrEqual(uint64(rand.Intn(size + 1000)))
	}
}

func BenchmarkSnapshotterSteadyState(b *testing.B) {
	cache := NewSnapshotter()
	// Pre-fill to simulate a warm cache.
	const warmup = 1000
	for i := uint64(0); i < warmup; i++ {
		cache.Add(i, fakeOrderedLister{rv: int(i)})
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rv := uint64(warmup + i)
		cache.Add(rv, fakeOrderedLister{rv: int(rv)})
		cache.RemoveLess(rv - warmup + 1)
		cache.GetLessOrEqual(rv - uint64(rand.Intn(warmup)))
	}
}
