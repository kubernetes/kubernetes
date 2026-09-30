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

package cacher

import (
	"fmt"
	"runtime"
	"testing"

	"k8s.io/apiserver/pkg/storage/cacher/store"
)

const benchmarkLazySnapshotWatchlists = 20

func benchmarkLazySnapshot(b *testing.B) (*store.WatchCacheStorage, store.Snapshot) {
	b.StopTimer()
	indexer := store.NewWatchCacheStorage(nil, nil)
	elements := make([]*store.Element, 0, 150000)
	for i := range 150000 {
		elements = append(elements, makeTestStoreElement(makeTestPod(fmt.Sprintf("pod-%d", i), 1000)))
	}
	if err := indexer.Replace(elements, 1000); err != nil {
		b.Fatal(err)
	}
	return indexer, indexer.LatestSnapshot()
}

// BenchmarkLazySnapshotCacheIntervalStreaming measures allocations while
// consuming snapshot events without retaining them.
func BenchmarkLazySnapshotCacheIntervalStreaming(b *testing.B) {
	indexer, snapshot := benchmarkLazySnapshot(b)

	runtime.GC()
	var before runtime.MemStats
	runtime.ReadMemStats(&before)
	b.StartTimer()
	for range b.N {
		for range benchmarkLazySnapshotWatchlists {
			interval := newCacheIntervalFromLazySnapshot(1000, snapshot)
			for {
				event, err := interval.Next()
				if err != nil {
					b.Fatal(err)
				}
				if event == nil {
					break
				}
			}
		}
	}
	b.StopTimer()
	var after runtime.MemStats
	runtime.ReadMemStats(&after)
	b.ReportMetric(float64(after.TotalAlloc-before.TotalAlloc)/float64(b.N)/1e6, "MB-allocated/op")
	runtime.KeepAlive(indexer)
	runtime.KeepAlive(snapshot)
}
