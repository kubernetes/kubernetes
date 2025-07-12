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

package pullmanager

import (
	"fmt"
	"runtime"
	"testing"

	"github.com/google/uuid"
	"k8s.io/apimachinery/pkg/util/rand"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
)

type namedAccessor struct {
	name         string
	accessorInit func(b testing.TB) PullRecordsAccessor
}

type recordAccessorBenchmark struct {
	namedInit             namedAccessor
	recordsInCache        int
	concurrencyMultiplier int
	cacheHit              bool
}

func BenchmarkPullRecordsAccessorsCacheHit(b *testing.B) {
	benchmarkAllPullAccessorsRead(b, true)
}

func BenchmarkPullRecordsAccessorCacheMiss(b *testing.B) {
	benchmarkAllPullAccessorsRead(b, false)
}

func BenchmarkPullRecordsAccessorsWrite(b *testing.B) {
	imageRequests := make([]string, 0, generatedTestRequestsNum)
	for range generatedTestRequestsNum {
		imageRequests = append(imageRequests, uuid.NewString())
	}

	createFSAccessor := namedAccessor{"FileSystemAccessor", setupFSRecordsAccessor}
	const cacheSize = 500
	createMemCachedAccessor := namedAccessor{
		"InMemoryCache",
		func(b testing.TB) PullRecordsAccessor {
			return setupInMemRecordsAccessor(b, cacheSize, false)
		},
	}

	runCheck := func(b *testing.B, accessor PullRecordsAccessor, idx uint) {
		imgRef := imageRequests[idx%generatedTestRequestsNum]

		if err := accessor.WriteImagePulledRecord(&kubeletconfig.ImagePulledRecord{
			ImageRef: imgRef,
			CredentialMapping: map[string]kubeletconfig.ImagePullCredentials{
				"test.repo/org/" + imgRef: {NodePodsAccessible: true},
			},
		}); err != nil {
			b.Fatalf("failed to write a record: %v", err)
		}
	}

	concurrencyMultipliers := []int{0, 1, 2, 8, 16}
	for _, namedInit := range []namedAccessor{
		createFSAccessor,
		createMemCachedAccessor,
	} {
		for _, concurrency := range concurrencyMultipliers {
			b.Run(fmt.Sprintf("Type=%s/ConcurrencyPerCPU=%d", namedInit.name, concurrency), func(b *testing.B) {
				accessor := namedInit.accessorInit(b)
				runBenchmark(b, accessor, concurrency, runCheck)
			})
		}
	}

}

func benchmarkAllPullAccessorsRead(b *testing.B, hit bool) {
	// Compare FS-cache with InMem-Cache authoritative
	recordsKepts := []int{10, 20, 50, 100, 200, 500, 2000}
	concurrencyMultipliers := []int{0, 1, 2, 8, 16}

	createFSAccessor := func(_ int) namedAccessor {
		return namedAccessor{"FileSystemAccessor", setupFSRecordsAccessor}
	}
	createMemCachedAuthoritativeAccessor := func(cacheSize int) namedAccessor {
		return namedAccessor{
			"InMemoryCache-Authoritative",
			func(b testing.TB) PullRecordsAccessor {
				return setupInMemRecordsAccessor(b, cacheSize, true)
			},
		}
	}

	for _, accessorInit := range []func(int) namedAccessor{
		createFSAccessor,
		createMemCachedAuthoritativeAccessor,
	} {
		for _, concurrency := range concurrencyMultipliers {
			for _, recordsKept := range recordsKepts {
				benchmarkPullAccessorCacheRead(b, recordAccessorBenchmark{
					namedInit:             accessorInit(recordsKept),
					recordsInCache:        recordsKept,
					concurrencyMultiplier: concurrency,
					cacheHit:              hit,
				})
			}
		}
	}

	// Benchmark dependency of InMem-Cache on TotalRecords/CacheSize ratio
	const nonAuthoritativeRecordsNumber = 2_000
	for _, concurrency := range concurrencyMultipliers {
		for _, cacheSize := range []int{2000, 1000, 500, 250, 125} {
			benchmarkPullAccessorCacheRead(b, recordAccessorBenchmark{
				namedInit: namedAccessor{
					fmt.Sprintf("InMemoryCache-NonAuthoritative/Records-CacheSize-Ratio=%d", nonAuthoritativeRecordsNumber/cacheSize),
					func(b testing.TB) PullRecordsAccessor { return setupInMemRecordsAccessor(b, cacheSize, false) },
				},
				recordsInCache:        nonAuthoritativeRecordsNumber,
				concurrencyMultiplier: concurrency,
				cacheHit:              hit,
			})
		}
	}
}

const (
	generatedTestRequestsNum = 4_000
)

func benchmarkPullAccessorCacheRead(b *testing.B, bc recordAccessorBenchmark) {
	b.Run(fmt.Sprintf("Type=%s/RecordsStored=%d/ConcurrencyPerCPU=%d", bc.namedInit.name, bc.recordsInCache, bc.concurrencyMultiplier), func(b *testing.B) {
		genRecords, genRequests := generateRecordsAndRequests(bc.recordsInCache, bc.cacheHit)

		accessor := bc.namedInit.accessorInit(b)
		for _, r := range genRecords {
			if err := accessor.WriteImagePulledRecord(r); err != nil {
				b.Fatalf("failed to prepare cache - write error: %v", err)
			}
		}

		runCheck := func(b *testing.B, accessor PullRecordsAccessor, recordIndex uint) {
			_, exists, err := accessor.GetImagePulledRecord(genRequests[recordIndex%generatedTestRequestsNum])
			if exists != bc.cacheHit || err != nil {
				b.Fatalf("no error expected (got %v) and record should've existed (got %t)", err, exists)
			}
		}
		runBenchmark(b, accessor, bc.concurrencyMultiplier, runCheck)
	})
}

func runBenchmark(b *testing.B, accessor PullRecordsAccessor, concurrencyMultiplier int, bf func(b *testing.B, accessor PullRecordsAccessor, idx uint)) {
	b.ReportAllocs()
	if concurrencyMultiplier == 0 {
		var idx uint
		for b.Loop() {
			bf(b, accessor, idx)
			idx++
		}
		return
	}

	// concurrencyMultiplier > 0
	b.SetParallelism(concurrencyMultiplier)
	b.ResetTimer()
	b.RunParallel(func(p *testing.PB) {
		idx := uint(rand.IntnRange(0, generatedTestRequestsNum))
		for p.Next() {
			bf(b, accessor, idx)
			idx++
		}
	})
	b.StopTimer()

}

func generateRecordsAndRequests(recordsNum int, generateHits bool) ([]*kubeletconfig.ImagePulledRecord, []string) {
	generatedPulledRecords := make([]*kubeletconfig.ImagePulledRecord, 0, recordsNum)
	for range recordsNum {
		imgRef := uuid.NewString()
		generatedPulledRecords = append(generatedPulledRecords, &kubeletconfig.ImagePulledRecord{
			ImageRef: string(imgRef),
			CredentialMapping: map[string]kubeletconfig.ImagePullCredentials{
				"test.repo/org/" + imgRef: {NodePodsAccessible: true},
			},
		})
	}

	generatedCacheRequests := make([]string, 0, generatedTestRequestsNum)
	for range generatedTestRequestsNum {
		var requestRef string
		if generateHits {
			requestRef = generatedPulledRecords[rand.IntnRange(0, recordsNum)].ImageRef
		} else {
			requestRef = uuid.NewString()
		}
		generatedCacheRequests = append(generatedCacheRequests, requestRef)
	}

	return generatedPulledRecords, generatedCacheRequests
}

func setupFSRecordsAccessor(t testing.TB) PullRecordsAccessor {
	t.Helper()

	tempDir := t.TempDir()
	accessor, err := NewFSPullRecordsAccessor(tempDir)
	if err != nil {
		t.Fatalf("failed to setup filesystem pull records accessor: %v", err)
	}
	return accessor
}

func setupInMemRecordsAccessor(t testing.TB, cacheSize int, authoritative bool) PullRecordsAccessor {
	t.Helper()

	fsAccessor := setupFSRecordsAccessor(t)
	memcacheAccessor := NewCachedPullRecordsAccessor(fsAccessor, int32(cacheSize), int32(cacheSize), int32(runtime.NumCPU()))
	memcacheAccessor.intents.authoritative.Store(authoritative)
	memcacheAccessor.pulledRecords.authoritative.Store(authoritative)

	return memcacheAccessor
}
