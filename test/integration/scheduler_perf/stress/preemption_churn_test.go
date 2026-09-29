/*
Copyright 2026 The Kubernetes Authors.

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

package stress

import (
	"fmt"
	"os"
	"runtime"
	"testing"

	_ "k8s.io/component-base/logs/json/register"
	perf "k8s.io/kubernetes/test/integration/scheduler_perf"
)

func TestMain(m *testing.M) {
	if err := perf.InitTests(); err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(1)
	}

	m.Run()
}

func TestSchedulerPerf(t *testing.T) {
	perf.RunIntegrationPerfScheduling(t, "performance-config.yaml")
}

func TestPreemptionChurnStress(t *testing.T) {
	// Profile heap and goroutine stability across preemption churn stress testing.
	runtime.GC()
	var mBefore runtime.MemStats
	runtime.ReadMemStats(&mBefore)
	goroutinesBefore := runtime.NumGoroutine()

	perf.RunIntegrationPerfScheduling(t, "performance-config.yaml")

	runtime.GC()
	var mAfter runtime.MemStats
	runtime.ReadMemStats(&mAfter)
	goroutinesAfter := runtime.NumGoroutine()

	t.Logf("Preemption Churn Stress Memory Profile:")
	t.Logf("  Allocated Heap Before: %.2f MB, After: %.2f MB",
		float64(mBefore.Alloc)/(1024*1024), float64(mAfter.Alloc)/(1024*1024))
	t.Logf("  Total Cumulative Alloc: %.2f MB",
		float64(mAfter.TotalAlloc-mBefore.TotalAlloc)/(1024*1024))
	t.Logf("  Total Mallocs: %d, Total Frees: %d",
		mAfter.Mallocs-mBefore.Mallocs, mAfter.Frees-mBefore.Frees)
	t.Logf("  Goroutines Before: %d, After: %d",
		goroutinesBefore, goroutinesAfter)
}

func BenchmarkPerfScheduling(b *testing.B) {
	perf.RunBenchmarkPerfScheduling(b, "performance-config.yaml", "stress", nil)
}

func BenchmarkPreemptionChurnStress(b *testing.B) {
	runtime.GC()
	var mBefore runtime.MemStats
	runtime.ReadMemStats(&mBefore)
	goroutinesBefore := runtime.NumGoroutine()

	perf.RunBenchmarkPerfScheduling(b, "performance-config.yaml", "stress", nil)

	runtime.GC()
	var mAfter runtime.MemStats
	runtime.ReadMemStats(&mAfter)
	goroutinesAfter := runtime.NumGoroutine()

	heapGrowthMB := float64(int64(mAfter.Alloc)-int64(mBefore.Alloc)) / (1024 * 1024)
	totalAllocMB := float64(mAfter.TotalAlloc-mBefore.TotalAlloc) / (1024 * 1024)

	b.ReportMetric(heapGrowthMB, "heap_growth_mb")
	b.ReportMetric(totalAllocMB, "total_alloc_mb")
	b.ReportMetric(float64(mAfter.Mallocs-mBefore.Mallocs), "mallocs")
	b.ReportMetric(float64(mAfter.NumGC-mBefore.NumGC), "gc_cycles")
	b.ReportMetric(float64(goroutinesAfter-goroutinesBefore), "goroutines_delta")
}

func BenchmarkPreemptionQueueChurn(b *testing.B) {
	perf.RunBenchmarkPerfScheduling(b, "performance-config.yaml", "stress", nil)
}
