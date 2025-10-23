/*
Copyright 2024 The Kubernetes Authors.

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

package scheduler

import (
	"runtime"
	"sync"
	"testing"

	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework"
)

// BenchmarkNodeStatusPool benchmarks the performance of nodeStatus allocation
// with and without sync.Pool to measure GC pressure reduction.
func BenchmarkNodeStatusPool(b *testing.B) {
	b.Run("WithPool", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		
		var wg sync.WaitGroup
		for i := 0; i < b.N; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				// Simulate getting from pool
				ns := nodeStatusPool.Get().(*nodeStatus)
				ns.node = "test-node"
				ns.status = fwk.NewStatus(fwk.Success)
				
				// Simulate some work
				_ = ns.node
				_ = ns.status
				
				// Return to pool
				ns.node = ""
				ns.status = nil
				nodeStatusPool.Put(ns)
			}()
		}
		wg.Wait()
	})
	
	b.Run("WithoutPool", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		
		var wg sync.WaitGroup
		for i := 0; i < b.N; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				// Direct allocation
				ns := &nodeStatus{
					node:   "test-node",
					status: fwk.NewStatus(fwk.Success),
				}
				
				// Simulate some work
				_ = ns.node
				_ = ns.status
			}()
		}
		wg.Wait()
	})
}

// BenchmarkCycleStatePool benchmarks the performance of CycleState allocation
// with and without sync.Pool to measure GC pressure reduction.
func BenchmarkCycleStatePool(b *testing.B) {
	b.Run("WithPool", func(b *testing.B) {
		b.ReportAllocs()
		
		// Measure initial GC stats
		var memStatsBefore runtime.MemStats
		runtime.ReadMemStats(&memStatsBefore)
		
		b.ResetTimer()
		
		for i := 0; i < b.N; i++ {
			state := framework.NewCycleState()
			state.SetRecordPluginMetrics(true)
			// Simulate some operations
			state.Write("test-key", &testStateData{value: i})
			_, _ = state.Read("test-key")
			// Recycle the state
			state.Recycle()
		}
		
		b.StopTimer()
		
		// Measure final GC stats
		var memStatsAfter runtime.MemStats
		runtime.ReadMemStats(&memStatsAfter)
		
		b.ReportMetric(float64(memStatsAfter.NumGC-memStatsBefore.NumGC), "gc-cycles")
		b.ReportMetric(float64(memStatsAfter.TotalAlloc-memStatsBefore.TotalAlloc)/float64(b.N), "alloc/op")
	})
	
	b.Run("WithoutPool", func(b *testing.B) {
		b.ReportAllocs()
		
		// Measure initial GC stats
		var memStatsBefore runtime.MemStats
		runtime.ReadMemStats(&memStatsBefore)
		
		b.ResetTimer()
		
		for i := 0; i < b.N; i++ {
			// Direct allocation without pool
			state := &framework.CycleState{}
			state.SetRecordPluginMetrics(true)
			// Simulate some operations
			state.Write("test-key", &testStateData{value: i})
			_, _ = state.Read("test-key")
		}
		
		b.StopTimer()
		
		// Measure final GC stats
		var memStatsAfter runtime.MemStats
		runtime.ReadMemStats(&memStatsAfter)
		
		b.ReportMetric(float64(memStatsAfter.NumGC-memStatsBefore.NumGC), "gc-cycles")
		b.ReportMetric(float64(memStatsAfter.TotalAlloc-memStatsBefore.TotalAlloc)/float64(b.N), "alloc/op")
	})
}

// testStateData is a simple implementation of StateData for testing
type testStateData struct {
	value int
}

func (t *testStateData) Clone() fwk.StateData {
	return &testStateData{value: t.value}
}

// BenchmarkSchedulingCycleMemory benchmarks memory allocations in a simulated scheduling cycle
func BenchmarkSchedulingCycleMemory(b *testing.B) {
	b.ReportAllocs()
	
	for i := 0; i < b.N; i++ {
		// Simulate a scheduling cycle with pool usage
		state := framework.NewCycleState()
		
		// Simulate filter phase with nodeStatus pool
		numNodes := 100
		result := make([]*nodeStatus, numNodes)
		for j := 0; j < numNodes; j++ {
			if j%2 == 0 { // Simulate 50% of nodes failing filter
				ns := nodeStatusPool.Get().(*nodeStatus)
				ns.node = "node-" + string(rune(j))
				ns.status = fwk.NewStatus(fwk.Unschedulable, "test reason")
				result[j] = ns
			}
		}
		
		// Cleanup
		for _, ns := range result {
			if ns != nil {
				ns.node = ""
				ns.status = nil
				nodeStatusPool.Put(ns)
			}
		}
		
		state.Recycle()
	}
}
