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

package cache

import (
	"context"
	"fmt"
	"runtime"
	"runtime/debug"
	"strings"
	"sync"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	apimruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	clientfeatures "k8s.io/client-go/features"
	clientfeaturestesting "k8s.io/client-go/features/testing"
)

// BenchmarkWatchListMemoryOptimizationFeatureGate is an informer based
// benchmark that validates the watchlist memory optimization feature gate.
//
// It measures total retained heap growth per iteration after GC.
//
// With optimization, the relist reuses objects from the first watchlist, so the
// total retained growth is roughly one store's worth of objects. Without
// optimization, relist duplicates those objects, so retained growth is roughly
// double.
func BenchmarkWatchListMemoryOptimizationFeatureGate(b *testing.B) {
	const numPods = 20000

	scenarios := []struct {
		name                string
		optimizationEnabled bool
	}{
		{
			name:                "WithWatchListMemoryOptimization",
			optimizationEnabled: true,
		},
		{
			name:                "WithoutWatchListMemoryOptimization",
			optimizationEnabled: false,
		},
	}

	for _, scenario := range scenarios {
		b.Run(scenario.name, func(b *testing.B) {
			clientfeaturestesting.SetFeatureDuringTest(b, clientfeatures.WatchListClient, true)
			clientfeaturestesting.SetFeatureDuringTest(b, clientfeatures.WatchListMemoryOptimization, scenario.optimizationEnabled)

			pods := make([]*v1.Pod, numPods)
			for i := 0; i < numPods; i++ {
				pods[i] = createLargePod(fmt.Sprintf("pod-%d", i), "1000")
			}

			var totalHeapGrowth uint64
			var totalHeapObjectsGrowth uint64

			for i := 0; i < b.N; i++ {
				func() {
					oldGCPercent := debug.SetGCPercent(-1)
					defer debug.SetGCPercent(oldGCPercent)

					runtime.GC()
					var memStart runtime.MemStats
					runtime.ReadMemStats(&memStart)

					fw1 := watch.NewFake()
					fw2 := watch.NewFake()
					lw := newFakeListWatcher(fw1, fw2)

					podInformer := NewSharedInformer(lw, &v1.Pod{}, 0)

					ctx := context.TODO()
					ctx, cancelPodInformer := context.WithCancel(ctx)
					defer cancelPodInformer()
					podInformerStopped := make(chan struct{})
					go func() {
						podInformer.Run(ctx.Done())
						close(podInformerStopped)
					}()

					b.Log("getting the list of pods and waiting for the pod informer to sync")
					for _, pod := range pods {
						fw1.Add(pod.DeepCopy())
					}
					fw1.Action(watch.Bookmark, &v1.Pod{
						ObjectMeta: metav1.ObjectMeta{
							ResourceVersion: "1000",
							Annotations: map[string]string{
								metav1.InitialEventsAnnotationKey: "true",
							},
						},
					})
					err := wait.PollUntilContextTimeout(ctx, 10*time.Millisecond, wait.ForeverTestTimeout, true, func(ctx context.Context) (bool, error) {
						return podInformer.HasSynced(), nil
					})
					if err != nil {
						b.Fatalf("failed to sync the pod informer: %v", err)
					}

					// keep references from the first watchlist to simulate real usage.
					initialWatchlistPods := make(map[string]*v1.Pod)
					for _, obj := range podInformer.GetStore().List() {
						pod := obj.(*v1.Pod)
						initialWatchlistPods[pod.Name] = pod
					}

					b.Log("triggering a relist via a synthetic watch error")
					fw1.Error(&metav1.Status{
						Status:  metav1.StatusFailure,
						Code:    410,
						Reason:  metav1.StatusReasonExpired,
						Message: "watch expired - triggering relist",
					})

					b.Log("getting the second list of pods and waiting for the pod informer to be on RV=2000")
					for _, pod := range pods {
						fw2.Add(pod.DeepCopy())
					}
					fw2.Action(watch.Bookmark, &v1.Pod{
						ObjectMeta: metav1.ObjectMeta{
							ResourceVersion: "2000",
							Annotations: map[string]string{
								metav1.InitialEventsAnnotationKey: "true",
							},
						},
					})
					err = wait.PollUntilContextTimeout(ctx, 10*time.Millisecond, wait.ForeverTestTimeout, true, func(ctx context.Context) (bool, error) {
						return podInformer.LastSyncResourceVersion() == "2000", nil
					})
					if err != nil {
						b.Fatalf("timeout waiting for relist, expected the pod informer to be on RV=2000: %v", err)
					}

					// relist allocates transient objects while decoding and comparing to the store.
					// A GC here clears those short-lived allocations so we measure retained heap.
					runtime.GC()
					var memAfter runtime.MemStats
					runtime.ReadMemStats(&memAfter)
					if memAfter.HeapInuse > memStart.HeapInuse {
						totalHeapGrowth += memAfter.HeapInuse - memStart.HeapInuse
					}
					if memAfter.HeapObjects > memStart.HeapObjects {
						totalHeapObjectsGrowth += memAfter.HeapObjects - memStart.HeapObjects
					}

					cancelPodInformer()
					select {
					case <-podInformerStopped:
					case <-time.After(wait.ForeverTestTimeout):
						b.Fatal("timeout waiting for the pod informer to stop")
					}

					// KeepAlive keeps these references live through the measurement.
					runtime.KeepAlive(initialWatchlistPods)
				}()
			}

			avgHeapGrowthMB := float64(totalHeapGrowth) / float64(b.N) / 1024 / 1024
			avgHeapObjectsGrowth := float64(totalHeapObjectsGrowth) / float64(b.N)

			b.ReportMetric(avgHeapGrowthMB, "MB-heap-growth/op")
			b.ReportMetric(avgHeapObjectsGrowth, "objects-growth/op")
		})
	}
}

type fakeLW struct {
	mu             sync.Mutex
	fakeWatcher1   *watch.FakeWatcher
	fakeWatcher2   *watch.FakeWatcher
	watchCallCount int
}

func (lw *fakeLW) List(_ metav1.ListOptions) (apimruntime.Object, error) {
	return nil, fmt.Errorf("unexpected LIST call")
}

func (lw *fakeLW) Watch(_ metav1.ListOptions) (watch.Interface, error) {
	lw.mu.Lock()
	defer lw.mu.Unlock()

	lw.watchCallCount++
	switch lw.watchCallCount {
	case 1:
		return lw.fakeWatcher1, nil
	case 2:
		return lw.fakeWatcher2, nil
	default:
		return nil, fmt.Errorf("unexpected WATCH call count: %d", lw.watchCallCount)
	}
}

func newFakeListWatcher(fw1, fw2 *watch.FakeWatcher) *fakeLW {
	return &fakeLW{
		fakeWatcher1: fw1,
		fakeWatcher2: fw2,
	}
}

// createLargePod creates a pod with substantial memory footprint.
func createLargePod(name string, resourceVersion string) *v1.Pod {
	annotations := map[string]string{
		"annotation1": strings.Repeat("a", 8*1024),
		"annotation2": strings.Repeat("b", 8*1024),
	}

	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:            name,
			Namespace:       "test-namespace",
			ResourceVersion: resourceVersion,
			Annotations:     annotations,
		},
	}
}
