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

package cacher

import (
	"context"
	"fmt"
	"math"
	"sort"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/apis/example"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/cacher/metrics"
	cachertesting "k8s.io/apiserver/pkg/storage/cacher/testing"
	"k8s.io/component-base/metrics/legacyregistry"
	testingclock "k8s.io/utils/clock/testing"
)

const dispatchDurationMetric = "apiserver_watch_events_dispatch_duration_seconds"

// TestDispatchStageSweep reproduces the fan-out at 100..10k watchers and reads
// the delay straight from Richa's staged metric
// (apiserver_watch_events_dispatch_duration_seconds), comparing the new
// stage="fanout" (the single dispatcher's serial fan-out cost) against
// stage="total" (end-to-end). No standalone metrics -- everything rides on
// #140336's WatcherMetricsObservers framework.
func TestDispatchStageSweep(t *testing.T) {
	metrics.Register()

	fmt.Printf("\n%-9s | %11s %11s\n", "watchers", "fanout p50", "fanout p99")
	fmt.Println("----------+-------------------------")

	for _, numWatchers := range []int{100, 1000, 2500, 5000, 10000} {
		fanBefore := snapshotStage(t, "fanout")

		runSweep(t, numWatchers)

		fan := deltaHist(snapshotStage(t, "fanout"), fanBefore)

		fmt.Printf("%-9d | %11s %11s\n",
			numWatchers,
			dur(histQuantile(fan, 0.50)), dur(histQuantile(fan, 0.99)))
	}
	fmt.Println("\nfanout = stage=\"fanout\" (dispatch -> c.input accept, the single dispatcher's serial cost)")
	fmt.Println("on apiserver_watch_events_dispatch_duration_seconds. p99 grows with watcher count:")
	fmt.Println("the serial single-dispatcher fan-out is the stage that scales with fan-out size.")
	fmt.Println("(stage=\"total\" is not measured here: this harness uses a fake clock, and post-")
	fmt.Println("#140851 sendWatchCacheEvent stamps sentAt from that clock, so total reads ~0.")
	fmt.Println("Compare fanout vs total in a real-clock scale/e2e run instead.)")
}

// runSweep registers numWatchers fast-draining watchers and dispatches a burst
// of events directly (fake clock avoids the bookmark-timer race).
func runSweep(t *testing.T, numWatchers int) {
	store := &cachertesting.MockStorage{}
	cacher, _, err := newTestCacherWithoutSyncing(store, testingclock.NewFakeClock(time.Now()))
	if err != nil {
		t.Fatalf("new cacher: %v", err)
	}
	defer cacher.Stop()
	if err := cacher.Wait(context.Background()); err != nil {
		t.Fatalf("wait: %v", err)
	}
	cacher.dispatchTimeoutBudget.returnUnused(100 * time.Millisecond)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	for i := 0; i < numWatchers; i++ {
		w, err := cacher.Watch(ctx, "/pods/", storage.ListOptions{
			ResourceVersion: "100", Predicate: storage.Everything, Recursive: true})
		if err != nil {
			t.Fatalf("watch %d: %v", i, err)
		}
		go func(w watch.Interface) {
			for range w.ResultChan() {
			}
		}(w)
	}
	time.Sleep(200 * time.Millisecond)

	makeEvent := func(rv uint64) *watchCacheEvent {
		pod := &example.Pod{ObjectMeta: metav1.ObjectMeta{
			Name: "pod-x", Namespace: "ns", ResourceVersion: fmt.Sprintf("%d", rv)}}
		return &watchCacheEvent{
			Type: watch.Modified, Object: pod, PrevObject: pod,
			ObjFields: podFields(pod), PrevObjFields: podFields(pod),
			Key: "/pods/ns/pod-x", ResourceVersion: rv,
			// RecordTime is required for the total stage to be observed.
			RecordTime: time.Now()}
	}
	const burst = 500
	for i := 0; i < burst; i++ {
		cacher.dispatchEvent(makeEvent(uint64(20000 + i)))
	}

	// The total stage is observed asynchronously in each watcher's process
	// goroutine; wait for the drain to settle so those samples land.
	prev := uint64(0)
	for i := 0; i < 40; i++ {
		time.Sleep(100 * time.Millisecond)
		now := snapshotStage(t, "total").count
		if now == prev && now > 0 {
			break
		}
		prev = now
	}
}

type histSnap struct {
	count   uint64
	sum     float64
	buckets []leCount // cumulative, sorted by le
}
type leCount struct {
	le float64
	c  uint64
}

// snapshotStage reads one stage of the staged dispatch_duration_seconds metric.
func snapshotStage(t *testing.T, stage string) histSnap {
	t.Helper()
	fams, err := legacyregistry.DefaultGatherer.Gather()
	if err != nil {
		t.Fatalf("gather: %v", err)
	}
	agg := map[float64]uint64{}
	s := histSnap{}
	for _, mf := range fams {
		if mf.GetName() != dispatchDurationMetric {
			continue
		}
		for _, m := range mf.GetMetric() {
			matched := false
			for _, lp := range m.GetLabel() {
				if lp.GetName() == "stage" && lp.GetValue() == stage {
					matched = true
					break
				}
			}
			if !matched {
				continue
			}
			h := m.GetHistogram()
			if h == nil {
				continue
			}
			s.count += h.GetSampleCount()
			s.sum += h.GetSampleSum()
			for _, b := range h.GetBucket() {
				agg[b.GetUpperBound()] += b.GetCumulativeCount()
			}
		}
	}
	for le, c := range agg {
		s.buckets = append(s.buckets, leCount{le, c})
	}
	sort.Slice(s.buckets, func(i, j int) bool { return s.buckets[i].le < s.buckets[j].le })
	return s
}

// deltaHist subtracts a prior snapshot to isolate this iteration.
func deltaHist(after, before histSnap) histSnap {
	d := histSnap{count: after.count - before.count, sum: after.sum - before.sum}
	bmap := map[float64]uint64{}
	for _, b := range before.buckets {
		bmap[b.le] = b.c
	}
	for _, a := range after.buckets {
		d.buckets = append(d.buckets, leCount{a.le, a.c - bmap[a.le]})
	}
	sort.Slice(d.buckets, func(i, j int) bool { return d.buckets[i].le < d.buckets[j].le })
	return d
}

// histQuantile linear-interpolates a quantile from cumulative buckets, the same
// way histogram_quantile() does in PromQL.
func histQuantile(h histSnap, q float64) float64 {
	if h.count == 0 {
		return 0
	}
	target := q * float64(h.count)
	prevLE, prevC := 0.0, uint64(0)
	for _, b := range h.buckets {
		if float64(b.c) >= target {
			if math.IsInf(b.le, 1) {
				return prevLE
			}
			frac := 0.0
			if b.c > prevC {
				frac = (target - float64(prevC)) / float64(b.c-prevC)
			}
			return prevLE + frac*(b.le-prevLE)
		}
		prevLE, prevC = b.le, b.c
	}
	return prevLE
}

func dur(seconds float64) string {
	return time.Duration(seconds * float64(time.Second)).Round(time.Microsecond).String()
}

func podFields(pod *example.Pod) fields.Set {
	return fields.Set{
		"metadata.name":      pod.Name,
		"metadata.namespace": pod.Namespace,
	}
}
