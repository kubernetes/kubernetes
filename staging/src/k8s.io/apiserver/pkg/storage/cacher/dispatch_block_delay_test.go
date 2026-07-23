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
	"sort"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	examplev1 "k8s.io/apiserver/pkg/apis/example/v1"
	"k8s.io/apiserver/pkg/storage"
	cachertesting "k8s.io/apiserver/pkg/storage/cacher/testing"
)

// TestBlockDelayBlastRadius quantifies the delay the single dispatcher's
// mailbox-full (c.input blocked) path injects into an INNOCENT, fast-draining
// watcher. The victim never blocks on its own; any latency it sees is
// head-of-line delay caused by the shared dispatcher stalling on OTHER
// watchers whose input buffers are full.
//
// This is the "blast radius" that #140851's per-watcher c.result handoff stage
// cannot see, and that #140336's `total` stage folds in but cannot isolate.
func TestBlockDelayBlastRadius(t *testing.T) {
	measure := func(t *testing.T, numSlow int) (p50, p99, max time.Duration) {
		backing := &cachertesting.MockStorage{}
		cacher, _, err := newTestCacher(backing)
		if err != nil {
			t.Fatalf("new cacher: %v", err)
		}
		defer cacher.Stop()
		// Realistic budget so the blocked path actually waits.
		cacher.dispatchTimeoutBudget.returnUnused(100 * time.Millisecond)

		makePod := func(rv int) *examplev1.Pod {
			return &examplev1.Pod{ObjectMeta: metav1.ObjectMeta{
				Name: "p", Namespace: "ns", ResourceVersion: fmt.Sprintf("%d", rv)}}
		}
		if err := cacher.watchCache.Add(makePod(1000)); err != nil {
			t.Fatalf("seed: %v", err)
		}

		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		// Victim: fast drainer, records delivery time per delivered event.
		victim, err := cacher.Watch(ctx, "/pods/ns", storage.ListOptions{
			ResourceVersion: "1000", Predicate: storage.Everything, Recursive: true})
		if err != nil {
			t.Fatalf("victim watch: %v", err)
		}
		const burst = 200
		delivered := make(chan time.Time, burst+16)
		go func() {
			for ev := range victim.ResultChan() {
				acc, aerr := metaAccessor(ev.Object)
				if aerr != nil || acc <= 1000 {
					continue // skip the seed / initial-list event
				}
				delivered <- time.Now()
			}
		}()

		// Slow watchers: never drain -> their c.input fills -> dispatcher is
		// forced into the blocking add() path when serving each event.
		for i := 0; i < numSlow; i++ {
			if _, err := cacher.Watch(ctx, "/pods/ns", storage.ListOptions{
				ResourceVersion: "1000", Predicate: storage.Everything, Recursive: true}); err != nil {
				t.Fatalf("slow watch %d: %v", i, err)
			}
		}
		time.Sleep(50 * time.Millisecond) // let watchers reach steady state

		// Inject a burst; latency = delivery - injection for each event.
		inject := make([]time.Time, 0, burst)
		for i := 0; i < burst; i++ {
			inject = append(inject, time.Now())
			if err := cacher.watchCache.Update(makePod(1001 + i)); err != nil {
				t.Fatalf("update: %v", err)
			}
		}

		lat := make([]time.Duration, 0, burst)
		timeout := time.After(30 * time.Second)
		for len(lat) < burst {
			select {
			case dt := <-delivered:
				lat = append(lat, dt.Sub(inject[len(lat)]))
			case <-timeout:
				t.Fatalf("only %d/%d events delivered to victim", len(lat), burst)
			}
		}
		sort.Slice(lat, func(i, j int) bool { return lat[i] < lat[j] })
		return lat[len(lat)/2], lat[(len(lat)*99)/100], lat[len(lat)-1]
	}

	bp50, bp99, bmax := measure(t, 0)
	t.Logf("baseline (0 slow watchers):   p50=%v p99=%v max=%v", bp50, bp99, bmax)
	sp50, sp99, smax := measure(t, 500)
	t.Logf("with 500 undrained watchers:  p50=%v p99=%v max=%v", sp50, sp99, smax)
	t.Logf("BLOCK-INDUCED DELAY on innocent watcher: p50 +%v, p99 +%v, max +%v",
		sp50-bp50, sp99-bp99, smax-bmax)
}

// metaAccessor returns the numeric resourceVersion of an object for filtering.
func metaAccessor(obj runtime.Object) (uint64, error) {
	m, err := meta.Accessor(obj)
	if err != nil {
		return 0, err
	}
	return storage.APIObjectVersioner{}.ParseResourceVersion(m.GetResourceVersion())
}
