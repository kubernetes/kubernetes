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
	"context"
	"fmt"
	"slices"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/apis/example"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/cacher/metrics"
	cachertesting "k8s.io/apiserver/pkg/storage/cacher/testing"
	compbasemetrics "k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/testutil"
)

// BenchmarkSlowWatcherTax measures event delivery latency through the cacher's
// dispatch path to a healthy watcher, alone and next to a companion watcher.
// Every scenario runs with the WatchCacheStallResume gate off and on.
// It needs wall-clock time; run it with:
//
//	go test ./staging/src/k8s.io/apiserver/pkg/storage/cacher/ -run xxx -bench BenchmarkSlowWatcherTax -benchtime 1x -v
func BenchmarkSlowWatcherTax(b *testing.B) {
	registry := compbasemetrics.NewKubeRegistry()
	// The cacher's own instruments; an unregistered vector records nothing.
	for _, m := range []compbasemetrics.Registerable{
		metrics.DispatchStageDuration, metrics.TerminatedWatchersCounter,
		metrics.WatcherStalls, metrics.WatcherDeferredEvents, metrics.WatcherCatchupRounds,
	} {
		if err := registry.Register(m); err != nil {
			b.Fatal(err)
		}
	}

	scenarios := []slowWatcherScenario{
		{name: "baseline", eventsPerSecond: 100},
		{name: "draining-companion", eventsPerSecond: 100, companion: true, drains: true, reconnects: true},
		{name: "stalled-once", eventsPerSecond: 100, companion: true},
		{name: "stalled-reconnecting", eventsPerSecond: 100, companion: true, reconnects: true},
		{name: "baseline-1000", eventsPerSecond: 1000},
		{name: "stalled-reconnecting-1000", eventsPerSecond: 1000, companion: true, reconnects: true},
	}
	for _, gateOn := range []bool{false, true} {
		gateName := "gate=off"
		if gateOn {
			gateName = "gate=on"
		}
		b.Run(gateName, func(b *testing.B) {
			for _, scenario := range scenarios {
				b.Run(scenario.name, func(b *testing.B) {
					// The cacher reads the gate at construction.
					setStallResumeGate(b, gateOn)
					var r slowWatcherResult
					for b.Loop() {
						r = runSlowWatcherScenario(b, registry, scenario)
					}
					b.ReportMetric(float64(r.percentile(0.5).Microseconds()), "p50-us")
					b.ReportMetric(float64(r.percentile(0.9).Microseconds()), "p90-us")
					b.ReportMetric(float64(r.percentile(0.99).Microseconds()), "p99-us")
					b.ReportMetric(float64(r.percentile(1.0).Microseconds()), "max-us")
					b.ReportMetric(float64(r.slowDispatches), "slow-dispatches")
					b.ReportMetric(float64(r.terminated), "force-closed")
					b.ReportMetric(float64(r.incomingHWM), "incoming-hwm")
					if gateOn {
						b.ReportMetric(r.stalls, "stalls")
						b.ReportMetric(r.deferredEvents, "deferred-events")
						b.ReportMetric(r.catchupRounds, "catchup-rounds")
					}
				})
			}
		})
	}
}

const (
	// slowWatcherScenarioDuration gives 1000 samples at 100 events/s, enough
	// to place p99 on a real sample rather than on the max.
	slowWatcherScenarioDuration = 10 * time.Second
	// slowWatcherReconnectEvery is slower than client-go, which re-watches
	// almost immediately, so the reconnecting scenarios are conservative.
	slowWatcherReconnectEvery = 500 * time.Millisecond
	// slowWatcherBudgetWarmup lets the dispatch budget fill from its empty
	// initial state (maxBudget / refreshPerSecond = 2s), as in a long-lived
	// production cacher when a client wedges.
	slowWatcherBudgetWarmup = 2500 * time.Millisecond
	// slowWatcherDrainGrace outwaits a blocked send still sleeping on its
	// budget timer at cacher stop, so its force close lands in this scenario.
	slowWatcherDrainGrace = 2 * maxBudget
	// slowDispatchGate is a bucket boundary of DispatchStageDuration.
	slowDispatchGate = 5 * time.Millisecond
)

type slowWatcherScenario struct {
	name            string
	eventsPerSecond int
	companion       bool // a second watcher exists
	drains          bool // the companion reads its result channel
	reconnects      bool // the companion re-dials every slowWatcherReconnectEvery, one alive at a time
}

type slowWatcherResult struct {
	sortedLatencies []time.Duration
	slowDispatches  uint64
	terminated      int
	incomingHWM     int64
	// Stall instruments, always zero with the gate off.
	stalls, deferredEvents, catchupRounds float64
}

func (r slowWatcherResult) percentile(p float64) time.Duration {
	return r.sortedLatencies[int(float64(len(r.sortedLatencies)-1)*p)]
}

func runSlowWatcherScenario(b *testing.B, registry compbasemetrics.KubeRegistry, scenario slowWatcherScenario) slowWatcherResult {
	totalEvents := scenario.eventsPerSecond * int(slowWatcherScenarioDuration/time.Second)

	// Sized so that injection never blocks and skews the injection timestamps.
	fw := watch.NewFakeWithChanSize(totalEvents+10, false)
	backing := &cachertesting.MockStorage{
		WatchFn: func(_ context.Context, _ string, _ storage.ListOptions) (watch.Interface, error) {
			return fw, nil
		},
	}
	cacher, _, err := newTestCacher(backing)
	if err != nil {
		b.Fatal(err)
	}
	defer cacher.Stop()

	time.Sleep(slowWatcherBudgetWarmup)
	before := snapshotSlowWatcherMetrics(b, registry)

	// The newest resourceVersion the injector has published. Companions
	// (re)connect from here, with no history to replay.
	var lastRV atomic.Int64
	lastRV.Store(100)
	newWatch := func() (watch.Interface, error) {
		return cacher.Watch(context.Background(), "/pods/ns", storage.ListOptions{
			ResourceVersion: fmt.Sprintf("%d", lastRV.Load()),
			Predicate:       storage.Everything,
		})
	}

	healthy, err := newWatch()
	if err != nil {
		b.Fatal(err)
	}
	defer healthy.Stop()

	if scenario.companion {
		stopCompanion := startSlowWatcherCompanion(b, scenario, newWatch)
		defer stopCompanion()
	}

	injected := make([]time.Time, totalEvents)
	stopInjector := make(chan struct{})
	var injector sync.WaitGroup
	injector.Go(func() {
		// Events go out on an absolute schedule from start, not on a ticker:
		// a ticker drops ticks when this goroutine falls behind, which would
		// lower the injection rate below eventsPerSecond.
		interval := time.Second / time.Duration(scenario.eventsPerSecond)
		start := time.Now()
		slot := time.NewTimer(interval)
		defer slot.Stop()
		for i := range totalEvents {
			slot.Reset(time.Until(start.Add(time.Duration(i+1) * interval)))
			select {
			case <-stopInjector:
				return
			case <-slot.C:
			}
			injected[i] = time.Now()
			fw.Add(&example.Pod{
				Name:            fmt.Sprintf("pod-%06d", i),
				Namespace:       "ns",
				ResourceVersion: fmt.Sprintf("%d", 101+i),
			})
			lastRV.Store(int64(101 + i))
		}
	})
	// Runs before cacher.Stop (defers are LIFO): the reflector closes the
	// fake watcher on stop, and fw.Add on a closed watcher panics.
	defer func() {
		close(stopInjector)
		injector.Wait()
	}()

	deadline := time.NewTimer(2 * slowWatcherScenarioDuration)
	defer deadline.Stop()
	latencies := make([]time.Duration, 0, totalEvents)
	for len(latencies) < totalEvents {
		var ev watch.Event
		var ok bool
		select {
		case ev, ok = <-healthy.ResultChan():
		case <-deadline.C:
			b.Fatalf("%s: received %d of %d events within %v", scenario.name, len(latencies), totalEvents, 2*slowWatcherScenarioDuration)
		}
		if !ok {
			b.Fatalf("%s: the healthy watcher was force closed after %d of %d events; the machine is too loaded to drain %d events/s",
				scenario.name, len(latencies), totalEvents, scenario.eventsPerSecond)
		}
		if ev.Type != watch.Added {
			continue
		}
		// Events arrive in injection order: one fake watcher channel, one
		// dispatch goroutine, one input channel per watcher. With the gate
		// on, a catch-up round streams the history in resourceVersion order
		// behind a strictly increasing resume position, so the order holds
		// even if this watcher stalls. So the k-th Added event is pod k, and
		// the loop bound keeps k below totalEvents.
		latencies = append(latencies, time.Since(injected[len(latencies)]))
	}
	slices.Sort(latencies)

	injector.Wait()
	cacher.Stop()
	time.Sleep(slowWatcherDrainGrace)
	// Deltas over this scenario only: the vectors are global and shared with other tests.
	after := snapshotSlowWatcherMetrics(b, registry)
	return slowWatcherResult{
		sortedLatencies: latencies,
		slowDispatches:  after.slowDispatches - before.slowDispatches,
		terminated:      after.terminated - before.terminated,
		incomingHWM:     atomic.LoadInt64((*int64)(&cacher.incomingHWM)),
		stalls:          after.stalls - before.stalls,
		deferredEvents:  after.deferredEvents - before.deferredEvents,
		catchupRounds:   after.catchupRounds - before.catchupRounds,
	}
}

// startSlowWatcherCompanion opens the companion watcher and, for a
// reconnecting scenario, re-dials it on a ticker so that exactly one
// companion is alive at any moment. The returned func stops everything it
// started.
func startSlowWatcherCompanion(b *testing.B, scenario slowWatcherScenario, newWatch func() (watch.Interface, error)) func() {
	var drainers sync.WaitGroup
	maybeDrain := func(w watch.Interface) {
		if !scenario.drains {
			// A stalled companion never reads.
			return
		}
		drainers.Go(func() {
			for range w.ResultChan() {
			}
		})
	}

	companion, err := newWatch()
	if err != nil {
		b.Fatal(err)
	}
	maybeDrain(companion)
	if !scenario.reconnects {
		return func() {
			companion.Stop()
			drainers.Wait()
		}
	}

	done := make(chan struct{})
	var reconnector sync.WaitGroup
	reconnector.Go(func() {
		ticker := time.NewTicker(slowWatcherReconnectEvery)
		defer ticker.Stop()
		for {
			select {
			case <-done:
				companion.Stop()
				return
			case <-ticker.C:
				companion.Stop()
				w, err := newWatch()
				if err != nil {
					// The cacher is shutting down; the next tick or done ends the loop.
					continue
				}
				companion = w
				maybeDrain(w)
			}
		}
	})
	return func() {
		close(done)
		reconnector.Wait()
		drainers.Wait()
	}
}

type slowWatcherMetrics struct {
	slowDispatches uint64 // stage="total" observations above slowDispatchGate
	terminated     int
	// WatchCacheStallResume instruments.
	stalls, deferredEvents, catchupRounds float64
}

func snapshotSlowWatcherMetrics(b *testing.B, registry compbasemetrics.KubeRegistry) slowWatcherMetrics {
	families, err := registry.Gather()
	if err != nil {
		b.Fatal(err)
	}
	var s slowWatcherMetrics
	// counterTotal sums a counter family over all its label sets. A closure
	// keeps the prometheus client_model types out of this file
	// (hack/verify-prometheus-imports.sh).
	counterTotal := func(name string) float64 {
		var total float64
		for _, mf := range families {
			if mf.GetName() != name {
				continue
			}
			for _, m := range mf.GetMetric() {
				total += m.GetCounter().GetValue()
			}
		}
		return total
	}
	s.stalls = counterTotal("apiserver_watch_cache_watcher_stalls_total")
	s.deferredEvents = counterTotal("apiserver_watch_cache_watcher_deferred_events_total")
	s.catchupRounds = counterTotal("apiserver_watch_cache_watcher_catchup_rounds_total")
	for _, mf := range families {
		switch mf.GetName() {
		case "apiserver_watch_events_dispatch_duration_seconds":
			for _, m := range mf.GetMetric() {
				if !testutil.LabelsMatch(m, map[string]string{"stage": "total"}) {
					continue
				}
				h := m.GetHistogram()
				var under uint64
				found := false
				for _, bucket := range h.GetBucket() {
					if bucket.GetUpperBound() == slowDispatchGate.Seconds() {
						under = bucket.GetCumulativeCount()
						found = true
					}
				}
				if !found {
					b.Fatalf("apiserver_watch_events_dispatch_duration_seconds has no bucket boundary at %v", slowDispatchGate)
				}
				s.slowDispatches += h.GetSampleCount() - under
			}
		case "apiserver_terminated_watchers_total":
			for _, m := range mf.GetMetric() {
				s.terminated += int(m.GetCounter().GetValue())
			}
		}
	}
	return s
}
