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
// It needs wall-clock time; run it with:
//
//	go test ./staging/src/k8s.io/apiserver/pkg/storage/cacher/ -run xxx -bench BenchmarkSlowWatcherTax -benchtime 1x -v
func BenchmarkSlowWatcherTax(b *testing.B) {
	registry := compbasemetrics.NewKubeRegistry()
	// The cacher's own instruments; an unregistered vector records nothing.
	for _, m := range []compbasemetrics.Registerable{metrics.DispatchStageDuration, metrics.TerminatedWatchersCounter} {
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
	for _, scenario := range scenarios {
		b.Run(scenario.name, func(b *testing.B) {
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
		})
	}
}

const (
	// slowWatcherScenarioDuration gives 1000 samples at 100 events/s, enough
	// to place p99 on a real sample rather than on the max.
	slowWatcherScenarioDuration = 10 * time.Second
	// slowWatcherReconnectEvery gives each stalled companion time to fill its
	// channels and block the dispatcher: up to 460 ms at 100 events/s.
	slowWatcherReconnectEvery = 500 * time.Millisecond
	// slowWatcherBudgetWarmup fills the dispatch budget, which starts empty.
	slowWatcherBudgetWarmup = maxBudget / refreshPerSecond * time.Second
	// slowWatcherDrainGrace lets the last force close land in this scenario: a
	// blocked send waits for up to maxBudget.
	slowWatcherDrainGrace = 2 * maxBudget
	// slowDispatchGate is a bucket boundary of DispatchStageDuration.
	slowDispatchGate = 5 * time.Millisecond
)

type slowWatcherScenario struct {
	name            string
	eventsPerSecond int
	companion       bool // a second watcher exists
	drains          bool // the companion reads its result channel
	reconnects      bool // a new companion replaces it every slowWatcherReconnectEvery
}

type slowWatcherResult struct {
	sortedLatencies []time.Duration
	slowDispatches  uint64
	terminated      int
	incomingHWM     int64
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

	// latestRV is the newest resourceVersion: 100 from the initial list, then
	// each injected event. watchFromLatestRV reads it on every call, so each
	// watch it opens, every companion reconnect included, starts at the newest
	// event and replays at most one event.
	var latestRV atomic.Int64
	latestRV.Store(100)
	watchFromLatestRV := func() (watch.Interface, error) {
		return cacher.Watch(context.Background(), "/pods/ns", storage.ListOptions{
			ResourceVersion: fmt.Sprintf("%d", latestRV.Load()),
			Predicate:       storage.Everything,
		})
	}

	healthy, err := watchFromLatestRV()
	if err != nil {
		b.Fatal(err)
	}
	defer healthy.Stop()

	// Both the call before cacher.Stop below and this defer run stopCompanion;
	// the defer also covers a b.Fatalf that ends the scenario early. stop
	// closes a channel, so the second call must do nothing.
	stopCompanion := func() {}
	if scenario.companion {
		stopCompanion = sync.OnceFunc(openCompanionWatcher(b, scenario, watchFromLatestRV))
	}
	defer stopCompanion()

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
			latestRV.Store(int64(101 + i))
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
		// dispatch goroutine, one input channel per watcher. So the k-th
		// Added event is pod k, and the loop bound keeps k below totalEvents.
		latencies = append(latencies, time.Since(injected[len(latencies)]))
	}
	slices.Sort(latencies)

	injector.Wait()
	// Before cacher.Stop, so that a failed reconnect is a real failure and no
	// "Terminating all watchers" log line splits the benchmark output.
	stopCompanion()
	healthy.Stop()
	cacher.Stop()
	time.Sleep(slowWatcherDrainGrace)
	// Deltas over this scenario only: the vectors are global and shared with other tests.
	after := snapshotSlowWatcherMetrics(b, registry)
	terminated := after.terminated - before.terminated
	if scenario.companion && !scenario.drains && terminated == 0 {
		b.Fatalf("%s: no companion was force closed, so none blocked the dispatcher and the scenario measured nothing", scenario.name)
	}
	return slowWatcherResult{
		sortedLatencies: latencies,
		slowDispatches:  after.slowDispatches - before.slowDispatches,
		terminated:      terminated,
		incomingHWM:     atomic.LoadInt64((*int64)(&cacher.incomingHWM)),
	}
}

// openCompanionWatcher opens the companion watcher and returns the func that stops it.
func openCompanionWatcher(b *testing.B, scenario slowWatcherScenario, newWatch func() (watch.Interface, error)) (stop func()) {
	var drained atomic.Int64
	var drainers sync.WaitGroup
	maybeDrain := func(w watch.Interface) {
		if !scenario.drains {
			return
		}
		drainers.Go(func() {
			for range w.ResultChan() {
				drained.Add(1)
			}
		})
	}

	companion, err := newWatch()
	if err != nil {
		b.Fatal(err)
	}
	maybeDrain(companion)

	done := make(chan struct{})
	var reconnector sync.WaitGroup
	if scenario.reconnects {
		reconnector.Go(func() {
			next := time.Now()
			for {
				next = next.Add(slowWatcherReconnectEvery)
				select {
				case <-done:
					return
				case <-time.After(time.Until(next)):
				}
				companion.Stop()
				w, err := newWatch()
				if err != nil {
					b.Errorf("%s: the companion failed to reconnect: %v", scenario.name, err)
					return
				}
				companion = w
				maybeDrain(w)
			}
		})
	}
	return func() {
		close(done)
		reconnector.Wait()
		companion.Stop()
		drainers.Wait()
		if scenario.drains && drained.Load() == 0 {
			b.Errorf("%s: the companion drained no events", scenario.name)
		}
	}
}

type slowWatcherMetrics struct {
	slowDispatches uint64 // stage="total" observations above slowDispatchGate
	terminated     int
}

func snapshotSlowWatcherMetrics(b *testing.B, registry compbasemetrics.KubeRegistry) slowWatcherMetrics {
	families, err := registry.Gather()
	if err != nil {
		b.Fatal(err)
	}
	var s slowWatcherMetrics
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
