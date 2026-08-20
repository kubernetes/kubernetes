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
	"sort"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	dto "github.com/prometheus/client_model/go"

	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/apis/example"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/cacher/metrics"
	cachertesting "k8s.io/apiserver/pkg/storage/cacher/testing"
	compbasemetrics "k8s.io/component-base/metrics"
)

// BenchmarkSlowWatcherTax measures how much one slow watch client delays
// event delivery to every other watcher of the same resource.
//
// A single goroutine (dispatchEvents) delivers every event to every
// watcher. Delivery is a non-blocking send into the watcher's input
// channel (chanSize 10 for an unindexed resource with the default history
// window). A watcher whose channel is full is retried with a blocking send
// bounded by a shared time budget (maxBudget 100ms, refilled at 50ms/s).
// While the dispatch goroutine blocks on one watcher, no other watcher
// receives anything. When the budget runs dry the blocked watcher is force
// closed and apiserver_terminated_watchers_total is incremented. The budget
// bounds the tax, it does not remove it: a wedged client that reconnects
// becomes a permanent duty-cycle stall of the dispatch goroutine, invisible
// at p50 and concentrated at p99 and max. At higher event rates the stalls
// back up the cacher's incoming channel (capacity 100), at which point the
// tax becomes staleness for every consumer of the cache, not only watchers.
//
// The benchmark builds a real Cacher over an in-memory storage for an
// unindexed resource, injects Added events through the cacher's own
// reflector, the same path etcd events take, and measures injection to
// delivery latency at one healthy watcher that always drains its channel.
// One healthy watcher is enough as the witness: the dispatch goroutine is
// the only delivery path, so its delay is everyone's delay.
//
// The slow companion never reads its result channel. A real wedged TCP
// peer looks healthier for a moment, because the serving goroutine and the
// kernel socket buffers absorb some events first, but once that runway
// fills this is the state the cacher sees. Reconnecting companions watch
// from the newest published resourceVersion, which is conservative: a real
// client's older last-acked resourceVersion would replay history into its
// unread buffer and jam sooner.
//
// Two evidence channels are reported per sub-benchmark: latency percentiles
// at the healthy watcher (p50-us, p90-us, p99-us, max-us) and the cacher's
// own instruments over the cell (slow-dispatches: stage="total" dispatch
// observations above slowDispatchGate, force-closed: terminated watchers,
// incoming-hwm: high water mark of the incoming channel).
//
// Every sub-benchmark runs one wall-clock cell per b.N iteration and
// reports the last iteration. The latency distribution needs wall-clock
// time, so the intended invocation is -benchtime 1x:
//
//	go test ./staging/src/k8s.io/apiserver/pkg/storage/cacher/ -run xxx -bench BenchmarkSlowWatcherTax -benchtime 1x -v
//
// Every cell runs once per WatchCacheStallResume gate state (gate=off and
// gate=on). With the gate on the dispatcher never blocks on a full
// watcher channel: the watcher catches up from the watch cache history
// ring on its own goroutine, and it is terminated only when the event it
// missed has already aged out of that history (an in-stream 410). The
// gate=on cells additionally report the stall instruments (stalls,
// deferred-events, catchup-rounds). To run one state only:
//
//	-bench 'BenchmarkSlowWatcherTax/gate=on'
func BenchmarkSlowWatcherTax(b *testing.B) {
	registry := compbasemetrics.NewKubeRegistry()
	// Registering creates the vectors; an unregistered vector discards
	// observations. The delta is computed per cell, so earlier tests that
	// touched the same global vectors do not matter.
	for _, m := range []compbasemetrics.Registerable{
		metrics.DispatchStageDuration, metrics.TerminatedWatchersCounter,
		metrics.WatcherStalls, metrics.WatcherDeferredEvents, metrics.WatcherCatchupRounds,
	} {
		if err := registry.Register(m); err != nil {
			b.Fatal(err)
		}
	}

	cells := []slowWatcherCell{
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
			for _, cell := range cells {
				b.Run(cell.name, func(b *testing.B) {
					// The cacher reads the gate at construction, so it is set
					// before the cell builds its cacher and restored by the
					// sub-benchmark's cleanup.
					setStallResumeGate(b, gateOn)
					var r slowWatcherResult
					for i := 0; i < b.N; i++ {
						r = runSlowWatcherCell(b, registry, cell)
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
					b.Logf("%s: %d of %d dispatches above %v, %d watcher(s) force closed, incoming high water mark %d",
						cell.name, r.slowDispatches, r.allDispatches, slowDispatchGate, r.terminated, r.incomingHWM)
				})
			}
		})
	}
}

const (
	// slowWatcherCellDuration gives 1000 samples at 100 events/s, enough to
	// place p99 on a real sample rather than on the max.
	slowWatcherCellDuration = 10 * time.Second
	// slowWatcherReconnectEvery is slower than client-go, which re-watches
	// almost immediately, so the reconnecting cells are conservative.
	slowWatcherReconnectEvery = 500 * time.Millisecond
	// slowWatcherBudgetWarmup lets the dispatch budget fill from its empty
	// initial state (maxBudget / refreshPerSecond = 2s). A long-lived
	// production cacher holds a full budget when a client wedges, and a
	// cold budget would understate the tax. Applied to every cell.
	slowWatcherBudgetWarmup = 2500 * time.Millisecond
	// slowWatcherDrainGrace outwaits a blocked send that is still sleeping
	// on its budget timer when the cacher stops, so its force close lands in
	// this cell's metrics delta and not in the next one.
	slowWatcherDrainGrace = 2 * maxBudget
	// slowDispatchGate is a bucket boundary of DispatchStageDuration.
	slowDispatchGate = 5 * time.Millisecond
)

type slowWatcherCell struct {
	name            string
	eventsPerSecond int
	companion       bool // a second watcher exists
	drains          bool // the companion reads its result channel
	reconnects      bool // the companion re-dials every slowWatcherReconnectEvery, one alive at a time
}

type slowWatcherResult struct {
	sortedLatencies []time.Duration
	slowDispatches  uint64
	allDispatches   uint64
	terminated      int
	incomingHWM     int64
	// Stall instruments, always zero with the gate off.
	stalls, deferredEvents, catchupRounds float64
}

func (r slowWatcherResult) percentile(p float64) time.Duration {
	return r.sortedLatencies[int(float64(len(r.sortedLatencies)-1)*p)]
}

func runSlowWatcherCell(b *testing.B, registry compbasemetrics.KubeRegistry, cell slowWatcherCell) slowWatcherResult {
	totalEvents := cell.eventsPerSecond * int(slowWatcherCellDuration/time.Second)

	// The injector must never block: a stall anywhere downstream must not
	// distort the injection timestamps.
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

	if cell.companion {
		stopCompanion := startSlowWatcherCompanion(b, cell, newWatch)
		defer stopCompanion()
	}

	injected := make([]time.Time, totalEvents)
	stopInjector := make(chan struct{})
	var injector sync.WaitGroup
	injector.Add(1)
	go func() {
		defer injector.Done()
		ticker := time.NewTicker(time.Second / time.Duration(cell.eventsPerSecond))
		defer ticker.Stop()
		for i := 0; i < totalEvents; i++ {
			select {
			case <-stopInjector:
				return
			case <-ticker.C:
			}
			injected[i] = time.Now()
			fw.Add(&example.Pod{ObjectMeta: metav1.ObjectMeta{
				Name:            fmt.Sprintf("pod-%06d", i),
				Namespace:       "ns",
				ResourceVersion: fmt.Sprintf("%d", 101+i),
			}})
			lastRV.Store(int64(101 + i))
		}
	}()
	// Runs before cacher.Stop (defers are LIFO): the reflector closes the
	// fake watcher on stop, and fw.Add on a closed watcher panics.
	defer func() {
		close(stopInjector)
		injector.Wait()
	}()

	deadline := time.NewTimer(2 * slowWatcherCellDuration)
	defer deadline.Stop()
	latencies := make([]time.Duration, 0, totalEvents)
	for len(latencies) < totalEvents {
		var ev watch.Event
		var ok bool
		select {
		case ev, ok = <-healthy.ResultChan():
		case <-deadline.C:
			b.Fatalf("%s: received %d of %d events within %v", cell.name, len(latencies), totalEvents, 2*slowWatcherCellDuration)
		}
		if !ok {
			b.Fatalf("%s: the healthy watcher was force closed after %d of %d events; the machine is too loaded to drain %d events/s",
				cell.name, len(latencies), totalEvents, cell.eventsPerSecond)
		}
		if ev.Type != watch.Added {
			continue
		}
		acc, err := meta.Accessor(ev.Object)
		if err != nil {
			b.Fatal(err)
		}
		var i int
		if n, err := fmt.Sscanf(acc.GetName(), "pod-%06d", &i); n != 1 || err != nil {
			b.Fatalf("unexpected object name %q", acc.GetName())
		}
		latencies = append(latencies, time.Since(injected[i]))
	}
	sort.Slice(latencies, func(i, j int) bool { return latencies[i] < latencies[j] })

	injector.Wait()
	cacher.Stop()
	time.Sleep(slowWatcherDrainGrace)
	after := snapshotSlowWatcherMetrics(b, registry)
	return slowWatcherResult{
		sortedLatencies: latencies,
		slowDispatches:  after.slowDispatches - before.slowDispatches,
		allDispatches:   after.allDispatches - before.allDispatches,
		terminated:      after.terminated - before.terminated,
		incomingHWM:     atomic.LoadInt64((*int64)(&cacher.incomingHWM)),
		stalls:          after.stalls - before.stalls,
		deferredEvents:  after.deferredEvents - before.deferredEvents,
		catchupRounds:   after.catchupRounds - before.catchupRounds,
	}
}

// startSlowWatcherCompanion opens the companion watcher and, for a
// reconnecting cell, re-dials it on a ticker so that exactly one companion
// is alive at any moment. The returned func stops everything it started.
func startSlowWatcherCompanion(b *testing.B, cell slowWatcherCell, newWatch func() (watch.Interface, error)) func() {
	var drainers sync.WaitGroup
	maybeDrain := func(w watch.Interface) {
		if !cell.drains {
			// A stalled companion never reads; see the benchmark comment.
			return
		}
		drainers.Add(1)
		go func() {
			defer drainers.Done()
			for range w.ResultChan() {
			}
		}()
	}

	companion, err := newWatch()
	if err != nil {
		b.Fatal(err)
	}
	maybeDrain(companion)
	if !cell.reconnects {
		return func() {
			companion.Stop()
			drainers.Wait()
		}
	}

	done := make(chan struct{})
	var reconnector sync.WaitGroup
	reconnector.Add(1)
	go func() {
		defer reconnector.Done()
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
	}()
	return func() {
		close(done)
		reconnector.Wait()
		drainers.Wait()
	}
}

type slowWatcherMetrics struct {
	slowDispatches uint64 // stage="total" observations above slowDispatchGate
	allDispatches  uint64 // one per delivered event per watcher
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
	for _, mf := range families {
		switch mf.GetName() {
		case "apiserver_watch_events_dispatch_duration_seconds":
			for _, m := range mf.GetMetric() {
				if !hasMetricLabel(m, "stage", "total") {
					continue
				}
				h := m.GetHistogram()
				s.allDispatches += h.GetSampleCount()
				var under uint64
				for _, bucket := range h.GetBucket() {
					if bucket.GetUpperBound() <= slowDispatchGate.Seconds() {
						under = bucket.GetCumulativeCount()
					}
				}
				s.slowDispatches += h.GetSampleCount() - under
			}
		case "apiserver_terminated_watchers_total":
			for _, m := range mf.GetMetric() {
				s.terminated += int(m.GetCounter().GetValue())
			}
		case "apiserver_watch_cache_watcher_stalls_total":
			s.stalls += sumCounters(mf)
		case "apiserver_watch_cache_watcher_deferred_events_total":
			s.deferredEvents += sumCounters(mf)
		case "apiserver_watch_cache_watcher_catchup_rounds_total":
			s.catchupRounds += sumCounters(mf)
		}
	}
	return s
}

func sumCounters(mf *dto.MetricFamily) float64 {
	var total float64
	for _, m := range mf.GetMetric() {
		total += m.GetCounter().GetValue()
	}
	return total
}

func hasMetricLabel(m *dto.Metric, name, value string) bool {
	for _, l := range m.GetLabel() {
		if l.GetName() == name && l.GetValue() == value {
			return true
		}
	}
	return false
}
