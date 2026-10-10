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

// Tests for the WatchCacheStallResume feature gate: a watcher whose input
// channel is full when an event is dispatched becomes unsynced and is served
// from the watch cache history by the dispatcher's sync passes; it is
// terminated only when its position has aged out of the history, and then
// with an in-stream 410.
//
// The tests run the cacher on a fake clock, so no pass runs on a timer
// unless the test steps the clock, and drive passes through the
// dispatcher hook, which runs a function on the dispatcher goroutine. The
// dispatcher goroutine is the only writer of a watcher's position and
// unsynced fields, so the hook is also the only race free way to read them.

import (
	"context"
	"fmt"
	goruntime "runtime"
	"slices"
	"strconv"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/apis/example"
	examplev1 "k8s.io/apiserver/pkg/apis/example/v1"
	"k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/cacher/metrics"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	compbasemetrics "k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/testutil"
	"k8s.io/utils/clock"
	testingclock "k8s.io/utils/clock/testing"

	cachertesting "k8s.io/apiserver/pkg/storage/cacher/testing"
)

var stallResumeMetricsOnce sync.Once

// ensureStallResumeMetrics instantiates the stall/resume metric vectors (they
// are lazily created on registration) so tests can read their values.
func ensureStallResumeMetrics() {
	stallResumeMetricsOnce.Do(func() {
		registry := compbasemetrics.NewKubeRegistry()
		for _, m := range []compbasemetrics.Registerable{
			metrics.WatcherStalls, metrics.WatcherDeferredEvents, metrics.WatcherCatchupRounds,
			metrics.WatcherCatchupEvents, metrics.TerminatedWatchersCounter,
		} {
			_ = registry.Register(m)
		}
	})
}

// stallResumeCounters is a snapshot of the stall/resume counters for pods.
type stallResumeCounters struct {
	stalls, deferred, rounds float64
	roundSamples             uint64
	roundSum                 float64
	expired, expiredInitial  float64
	unresponsive             float64
}

func readStallResumeCounters(t testing.TB) stallResumeCounters {
	t.Helper()
	ensureStallResumeMetrics()
	read := func(m compbasemetrics.CounterMetric) float64 {
		v, err := testutil.GetCounterMetricValue(m)
		if err != nil {
			t.Fatalf("reading counter: %v", err)
		}
		return v
	}
	histogram := metrics.WatcherCatchupEvents.WithLabelValues("", "pods")
	roundSamples, err := testutil.GetHistogramMetricCount(histogram)
	if err != nil {
		t.Fatalf("reading histogram: %v", err)
	}
	roundSum, err := testutil.GetHistogramMetricValue(histogram)
	if err != nil {
		t.Fatalf("reading histogram: %v", err)
	}
	return stallResumeCounters{
		roundSamples:   roundSamples,
		roundSum:       roundSum,
		stalls:         read(metrics.WatcherStalls.WithLabelValues("", "pods")),
		deferred:       read(metrics.WatcherDeferredEvents.WithLabelValues("", "pods")),
		rounds:         read(metrics.WatcherCatchupRounds.WithLabelValues("", "pods")),
		expired:        read(metrics.TerminatedWatchersCounter.WithLabelValues("", "pods", metrics.TerminationReasonResourceExpired)),
		expiredInitial: read(metrics.TerminatedWatchersCounter.WithLabelValues("", "pods", metrics.TerminationReasonResourceExpiredInitial)),
		unresponsive:   read(metrics.TerminatedWatchersCounter.WithLabelValues("", "pods", metrics.TerminationReasonUnresponsive)),
	}
}

// setStallResumeGate enables or disables WatchCacheStallResume for a test.
func setStallResumeGate(t testing.TB, enabled bool) {
	t.Helper()
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.WatchCacheStallResume, enabled)
}

// newStallResumeCacher enables the feature gate for the duration of the test
// and builds a cacher over MockStorage on a fake clock, seeded with one pod
// at resourceVersion 100 so watches can start there.
func newStallResumeCacher(t *testing.T, mutators ...func(*Config)) (*Cacher, *testingclock.FakeClock) {
	t.Helper()
	setStallResumeGate(t, true)
	ensureStallResumeMetrics()
	clk := testingclock.NewFakeClock(time.Now())
	cacher, _, err := newTestCacherWithoutSyncing(&cachertesting.MockStorage{}, clk, mutators...)
	if err != nil {
		t.Fatalf("Couldn't create cacher: %v", err)
	}
	if err := cacher.Wait(context.Background()); err != nil {
		t.Fatalf("cacher never became ready: %v", err)
	}
	if cacher.stall == nil {
		t.Fatalf("expected the cacher to run in stall/resume mode")
	}
	t.Cleanup(cacher.Stop)
	stallResumeAddPods(t, cacher, "ns", 100, 100)
	waitDispatched(t, cacher)
	return cacher, clk
}

// waitDispatched polls until the dispatcher has dispatched everything the
// writer appended so far, so a watch opened next sees no backlog older than
// itself in its input.
func waitDispatched(t testing.TB, c *Cacher) {
	t.Helper()
	waitDispatcherState(t, c, nil, "incoming drained", func(*cacheWatcher) bool { return true })
}

// stallResumePod builds pod-<rv> in the namespace with the given resourceVersion.
func stallResumePod(namespace string, rv uint64) *examplev1.Pod {
	return &examplev1.Pod{
		Name:            fmt.Sprintf("pod-%d", rv),
		Namespace:       namespace,
		ResourceVersion: strconv.FormatUint(rv, 10),
	}
}

// stallResumeAddPods adds pod-<from>..pod-<to> in the namespace, each at its own RV.
func stallResumeAddPods(t testing.TB, c *Cacher, namespace string, from, to uint64) {
	t.Helper()
	for rv := from; rv <= to; rv++ {
		if err := c.watchCache.Add(stallResumePod(namespace, rv)); err != nil {
			t.Fatalf("failed to add a pod: %v", err)
		}
	}
}

// stallResumeWatch opens a watch on the namespace from the given
// resourceVersion. It is namespace scoped for the dispatcher (the request
// context names the namespace), which is how a client-go informer for one
// namespace registers.
func stallResumeWatch(t testing.TB, c *Cacher, namespace string, rv uint64) *cacheWatcher {
	t.Helper()
	pred := storage.Everything
	pred.AllowWatchBookmarks = true
	ctx := request.WithNamespace(context.Background(), namespace)
	w, err := c.Watch(ctx, "/pods/"+namespace, storage.ListOptions{ResourceVersion: strconv.FormatUint(rv, 10), Predicate: pred, Recursive: true})
	if err != nil {
		t.Fatalf("Failed to create watch: %v", err)
	}
	cw, ok := w.(*cacheWatcher)
	if !ok {
		t.Fatalf("expected a *cacheWatcher, got %T", w)
	}
	t.Cleanup(cw.Stop)
	return cw
}

// dispatcherDo runs fn on the dispatcher goroutine and returns once it is
// done. fn gets a func that runs one sync pass.
func dispatcherDo(t testing.TB, c *Cacher, fn func(runPass syncPassFunc)) {
	t.Helper()
	done := make(chan struct{})
	wrapped := func(runPass syncPassFunc) {
		defer close(done)
		fn(runPass)
	}
	select {
	case c.stall.dispatcherHook <- wrapped:
	case <-c.stopCh:
		t.Fatalf("the cacher is stopped")
	case <-time.After(10 * time.Second):
		t.Fatalf("the dispatcher goroutine never took the hook")
	}
	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatalf("the hook never finished")
	}
}

// runSyncPass runs one sync pass on the dispatcher goroutine.
func runSyncPass(t testing.TB, c *Cacher) (served, expired int) {
	t.Helper()
	dispatcherDo(t, c, func(runPass syncPassFunc) { served, expired = runPass() })
	return served, expired
}

// watcherState is a dispatcher-side snapshot of one watcher.
type watcherState struct {
	position uint64
	unsynced bool
	expired  bool
	inSet    bool
}

func snapshot(t testing.TB, c *Cacher, w *cacheWatcher) watcherState {
	t.Helper()
	var st watcherState
	dispatcherDo(t, c, func(syncPassFunc) {
		_, st.inSet = c.stall.unsynced[w]
		st.position, st.unsynced, st.expired = w.position, w.unsynced, w.expired
	})
	return st
}

// schedulerState reads the pass scheduler's state on the dispatcher goroutine.
func schedulerState(t testing.TB, c *Cacher) (period time.Duration, armed bool) {
	t.Helper()
	dispatcherDo(t, c, func(syncPassFunc) {
		period, armed = c.stall.passPeriod, c.stall.passArmed
	})
	return period, armed
}

// waitDispatcherState polls cond, evaluated on the dispatcher goroutine,
// until it holds for every watcher.
func waitDispatcherState(t testing.TB, c *Cacher, ws []*cacheWatcher, what string, cond func(*cacheWatcher) bool) {
	t.Helper()
	if err := wait.PollUntilContextTimeout(context.Background(), time.Millisecond, 10*time.Second, true, func(context.Context) (bool, error) {
		ok := true
		dispatcherDo(t, c, func(syncPassFunc) {
			// The hook runs between dispatches, so an empty incoming queue
			// here means every appended event was dispatched.
			ok = len(c.incoming) == 0
			for _, w := range ws {
				ok = ok && cond(w)
			}
		})
		return ok, nil
	}); err != nil {
		t.Fatalf("watchers never reached the state %q: %v", what, err)
	}
}

// fillWatchers writes events selected for every watcher in ws, starting at
// rv, until each holds cap(result) events in result, one in the goroutine's
// hand and cap(input) in input, and returns the next unused rv. The next
// selected event then misses every one of them, at exactly that rv. add
// writes one event at the given rv.
func fillWatchers(t testing.TB, c *Cacher, ws []*cacheWatcher, add func(rv uint64), rv uint64) uint64 {
	t.Helper()
	resultCap, inputCap := cap(ws[0].result), cap(ws[0].input)
	for range resultCap {
		add(rv)
		rv++
	}
	last := rv - 1
	waitDispatcherState(t, c, ws, "result full", func(w *cacheWatcher) bool {
		return w.position == last && len(w.result) == cap(w.result) && len(w.input) == 0
	})
	add(rv)
	inHand := rv
	rv++
	waitDispatcherState(t, c, ws, "one event in hand", func(w *cacheWatcher) bool {
		return w.position == inHand && len(w.input) == 0
	})
	for range inputCap {
		add(rv)
		rv++
	}
	last = rv - 1
	waitDispatcherState(t, c, ws, "input full", func(w *cacheWatcher) bool {
		return w.position == last && len(w.input) == cap(w.input)
	})
	return rv
}

// wedgeWatchers fills the watchers and writes the one event that misses all
// of them, then returns the next unused rv. Afterwards every watcher is
// unsynced at position rv-1.
func wedgeWatchers(t testing.TB, c *Cacher, ws []*cacheWatcher, add func(rv uint64), rv uint64) uint64 {
	t.Helper()
	rv = fillWatchers(t, c, ws, add, rv)
	add(rv)
	waitDispatcherState(t, c, ws, "unsynced", func(w *cacheWatcher) bool {
		return w.unsynced && w.position == rv-1
	})
	return rv + 1
}

// waitInputDrained polls until the watcher goroutine has taken everything
// out of input, so the next pass finds room.
func waitInputDrained(t testing.TB, c *Cacher, ws ...*cacheWatcher) {
	t.Helper()
	waitDispatcherState(t, c, ws, "input drained", func(w *cacheWatcher) bool {
		return len(w.input) == 0
	})
}

// collected is what a reader saw on one watch.
type collected struct {
	rvs       []uint64
	objects   map[uint64]runtime.Object
	types     map[uint64]watch.EventType
	bookmarks []uint64
	errors    []*metav1.Status
	closed    bool
}

// readUntil reads the watch until it has seen the object at resourceVersion
// until, the channel closes or the timeout fires. Bookmarks and errors are
// recorded, not counted.
func readUntil(t testing.TB, w watch.Interface, until uint64, timeout time.Duration) collected {
	t.Helper()
	got := collected{objects: map[uint64]runtime.Object{}, types: map[uint64]watch.EventType{}}
	deadline := time.After(timeout)
	for {
		select {
		case ev, ok := <-w.ResultChan():
			if !ok {
				got.closed = true
				return got
			}
			if ev.Type == watch.Error {
				status, ok := ev.Object.(*metav1.Status)
				if !ok {
					t.Fatalf("error event without a Status: %#v", ev.Object)
				}
				got.errors = append(got.errors, status)
				continue
			}
			rv, err := storage.APIObjectVersioner{}.ObjectResourceVersion(ev.Object)
			if err != nil {
				t.Fatalf("parsing resource version: %v", err)
			}
			if ev.Type == watch.Bookmark {
				got.bookmarks = append(got.bookmarks, rv)
				continue
			}
			got.rvs = append(got.rvs, rv)
			got.objects[rv] = ev.Object
			got.types[rv] = ev.Type
			if rv >= until {
				return got
			}
		case <-deadline:
			t.Fatalf("timed out waiting for events; got %d object events so far (last %v)", len(got.rvs), got.rvs)
		}
	}
}

// readAll reads everything the watch currently holds, until it is idle for
// idle, and returns what it saw. It does not wait for a close.
func readAll(t testing.TB, w watch.Interface, idle time.Duration) collected {
	t.Helper()
	got := collected{objects: map[uint64]runtime.Object{}, types: map[uint64]watch.EventType{}}
	for {
		select {
		case ev, ok := <-w.ResultChan():
			if !ok {
				got.closed = true
				return got
			}
			if ev.Type == watch.Error {
				status, ok := ev.Object.(*metav1.Status)
				if !ok {
					t.Fatalf("error event without a Status: %#v", ev.Object)
				}
				got.errors = append(got.errors, status)
				continue
			}
			rv, err := storage.APIObjectVersioner{}.ObjectResourceVersion(ev.Object)
			if err != nil {
				t.Fatalf("parsing resource version: %v", err)
			}
			if ev.Type == watch.Bookmark {
				got.bookmarks = append(got.bookmarks, rv)
				continue
			}
			got.rvs = append(got.rvs, rv)
			got.objects[rv] = ev.Object
			got.types[rv] = ev.Type
		case <-time.After(idle):
			return got
		}
	}
}

// assertExactSequence verifies rvs is exactly from, from+1, ..., to.
func assertExactSequence(t testing.TB, rvs []uint64, from, to uint64) {
	t.Helper()
	want := int(to - from + 1)
	if len(rvs) != want {
		t.Fatalf("expected %d object events (%d..%d), got %d: %v", want, from, to, len(rvs), rvs)
	}
	for i, rv := range rvs {
		if rv != from+uint64(i) {
			t.Fatalf("event %d: expected resourceVersion %d, got %d (sequence: %v)", i, from+uint64(i), rv, rvs)
		}
	}
}

// assertContiguous verifies every consecutive pair in rvs differs by one.
func assertContiguous(t testing.TB, rvs []uint64) {
	t.Helper()
	for i := 1; i < len(rvs); i++ {
		if rvs[i] != rvs[i-1]+1 {
			t.Fatalf("event %d: resourceVersion %d follows %d", i, rvs[i], rvs[i-1])
		}
	}
}

// pinHistoryCapacity fixes the watch cache history capacity so events age
// out deterministically once more than capacity newer events exist.
func pinHistoryCapacity(t *testing.T, c *Cacher, capacity int) {
	t.Helper()
	wc := c.watchCache
	wc.Lock()
	defer wc.Unlock()
	wc.history.lowerBoundCapacity = capacity
	wc.history.upperBoundCapacity = capacity
	if wc.history.capacity != capacity {
		wc.history.doCacheResizeLocked(capacity)
	}
}

// expectExpiredThenClose reads the watch until it closes and requires the
// sequence to end with exactly one 410 ResourceExpired ERROR event followed
// by the channel close. It returns the object RVs delivered before the error.
func expectExpiredThenClose(t *testing.T, w watch.Interface) []uint64 {
	t.Helper()
	got := readUntil(t, w, ^uint64(0), 15*time.Second)
	if !got.closed {
		t.Fatalf("watch did not close")
	}
	if len(got.errors) != 1 {
		t.Fatalf("expected exactly one error event, got %d", len(got.errors))
	}
	status := got.errors[0]
	if !apierrors.IsResourceExpired(apierrors.FromObject(status)) || status.Code != 410 {
		t.Fatalf("expected a 410 ResourceExpired status, got %#v", status)
	}
	return got.rvs
}

// resyncWhileReading runs passes until the watchers are synced again while
// readers drain them, and returns what each reader saw.
func resyncWhileReading(t *testing.T, c *Cacher, ws []*cacheWatcher, until uint64) []collected {
	t.Helper()
	results := make([]collected, len(ws))
	var readers sync.WaitGroup
	for i, w := range ws {
		readers.Go(func() { results[i] = readUntil(t, w, until, 20*time.Second) })
	}
	deadline := time.Now().Add(20 * time.Second)
	for {
		runSyncPass(t, c)
		synced := true
		dispatcherDo(t, c, func(syncPassFunc) {
			for _, w := range ws {
				synced = synced && !w.unsynced
			}
		})
		if synced {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("watchers never resynced")
		}
	}
	readers.Wait()
	return results
}

// TestStallResumeSurvivesStall is the core property: a client that stops
// reading while the writer produces far more than the watcher's buffers can
// hold is not terminated, loses nothing and sees strict resourceVersion order
// once it resumes; the counters record one stall, one resync and every
// event the passes pushed.
func TestStallResumeSurvivesStall(t *testing.T) {
	cacher, clk := newStallResumeCacher(t)
	w := stallResumeWatch(t, cacher, "ns", 100)
	before := readStallResumeCounters(t)
	add := func(rv uint64) { stallResumeAddPods(t, cacher, "ns", rv, rv) }

	next := wedgeWatchers(t, cacher, []*cacheWatcher{w}, add, 101)
	missed := next - 1
	stallResumeAddPods(t, cacher, "ns", next, 500)

	st := snapshot(t, cacher, w)
	if !st.unsynced || !st.inSet || st.position != missed-1 {
		t.Fatalf("unexpected state after the miss: %+v", st)
	}
	// The inline pass at the stall found no candidate (the input was
	// full), but the stall keeps the short period so the first catch-up
	// is not delayed by the idle backoff.
	if period, armed := schedulerState(t, cacher); period != syncPassPeriod || !armed {
		t.Fatalf("expected the short pass period and an armed timer right after a stall, got %v armed=%v", period, armed)
	}

	got := resyncWhileReading(t, cacher, []*cacheWatcher{w}, 500)[0]
	assertExactSequence(t, got.rvs, 101, 500)
	if len(got.bookmarks) != 0 {
		t.Fatalf("no bookmark is delivered while unsynced, got %v", got.bookmarks)
	}
	st = snapshot(t, cacher, w)
	if st.unsynced || st.inSet || st.position != 500 {
		t.Fatalf("unexpected state after the resync: %+v", st)
	}

	after := readStallResumeCounters(t)
	pushed := float64(500 - (missed - 1))
	if after.stalls-before.stalls != 1 {
		t.Errorf("stalls: want +1, got %v", after.stalls-before.stalls)
	}
	if after.rounds-before.rounds != 1 {
		t.Errorf("catch-up rounds: want +1, got %v", after.rounds-before.rounds)
	}
	if after.deferred-before.deferred != pushed {
		t.Errorf("deferred events: want +%v, got %v", pushed, after.deferred-before.deferred)
	}
	if after.roundSamples-before.roundSamples != 1 || after.roundSum-before.roundSum != pushed {
		t.Errorf("catch-up histogram: want one sample of %v, got %d samples summing to %v", pushed, after.roundSamples-before.roundSamples, after.roundSum-before.roundSum)
	}
	if after.expired != before.expired || after.expiredInitial != before.expiredInitial || after.unresponsive != before.unresponsive {
		t.Errorf("no termination expected, counters went from %+v to %+v", before, after)
	}
	if period, armed := schedulerState(t, cacher); period != syncPassPeriod || armed {
		t.Errorf("expected the short pass period and a stopped timer after the resync, got %v armed=%v", period, armed)
	}

	// Synced again: a bookmark fired by the timer carries the newest RV.
	clk.Step(2 * time.Minute)
	got = readAll(t, w, 300*time.Millisecond)
	if len(got.bookmarks) != 1 || got.bookmarks[0] != 500 || len(got.rvs) != 0 {
		t.Fatalf("expected one bookmark at 500 and nothing else, got bookmarks %v objects %v", got.bookmarks, got.rvs)
	}
}

// TestStallResumeBookmarkBelowPositionSkipped checks the live path after a
// resync that scanned past events still queued in c.incoming: a bookmark
// below the position is dropped, the queued object events are dropped as
// duplicates, and the next bookmark at the position is delivered.
func TestStallResumeBookmarkBelowPositionSkipped(t *testing.T) {
	cacher, clk := newStallResumeCacher(t)
	w := stallResumeWatch(t, cacher, "ns", 100)
	add := func(rv uint64) { stallResumeAddPods(t, cacher, "ns", rv, rv) }
	next := wedgeWatchers(t, cacher, []*cacheWatcher{w}, add, 101)
	// Drain the client so the watcher has room for the whole backlog.
	first := readAll(t, w, 100*time.Millisecond)
	waitInputDrained(t, cacher, w)

	bookmark := func(rv uint64) *watchCacheEvent {
		ev := &watchCacheEvent{Type: watch.Bookmark, Object: &example.Pod{}, ResourceVersion: rv}
		if err := (storage.APIObjectVersioner{}).UpdateObject(ev.Object, rv); err != nil {
			t.Fatal(err)
		}
		return ev
	}
	// While the dispatcher is held inside the hook, the writer appends
	// events the dispatcher has not seen; the pass then scans to the
	// history end, above the dispatcher's last processed RV.
	last := next + 4
	dispatcherDo(t, cacher, func(runPass syncPassFunc) {
		for rv := next; rv <= last; rv++ {
			add(rv)
		}
		if served, _ := runPass(); served != 1 {
			t.Errorf("expected the watcher to be served, got %d", served)
		}
		if w.unsynced || w.position != last {
			t.Errorf("expected a synced watcher at %d, got unsynced=%v position=%d", last, w.unsynced, w.position)
		}
		// A bookmark at the dispatcher's own last processed RV is below the
		// position and must not reach the client. Stepping the clock expires
		// the watcher's bookmark bucket so the dispatch selects it.
		clk.Step(2 * time.Minute)
		cacher.dispatchEvent(bookmark(next - 1))
		if w.position != last {
			t.Errorf("a skipped bookmark moved the position to %d", w.position)
		}
		// A bookmark at the position is delivered.
		clk.Step(2 * time.Minute)
		cacher.dispatchEvent(bookmark(last))
	})
	got := readUntil(t, w, last, 10*time.Second)
	assertExactSequence(t, slices.Concat(first.rvs, got.rvs), 101, last)
	// The events the hook appended are still in c.incoming; once dispatched
	// they are duplicates of what the pass pushed and must not be delivered.
	waitDispatched(t, cacher)
	extra := readAll(t, w, 300*time.Millisecond)
	if len(extra.rvs) != 0 {
		t.Fatalf("duplicate delivery of %v", extra.rvs)
	}
	if bookmarks := slices.Concat(got.bookmarks, extra.bookmarks); len(bookmarks) != 1 || bookmarks[0] != last {
		t.Fatalf("expected exactly the bookmark at %d, got %v", last, bookmarks)
	}
}

// TestStallResumeCohortSharesWrappedEvents: two watchers at different
// positions served by one cohort receive the same wrapped object for the
// events both are owed, and a member that rejected a push mid window keeps
// its last accepted RV and is served from there by the next pass.
func TestStallResumeCohortSharesWrappedEvents(t *testing.T) {
	cacher, _ := newStallResumeCacher(t)
	a := stallResumeWatch(t, cacher, "ns", 100)
	b := stallResumeWatch(t, cacher, "ns", 100)
	add := func(rv uint64) { stallResumeAddPods(t, cacher, "ns", rv, rv) }
	next := wedgeWatchers(t, cacher, []*cacheWatcher{a, b}, add, 101)
	missed := next - 1
	stallResumeAddPods(t, cacher, "ns", next, 300)

	// Give a five slots of room: it accepts five events and rejects the
	// sixth, keeping its position at the last accepted RV.
	gotA := collected{objects: map[uint64]runtime.Object{}}
	for range 5 {
		ev := <-a.ResultChan()
		rv, _ := storage.APIObjectVersioner{}.ObjectResourceVersion(ev.Object)
		gotA.rvs = append(gotA.rvs, rv)
		gotA.objects[rv] = ev.Object
	}
	waitDispatcherState(t, cacher, []*cacheWatcher{a}, "five slots free", func(w *cacheWatcher) bool {
		return len(w.result) == cap(w.result) && len(w.input) == cap(w.input)-5
	})
	if served, _ := runSyncPass(t, cacher); served != 1 {
		t.Fatalf("expected only a to be served (b has no room), got %d", served)
	}
	st := snapshot(t, cacher, a)
	if !st.unsynced || st.position != missed+4 {
		t.Fatalf("expected a still unsynced at %d (five accepted pushes), got %+v", missed+4, st)
	}
	if stB := snapshot(t, cacher, b); !stB.unsynced || stB.position != missed-1 {
		t.Fatalf("expected b untouched at %d, got %+v", missed-1, stB)
	}

	// Drain both, then one pass serves both in one cohort led by b.
	restA := readAll(t, a, 100*time.Millisecond)
	gotB := readAll(t, b, 100*time.Millisecond)
	waitInputDrained(t, cacher, a, b)
	if served, _ := runSyncPass(t, cacher); served != 2 {
		t.Fatalf("expected both watchers served, got %d", served)
	}
	moreA := readAll(t, a, 100*time.Millisecond)
	moreB := readAll(t, b, 100*time.Millisecond)

	seqA := slices.Concat(gotA.rvs, restA.rvs, moreA.rvs)
	seqB := slices.Concat(gotB.rvs, moreB.rvs)
	assertContiguous(t, seqA)
	assertContiguous(t, seqB)
	if seqA[0] != 101 || seqB[0] != 101 {
		t.Fatalf("sequences must start at 101: a=%v b=%v", seqA[0], seqB[0])
	}
	// b was owed missed..; a was owed missed+5..; the cohort scan pushed
	// cap(input) events to each, so the overlap is missed+5..missed+9.
	shared := 0
	for rv := missed + 5; rv <= missed+9; rv++ {
		objA, okA := moreA.objects[rv]
		objB, okB := moreB.objects[rv]
		if !okA || !okB {
			t.Fatalf("event %d expected on both watchers this pass (a=%v b=%v)", rv, okA, okB)
		}
		if _, isCaching := objA.(*cachingObject); !isCaching {
			t.Fatalf("event %d: expected a *cachingObject, got %T", rv, objA)
		}
		if objA != objB {
			t.Fatalf("event %d: the two watchers got different objects", rv)
		}
		shared++
	}
	if shared != 5 {
		t.Fatalf("expected 5 shared events, got %d", shared)
	}
	// The events a got from the earlier pass were wrapped by that pass, not
	// shared with b's copies from this one.
	for rv := missed; rv < missed+5; rv++ {
		if restA.objects[rv] == moreB.objects[rv] {
			t.Fatalf("event %d: expected separate wrapped copies across passes", rv)
		}
	}
}

// TestStallResumeTriggerScopedMember: a trigger-indexed watcher (spec.nodeName)
// receives from the history exactly the events of its node, in order, and
// its position follows the scan across a window that holds nothing for it.
func TestStallResumeTriggerScopedMember(t *testing.T) {
	cacher, _ := newStallResumeCacher(t, func(cfg *Config) {
		cfg.IndexerFuncs = map[string]storage.IndexerFunc{
			"spec.nodeName": func(obj runtime.Object) string {
				if pod, ok := obj.(*example.Pod); ok {
					return pod.Spec.NodeName
				}
				return ""
			},
		}
	})
	nodePod := func(name, node string, rv uint64) *example.Pod {
		return &example.Pod{
			Name: name, Namespace: "ns", ResourceVersion: strconv.FormatUint(rv, 10),
			Spec: example.PodSpec{NodeName: node},
		}
	}
	pred := storage.SelectionPredicate{
		Label:       labels.Everything(),
		Field:       fields.OneTermEqualSelector("spec.nodeName", "node-1"),
		IndexFields: []string{"spec.nodeName"},
	}
	wi, err := cacher.Watch(context.Background(), "/pods/ns", storage.ListOptions{ResourceVersion: "100", Predicate: pred, Recursive: true})
	if err != nil {
		t.Fatalf("Failed to create watch: %v", err)
	}
	w, ok := wi.(*cacheWatcher)
	if !ok {
		t.Fatalf("expected a *cacheWatcher, got %T", wi)
	}
	defer w.Stop()
	if !w.triggerSupported || w.triggerValue != "node-1" || cap(w.input) != 10 {
		t.Fatalf("expected a trigger-scoped watcher on node-1 with input cap 10, got supported=%v value=%q cap=%d", w.triggerSupported, w.triggerValue, cap(w.input))
	}

	var want []uint64
	addNode := func(node string) func(rv uint64) {
		return func(rv uint64) {
			if err := cacher.watchCache.Add(nodePod(fmt.Sprintf("pod-%d", rv), node, rv)); err != nil {
				t.Fatal(err)
			}
			if node == "node-1" {
				want = append(want, rv)
			}
		}
	}
	next := wedgeWatchers(t, cacher, []*cacheWatcher{w}, addNode("node-1"), 101)
	// A window with nothing for node-1, then one node-1 pod moves away:
	// selected through its previous trigger value, delivered as DELETED.
	for rv := next; rv < next+200; rv++ {
		addNode("node-0")(rv)
	}
	moved := next + 200
	if err := cacher.watchCache.Update(nodePod(fmt.Sprintf("pod-%d", want[0]), "node-2", moved)); err != nil {
		t.Fatal(err)
	}
	want = append(want, moved)
	for rv := moved + 1; rv <= moved+50; rv++ {
		addNode("node-0")(rv)
	}

	got := resyncWhileReading(t, cacher, []*cacheWatcher{w}, moved)[0]
	if len(got.rvs) != len(want) {
		t.Fatalf("want %v, got %v", want, got.rvs)
	}
	for i := range want {
		if got.rvs[i] != want[i] {
			t.Fatalf("event %d: want RV %d got %d", i, want[i], got.rvs[i])
		}
	}
	if got.types[moved] != watch.Deleted {
		t.Fatalf("expected the moved pod as DELETED, got %v", got.types[moved])
	}
	if st := snapshot(t, cacher, w); st.unsynced || st.position != moved+50 {
		t.Fatalf("expected the position at the history end %d, got %+v", moved+50, st)
	}
	if extra := readAll(t, w, 100*time.Millisecond); len(extra.rvs) != 0 {
		t.Fatalf("unexpected extra events %v", extra.rvs)
	}
}

// TestStallResumeNamespaceScopedMember: a namespace scoped watcher receives
// from the history exactly its namespace's events, and crosses a window of
// other namespaces' churn with its position following the scan.
func TestStallResumeNamespaceScopedMember(t *testing.T) {
	cacher, _ := newStallResumeCacher(t)
	w := stallResumeWatch(t, cacher, "ns-a", 100)
	if w.scope != (namespacedName{namespace: "ns-a"}) {
		t.Fatalf("expected a namespace scoped watcher, got %+v", w.scope)
	}
	addA := func(rv uint64) { stallResumeAddPods(t, cacher, "ns-a", rv, rv) }
	next := wedgeWatchers(t, cacher, []*cacheWatcher{w}, addA, 101)
	stallResumeAddPods(t, cacher, "ns-b", next, next+299)
	tail := next + 300
	stallResumeAddPods(t, cacher, "ns-a", tail, tail+2)
	stallResumeAddPods(t, cacher, "ns-b", tail+3, tail+100)

	got := resyncWhileReading(t, cacher, []*cacheWatcher{w}, tail+2)[0]
	want := make([]uint64, 0, next-101+3)
	for rv := uint64(101); rv < next; rv++ {
		want = append(want, rv)
	}
	want = append(want, tail, tail+1, tail+2)
	if len(got.rvs) != len(want) {
		t.Fatalf("want %v, got %v", want, got.rvs)
	}
	for i := range want {
		if got.rvs[i] != want[i] {
			t.Fatalf("event %d: want RV %d got %d", i, want[i], got.rvs[i])
		}
	}
	if st := snapshot(t, cacher, w); st.unsynced || st.position != tail+100 {
		t.Fatalf("expected the position at the history end %d, got %+v", tail+100, st)
	}
}

// TestStallResumeExpiry: a watcher whose position aged out of the history
// gets its backlog, then exactly one 410 ERROR event, then the close; the
// reason is resource_expired for a watcher that served the live stream and
// resource_expired_initial for one still on its initial interval.
func TestStallResumeExpiry(t *testing.T) {
	t.Run("live watcher ages out", func(t *testing.T) {
		cacher, _ := newStallResumeCacher(t)
		pinHistoryCapacity(t, cacher, 100)
		w := stallResumeWatch(t, cacher, "ns", 100)
		before := readStallResumeCounters(t)
		add := func(rv uint64) { stallResumeAddPods(t, cacher, "ns", rv, rv) }
		next := wedgeWatchers(t, cacher, []*cacheWatcher{w}, add, 101)
		// The watcher holds 101..next-2; push next-1..next+199 past the
		// pinned history so its position is below the oldest servable RV.
		stallResumeAddPods(t, cacher, "ns", next, next+199)
		if _, expired := runSyncPass(t, cacher); expired != 1 {
			t.Fatalf("expected the watcher to expire, got %d", expired)
		}
		if st := snapshot(t, cacher, w); !st.expired || st.inSet || w.expiredReason != metrics.TerminationReasonResourceExpired {
			t.Fatalf("unexpected state: %+v reason=%q", st, w.expiredReason)
		}
		delivered := expectExpiredThenClose(t, w)
		assertExactSequence(t, delivered, 101, next-2)
		after := readStallResumeCounters(t)
		if after.expired-before.expired != 1 || after.expiredInitial-before.expiredInitial != 0 {
			t.Errorf("expected terminated{resource_expired} +1 only, got expired %v initial %v", after.expired-before.expired, after.expiredInitial-before.expiredInitial)
		}
	})

	t.Run("initial interval watcher ages out", func(t *testing.T) {
		cacher, _ := newStallResumeCacher(t)
		pinHistoryCapacity(t, cacher, 100)
		// More initial objects than the result buffer, so the goroutine is
		// still on its initial interval when it stalls; input fills up
		// untouched.
		stallResumeAddPods(t, cacher, "ns", 101, 130)
		waitDispatched(t, cacher)
		w := stallResumeWatch(t, cacher, "ns", 0)
		before := readStallResumeCounters(t)
		waitDispatcherState(t, cacher, []*cacheWatcher{w}, "result full on the initial interval", func(w *cacheWatcher) bool {
			return len(w.result) == cap(w.result)
		})
		stallResumeAddPods(t, cacher, "ns", 131, 131+uint64(cap(w.input)))
		waitDispatcherState(t, cacher, []*cacheWatcher{w}, "unsynced", func(w *cacheWatcher) bool { return w.unsynced })
		if w.live.Load() {
			t.Fatalf("the watcher must not be live while on its initial interval")
		}
		stallResumeAddPods(t, cacher, "ns", 132+uint64(cap(w.input)), 500)
		if _, expired := runSyncPass(t, cacher); expired != 1 {
			t.Fatalf("expected the watcher to expire, got %d", expired)
		}
		if w.expiredReason != metrics.TerminationReasonResourceExpiredInitial {
			t.Fatalf("expected reason %q, got %q", metrics.TerminationReasonResourceExpiredInitial, w.expiredReason)
		}
		delivered := expectExpiredThenClose(t, w)
		// The initial listing (31 pods at RV 130 or below, all delivered at the
		// list RV) and then the input backlog, in order.
		if len(delivered) != 31+cap(w.input) {
			t.Fatalf("expected the 31 initial objects and %d live events before the 410, got %d: %v", cap(w.input), len(delivered), delivered)
		}
		assertExactSequence(t, delivered[31:], 131, 130+uint64(cap(w.input)))
		after := readStallResumeCounters(t)
		if after.expiredInitial-before.expiredInitial != 1 || after.expired-before.expired != 0 {
			t.Errorf("expected terminated{resource_expired_initial} +1 only, got initial %v expired %v", after.expiredInitial-before.expiredInitial, after.expired-before.expired)
		}
	})
}

// TestStallResumeStopDuringPass: a client Stop that lands while a pass is
// dispatching, followed by the pass's own draining forget of the same
// watcher, closes done and lets the goroutine exit. Without the hardStop
// rule the draining forget would reopen the drain window and leak the
// goroutine on its full result channel.
func TestStallResumeStopDuringPass(t *testing.T) {
	t.Run("stop lands between step 1 and forget", func(t *testing.T) {
		cacher, _ := newStallResumeCacher(t)
		w := stallResumeWatch(t, cacher, "ns", 100)
		add := func(rv uint64) { stallResumeAddPods(t, cacher, "ns", rv, rv) }
		wedgeWatchers(t, cacher, []*cacheWatcher{w}, add, 101)
		// The exact interleaving of a pass: dispatching is set, the client
		// stops, the pass forgets the expired watcher in drain mode, then
		// finishDispatching runs the deferred stops.
		dispatcherDo(t, cacher, func(syncPassFunc) {
			cacher.Lock()
			cacher.dispatching = true
			cacher.Unlock()
			w.Stop()
			w.forget(true)
			cacher.finishDispatching()
		})
		cacher.RLock()
		doneClosed := w.isDoneChannelClosedLocked()
		drain := w.drainInputBuffer
		cacher.RUnlock()
		if !doneClosed || drain {
			t.Fatalf("expected done closed and no drain mode, got doneClosed=%v drain=%v", doneClosed, drain)
		}
		// The goroutine exits: the result channel closes behind the
		// buffered prefix, with no error event.
		if got := readAll(t, w, 10*time.Second); !got.closed || len(got.errors) != 0 {
			t.Fatalf("expected the goroutine to exit with a clean close, got closed=%v errors=%d", got.closed, len(got.errors))
		}
	})

	t.Run("stop before the pass drops the member", func(t *testing.T) {
		cacher, _ := newStallResumeCacher(t)
		pinHistoryCapacity(t, cacher, 100)
		w := stallResumeWatch(t, cacher, "ns", 100)
		add := func(rv uint64) { stallResumeAddPods(t, cacher, "ns", rv, rv) }
		next := wedgeWatchers(t, cacher, []*cacheWatcher{w}, add, 101)
		stallResumeAddPods(t, cacher, "ns", next, next+199)
		before := readStallResumeCounters(t)
		w.Stop()
		if served, expired := runSyncPass(t, cacher); served != 0 || expired != 0 {
			t.Fatalf("a stopped watcher must be dropped, not served or expired: served=%d expired=%d", served, expired)
		}
		if st := snapshot(t, cacher, w); st.inSet || st.expired {
			t.Fatalf("expected the watcher out of the set and not expired, got %+v", st)
		}
		if after := readStallResumeCounters(t); after.expired != before.expired || after.expiredInitial != before.expiredInitial {
			t.Fatalf("a stopped watcher must not be counted as expired")
		}
	})
}

// TestStallResumeTerminateAllWithUnsyncedMember: a relist (terminateAllWatchers)
// stops an unsynced watcher without forgetting it; the next pass drops it
// from the set instead of pushing to its closed input.
func TestStallResumeTerminateAllWithUnsyncedMember(t *testing.T) {
	cacher, _ := newStallResumeCacher(t)
	w := stallResumeWatch(t, cacher, "ns", 100)
	add := func(rv uint64) { stallResumeAddPods(t, cacher, "ns", rv, rv) }
	wedgeWatchers(t, cacher, []*cacheWatcher{w}, add, 101)
	// Drain so the watcher would be a candidate with room.
	readAll(t, w, 100*time.Millisecond)
	waitInputDrained(t, cacher, w)
	cacher.terminateAllWatchers()
	if served, expired := runSyncPass(t, cacher); served != 0 || expired != 0 {
		t.Fatalf("expected nothing served or expired, got served=%d expired=%d", served, expired)
	}
	if st := snapshot(t, cacher, w); st.inSet {
		t.Fatalf("expected the stopped watcher dropped from the set, got %+v", st)
	}
	if got := readAll(t, w, 5*time.Second); !got.closed || len(got.errors) != 0 {
		t.Fatalf("expected a clean close, got closed=%v errors=%d", got.closed, len(got.errors))
	}
}

// TestStallResumeCohortsUnderBudget: three watchers whose positions are
// further apart than the scan budget, each scoped so that most of the
// history holds nothing for it. The passes are deterministic: the low pair
// forms one cohort and is served every pass, the lead is re-picked at the
// cursor, the cursor is the last scanned RV, positions follow the scan, and
// the far watcher joins the cohort once the scan reaches it. Every sequence
// is exact.
func TestStallResumeCohortsUnderBudget(t *testing.T) {
	cacher, _ := newStallResumeCacher(t)
	pinHistoryCapacity(t, cacher, 10000)
	w1 := stallResumeWatch(t, cacher, "ns-a", 100)
	w2 := stallResumeWatch(t, cacher, "ns-a", 100)
	w3 := stallResumeWatch(t, cacher, "ns-b", 100)
	addA := func(rv uint64) { stallResumeAddPods(t, cacher, "ns-a", rv, rv) }
	addB := func(rv uint64) { stallResumeAddPods(t, cacher, "ns-b", rv, rv) }
	before := readStallResumeCounters(t)

	nextA := wedgeWatchers(t, cacher, []*cacheWatcher{w1, w2}, addA, 101)
	missedA := nextA - 1
	churn1End := nextA + 4999
	stallResumeAddPods(t, cacher, "ns-c", nextA, churn1End)
	nextB := wedgeWatchers(t, cacher, []*cacheWatcher{w3}, addB, churn1End+1)
	missedB := nextB - 1
	end := nextB + 149
	stallResumeAddPods(t, cacher, "ns-c", nextB, end)

	// Everyone drains; each holds cap(result)+1+cap(input) events.
	got1 := readAll(t, w1, 100*time.Millisecond)
	got2 := readAll(t, w2, 100*time.Millisecond)
	got3 := readAll(t, w3, 100*time.Millisecond)
	waitInputDrained(t, cacher, w1, w2, w3)

	type passExpectation struct {
		served     int
		cursor     uint64
		pos1       uint64
		pos3       uint64
		synced     bool
		historyEnd bool
		leadNote   string
	}
	// One cohort per pass: the scan budget less the interval open cost.
	budget := uint64(syncScanBudget - syncIntervalOpenCost)
	expectations := []passExpectation{
		// Lead w1 at missedA-1 scans budget events; w2 shares the cohort;
		// w3 is above the window.
		{served: 2, cursor: missedA - 1 + budget, pos1: missedA - 1 + budget, pos3: missedB - 1, leadNote: "w1 from its miss"},
		// Lead w1 again (at the cursor); another budget worth.
		{served: 2, cursor: missedA - 1 + 2*budget, pos1: missedA - 1 + 2*budget, pos3: missedB - 1, leadNote: "w1 at the cursor"},
		// The scan reaches the history end; w3 is below it, so it joins and
		// everyone resyncs at the end.
		{served: 3, cursor: end, pos1: end, pos3: end, synced: true, historyEnd: true, leadNote: "w1 at the cursor, w3 joins"},
	}
	for i, want := range expectations {
		served, expired := runSyncPass(t, cacher)
		if served != want.served || expired != 0 {
			t.Fatalf("pass %d (%s): served=%d expired=%d, want served=%d", i+1, want.leadNote, served, expired, want.served)
		}
		var cursor uint64
		var st1, st2, st3 watcherState
		dispatcherDo(t, cacher, func(syncPassFunc) {
			cursor = cacher.stall.syncCursor
			for _, p := range []struct {
				w  *cacheWatcher
				st *watcherState
			}{{w1, &st1}, {w2, &st2}, {w3, &st3}} {
				_, p.st.inSet = cacher.stall.unsynced[p.w]
				p.st.position, p.st.unsynced = p.w.position, p.w.unsynced
			}
		})
		if cursor != want.cursor {
			t.Fatalf("pass %d: cursor %d, want %d", i+1, cursor, want.cursor)
		}
		if st1.position != want.pos1 || st2.position != want.pos1 || st3.position != want.pos3 {
			t.Fatalf("pass %d: positions w1=%d w2=%d w3=%d, want %d %d %d", i+1, st1.position, st2.position, st3.position, want.pos1, want.pos1, want.pos3)
		}
		if st1.unsynced == want.synced || st2.unsynced == want.synced || st3.unsynced == want.synced {
			t.Fatalf("pass %d: unsynced w1=%v w2=%v w3=%v, want all %v", i+1, st1.unsynced, st2.unsynced, st3.unsynced, !want.synced)
		}
	}

	more1 := readAll(t, w1, 100*time.Millisecond)
	more2 := readAll(t, w2, 100*time.Millisecond)
	more3 := readAll(t, w3, 100*time.Millisecond)
	assertExactSequence(t, slices.Concat(got1.rvs, more1.rvs), 101, missedA)
	assertExactSequence(t, slices.Concat(got2.rvs, more2.rvs), 101, missedA)
	assertExactSequence(t, slices.Concat(got3.rvs, more3.rvs), churn1End+1, missedB)
	after := readStallResumeCounters(t)
	if after.stalls-before.stalls != 3 || after.rounds-before.rounds != 3 || after.deferred-before.deferred != 3 {
		t.Errorf("expected 3 stalls, 3 resyncs and 3 deferred events (one missed event each), got stalls=%v rounds=%v deferred=%v",
			after.stalls-before.stalls, after.rounds-before.rounds, after.deferred-before.deferred)
	}
}

// TestStallResumePassTimer: with no event traffic, the pass timer on the
// cacher's clock drives the resync: the short period right after a stall
// (even though the inline pass at the stall served nothing), the idle
// period after a later pass that served nothing.
func TestStallResumePassTimer(t *testing.T) {
	cacher, clk := newStallResumeCacher(t)
	w := stallResumeWatch(t, cacher, "ns", 100)
	before := readStallResumeCounters(t)
	add := func(rv uint64) { stallResumeAddPods(t, cacher, "ns", rv, rv) }
	next := wedgeWatchers(t, cacher, []*cacheWatcher{w}, add, 101)
	stallResumeAddPods(t, cacher, "ns", next, next+4)

	// The stall keeps the short period; with the input still full, a
	// short step runs a pass that serves nothing and backs off to the
	// idle period.
	if period, _ := schedulerState(t, cacher); period != syncPassPeriod {
		t.Fatalf("expected the short pass period right after the stall, got %v", period)
	}
	clk.Step(syncPassPeriod)
	if err := wait.PollUntilContextTimeout(context.Background(), time.Millisecond, 10*time.Second, true, func(context.Context) (bool, error) {
		period, _ := schedulerState(t, cacher)
		return period == syncPassIdlePeriod, nil
	}); err != nil {
		t.Fatalf("expected the idle period after a pass that served nothing: %v", err)
	}
	readAll(t, w, 100*time.Millisecond)
	waitInputDrained(t, cacher, w)

	// The timer is armed at the idle period now, so a short step must not
	// fire it.
	clk.Step(syncPassPeriod)
	if st := snapshot(t, cacher, w); !st.unsynced {
		t.Fatalf("a %v step fired the idle timer", syncPassPeriod)
	}
	clk.Step(syncPassIdlePeriod)
	if err := wait.PollUntilContextTimeout(context.Background(), time.Millisecond, 10*time.Second, true, func(context.Context) (bool, error) {
		return readStallResumeCounters(t).rounds-before.rounds == 1, nil
	}); err != nil {
		t.Fatalf("the timer never drove a resync: %v", err)
	}
	got := readUntil(t, w, next+4, 10*time.Second)
	assertExactSequence(t, got.rvs, next-1, next+4)
	if st := snapshot(t, cacher, w); st.unsynced || st.position != next+4 {
		t.Fatalf("unexpected state after the timer pass: %+v", st)
	}
	if _, armed := schedulerState(t, cacher); armed {
		t.Fatalf("the pass timer must be stopped while nothing is unsynced")
	}
}

// TestStallResumeGateOff: with the gate off the cacher has no stall state,
// registers no pass, and a watcher with a full input is force closed as
// unresponsive exactly as before.
func TestStallResumeGateOff(t *testing.T) {
	setStallResumeGate(t, false)
	ensureStallResumeMetrics()
	cacher, _, err := newTestCacher(&cachertesting.MockStorage{})
	if err != nil {
		t.Fatalf("Couldn't create cacher: %v", err)
	}
	defer cacher.Stop()
	if cacher.stall != nil {
		t.Fatalf("expected no stall state with the gate off")
	}
	stallResumeAddPods(t, cacher, "ns", 100, 100)
	w := stallResumeWatch(t, cacher, "ns", 100)
	if w.stallMetrics != nil {
		t.Fatalf("expected no stall metrics on the watcher with the gate off")
	}
	before := readStallResumeCounters(t)
	// Far more than the watcher's buffering while the client is silent;
	// the dispatch budget runs out and the watcher is closed.
	stallResumeAddPods(t, cacher, "ns", 101, 400)
	got := readUntil(t, w, ^uint64(0), 30*time.Second)
	if !got.closed || len(got.errors) != 0 {
		t.Fatalf("expected a bare close, got closed=%v errors=%d", got.closed, len(got.errors))
	}
	assertContiguous(t, got.rvs)
	after := readStallResumeCounters(t)
	if after.unresponsive-before.unresponsive != 1 {
		t.Errorf("expected terminated{unresponsive} +1, got %v", after.unresponsive-before.unresponsive)
	}
	if after.stalls != before.stalls || after.deferred != before.deferred || after.rounds != before.rounds {
		t.Errorf("stall counters must not move with the gate off: %+v -> %+v", before, after)
	}
	if w.position != 0 || w.unsynced || w.live.Load() {
		t.Errorf("gate off must not touch the stall fields: position=%d unsynced=%v live=%v", w.position, w.unsynced, w.live.Load())
	}
}

// TestStallResumeCacherStopWithUnsyncedWatcher stops the whole Cacher while a
// watcher is unsynced: the client gets an in-order prefix and a clean close.
func TestStallResumeCacherStopWithUnsyncedWatcher(t *testing.T) {
	cacher, _ := newStallResumeCacher(t)
	w := stallResumeWatch(t, cacher, "ns", 100)
	add := func(rv uint64) { stallResumeAddPods(t, cacher, "ns", rv, rv) }
	wedgeWatchers(t, cacher, []*cacheWatcher{w}, add, 101)
	cacher.Stop()
	got := readAll(t, w, 5*time.Second)
	if !got.closed || len(got.errors) != 0 {
		t.Fatalf("expected a clean close, got closed=%v errors=%d", got.closed, len(got.errors))
	}
	assertContiguous(t, got.rvs)
}

// TestStallResumeRealClock runs the mechanism end to end on the real clock:
// timer driven passes resync a watcher without any test hook.
func TestStallResumeRealClock(t *testing.T) {
	setStallResumeGate(t, true)
	ensureStallResumeMetrics()
	cacher, _, err := newTestCacherWithoutSyncing(&cachertesting.MockStorage{}, clock.RealClock{})
	if err != nil {
		t.Fatalf("Couldn't create cacher: %v", err)
	}
	defer cacher.Stop()
	if err := cacher.Wait(context.Background()); err != nil {
		t.Fatal(err)
	}
	stallResumeAddPods(t, cacher, "ns", 100, 100)
	w := stallResumeWatch(t, cacher, "ns", 100)
	before := readStallResumeCounters(t)
	stallResumeAddPods(t, cacher, "ns", 101, 1000)
	got := readUntil(t, w, 1000, 20*time.Second)
	assertExactSequence(t, got.rvs, 101, 1000)
	// The client can read the last event before the pass that pushed it
	// reaches its resync section, which is where the round is counted;
	// the resync can even land in the next pass when the retry rounds
	// drained the member. Wait for the count rather than read it once.
	if err := wait.PollUntilContextTimeout(context.Background(), time.Millisecond, 10*time.Second, true, func(context.Context) (bool, error) {
		return readStallResumeCounters(t).rounds-before.rounds >= 1, nil
	}); err != nil {
		t.Fatalf("the watcher did not resync after the schedule: %v", err)
	}
	after := readStallResumeCounters(t)
	if after.stalls-before.stalls < 1 || after.rounds-before.rounds < 1 {
		t.Fatalf("the schedule did not stall and resync the watcher: stalls=%v rounds=%v", after.stalls-before.stalls, after.rounds-before.rounds)
	}
	if after.expired != before.expired || after.expiredInitial != before.expiredInitial || after.unresponsive != before.unresponsive {
		t.Fatalf("no termination expected")
	}
}

// scopedWatch opens a watch with the given context, key and predicate and
// returns the cacheWatcher behind it.
func scopedWatch(t testing.TB, c *Cacher, ctx context.Context, key string, pred storage.SelectionPredicate) *cacheWatcher {
	t.Helper()
	w, err := c.Watch(ctx, key, storage.ListOptions{ResourceVersion: "100", Predicate: pred, Recursive: true})
	if err != nil {
		t.Fatalf("Failed to create watch: %v", err)
	}
	cw, ok := w.(*cacheWatcher)
	if !ok {
		t.Fatalf("expected a *cacheWatcher, got %T", w)
	}
	t.Cleanup(cw.Stop)
	return cw
}

// updatePod writes one MODIFIED event for the named pod at the given
// resourceVersion.
func updatePod(t testing.TB, c *Cacher, namespace, name string, rv uint64) {
	t.Helper()
	pod := stallResumePod(namespace, rv)
	pod.Name = name
	if err := c.watchCache.Update(pod); err != nil {
		t.Fatalf("failed to update a pod: %v", err)
	}
}

// detachedWatcher builds a watcher with no goroutine and no registration
// in the dispatch indexes: only the sync passes see it, once
// registerUnsynced adds it, so a test controls its input and position
// exactly.
func detachedWatcher(c *Cacher, chanSize int, scope namespacedName) *cacheWatcher {
	filter := func(string, labels.Set, fields.Set, runtime.Object) bool { return true }
	w := newCacheWatcher(chanSize, filter, emptyFunc, storage.APIObjectVersioner{}, c.clock.Now().Add(time.Hour), false, c.groupResource, metrics.NewNoopWatcherMetricsObservers(), c.clock, "")
	w.stallMetrics = c.stall.metrics
	w.scope = scope
	return w
}

// registerUnsynced puts the watchers in the unsynced set at the positions
// the function returns, on the dispatcher goroutine.
func registerUnsynced(t testing.TB, c *Cacher, ws []*cacheWatcher, position func(i int) uint64) {
	t.Helper()
	dispatcherDo(t, c, func(syncPassFunc) {
		for i, w := range ws {
			w.position, w.unsynced = position(i), true
			c.stall.unsynced[w] = struct{}{}
		}
	})
}

// drainInput takes everything out of a detached watcher's input and returns
// the resourceVersions in order.
func drainInput(w *cacheWatcher) []uint64 {
	var rvs []uint64
	for {
		select {
		case ev := <-w.input:
			rvs = append(rvs, ev.ResourceVersion)
		default:
			return rvs
		}
	}
}

// indexerHook lets a test run a function on the dispatcher goroutine in the
// middle of a sync pass: the cacher's trigger indexer is called for every
// scanned event, before the event is offered. The hook sees the scanned
// object and is inert until armed.
type indexerHook struct {
	fn atomic.Pointer[func(obj runtime.Object)]
}

func (h *indexerHook) config(cfg *Config) {
	cfg.IndexerFuncs = map[string]storage.IndexerFunc{
		"spec.nodeName": func(obj runtime.Object) string {
			if fn := h.fn.Load(); fn != nil {
				(*fn)(obj)
			}
			return ""
		},
	}
}

func objectRV(t testing.TB, obj runtime.Object) uint64 {
	t.Helper()
	rv, err := storage.APIObjectVersioner{}.ObjectResourceVersion(obj)
	if err != nil {
		t.Fatalf("parsing resource version: %v", err)
	}
	return rv
}

// TestStallResumeManyTinyCohortsCost: 512 unsynced watchers at distinct
// positions with one free input slot each is the worst case for the pass
// (every cohort is one member and one tiny interval). The lookup maps are
// built once per pass and every interval open is charged to the scan
// budget, so a pass stays cheap in time and allocation, and the cursor
// serves every member over the following passes.
func TestStallResumeManyTinyCohortsCost(t *testing.T) {
	cacher, _ := newStallResumeCacher(t)
	pinHistoryCapacity(t, cacher, 10000)
	const members = maxWatchersPerSync
	// Positions three apart: the lead accepts one event, rejects the next
	// and the scan stops before the next member's position.
	position := func(i int) uint64 { return 100 + 3*uint64(i) }
	stallResumeAddPods(t, cacher, "ns", 101, position(members-1)+100)
	waitDispatched(t, cacher)
	ws := make([]*cacheWatcher, members)
	for i := range ws {
		ws[i] = detachedWatcher(cacher, 1, namespacedName{})
	}
	registerUnsynced(t, cacher, ws, position)

	// Each cohort costs the interval open plus the events it scans: the
	// accepted one, the rejected one and the owed events the lead can
	// record (its pending cap of syncPushRetryRounds slots, less the
	// rejected event already in it).
	cohortsPerPass := syncScanBudget / (syncIntervalOpenCost + 1 + syncPushRetryRounds)
	passes := (members + cohortsPerPass - 1) / cohortsPerPass
	minWall, maxWall := time.Duration(-1), time.Duration(0)
	var maxAlloc uint64
	for pass := range passes {
		wantServed := min(cohortsPerPass, members-pass*cohortsPerPass)
		dispatcherDo(t, cacher, func(runPass syncPassFunc) {
			var before, after goruntime.MemStats
			goruntime.ReadMemStats(&before)
			start := time.Now()
			served, expired := runPass()
			wall := time.Since(start)
			goruntime.ReadMemStats(&after)
			if served != wantServed || expired != 0 {
				t.Errorf("pass %d: served=%d expired=%d, want served=%d", pass+1, served, expired, wantServed)
			}
			if cacher.stall.passPeriod != syncPassBusyPeriod {
				t.Errorf("pass %d: period %v, want the busy period %v (a member is still owed events)", pass+1, cacher.stall.passPeriod, syncPassBusyPeriod)
			}
			alloc := after.TotalAlloc - before.TotalAlloc
			if minWall < 0 || wall < minWall {
				minWall = wall
			}
			maxWall = max(maxWall, wall)
			maxAlloc = max(maxAlloc, alloc)
		})
	}
	t.Logf("%d passes of up to %d one-member cohorts: wall min %v max %v, alloc max %d bytes", passes, cohortsPerPass, minWall, maxWall, maxAlloc)
	if minWall > 2*time.Millisecond {
		t.Errorf("the cheapest pass took %v, want under 2ms", minWall)
	}
	if maxAlloc > 500<<10 {
		t.Errorf("a pass allocated %d bytes, want under 500KB", maxAlloc)
	}
	// Every member got exactly its one event, in cursor order.
	dispatcherDo(t, cacher, func(syncPassFunc) {
		for i, w := range ws {
			if w.catchupEvents != 1 || w.position != position(i)+1 || !w.unsynced {
				t.Fatalf("member %d: catchupEvents=%d position=%d unsynced=%v, want 1 event and position %d", i, w.catchupEvents, w.position, w.unsynced, position(i)+1)
			}
		}
	})
	if served, _ := runSyncPass(t, cacher); served != 0 {
		t.Fatalf("every input is full, want nothing served, got %d", served)
	}
	if period, _ := schedulerState(t, cacher); period != syncPassIdlePeriod {
		t.Fatalf("expected the idle period after a pass that served nothing, got %v", period)
	}
}

// TestStallResumePassPeriod: the period after a pass follows what the pass
// did, and the busy period scales with the measured pass duration so passes
// take at most about a tenth of the dispatcher's time, capped at the normal
// period so a descheduled pass cannot amplify.
func TestStallResumePassPeriod(t *testing.T) {
	for _, tc := range []struct {
		name            string
		served, expired int
		busy            bool
		duration        time.Duration
		want            time.Duration
	}{
		{name: "busy, instant pass", busy: true, want: syncPassBusyPeriod},
		{name: "busy, short pass", busy: true, duration: 50 * time.Microsecond, want: syncPassBusyPeriod},
		{name: "busy, 1ms pass", served: 5, busy: true, duration: time.Millisecond, want: 9 * time.Millisecond},
		{name: "busy, 5ms pass, capped", busy: true, duration: 5 * time.Millisecond, want: syncPassPeriod},
		{name: "busy, descheduled 14ms pass, capped", busy: true, duration: 14 * time.Millisecond, want: syncPassPeriod},
		{name: "nothing done", want: syncPassIdlePeriod},
		{name: "nothing done, slow pass", duration: 5 * time.Millisecond, want: syncPassIdlePeriod},
		{name: "served", served: 1, want: syncPassPeriod},
		{name: "expired", expired: 1, want: syncPassPeriod},
		{name: "served, slow pass", served: 1, duration: 5 * time.Millisecond, want: syncPassPeriod},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := nextPassPeriod(tc.served, tc.expired, tc.busy, tc.duration); got != tc.want {
				t.Fatalf("nextPassPeriod(%d, %d, %v, %v) = %v, want %v", tc.served, tc.expired, tc.busy, tc.duration, got, tc.want)
			}
		})
	}
}

// TestStallResumePushBudget: 512 draining members in one cohort would take
// 512 times cap(input) pushes in one pass; the push budget cuts the
// followers after the first event instead, while the lead goes on to its
// room, every follower's position follows the scan to the cut, the pass
// selects the busy period, and the next pass carries on from there.
func TestStallResumePushBudget(t *testing.T) {
	cacher, _ := newStallResumeCacher(t)
	pinHistoryCapacity(t, cacher, 10000)
	const members = maxWatchersPerSync
	stallResumeAddPods(t, cacher, "ns", 101, 2100)
	waitDispatched(t, cacher)
	ws := make([]*cacheWatcher, members)
	for i := range ws {
		ws[i] = detachedWatcher(cacher, 100, namespacedName{})
	}
	registerUnsynced(t, cacher, ws, func(int) uint64 { return 100 })
	room := uint64(cap(ws[0].input))
	// The followers all join at the first scanned event, so the cut comes
	// after the first event that brings the pushes to the budget or
	// above; the lead then goes on alone to its room.
	cutAfter := func(followers int) uint64 { return uint64((syncPushBudget + followers - 1) / followers) }
	before := readStallResumeCounters(t)
	var lead *cacheWatcher
	dispatcherDo(t, cacher, func(runPass syncPassFunc) {
		if served, expired := runPass(); served != members || expired != 0 {
			t.Fatalf("served=%d expired=%d, want every member served", served, expired)
		}
		if cacher.stall.passPeriod != syncPassBusyPeriod {
			t.Fatalf("period %v, want the busy period (the push budget ran out)", cacher.stall.passPeriod)
		}
		for _, w := range ws {
			switch {
			case w.position == 100+room && lead == nil:
				lead = w
			case w.position != 100+cutAfter(members) || !w.unsynced:
				t.Fatalf("member at %d unsynced=%v, want a follower at the cut (%d) or one lead at %d", w.position, w.unsynced, 100+cutAfter(members), 100+room)
			}
		}
		if lead == nil {
			t.Fatalf("no member got its whole room")
		}
	})
	if got, want := readStallResumeCounters(t).deferred-before.deferred, float64(members)*float64(cutAfter(members))+float64(room-cutAfter(members)); got != want {
		t.Fatalf("deferred events %v, want %v", got, want)
	}
	assertExactSequence(t, drainInput(lead), 101, 100+room)
	for _, w := range ws {
		if w == lead {
			continue
		}
		assertExactSequence(t, drainInput(w), 101, 100+cutAfter(members))
	}
	// Next pass: the drained lead sits above the cursor's wrap, so a
	// follower leads from the cut and gets its room past the budget, the
	// followers below the cut get what the budget allows, and the old
	// lead above the cut is not reached.
	if served, _ := runSyncPass(t, cacher); served != members-1 {
		t.Fatalf("pass 2: served=%d, want %d", served, members-1)
	}
	cut := 100 + cutAfter(members)
	leads := 0
	for _, w := range ws {
		if w == lead {
			if w.position != 100+room {
				t.Fatalf("pass 2: the old lead moved to %d", w.position)
			}
			continue
		}
		if w.position == cut+room {
			leads++
		} else if w.position != cut+cutAfter(members-1) {
			t.Fatalf("pass 2: follower at %d, want %d", w.position, cut+cutAfter(members-1))
		}
	}
	if leads != 1 {
		t.Fatalf("pass 2: %d members got their whole room, want 1", leads)
	}
}

// TestStallResumePushBudgetKeepsLead: a lead whose scan picks up a dense
// band of followers just above it is not cut short when they spend the
// push budget on the first event: the scan goes on for the lead alone up
// to its room, and the followers' positions stop at the cut.
func TestStallResumePushBudgetKeepsLead(t *testing.T) {
	cacher, _ := newStallResumeCacher(t)
	pinHistoryCapacity(t, cacher, 10000)
	stallResumeAddPods(t, cacher, "ns", 101, 1000)
	waitDispatched(t, cacher)
	lead := detachedWatcher(cacher, 10, namespacedName{})
	followers := make([]*cacheWatcher, maxWatchersPerSync)
	for i := range followers {
		followers[i] = detachedWatcher(cacher, 10, namespacedName{})
	}
	// The lead is one below the band, so the band joins at the second
	// scanned event and spends the budget on it.
	registerUnsynced(t, cacher, []*cacheWatcher{lead}, func(int) uint64 { return 100 })
	registerUnsynced(t, cacher, followers, func(int) uint64 { return 101 })
	// The candidate cap drops one follower (equal keys, cursor order).
	if served, _ := runSyncPass(t, cacher); served != maxWatchersPerSync {
		t.Fatalf("served=%d, want %d", served, maxWatchersPerSync)
	}
	assertExactSequence(t, drainInput(lead), 101, 110)
	dispatcherDo(t, cacher, func(syncPassFunc) {
		if lead.position != 110 || !lead.unsynced {
			t.Fatalf("lead: position=%d unsynced=%v, want blocked at 110", lead.position, lead.unsynced)
		}
	})
	// The first event goes to the lead alone, each later one to the lead
	// and the 511 followers, until the pushes reach the budget.
	cutAt := 101 + uint64((syncPushBudget-1+maxWatchersPerSync)/(maxWatchersPerSync+1))
	left := 0
	for i, w := range followers {
		got := drainInput(w)
		switch {
		case w.servedPass == 0 && len(got) == 0:
			left++
		case w.position != cutAt:
			t.Fatalf("follower %d: position=%d, want position %d", i, w.position, cutAt)
		default:
			assertExactSequence(t, got, 102, cutAt)
		}
	}
	if left != 1 {
		t.Fatalf("expected one follower left out by the candidate cap, got %d", left)
	}
}

// TestStallResumeBusyPeriodDrivesInlinePass: a pass that leaves a member
// owed events on a full input selects the busy period, and the inline check
// after a dispatched event runs the next pass once that period, not the
// normal one, has elapsed.
func TestStallResumeBusyPeriodDrivesInlinePass(t *testing.T) {
	cacher, clk := newStallResumeCacher(t)
	w := stallResumeWatch(t, cacher, "ns", 100)
	add := func(rv uint64) { stallResumeAddPods(t, cacher, "ns", rv, rv) }
	next := wedgeWatchers(t, cacher, []*cacheWatcher{w}, add, 101)
	stallResumeAddPods(t, cacher, "ns", next, next+50)
	// One slot of room: the pass pushes one event and is rejected on the
	// next, leaving the member blocked.
	<-w.ResultChan()
	waitDispatcherState(t, cacher, []*cacheWatcher{w}, "one slot free", func(w *cacheWatcher) bool {
		return len(w.input) == cap(w.input)-1
	})
	dispatcherDo(t, cacher, func(runPass syncPassFunc) {
		if served, _ := runPass(); served != 1 {
			t.Fatalf("expected the watcher served, got %d", served)
		}
		if cacher.stall.passPeriod != syncPassBusyPeriod {
			t.Fatalf("expected the busy period %v after a pass that left the member blocked, got %v", syncPassBusyPeriod, cacher.stall.passPeriod)
		}
		// Less than the normal period, exactly the busy one: the inline
		// check after a dispatch runs a pass.
		clk.Step(syncPassBusyPeriod)
		cacher.syncAfterDispatch()
		if !cacher.stall.lastPass.Equal(clk.Now()) {
			t.Fatalf("the inline check did not run a pass after the busy period: lastPass=%v now=%v", cacher.stall.lastPass, clk.Now())
		}
	})
}

// TestStallResumeClusterWideWatcher: a watcher with no namespace, no name
// and no trigger (the shape of an informer or a proxy watching everything)
// is served every event of every namespace from the history, in order.
func TestStallResumeClusterWideWatcher(t *testing.T) {
	cacher, _ := newStallResumeCacher(t)
	w := scopedWatch(t, cacher, context.Background(), "/pods", storage.Everything)
	if w.scope != (namespacedName{}) || w.triggerSupported {
		t.Fatalf("expected a cluster-wide unscoped watcher, got scope=%+v trigger=%v", w.scope, w.triggerSupported)
	}
	add := func(rv uint64) { stallResumeAddPods(t, cacher, "ns-a", rv, rv) }
	next := wedgeWatchers(t, cacher, []*cacheWatcher{w}, add, 101)
	stallResumeAddPods(t, cacher, "ns-b", next, next+199)
	stallResumeAddPods(t, cacher, "ns-c", next+200, next+299)
	end := next + 299

	got := resyncWhileReading(t, cacher, []*cacheWatcher{w}, end)[0]
	assertExactSequence(t, got.rvs, 101, end)
	if st := snapshot(t, cacher, w); st.unsynced || st.position != end {
		t.Fatalf("expected a synced watcher at %d, got %+v", end, st)
	}
}

// TestStallResumeNameScopedWatcher: a cluster-wide watcher scoped by name
// (a field selector on metadata.name) receives from the history exactly the
// events of that name, from every namespace, and crosses the churn of other
// names with its position following the scan.
func TestStallResumeNameScopedWatcher(t *testing.T) {
	cacher, _ := newStallResumeCacher(t)
	pred := storage.SelectionPredicate{Label: labels.Everything(), Field: fields.OneTermEqualSelector("metadata.name", "target")}
	w := scopedWatch(t, cacher, context.Background(), "/pods", pred)
	if w.scope != (namespacedName{name: "target"}) {
		t.Fatalf("expected a name scoped watcher, got %+v", w.scope)
	}
	var want []uint64
	target := func(namespace string) func(rv uint64) {
		return func(rv uint64) {
			updatePod(t, cacher, namespace, "target", rv)
			want = append(want, rv)
		}
	}
	next := wedgeWatchers(t, cacher, []*cacheWatcher{w}, target("ns-a"), 101)
	stallResumeAddPods(t, cacher, "ns-a", next, next+199)
	target("ns-b")(next + 200)
	stallResumeAddPods(t, cacher, "ns-b", next+201, next+300)
	target("ns-a")(next + 301)
	end := next + 350
	stallResumeAddPods(t, cacher, "ns-c", next+302, end)

	got := resyncWhileReading(t, cacher, []*cacheWatcher{w}, next+301)[0]
	if !slices.Equal(got.rvs, want) {
		t.Fatalf("want %v, got %v", want, got.rvs)
	}
	if st := snapshot(t, cacher, w); st.unsynced || st.position != end {
		t.Fatalf("expected the position at the history end %d, got %+v", end, st)
	}
	if extra := readAll(t, w, 100*time.Millisecond); len(extra.rvs) != 0 {
		t.Fatalf("unexpected extra events %v", extra.rvs)
	}
}

// TestStallResumeNamespaceNameScopedWatcher: a watcher scoped by namespace
// and name receives from the history exactly that one object's events; the
// same name in another namespace and other names in its namespace are
// crossed with the position following the scan.
func TestStallResumeNamespaceNameScopedWatcher(t *testing.T) {
	cacher, _ := newStallResumeCacher(t)
	pred := storage.SelectionPredicate{Label: labels.Everything(), Field: fields.OneTermEqualSelector("metadata.name", "target")}
	w := scopedWatch(t, cacher, request.WithNamespace(context.Background(), "ns-a"), "/pods/ns-a", pred)
	if w.scope != (namespacedName{namespace: "ns-a", name: "target"}) {
		t.Fatalf("expected a namespace and name scoped watcher, got %+v", w.scope)
	}
	var want []uint64
	target := func(rv uint64) {
		updatePod(t, cacher, "ns-a", "target", rv)
		want = append(want, rv)
	}
	next := wedgeWatchers(t, cacher, []*cacheWatcher{w}, target, 101)
	stallResumeAddPods(t, cacher, "ns-a", next, next+199)
	updatePod(t, cacher, "ns-b", "target", next+200)
	stallResumeAddPods(t, cacher, "ns-b", next+201, next+300)
	target(next + 301)
	end := next + 350
	stallResumeAddPods(t, cacher, "ns-a", next+302, end)

	got := resyncWhileReading(t, cacher, []*cacheWatcher{w}, next+301)[0]
	if !slices.Equal(got.rvs, want) {
		t.Fatalf("want %v, got %v", want, got.rvs)
	}
	if st := snapshot(t, cacher, w); st.unsynced || st.position != end {
		t.Fatalf("expected the position at the history end %d, got %+v", end, st)
	}
	if extra := readAll(t, w, 100*time.Millisecond); len(extra.rvs) != 0 {
		t.Fatalf("unexpected extra events %v", extra.rvs)
	}
}

// TestStallResumeRetryRounds: a member whose input drains while its cohort
// is still being scanned is re-offered what it is owed after the scan, so
// one pass pushes more than cap(input) events to it; a re-offer that is
// rejected again leaves the member blocked at its last accepted
// resourceVersion, and the pass selects the busy period. The drain happens
// on the dispatcher goroutine, from inside the scan, so the sequence is
// exact.
func TestStallResumeRetryRounds(t *testing.T) {
	hook := &indexerHook{}
	cacher, _ := newStallResumeCacher(t, hook.config)
	pinHistoryCapacity(t, cacher, 10000)
	const end = 1100
	stallResumeAddPods(t, cacher, "ns", 101, end)
	waitDispatched(t, cacher)
	// a takes every event; b, scoped to an empty namespace, is offered
	// nothing but keeps the scan going after a blocks.
	a := detachedWatcher(cacher, 100, namespacedName{})
	b := detachedWatcher(cacher, 100, namespacedName{namespace: "none"})
	registerUnsynced(t, cacher, []*cacheWatcher{a, b}, func(int) uint64 { return 100 })
	inputCap := uint64(cap(a.input))

	// Once a has rejected a push (its input is full and the scan is past
	// position+1), drain it completely, once.
	var drained []uint64
	drain := func(obj runtime.Object) {
		if len(a.input) == cap(a.input) && objectRV(t, obj) > a.position+1 && drained == nil {
			drained = drainInput(a)
		}
	}
	hook.fn.Store(&drain)
	before := readStallResumeCounters(t)
	dispatcherDo(t, cacher, func(runPass syncPassFunc) {
		if served, expired := runPass(); served != 2 || expired != 0 {
			t.Fatalf("served=%d expired=%d, want both members served", served, expired)
		}
		if a.position != 100+2*inputCap || !a.unsynced || a.catchupEvents != int(2*inputCap) {
			t.Fatalf("a: position=%d unsynced=%v catchupEvents=%d, want blocked at %d after two rounds of %d",
				a.position, a.unsynced, a.catchupEvents, 100+2*inputCap, inputCap)
		}
		if b.position != end || b.unsynced {
			t.Fatalf("b: position=%d unsynced=%v, want synced at the history end %d", b.position, b.unsynced, end)
		}
		if cacher.stall.passPeriod != syncPassBusyPeriod {
			t.Fatalf("expected the busy period after a pass that left a member blocked, got %v", cacher.stall.passPeriod)
		}
	})
	hook.fn.Store(nil)
	assertExactSequence(t, drained, 101, 100+inputCap)
	assertExactSequence(t, drainInput(a), 101+inputCap, 100+2*inputCap)
	if got := readStallResumeCounters(t).deferred - before.deferred; got != float64(2*inputCap) {
		t.Fatalf("deferred events: want %d, got %v", 2*inputCap, got)
	}

	// Nothing drains during the next pass: exactly cap(input) more, and
	// the member stays blocked at the last accepted resourceVersion.
	dispatcherDo(t, cacher, func(runPass syncPassFunc) {
		if served, _ := runPass(); served != 1 {
			t.Fatalf("served=%d, want a alone", served)
		}
		if a.position != 100+3*inputCap || !a.unsynced {
			t.Fatalf("a: position=%d unsynced=%v, want blocked at %d", a.position, a.unsynced, 100+3*inputCap)
		}
	})
	assertExactSequence(t, drainInput(a), 101+2*inputCap, 100+3*inputCap)
}

// TestStallResumeCandidateCap: one pass serves at most maxWatchersPerSync
// candidates; the one left over is served by the next pass.
func TestStallResumeCandidateCap(t *testing.T) {
	cacher, _ := newStallResumeCacher(t)
	stallResumeAddPods(t, cacher, "ns", 101, 101)
	waitDispatched(t, cacher)
	ws := make([]*cacheWatcher, maxWatchersPerSync+1)
	for i := range ws {
		ws[i] = detachedWatcher(cacher, 10, namespacedName{})
	}
	registerUnsynced(t, cacher, ws, func(int) uint64 { return 100 })
	before := readStallResumeCounters(t)
	if served, expired := runSyncPass(t, cacher); served != maxWatchersPerSync || expired != 0 {
		t.Fatalf("served=%d expired=%d, want %d served", served, expired, maxWatchersPerSync)
	}
	if got := readStallResumeCounters(t).deferred - before.deferred; got != float64(maxWatchersPerSync) {
		t.Fatalf("deferred events: want %d, got %v", maxWatchersPerSync, got)
	}
	unsynced := 0
	dispatcherDo(t, cacher, func(syncPassFunc) {
		unsynced = len(cacher.stall.unsynced)
	})
	if unsynced != 1 {
		t.Fatalf("expected one candidate left over, got %d", unsynced)
	}
	if served, _ := runSyncPass(t, cacher); served != 1 {
		t.Fatalf("served=%d, want the left over candidate", served)
	}
	for i, w := range ws {
		if got := drainInput(w); len(got) != 1 || got[0] != 101 {
			t.Fatalf("watcher %d: got %v, want [101]", i, got)
		}
	}
}

// TestStallResumeLeadExpiresMidPass: the history advances past a lead's
// position between step 1 (the expiry check) and its cohort's interval
// open; the lead counts as served with nothing offered, keeps its position,
// and the next pass expires it.
func TestStallResumeLeadExpiresMidPass(t *testing.T) {
	hook := &indexerHook{}
	cacher, _ := newStallResumeCacher(t, hook.config)
	pinHistoryCapacity(t, cacher, 100)
	stallResumeAddPods(t, cacher, "ns", 101, 150)
	waitDispatched(t, cacher)
	// lead sits above low's window (ten accepted and twenty recorded as
	// owed), so it gets its own cohort.
	low := detachedWatcher(cacher, 10, namespacedName{})
	lead := detachedWatcher(cacher, 10, namespacedName{})
	registerUnsynced(t, cacher, []*cacheWatcher{low, lead}, func(i int) uint64 { return []uint64{100, 140}[i] })

	// During low's cohort (the first scanned event), 95 new events evict
	// 101..145 from the 100 event history; lead's position 140 is then
	// below the oldest servable resourceVersion, 146.
	fired := false
	advance := func(runtime.Object) {
		if fired {
			return
		}
		fired = true
		stallResumeAddPods(t, cacher, "ns", 200, 294)
	}
	hook.fn.Store(&advance)
	if served, expired := runSyncPass(t, cacher); served != 2 || expired != 0 {
		t.Fatalf("served=%d expired=%d, want both served (the lead with nothing) and none expired", served, expired)
	}
	hook.fn.Store(nil)
	if st := snapshot(t, cacher, lead); st.expired || !st.inSet || st.position != 140 {
		t.Fatalf("expected the lead unexpired in the set at 140 after the pass, got %+v", st)
	}
	if got := drainInput(lead); len(got) != 0 {
		t.Fatalf("the lead must be offered nothing, got %v", got)
	}
	waitDispatched(t, cacher)
	if _, expired := runSyncPass(t, cacher); expired != 2 {
		t.Fatalf("expected both members expired by the next pass, got %d", expired)
	}
	if st := snapshot(t, cacher, lead); !st.expired || st.inSet {
		t.Fatalf("expected the lead expired and out of the set, got %+v", st)
	}
}

// TestStallResumeTerminateAllDuringPass: a relist stops an unsynced watcher
// while a pass is dispatching, and the pass then forgets the same watcher in
// drain mode: the non-draining stop is latched in stopWatcherLocked, so the
// draining forget cannot reopen the drain window and done closes at
// finishDispatching.
func TestStallResumeTerminateAllDuringPass(t *testing.T) {
	cacher, _ := newStallResumeCacher(t)
	w := stallResumeWatch(t, cacher, "ns", 100)
	add := func(rv uint64) { stallResumeAddPods(t, cacher, "ns", rv, rv) }
	wedgeWatchers(t, cacher, []*cacheWatcher{w}, add, 101)
	dispatcherDo(t, cacher, func(syncPassFunc) {
		cacher.Lock()
		cacher.dispatching = true
		cacher.Unlock()
		cacher.terminateAllWatchers()
		w.forget(true)
		cacher.finishDispatching()
	})
	cacher.RLock()
	doneClosed := w.isDoneChannelClosedLocked()
	drain := w.drainInputBuffer
	cacher.RUnlock()
	if !doneClosed || drain {
		t.Fatalf("expected done closed and no drain mode, got doneClosed=%v drain=%v", doneClosed, drain)
	}
	if got := readAll(t, w, 10*time.Second); !got.closed || len(got.errors) != 0 {
		t.Fatalf("expected the goroutine to exit with a clean close, got closed=%v errors=%d", got.closed, len(got.errors))
	}
}

// TestStallResumeLeadByRoom: the lead of a cohort is the candidate with
// the most room in its input, not the lowest position: a client draining
// at full speed is served before one that frees a few slots per pass,
// so it catches up instead of sharing the push budget with every laggard.
func TestStallResumeLeadByRoom(t *testing.T) {
	cacher, _ := newStallResumeCacher(t)
	pinHistoryCapacity(t, cacher, 10000)
	stallResumeAddPods(t, cacher, "ns", 101, 340)
	waitDispatched(t, cacher)
	// laggard: low position, ten free slots; drainer: higher position,
	// an empty input with room for the 90 events it is behind. The
	// laggard would be the lead by position and, as the cohort scan stops
	// once every active member is blocked, would keep the drainer out of
	// its cohort.
	laggard := detachedWatcher(cacher, 100, namespacedName{})
	for range 90 {
		laggard.input <- &watchCacheEvent{}
	}
	drainer := detachedWatcher(cacher, 100, namespacedName{})
	registerUnsynced(t, cacher, []*cacheWatcher{laggard, drainer}, func(i int) uint64 { return []uint64{100, 250}[i] })

	dispatcherDo(t, cacher, func(runPass syncPassFunc) {
		if served, _ := runPass(); served != 2 {
			t.Fatalf("served=%d, want both", served)
		}
		if drainer.unsynced || drainer.position != 340 {
			t.Fatalf("drainer: unsynced=%v position=%d, want synced at the history end 340", drainer.unsynced, drainer.position)
		}
		if !laggard.unsynced || laggard.position != 110 {
			t.Fatalf("laggard: unsynced=%v position=%d, want blocked at 110", laggard.unsynced, laggard.position)
		}
	})
	assertExactSequence(t, drainInput(drainer), 251, 340)
	got := drainInput(laggard)
	assertExactSequence(t, got[90:], 101, 110)
}

// TestStallResumeLeadEager: eagerness depends only on the member's own
// behaviour. After a service, the first pass in which the member has room
// judges it: a fast drainer has an empty input, a slow one has freed a
// slot or a few. A member never served since it went unsynced is eager.
// Among equal room an eager member leads before a slow one whatever the
// cursor says, and the candidate cap keeps it; being left out of a pass
// by the cap does not change the judgement.
func TestStallResumeLeadEager(t *testing.T) {
	cacher, _ := newStallResumeCacher(t)
	pinHistoryCapacity(t, cacher, 10000)
	stallResumeAddPods(t, cacher, "ns", 101, 6000)
	waitDispatched(t, cacher)
	slow := detachedWatcher(cacher, 10, namespacedName{})
	fast := detachedWatcher(cacher, 10, namespacedName{})
	registerUnsynced(t, cacher, []*cacheWatcher{slow, fast}, func(i int) uint64 { return []uint64{100, 500}[i] })
	// Pass 1: both unserved, equal room; each gets its ten. fast empties
	// its input, slow frees half.
	runSyncPass(t, cacher)
	assertExactSequence(t, drainInput(fast), 501, 510)
	for range 5 {
		<-slow.input
	}
	// 512 members judged slow, with room, above the cursor: the cursor
	// rule alone would pick them first and the cap would keep them all.
	band := make([]*cacheWatcher, maxWatchersPerSync)
	for i := range band {
		band[i] = detachedWatcher(cacher, 10, namespacedName{})
	}
	registerUnsynced(t, cacher, band, func(int) uint64 { return 5000 })
	dispatcherDo(t, cacher, func(syncPassFunc) {
		for _, w := range band {
			w.servedPass, w.judged, w.fastDrainer = 1, true, false
		}
	})
	// Pass 2 judges fast and slow; fast, eager, leads and is kept by the
	// cap; slow is left out.
	runSyncPass(t, cacher)
	dispatcherDo(t, cacher, func(syncPassFunc) {
		if fast.judged || !fast.fastDrainer || fast.servedPass != 2 || fast.position != 520 {
			t.Errorf("fast: judged=%v fastDrainer=%v servedPass=%d position=%d, want judged fast and served", fast.judged, fast.fastDrainer, fast.servedPass, fast.position)
		}
		if !slow.judged || slow.fastDrainer || slow.servedPass != 1 || slow.position != 110 {
			t.Errorf("slow: judged=%v fastDrainer=%v servedPass=%d position=%d, want judged slow and left out", slow.judged, slow.fastDrainer, slow.servedPass, slow.position)
		}
		if band[0].servedPass != 2 {
			t.Errorf("band: servedPass=%d, want 2", band[0].servedPass)
		}
	})
	assertExactSequence(t, drainInput(fast), 511, 520)
	assertExactSequence(t, drainInput(slow), 106, 110)
	// Pass 3: slow now has full room too, but its judgement stands and
	// it sits below the cursor: left out again, judgement unchanged.
	runSyncPass(t, cacher)
	dispatcherDo(t, cacher, func(syncPassFunc) {
		if fast.servedPass != 3 || fast.position != 530 {
			t.Errorf("fast: servedPass=%d position=%d, want served first again", fast.servedPass, fast.position)
		}
		if slow.servedPass != 1 || !slow.judged || slow.fastDrainer {
			t.Errorf("slow: servedPass=%d judged=%v fastDrainer=%v, want left out with its judgement kept", slow.servedPass, slow.judged, slow.fastDrainer)
		}
	})
	assertExactSequence(t, drainInput(fast), 521, 530)
	// Once the band has no room, slow is served: it was left out, not
	// demoted.
	for _, w := range band {
		for len(w.input) < cap(w.input) {
			w.input <- &watchCacheEvent{}
		}
	}
	runSyncPass(t, cacher)
	assertExactSequence(t, drainInput(slow), 111, 120)
	assertExactSequence(t, drainInput(fast), 531, 540)
}

// TestStallResumeCandidateCapKeepsPreferred: when more than
// maxWatchersPerSync watchers have room, the candidates are the ones the
// lead choice prefers, not the lowest positions: a client draining at full
// speed above 512 laggards is served in the first pass.
func TestStallResumeCandidateCapKeepsPreferred(t *testing.T) {
	cacher, _ := newStallResumeCacher(t)
	pinHistoryCapacity(t, cacher, 10000)
	stallResumeAddPods(t, cacher, "ns", 101, 3000)
	waitDispatched(t, cacher)
	laggards := make([]*cacheWatcher, maxWatchersPerSync)
	for i := range laggards {
		laggards[i] = detachedWatcher(cacher, 10, namespacedName{})
		for range 5 {
			laggards[i].input <- &watchCacheEvent{}
		}
	}
	drainer := detachedWatcher(cacher, 10, namespacedName{})
	registerUnsynced(t, cacher, laggards, func(i int) uint64 { return 100 + uint64(i) })
	registerUnsynced(t, cacher, []*cacheWatcher{drainer}, func(int) uint64 { return 2000 })
	// The drainer, served first, takes its ten; the laggards take what
	// the push budget leaves.
	served, _ := runSyncPass(t, cacher)
	assertExactSequence(t, drainInput(drainer), 2001, 2010)
	laggardsServed := 0
	for _, w := range laggards {
		if w.servedPass != 0 {
			laggardsServed++
		}
	}
	if served < 2 || laggardsServed != served-1 {
		t.Fatalf("served=%d laggards served=%d, want the drainer and served-1 laggards", served, laggardsServed)
	}
}

// drainReal takes every real event (resourceVersion above 0) out of a
// detached watcher's input and leaves the filler events in place, so the
// watcher keeps the same free share it had before the pass.
func drainReal(w *cacheWatcher) []uint64 {
	var rvs []uint64
	filler := 0
	for {
		select {
		case ev := <-w.input:
			if ev.ResourceVersion == 0 {
				filler++
			} else {
				rvs = append(rvs, ev.ResourceVersion)
			}
			continue
		default:
		}
		break
	}
	for range filler {
		w.input <- &watchCacheEvent{}
	}
	return rvs
}

// TestStallResumeCrossCapacityFairness: 512 watchers with a 1000 slot
// input that keep 11 slots free (a wide pod watcher reading slowly) and
// three 10 slot watchers with an empty input at a low position. Room is a
// share of the capacity, so the small watchers are served in the first
// pass and every pass after, instead of losing to 11 free slots forever.
func TestStallResumeCrossCapacityFairness(t *testing.T) {
	cacher, _ := newStallResumeCacher(t)
	pinHistoryCapacity(t, cacher, 10000)
	stallResumeAddPods(t, cacher, "ns", 101, 3000)
	waitDispatched(t, cacher)
	wide := make([]*cacheWatcher, maxWatchersPerSync)
	for i := range wide {
		wide[i] = detachedWatcher(cacher, 1000, namespacedName{})
		for range 1000 - 11 {
			wide[i].input <- &watchCacheEvent{}
		}
	}
	small := make([]*cacheWatcher, 3)
	for i := range small {
		small[i] = detachedWatcher(cacher, 10, namespacedName{})
	}
	registerUnsynced(t, cacher, wide, func(i int) uint64 { return 1500 + uint64(i%7) })
	registerUnsynced(t, cacher, small, func(i int) uint64 { return 100 + uint64(i) })
	for pass := 1; pass <= 5; pass++ {
		runSyncPass(t, cacher)
		for _, w := range wide {
			drainReal(w)
		}
		for i, w := range small {
			got := drainReal(w)
			if len(got) != 10 {
				t.Fatalf("pass %d: small watcher %d got %v, want ten events", pass, i, got)
			}
		}
		dispatcherDo(t, cacher, func(syncPassFunc) {
			for i, w := range small {
				if w.servedPass != cacher.stall.passCount || w.expired {
					t.Fatalf("pass %d: small watcher %d servedPass=%d expired=%v", pass, i, w.servedPass, w.expired)
				}
			}
		})
	}
}

// TestStallResumeOnePassHiccup: 512 equal watchers in a lockstep band that
// read exactly what each pass gives them, and one watcher far behind them
// that drains fully after every pass, except once. Missing one pass with a
// full input does not demote it: it is served again on the very next pass
// it has room and every pass after, and never expires.
func TestStallResumeOnePassHiccup(t *testing.T) {
	cacher, _ := newStallResumeCacher(t)
	pinHistoryCapacity(t, cacher, 10000)
	stallResumeAddPods(t, cacher, "ns", 101, 3000)
	waitDispatched(t, cacher)
	band := make([]*cacheWatcher, maxWatchersPerSync)
	for i := range band {
		band[i] = detachedWatcher(cacher, 10, namespacedName{})
	}
	behind := detachedWatcher(cacher, 10, namespacedName{})
	registerUnsynced(t, cacher, band, func(int) uint64 { return 1500 })
	registerUnsynced(t, cacher, []*cacheWatcher{behind}, func(int) uint64 { return 100 })
	next := uint64(3001)
	// The client does not read after pass 2, so behind is full during
	// pass 3.
	const hiccupPass = 3
	for pass := 1; pass <= 40; pass++ {
		runSyncPass(t, cacher)
		for _, w := range band {
			drainReal(w)
		}
		dispatcherDo(t, cacher, func(syncPassFunc) {
			if want := pass != hiccupPass; (behind.servedPass == cacher.stall.passCount) != want {
				t.Fatalf("pass %d: behind servedPass=%d passCount=%d, want served=%v", pass, behind.servedPass, cacher.stall.passCount, want)
			}
			if behind.expired {
				t.Fatalf("pass %d: behind expired", pass)
			}
			for _, w := range band {
				if !w.unsynced {
					w.unsynced, w.servedPass = true, 0
					cacher.stall.unsynced[w] = struct{}{}
				}
			}
		})
		if pass != hiccupPass-1 {
			if got := drainReal(behind); len(got) != 10 {
				t.Fatalf("pass %d: behind got %v, want ten events", pass, got)
			}
		}
		stallResumeAddPods(t, cacher, "ns", next, next+19)
		next += 20
	}
}
