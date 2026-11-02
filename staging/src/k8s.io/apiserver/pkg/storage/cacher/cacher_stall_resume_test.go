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

// The tests run on a fake clock and drive passes through the dispatcher hook.
// The dispatcher goroutine is the only writer of a watcher's position and
// unsynced fields, so the hook is the race free way to read them. A function
// that runs on that goroutine must not call t.Fatal.

import (
	"context"
	"fmt"
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
			metrics.WatcherStalls, metrics.WatcherDeferredEvents,
			metrics.WatcherCatchupEvents, metrics.WatcherSkippedEvents, metrics.StalledWatchers,
			metrics.TerminatedWatchersCounter,
		} {
			_ = registry.Register(m)
		}
	})
}

// stallResumeCounters is a snapshot of the stall/resume counters for pods.
type stallResumeCounters struct {
	stalls, deferred        float64
	roundSamples            uint64
	roundSum                float64
	expired, expiredInitial float64
	unresponsive, skipped   float64
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
		expired:        read(metrics.TerminatedWatchersCounter.WithLabelValues("", "pods", metrics.TerminationReasonResourceExpired)),
		expiredInitial: read(metrics.TerminatedWatchersCounter.WithLabelValues("", "pods", metrics.TerminationReasonResourceExpiredInitial)),
		unresponsive:   read(metrics.TerminatedWatchersCounter.WithLabelValues("", "pods", metrics.TerminationReasonUnresponsive)),
		skipped:        read(metrics.WatcherSkippedEvents.WithLabelValues("", "pods")),
	}
}

func stalledWatchersGauge(t testing.TB) float64 {
	t.Helper()
	ensureStallResumeMetrics()
	v, err := testutil.GetGaugeMetricValue(metrics.StalledWatchers.WithLabelValues("", "pods"))
	if err != nil {
		t.Fatalf("reading the stalled watchers gauge: %v", err)
	}
	return v
}

// waitStalledWatchersGauge polls because a stopped Cacher withdraws its count
// when its dispatcher exits, after Stop returned.
func waitStalledWatchersGauge(t testing.TB, what string, want float64) {
	t.Helper()
	if err := wait.PollUntilContextTimeout(context.Background(), time.Millisecond, 10*time.Second, true, func(context.Context) (bool, error) {
		return stalledWatchersGauge(t) == want, nil
	}); err != nil {
		t.Fatalf("%s: stalled watchers gauge %v, want %v", what, stalledWatchersGauge(t), want)
	}
}

// setStallResumeGate enables or disables WatchCacheStallResume for a test.
func setStallResumeGate(t testing.TB, enabled bool) {
	t.Helper()
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.WatchCacheStallResume, enabled)
}

// newStallResumeCacher enables the feature gate for the duration of the test
// and builds a cacher over MockStorage on a fake clock. Its list
// resourceVersion is 100, so watches can start there.
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
			t.Errorf("failed to add a pod: %v", err)
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
	return startedWatcher(t, w)
}

// startedWatcher returns the cacheWatcher behind w once its goroutine has taken
// the initial interval, which holds one input slot until then.
func startedWatcher(t testing.TB, w watch.Interface) *cacheWatcher {
	t.Helper()
	cw, ok := w.(*cacheWatcher)
	if !ok {
		t.Fatalf("expected a *cacheWatcher, got %T", w)
	}
	t.Cleanup(cw.Stop)
	if err := wait.PollUntilContextTimeout(context.Background(), 100*time.Microsecond, 10*time.Second, true, func(context.Context) (bool, error) {
		return len(cw.input) == 0, nil
	}); err != nil {
		t.Fatalf("the watcher goroutine never took the initial interval: %v", err)
	}
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

// readEvents reads the watch until it has seen the object at resourceVersion
// until, the channel closes or the timeout fires. Bookmarks and errors are
// recorded, not counted. It is safe to call from any goroutine.
func readEvents(w watch.Interface, until uint64, timeout time.Duration) (collected, error) {
	got := collected{objects: map[uint64]runtime.Object{}, types: map[uint64]watch.EventType{}}
	deadline := time.After(timeout)
	for {
		select {
		case ev, ok := <-w.ResultChan():
			if !ok {
				got.closed = true
				return got, nil
			}
			if ev.Type == watch.Error {
				status, ok := ev.Object.(*metav1.Status)
				if !ok {
					return got, fmt.Errorf("error event without a Status: %#v", ev.Object)
				}
				got.errors = append(got.errors, status)
				continue
			}
			rv, err := storage.APIObjectVersioner{}.ObjectResourceVersion(ev.Object)
			if err != nil {
				return got, fmt.Errorf("parsing resource version: %w", err)
			}
			if ev.Type == watch.Bookmark {
				got.bookmarks = append(got.bookmarks, rv)
				continue
			}
			got.rvs = append(got.rvs, rv)
			got.objects[rv] = ev.Object
			got.types[rv] = ev.Type
			if rv >= until {
				return got, nil
			}
		case <-deadline:
			return got, fmt.Errorf("timed out waiting for events; got %d object events so far (last %v)", len(got.rvs), got.rvs)
		}
	}
}

// readUntil is readEvents for the test goroutine.
func readUntil(t testing.TB, w watch.Interface, until uint64, timeout time.Duration) collected {
	t.Helper()
	got, err := readEvents(w, until, timeout)
	if err != nil {
		t.Fatal(err)
	}
	return got
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
		readers.Go(func() {
			var err error
			if results[i], err = readEvents(w, until, 20*time.Second); err != nil {
				t.Error(err)
			}
		})
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
// events both still need, and a member that rejected a push mid window keeps
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
	// b needs missed..; a needs missed+5..; the cohort scan pushed
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
	w := startedWatcher(t, wi)
	if w.selector.indexValue == nil || *w.selector.indexValue != "node-1" || cap(w.input) != 10 {
		t.Fatalf("expected a trigger-scoped watcher on node-1 with input cap 10, got selector=%+v cap=%d", w.selector, cap(w.input))
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

// TestStallResumeScopes: a watcher receives from the history exactly the events
// its scope selects, in order, and its position follows the scan across events
// it does not select.
func TestStallResumeScopes(t *testing.T) {
	for _, tc := range []struct {
		name            string
		namespace, pick string
		wantScope       namespacedName
	}{
		{name: "cluster wide"},
		{name: "namespace", namespace: "ns-a", wantScope: namespacedName{namespace: "ns-a"}},
		{name: "name", pick: "target", wantScope: namespacedName{name: "target"}},
		{name: "namespace and name", namespace: "ns-a", pick: "target", wantScope: namespacedName{namespace: "ns-a", name: "target"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cacher, _ := newStallResumeCacher(t)
			ctx, key, pred := context.Background(), "/pods", storage.Everything
			if tc.namespace != "" {
				ctx, key = request.WithNamespace(ctx, tc.namespace), "/pods/"+tc.namespace
			}
			if tc.pick != "" {
				pred = storage.SelectionPredicate{Label: labels.Everything(), Field: fields.OneTermEqualSelector("metadata.name", tc.pick)}
			}
			w := scopedWatch(t, cacher, ctx, key, pred)
			if w.selector.scope != tc.wantScope {
				t.Fatalf("expected the scope %+v, got %+v", tc.wantScope, w.selector.scope)
			}

			var want []uint64
			write := func(namespace, name string, rv uint64) {
				updatePod(t, cacher, namespace, name, rv)
				if (tc.namespace == "" || namespace == tc.namespace) && (tc.pick == "" || name == tc.pick) {
					want = append(want, rv)
				}
			}
			rv := wedgeWatchers(t, cacher, []*cacheWatcher{w}, func(rv uint64) { write("ns-a", "target", rv) }, 101)
			for _, batch := range []struct {
				namespace, name string
				count           int
			}{{"ns-a", "other", 200}, {"ns-b", "target", 1}, {"ns-b", "other", 100}, {"ns-a", "target", 1}, {"ns-c", "other", 50}} {
				for range batch.count {
					write(batch.namespace, batch.name, rv)
					rv++
				}
			}
			end := rv - 1

			got := resyncWhileReading(t, cacher, []*cacheWatcher{w}, want[len(want)-1])[0]
			if !slices.Equal(got.rvs, want) {
				t.Fatalf("want %v, got %v", want, got.rvs)
			}
			if st := snapshot(t, cacher, w); st.unsynced || st.position != end {
				t.Fatalf("expected the position at the history end %d, got %+v", end, st)
			}
			if extra := readAll(t, w, 100*time.Millisecond); len(extra.rvs) != 0 {
				t.Fatalf("unexpected extra events %v", extra.rvs)
			}
		})
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
		// The initial listing (30 pods, all delivered at the list RV) and
		// then the input backlog, in order.
		if len(delivered) != 30+cap(w.input) {
			t.Fatalf("expected the 30 initial objects and %d live events before the 410, got %d: %v", cap(w.input), len(delivered), delivered)
		}
		assertExactSequence(t, delivered[30:], 131, 130+uint64(cap(w.input)))
		after := readStallResumeCounters(t)
		if after.expiredInitial-before.expiredInitial != 1 || after.expired-before.expired != 0 {
			t.Errorf("expected terminated{resource_expired_initial} +1 only, got initial %v expired %v", after.expiredInitial-before.expiredInitial, after.expired-before.expired)
		}
	})
}

// TestStallResumeStopOrders runs every order of a client Stop, a relist
// (terminateAllWatchers) and the draining stop of an expiring pass through a
// real pass. The watcher's forget function is wrapped, so the other stop lands
// exactly around the pass's forget(true), while the Cacher is dispatching.
// A stop that nobody follows with a read must end the goroutine with no read
// at all; an expiry that no client Stop overrides must deliver the backlog
// and the 410.
func TestStallResumeStopOrders(t *testing.T) {
	for _, tc := range []struct {
		name string
		// before runs ahead of the pass, outside a dispatch.
		before func(c *Cacher, w *cacheWatcher)
		// during replaces the pass's forget(true); expire is that call.
		during func(c *Cacher, w *cacheWatcher, expire func())
		// after runs once the pass is over, outside a dispatch.
		after func(c *Cacher, w *cacheWatcher)
		// clientStops is how many times the functions above call Stop.
		clientStops int32
		// wantExpired is what the pass reports.
		wantExpired int
		// wantDrained: done stays open and the client gets the backlog and
		// the 410. Otherwise done is closed and the goroutine exits unread.
		wantDrained bool
	}{
		{
			name:        "expiry alone",
			during:      func(_ *Cacher, _ *cacheWatcher, expire func()) { expire() },
			wantExpired: 1,
			wantDrained: true,
		},
		{
			name:        "client Stop, then expiry in the same pass",
			during:      func(_ *Cacher, w *cacheWatcher, expire func()) { w.Stop(); expire() },
			clientStops: 1,
			wantExpired: 1,
		},
		{
			name:        "terminateAllWatchers, then expiry in the same pass",
			during:      func(c *Cacher, _ *cacheWatcher, expire func()) { c.terminateAllWatchers(); expire() },
			wantExpired: 1,
		},
		{
			name:        "expiry, then client Stop in the same pass",
			during:      func(_ *Cacher, w *cacheWatcher, expire func()) { expire(); w.Stop() },
			clientStops: 1,
			wantExpired: 1,
		},
		{
			// The expiry unregistered the watcher, so the relist does not
			// see it and the drain goes on.
			name:        "expiry, then terminateAllWatchers in the same pass",
			during:      func(c *Cacher, _ *cacheWatcher, expire func()) { expire(); c.terminateAllWatchers() },
			wantExpired: 1,
			wantDrained: true,
		},
		{
			// Production code stops a watcher with draining once. A second
			// draining stop is not a non-draining one, so the drain goes on.
			name:        "expiry, then a second draining stop in the same pass",
			during:      func(_ *Cacher, _ *cacheWatcher, expire func()) { expire(); expire() },
			wantExpired: 1,
			wantDrained: true,
		},
		{
			name:        "expiry, then client Stop after the pass",
			during:      func(_ *Cacher, _ *cacheWatcher, expire func()) { expire() },
			after:       func(_ *Cacher, w *cacheWatcher) { w.Stop() },
			clientStops: 1,
			wantExpired: 1,
		},
		{
			// The pass drops a stopped member; a draining stop that still
			// came finds done closed and cannot open it again.
			name:        "client Stop outside a dispatch, then a draining stop",
			before:      func(_ *Cacher, w *cacheWatcher) { w.Stop() },
			after:       func(_ *Cacher, w *cacheWatcher) { w.forget(true) },
			clientStops: 1,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cacher, _ := newStallResumeCacher(t)
			pinHistoryCapacity(t, cacher, 100)
			w := stallResumeWatch(t, cacher, "ns", 100)
			add := func(rv uint64) { stallResumeAddPods(t, cacher, "ns", rv, rv) }
			next := wedgeWatchers(t, cacher, []*cacheWatcher{w}, add, 101)
			stallResumeAddPods(t, cacher, "ns", next, next+199)
			waitDispatched(t, cacher)

			// The goroutine stops its watcher when it exits, so one
			// forget(false) more than the client's reports the exit without
			// a read.
			var passForgets, nonDrainingStops atomic.Int32
			forget := w.forget
			w.forget = func(drain bool) {
				if !drain {
					forget(false)
					nonDrainingStops.Add(1)
					return
				}
				if tc.during != nil && passForgets.Add(1) == 1 {
					tc.during(cacher, w, func() { forget(true) })
					return
				}
				forget(true)
			}

			if tc.before != nil {
				tc.before(cacher, w)
			}
			_, expired := runSyncPass(t, cacher)
			if tc.after != nil {
				tc.after(cacher, w)
			}
			// A later pass has nothing left to stop.
			_, expiredAgain := runSyncPass(t, cacher)
			if expired != tc.wantExpired || expiredAgain != 0 {
				t.Fatalf("expected %d expired and then none, got %d and %d", tc.wantExpired, expired, expiredAgain)
			}
			if tc.during != nil && passForgets.Load() != 1 {
				t.Fatalf("expected one draining stop from the passes, got %d", passForgets.Load())
			}

			cacher.RLock()
			stopped, doneClosed := w.stopped, w.isDoneChannelClosedLocked()
			cacher.RUnlock()
			if !stopped || doneClosed == tc.wantDrained {
				t.Fatalf("expected stopped and doneClosed=%v, got stopped=%v doneClosed=%v", !tc.wantDrained, stopped, doneClosed)
			}

			if tc.wantDrained {
				assertExactSequence(t, expectExpiredThenClose(t, w), 101, next-2)
				return
			}
			if err := wait.PollUntilContextTimeout(context.Background(), time.Millisecond, 10*time.Second, true, func(context.Context) (bool, error) {
				return nonDrainingStops.Load() > tc.clientStops, nil
			}); err != nil {
				t.Fatalf("the watcher goroutine is still blocked with nobody reading")
			}
			// Only what result already held is left, and no 410.
			got := readAll(t, w, 10*time.Second)
			if !got.closed || len(got.errors) != 0 {
				t.Fatalf("expected a clean close, got closed=%v errors=%d", got.closed, len(got.errors))
			}
			assertExactSequence(t, got.rvs, 101, 100+uint64(cap(w.result)))
		})
	}
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

// TestStallResumeStalledWatchersGauge: the gauge is the size of the unsynced
// set. It rises on a stall and falls on a resync, on an expiry, when a pass
// drops a stopped member, and when the Cacher stops.
func TestStallResumeStalledWatchersGauge(t *testing.T) {
	// wedged returns a Cacher with one watcher that has just gone unsynced.
	wedged := func(t *testing.T) (c *Cacher, w *cacheWatcher, next uint64) {
		waitStalledWatchersGauge(t, "before the stall", 0)
		c, _ = newStallResumeCacher(t)
		w = stallResumeWatch(t, c, "ns", 100)
		add := func(rv uint64) { stallResumeAddPods(t, c, "ns", rv, rv) }
		next = wedgeWatchers(t, c, []*cacheWatcher{w}, add, 101)
		if got := stalledWatchersGauge(t); got != 1 {
			t.Fatalf("after the stall: gauge %v, want 1", got)
		}
		return c, w, next
	}

	t.Run("resync", func(t *testing.T) {
		c, w, next := wedged(t)
		stallResumeAddPods(t, c, "ns", next, next+50)
		resyncWhileReading(t, c, []*cacheWatcher{w}, next+50)
		if got := stalledWatchersGauge(t); got != 0 {
			t.Fatalf("after the resync: gauge %v, want 0", got)
		}
	})

	t.Run("expiry", func(t *testing.T) {
		c, w, next := wedged(t)
		pinHistoryCapacity(t, c, 100)
		stallResumeAddPods(t, c, "ns", next, next+199)
		if _, expired := runSyncPass(t, c); expired != 1 {
			t.Fatalf("expected the watcher to expire, got %d", expired)
		}
		if got := stalledWatchersGauge(t); got != 0 {
			t.Fatalf("after the expiry: gauge %v, want 0", got)
		}
		expectExpiredThenClose(t, w)
	})

	t.Run("stopped member dropped", func(t *testing.T) {
		c, w, _ := wedged(t)
		w.Stop()
		runSyncPass(t, c)
		if got := stalledWatchersGauge(t); got != 0 {
			t.Fatalf("after the pass: gauge %v, want 0", got)
		}
	})

	t.Run("cacher stop", func(t *testing.T) {
		c, _, _ := wedged(t)
		c.Stop()
		waitStalledWatchersGauge(t, "after the stop", 0)
	})
}

// TestStallResumeStalledWatchersGaugeSharedSeries: two Cachers of one resource
// (a CRD with two served versions) share a series. It reads the sum, and a
// stopped Cacher withdraws only its own count.
func TestStallResumeStalledWatchersGaugeSharedSeries(t *testing.T) {
	waitStalledWatchersGauge(t, "before the test", 0)
	wedge := func(c *Cacher, n int) {
		t.Helper()
		ws := make([]*cacheWatcher, n)
		for i := range ws {
			ws[i] = stallResumeWatch(t, c, "ns", 100)
		}
		add := func(rv uint64) { stallResumeAddPods(t, c, "ns", rv, rv) }
		wedgeWatchers(t, c, ws, add, 101)
	}
	a, _ := newStallResumeCacher(t)
	b, _ := newStallResumeCacher(t)
	wedge(a, 2)
	wedge(b, 3)
	if got := stalledWatchersGauge(t); got != 5 {
		t.Fatalf("gauge %v, want the sum 5", got)
	}
	b.Stop()
	waitStalledWatchersGauge(t, "after one stop", 2)
	a.Stop()
	waitStalledWatchersGauge(t, "after both stops", 0)
}

// TestStallResumeSkippedEventsCounter: a watcher that resynced at the history
// end while events were queued for the dispatcher has each of them skipped
// once and delivered once. An unsynced watcher moves the counter by nothing.
func TestStallResumeSkippedEventsCounter(t *testing.T) {
	t.Run("events queued at the resync", func(t *testing.T) {
		cacher, _ := newStallResumeCacher(t)
		w := stallResumeWatch(t, cacher, "ns", 100)
		add := func(rv uint64) { stallResumeAddPods(t, cacher, "ns", rv, rv) }
		next := wedgeWatchers(t, cacher, []*cacheWatcher{w}, add, 101)
		first := readAll(t, w, 100*time.Millisecond)
		waitInputDrained(t, cacher, w)
		before := readStallResumeCounters(t)
		const queued = 5
		last := next + queued - 1
		dispatcherDo(t, cacher, func(runPass syncPassFunc) {
			for rv := next; rv <= last; rv++ {
				add(rv)
			}
			runPass()
			if w.unsynced || w.position != last {
				t.Errorf("expected a synced watcher at %d, got unsynced=%v position=%d", last, w.unsynced, w.position)
			}
		})
		waitDispatched(t, cacher)
		if got := readStallResumeCounters(t).skipped - before.skipped; got != queued {
			t.Fatalf("skipped events +%v, want +%d", got, queued)
		}
		got := readUntil(t, w, last, 10*time.Second)
		assertExactSequence(t, slices.Concat(first.rvs, got.rvs), 101, last)
		if extra := readAll(t, w, 300*time.Millisecond); len(extra.rvs) != 0 {
			t.Fatalf("duplicate delivery of %v", extra.rvs)
		}
	})

	t.Run("events dispatched while unsynced", func(t *testing.T) {
		cacher, _ := newStallResumeCacher(t)
		w := stallResumeWatch(t, cacher, "ns", 100)
		add := func(rv uint64) { stallResumeAddPods(t, cacher, "ns", rv, rv) }
		next := wedgeWatchers(t, cacher, []*cacheWatcher{w}, add, 101)
		before := readStallResumeCounters(t)
		stallResumeAddPods(t, cacher, "ns", next, next+50)
		waitDispatched(t, cacher)
		resyncWhileReading(t, cacher, []*cacheWatcher{w}, next+50)
		stallResumeAddPods(t, cacher, "ns", next+51, next+60)
		got := readUntil(t, w, next+60, 10*time.Second)
		assertExactSequence(t, got.rvs, next+51, next+60)
		if after := readStallResumeCounters(t); after.skipped != before.skipped {
			t.Fatalf("skipped events +%v, want +0", after.skipped-before.skipped)
		}
	})
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
	if after.stalls-before.stalls != 3 || after.roundSamples-before.roundSamples != 3 || after.deferred-before.deferred != 3 {
		t.Errorf("expected 3 stalls, 3 resyncs and 3 deferred events (one missed event each), got stalls=%v resyncs=%v deferred=%v",
			after.stalls-before.stalls, after.roundSamples-before.roundSamples, after.deferred-before.deferred)
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

	// With the input still full, a pass after the short period serves nothing
	// and backs off to the idle period.
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
		return readStallResumeCounters(t).roundSamples-before.roundSamples == 1, nil
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
	if after.stalls != before.stalls || after.deferred != before.deferred || after.roundSamples != before.roundSamples {
		t.Errorf("stall counters must not move with the gate off: %+v -> %+v", before, after)
	}
	if w.position != 0 || w.unsynced {
		t.Errorf("gate off must not touch the stall fields: position=%d unsynced=%v", w.position, w.unsynced)
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

// scopedWatch opens a watch with the given context, key and predicate and
// returns the cacheWatcher behind it.
func scopedWatch(t testing.TB, c *Cacher, ctx context.Context, key string, pred storage.SelectionPredicate) *cacheWatcher {
	t.Helper()
	w, err := c.Watch(ctx, key, storage.ListOptions{ResourceVersion: "100", Predicate: pred, Recursive: true})
	if err != nil {
		t.Fatalf("Failed to create watch: %v", err)
	}
	return startedWatcher(t, w)
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
	w := newCacheWatcher(chanSize, filterAny, emptyFunc, storage.APIObjectVersioner{}, c.clock.Now().Add(time.Hour), false, c.groupResource, metrics.NewNoopWatcherMetricsObservers(), c.clock, "", watchSelector{scope: scope}, 0, 0)
	w.stallMetrics = c.stall.metrics
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
			c.stall.metrics.StalledWatchers.Inc()
		}
	})
}

// drainInput takes everything out of a detached watcher's input and returns
// the resourceVersions in order.
func drainInput(w *cacheWatcher) []uint64 {
	var rvs []uint64
	for {
		select {
		case item := <-w.input:
			rvs = append(rvs, item.event.ResourceVersion)
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
		t.Errorf("parsing resource version: %v", err)
	}
	return rv
}

// TestStallResumePassPeriod: the period after a pass follows what the pass did.
// The busy period is nine times the pass duration, from syncPassBusyPeriod up to
// syncPassPeriod. A pass that finds nothing goes idle only after a normal period.
func TestStallResumePassPeriod(t *testing.T) {
	for _, tc := range []struct {
		name            string
		served, expired int
		busy            bool
		duration        time.Duration
		previous        time.Duration
		want            time.Duration
	}{
		{name: "busy, instant pass", busy: true, want: syncPassBusyPeriod},
		{name: "busy, short pass", busy: true, duration: 50 * time.Microsecond, want: syncPassBusyPeriod},
		{name: "busy, 1ms pass", served: 5, busy: true, duration: time.Millisecond, want: 9 * time.Millisecond},
		{name: "busy, 5ms pass, capped", busy: true, duration: 5 * time.Millisecond, want: syncPassPeriod},
		{name: "busy, descheduled 14ms pass, capped", busy: true, duration: 14 * time.Millisecond, want: syncPassPeriod},
		{name: "nothing done", previous: syncPassPeriod, want: syncPassIdlePeriod},
		{name: "nothing done, slow pass", duration: 5 * time.Millisecond, previous: syncPassPeriod, want: syncPassIdlePeriod},
		{name: "served", served: 1, previous: syncPassPeriod, want: syncPassPeriod},
		{name: "expired", expired: 1, previous: syncPassPeriod, want: syncPassPeriod},
		{name: "nothing done after the busy period", previous: syncPassBusyPeriod, want: syncPassPeriod},
		{name: "nothing done after the idle period", previous: syncPassIdlePeriod, want: syncPassIdlePeriod},
		{name: "served, slow pass", served: 1, duration: 5 * time.Millisecond, previous: syncPassPeriod, want: syncPassPeriod},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := nextPassPeriod(tc.served, tc.expired, tc.busy, tc.duration, tc.previous); got != tc.want {
				t.Fatalf("nextPassPeriod(%d, %d, %v, %v, %v) = %v, want %v", tc.served, tc.expired, tc.busy, tc.duration, tc.previous, got, tc.want)
			}
		})
	}
}

// TestStallResumePushBudget: 512 draining members in one cohort would take
// 512 times cap(input) pushes in one pass. The push budget ends the scan and
// the pass instead. It is checked between events, so every member of the
// cohort got every scanned event, every position follows the scan, the pass
// selects the busy period, and the next pass carries on from there: no
// member skips an event.
func TestStallResumePushBudget(t *testing.T) {
	cacher, _ := newStallResumeCacher(t)
	const members, end = maxWatchersPerSync - 1, 150
	stallResumeAddPods(t, cacher, "ns", 101, end)
	waitDispatched(t, cacher)
	ws := make([]*cacheWatcher, members)
	for i := range ws {
		ws[i] = detachedWatcher(cacher, 100, namespacedName{})
	}
	// far has room too, but its position is above what the budget lets the
	// first cohort scan, and the spent budget leaves no second cohort.
	far := detachedWatcher(cacher, 100, namespacedName{})
	registerUnsynced(t, cacher, ws, func(int) uint64 { return 100 })
	registerUnsynced(t, cacher, []*cacheWatcher{far}, func(int) uint64 { return end - 10 })
	// Every member joins at the first scanned event, so the scan stops
	// after the first event that brings the pushes to the budget or above.
	const perPass = (syncPushBudget + members - 1) / members
	before := readStallResumeCounters(t)
	dispatcherDo(t, cacher, func(runPass syncPassFunc) {
		if served, expired := runPass(); served != members || expired != 0 {
			t.Errorf("served=%d expired=%d, want the %d members of the first cohort", served, expired, members)
		}
		if cacher.stall.passPeriod != syncPassBusyPeriod {
			t.Errorf("period %v, want the busy period (the push budget ran out)", cacher.stall.passPeriod)
		}
		if cacher.stall.syncCursor != 100+perPass {
			t.Errorf("cursor %d, want the last scanned resourceVersion %d", cacher.stall.syncCursor, 100+perPass)
		}
		for i, w := range ws {
			if w.position != 100+perPass || !w.unsynced || len(w.input) != perPass {
				t.Errorf("member %d: position=%d unsynced=%v len(input)=%d, want %d events and position %d", i, w.position, w.unsynced, len(w.input), perPass, 100+perPass)
			}
		}
		if far.position != end-10 || len(far.input) != 0 {
			t.Errorf("far: position=%d len(input)=%d, want it untouched", far.position, len(far.input))
		}
	})
	if got := readStallResumeCounters(t).deferred - before.deferred; got != members*perPass {
		t.Fatalf("deferred events %v, want %d", got, members*perPass)
	}

	// The members sit at the cursor, so they lead again; far joins once
	// the scan reaches it, and everyone resyncs at the history end.
	for pass := 2; ; pass++ {
		runSyncPass(t, cacher)
		unsynced := 0
		dispatcherDo(t, cacher, func(syncPassFunc) { unsynced = len(cacher.stall.unsynced) })
		if unsynced == 0 {
			break
		}
		if pass > (end-100)/perPass+2 {
			t.Fatalf("%d watchers still unsynced after %d passes", unsynced, pass)
		}
	}
	for _, w := range ws {
		assertExactSequence(t, drainInput(w), 101, end)
	}
	assertExactSequence(t, drainInput(far), end-9, end)
}

// TestStallResumeBudgetSpentAtHistoryEnd: a budget that runs out exactly on
// the last event of the history does not hide the history end: the cohort
// resyncs in that pass. With one more event in the history it stays
// unsynced at the last scanned resourceVersion.
func TestStallResumeBudgetSpentAtHistoryEnd(t *testing.T) {
	for _, tc := range []struct {
		name    string
		members int
		scope   namespacedName
		// scanned is how many events the budget under test lets one pass scan.
		scanned uint64
	}{
		// Nothing is selected, so only the scan budget counts.
		{name: "scan budget", members: 1, scope: namespacedName{namespace: "none"}, scanned: syncScanBudget - syncIntervalOpenCost},
		{name: "push budget", members: 500, scanned: syncPushBudget / 500},
	} {
		for _, extra := range []uint64{0, 1} {
			t.Run(fmt.Sprintf("%s, %d events past it", tc.name, extra), func(t *testing.T) {
				cacher, _ := newStallResumeCacher(t)
				pinHistoryCapacity(t, cacher, 10000)
				stallResumeAddPods(t, cacher, "ns", 101, 100+tc.scanned+extra)
				waitDispatched(t, cacher)
				ws := make([]*cacheWatcher, tc.members)
				for i := range ws {
					ws[i] = detachedWatcher(cacher, 10, tc.scope)
				}
				registerUnsynced(t, cacher, ws, func(int) uint64 { return 100 })
				dispatcherDo(t, cacher, func(runPass syncPassFunc) {
					if served, _ := runPass(); served != tc.members {
						t.Errorf("served=%d, want %d", served, tc.members)
					}
					for i, w := range ws {
						if w.position != 100+tc.scanned || w.unsynced != (extra > 0) {
							t.Errorf("member %d: position=%d unsynced=%v, want position %d and unsynced=%v", i, w.position, w.unsynced, 100+tc.scanned, extra > 0)
						}
					}
				})
			})
		}
	}
}

// TestStallResumeBusyPeriodDrivesInlinePass: a pass that leaves a member
// blocked on a full input selects the busy period, and the inline check
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
			t.Errorf("expected the watcher served, got %d", served)
		}
		if cacher.stall.passPeriod != syncPassBusyPeriod {
			t.Errorf("expected the busy period %v after a pass that left the member blocked, got %v", syncPassBusyPeriod, cacher.stall.passPeriod)
		}
		// Less than the normal period, exactly the busy one: the inline
		// check after a dispatch runs a pass.
		clk.Step(syncPassBusyPeriod)
		cacher.syncAfterDispatch()
		if !cacher.stall.lastPass.Equal(clk.Now()) {
			t.Errorf("the inline check did not run a pass after the busy period: lastPass=%v now=%v", cacher.stall.lastPass, clk.Now())
		}
	})
}

// TestStallResumeRejectedPushEndsOffering: a rejected push ends the offering
// to that member for the pass, even when its input drains while the scan
// goes on for a cohort mate; anything pushed then would leave a gap behind
// the rejected event. The member keeps its last accepted resourceVersion,
// the pass selects the busy period, and the next pass resumes from there
// with no gap and no duplicate. The drain happens on the dispatcher
// goroutine, from inside the scan, so the sequence is exact.
func TestStallResumeRejectedPushEndsOffering(t *testing.T) {
	hook := &indexerHook{}
	cacher, _ := newStallResumeCacher(t, hook.config)
	pinHistoryCapacity(t, cacher, 10000)
	const end = 400
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
	var got []uint64
	drain := func(obj runtime.Object) {
		if len(a.input) == cap(a.input) && objectRV(t, obj) > a.position+1 && got == nil {
			got = drainInput(a)
		}
	}
	hook.fn.Store(&drain)
	before := readStallResumeCounters(t)
	dispatcherDo(t, cacher, func(runPass syncPassFunc) {
		if served, expired := runPass(); served != 2 || expired != 0 {
			t.Errorf("served=%d expired=%d, want both members served", served, expired)
		}
		if a.position != 100+inputCap || !a.unsynced || len(a.input) != 0 {
			t.Errorf("a: position=%d unsynced=%v len(input)=%d, want blocked at %d and nothing offered after the rejected push",
				a.position, a.unsynced, len(a.input), 100+inputCap)
		}
		if b.position != end || b.unsynced {
			t.Errorf("b: position=%d unsynced=%v, want synced at the history end %d", b.position, b.unsynced, end)
		}
		if cacher.stall.passPeriod != syncPassBusyPeriod {
			t.Errorf("expected the busy period after a pass that left a member blocked, got %v", cacher.stall.passPeriod)
		}
	})
	hook.fn.Store(nil)
	assertExactSequence(t, got, 101, 100+inputCap)
	if got := readStallResumeCounters(t).deferred - before.deferred; got != float64(inputCap) {
		t.Fatalf("deferred events: want %d, got %v", inputCap, got)
	}

	// Each later pass pushes cap(input) more from the last accepted
	// resourceVersion; the last one reaches the history end.
	for pass := uint64(2); pass <= (end-100)/inputCap; pass++ {
		if served, _ := runSyncPass(t, cacher); served != 1 {
			t.Fatalf("pass %d: served=%d, want a alone", pass, served)
		}
		if st := snapshot(t, cacher, a); st.position != 100+pass*inputCap {
			t.Fatalf("pass %d: a at %d, want %d", pass, st.position, 100+pass*inputCap)
		}
		got = append(got, drainInput(a)...)
	}
	assertExactSequence(t, got, 101, end)
	if st := snapshot(t, cacher, a); st.unsynced {
		t.Fatalf("expected a synced at the history end, got %+v", st)
	}
}

// TestStallResumeLeadOrder: the lead is the first candidate at or above the
// cursor, else the first one. 100 members at distinct positions take one event
// and reject the next, so a cohort is one member, it stays below the cursor,
// and the scan budget pays for some of them per pass. In this shape each pass
// serves the next ones in position order and wraps at the top.
func TestStallResumeLeadOrder(t *testing.T) {
	cacher, _ := newStallResumeCacher(t)
	pinHistoryCapacity(t, cacher, 10000)
	const members, passes = 100, 14
	// Far enough apart that no scan reaches the next member's position.
	position := func(i int) uint64 { return 100 + 20*uint64(i) }
	stallResumeAddPods(t, cacher, "ns", 101, position(members))
	waitDispatched(t, cacher)
	ws := make([]*cacheWatcher, members)
	for i := range ws {
		ws[i] = detachedWatcher(cacher, 1, namespacedName{})
	}
	registerUnsynced(t, cacher, ws, position)

	// The interval open, the accepted event and the rejected event.
	const cohortsPerPass = syncScanBudget / (syncIntervalOpenCost + 2)
	const fullTurn = (members + cohortsPerPass - 1) / cohortsPerPass
	got := make([][]uint64, members)
	lastServed := make([]int, members)
	turn := 0
	for pass := 1; pass <= passes; pass++ {
		if served, _ := runSyncPass(t, cacher); served != cohortsPerPass {
			t.Fatalf("pass %d: served=%d, want %d", pass, served, cohortsPerPass)
		}
		want := map[int]bool{}
		for range cohortsPerPass {
			want[turn] = true
			turn = (turn + 1) % members
		}
		for i, w := range ws {
			rvs := drainInput(w)
			if (len(rvs) == 1) != want[i] {
				t.Fatalf("pass %d: member %d got %v, want served=%v", pass, i, rvs, want[i])
			}
			got[i] = append(got[i], rvs...)
			if want[i] {
				lastServed[i] = pass
			}
			if pass-lastServed[i] > fullTurn {
				t.Fatalf("pass %d: member %d was last served in pass %d, want a turn every %d passes", pass, i, lastServed[i], fullTurn)
			}
		}
	}
	for i, rvs := range got {
		assertExactSequence(t, rvs, position(i)+1, position(i)+uint64(len(rvs)))
	}
}

// TestStallResumeCandidateCapRotation: with more than maxWatchersPerSync
// candidates a pass keeps the positions at or above the cursor, then the lowest
// ones. Here the ones left out sit just below the cursor and the next pass
// serves them.
func TestStallResumeCandidateCapRotation(t *testing.T) {
	cacher, _ := newStallResumeCacher(t)
	pinHistoryCapacity(t, cacher, 10000)
	const members, cursor, end = maxWatchersPerSync + 88, 400, 1000
	stallResumeAddPods(t, cacher, "ns", 101, end)
	waitDispatched(t, cacher)
	// Scoped to an empty namespace: nothing is pushed, positions follow the
	// scan, and the scan budget covers both cohorts to the history end.
	ws := make([]*cacheWatcher, members)
	for i := range ws {
		ws[i] = detachedWatcher(cacher, 10, namespacedName{namespace: "none"})
	}
	position := func(i int) uint64 { return 100 + uint64(i) }
	registerUnsynced(t, cacher, ws, position)
	// 300 members sit at or above the cursor; the wrap keeps the lowest 212.
	leftOut := func(i int) bool {
		return position(i) < cursor && i >= maxWatchersPerSync-(members-(cursor-100))
	}
	dispatcherDo(t, cacher, func(runPass syncPassFunc) {
		cacher.stall.syncCursor = cursor
		if served, expired := runPass(); served != maxWatchersPerSync || expired != 0 {
			t.Errorf("served=%d expired=%d, want %d served", served, expired, maxWatchersPerSync)
		}
		for i, w := range ws {
			switch {
			case leftOut(i) && (!w.unsynced || w.position != position(i)):
				t.Errorf("member %d: position=%d unsynced=%v, want it left out at %d", i, w.position, w.unsynced, position(i))
			case !leftOut(i) && (w.unsynced || w.position != end):
				t.Errorf("member %d: position=%d unsynced=%v, want it synced at %d", i, w.position, w.unsynced, end)
			}
		}
		if served, _ := runPass(); served != members-maxWatchersPerSync {
			t.Errorf("pass 2: served=%d, want the %d members left out", served, members-maxWatchersPerSync)
		}
		if len(cacher.stall.unsynced) != 0 {
			t.Errorf("pass 2: %d members still unsynced", len(cacher.stall.unsynced))
		}
	})
}

// TestStallResumeExpiryBoundary: a position one below the oldest servable
// resourceVersion is served, the one below that is expired. The boundary is
// the first event in the history once events were dropped, and the list
// resourceVersion plus one right after a relist, when the history is empty.
func TestStallResumeExpiryBoundary(t *testing.T) {
	check := func(t *testing.T, cacher *Cacher, oldest, end uint64) {
		t.Helper()
		servable := detachedWatcher(cacher, 1000, namespacedName{})
		tooOld := detachedWatcher(cacher, 1000, namespacedName{})
		registerUnsynced(t, cacher, []*cacheWatcher{servable, tooOld}, func(i int) uint64 { return oldest - 1 - uint64(i) })
		if served, expired := runSyncPass(t, cacher); served != 1 || expired != 1 {
			t.Fatalf("served=%d expired=%d, want one of each", served, expired)
		}
		if st := snapshot(t, cacher, servable); st.expired || st.unsynced || st.position != end {
			t.Fatalf("position %d must be served to the history end %d, got %+v", oldest-1, end, st)
		}
		if got := drainInput(servable); len(got) != int(end-oldest+1) {
			t.Fatalf("expected the events %d..%d, got %v", oldest, end, got)
		} else if len(got) > 0 {
			assertExactSequence(t, got, oldest, end)
		}
		if st := snapshot(t, cacher, tooOld); !st.expired || st.inSet || len(tooOld.input) != 0 {
			t.Fatalf("position %d must be expired with nothing offered, got %+v", oldest-2, st)
		}
	}
	t.Run("events were dropped", func(t *testing.T) {
		cacher, _ := newStallResumeCacher(t)
		pinHistoryCapacity(t, cacher, 100)
		stallResumeAddPods(t, cacher, "ns", 101, 300)
		waitDispatched(t, cacher)
		check(t, cacher, 201, 300)
	})
	t.Run("empty history after a relist", func(t *testing.T) {
		cacher, _ := newStallResumeCacher(t)
		stallResumeAddPods(t, cacher, "ns", 101, 300)
		waitDispatched(t, cacher)
		if err := cacher.watchCache.Replace(nil, "500"); err != nil {
			t.Fatal(err)
		}
		check(t, cacher, 501, 500)
	})
}

// TestStallResumeIntervalInvalidatedMidScan: the history moves past a
// cohort's interval while it is scanned. What was scanned stands, the
// member is not taken for caught up, and the next pass decides from its
// position.
func TestStallResumeIntervalInvalidatedMidScan(t *testing.T) {
	hook := &indexerHook{}
	cacher, _ := newStallResumeCacher(t, hook.config)
	pinHistoryCapacity(t, cacher, 1000)
	stallResumeAddPods(t, cacher, "ns", 101, 400)
	waitDispatched(t, cacher)
	w := detachedWatcher(cacher, 1000, namespacedName{})
	registerUnsynced(t, cacher, []*cacheWatcher{w}, func(int) uint64 { return 100 })

	// The interval copies bufferSize events per read. During the first
	// scanned event the history shrinks to 50 events and takes a new one,
	// so the second read finds its start index gone.
	fired := false
	shrink := func(runtime.Object) {
		if !fired {
			fired = true
			pinHistoryCapacity(t, cacher, 50)
			stallResumeAddPods(t, cacher, "ns", 401, 401)
		}
	}
	hook.fn.Store(&shrink)
	if served, expired := runSyncPass(t, cacher); served != 1 || expired != 0 {
		t.Fatalf("served=%d expired=%d, want the member served", served, expired)
	}
	hook.fn.Store(nil)
	if st := snapshot(t, cacher, w); !st.unsynced || st.expired || st.position != 100+bufferSize {
		t.Fatalf("expected the member unsynced at %d, got %+v", 100+bufferSize, st)
	}
	assertExactSequence(t, drainInput(w), 101, 100+bufferSize)
	if _, expired := runSyncPass(t, cacher); expired != 1 {
		t.Fatalf("expected the next pass to expire the member, got %d", expired)
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
	// lead sits above low's window (ten accepted and one rejected), so it
	// gets its own cohort.
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

// holdDispatcher blocks the dispatcher goroutine until the returned func is
// called, so the events a test writes meanwhile wait in c.incoming.
func holdDispatcher(t testing.TB, c *Cacher) (release func()) {
	t.Helper()
	resume := make(chan struct{})
	select {
	case c.stall.dispatcherHook <- func(syncPassFunc) { <-resume }:
	case <-time.After(10 * time.Second):
		t.Fatalf("the dispatcher goroutine never took the hook")
	}
	release = sync.OnceFunc(func() { close(resume) })
	t.Cleanup(release)
	return release
}

// TestStallResumeWatchOpenedBehindHistory: a Watch registers while the
// dispatcher is behind the history, so events the initial interval already
// covers are still to be dispatched. The live path must not offer them: they
// would fill the input, and a rejected one would put the position below what
// the client gets from the interval, so the watcher would expire too early.
// The first event above the interval must still arrive, exactly once.
func TestStallResumeWatchOpenedBehindHistory(t *testing.T) {
	for _, tc := range []struct {
		name string
		// rv is where the watch starts.
		rv uint64
	}{
		{name: "history interval", rv: 100},
		{name: "store interval", rv: 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cacher, _ := newStallResumeCacher(t)
			pinHistoryCapacity(t, cacher, 100)

			// 101..140 are in the history and wait in c.incoming when the
			// watch registers. 141 is the first event above its interval.
			release := holdDispatcher(t, cacher)
			stallResumeAddPods(t, cacher, "ns", 101, 140)
			w := stallResumeWatch(t, cacher, "ns", tc.rv)
			ws := []*cacheWatcher{w}
			stallResumeAddPods(t, cacher, "ns", 141, 141)
			release()
			// The client reads nothing, so the goroutine stays on the initial
			// interval, which it has copied by now, and never touches input.
			waitDispatcherState(t, cacher, ws, "result full on the initial interval", func(w *cacheWatcher) bool {
				return len(w.result) == cap(w.result)
			})
			if st := snapshot(t, cacher, w); st.unsynced || st.position != 141 || len(w.input) != 1 {
				t.Errorf("expected a synced watcher at 141 that holds only that event, got %+v len(input)=%d", st, len(w.input))
			}

			// Stall it: 141..150 fill the input and 151 misses.
			stallResumeAddPods(t, cacher, "ns", 142, 151)
			waitDispatcherState(t, cacher, ws, "unsynced", func(w *cacheWatcher) bool { return w.unsynced })
			// The history drops 101..140, exactly what the interval covers.
			stallResumeAddPods(t, cacher, "ns-other", 152, 240)
			waitDispatched(t, cacher)
			if oldest, ok := cacher.stall.src.oldest(); !ok || oldest != 141 {
				t.Fatalf("expected the history to start at 141, got %d %v", oldest, ok)
			}
			if _, expired := runSyncPass(t, cacher); expired != 0 {
				t.Fatalf("the watcher expired at position %d, but its client gets everything up to 140 and the history serves from 141", snapshot(t, cacher, w).position)
			}

			got := resyncWhileReading(t, cacher, ws, 151)[0]
			if len(got.errors) != 0 {
				t.Fatalf("unexpected error event: %+v", got.errors[0])
			}
			assertExactSequence(t, got.rvs, 101, 151)
			if st := snapshot(t, cacher, w); st.unsynced || st.position != 240 {
				t.Fatalf("expected a synced watcher at the history end 240, got %+v", st)
			}
		})
	}
}

// TestStallResumeStallOnlyShortensPeriod: a new stall ends the idle backoff,
// but it does not lengthen a busy period that is shorter than the normal one.
// In both cases the timer fires after the expected period.
func TestStallResumeStallOnlyShortensPeriod(t *testing.T) {
	for _, tc := range []struct {
		name string
		// freeSlots is the room in the first watcher's input at the pass.
		freeSlots    int
		before, want time.Duration
	}{
		{name: "busy period stays", freeSlots: 1, before: syncPassBusyPeriod, want: syncPassBusyPeriod},
		{name: "idle period ends", freeSlots: 0, before: syncPassIdlePeriod, want: syncPassPeriod},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cacher, clk := newStallResumeCacher(t)
			w1 := stallResumeWatch(t, cacher, "ns-a", 100)
			w2 := stallResumeWatch(t, cacher, "ns-b", 100)
			next := wedgeWatchers(t, cacher, []*cacheWatcher{w1}, func(rv uint64) { stallResumeAddPods(t, cacher, "ns-a", rv, rv) }, 101)
			stallResumeAddPods(t, cacher, "ns-a", next, next+50)
			for range tc.freeSlots {
				<-w1.ResultChan()
			}
			waitDispatcherState(t, cacher, []*cacheWatcher{w1}, "the free slots", func(w *cacheWatcher) bool {
				return len(w.input) == cap(w.input)-tc.freeSlots
			})
			// With one free slot the pass leaves w1 blocked (busy); with none
			// it serves nothing (idle).
			runSyncPass(t, cacher)
			if period, armed := schedulerState(t, cacher); period != tc.before || !armed {
				t.Fatalf("expected the period %v and an armed timer after the pass, got %v armed=%v", tc.before, period, armed)
			}

			// The clock stands still, so the stall of w2 runs no inline pass.
			wedgeWatchers(t, cacher, []*cacheWatcher{w2}, func(rv uint64) { stallResumeAddPods(t, cacher, "ns-b", rv, rv) }, next+51)
			if period, armed := schedulerState(t, cacher); period != tc.want || !armed {
				t.Fatalf("expected the period %v and an armed timer after the stall, got %v armed=%v", tc.want, period, armed)
			}
			clk.Step(tc.want)
			if err := wait.PollUntilContextTimeout(context.Background(), time.Millisecond, 5*time.Second, true, func(context.Context) (bool, error) {
				var ran bool
				dispatcherDo(t, cacher, func(syncPassFunc) { ran = cacher.stall.lastPass.Equal(clk.Now()) })
				return ran, nil
			}); err != nil {
				t.Fatalf("no pass ran %v after the last one", tc.want)
			}
		})
	}
}

// TestStallResumeClientStopMidScan: a client Stop lands in the middle of a scan,
// on a candidate with room. The stop must wait for finishDispatching: closing
// input at once makes the next push panic.
func TestStallResumeClientStopMidScan(t *testing.T) {
	hook := &indexerHook{}
	cacher, _ := newStallResumeCacher(t, hook.config)
	w := stallResumeWatch(t, cacher, "ns", 100)
	add := func(rv uint64) { stallResumeAddPods(t, cacher, "ns", rv, rv) }
	next := wedgeWatchers(t, cacher, []*cacheWatcher{w}, add, 101)
	stallResumeAddPods(t, cacher, "ns", next, next+4)
	readAll(t, w, 100*time.Millisecond)
	waitInputDrained(t, cacher, w)

	stopped := false
	stop := func(runtime.Object) {
		if !stopped {
			stopped = true
			w.Stop()
		}
	}
	hook.fn.Store(&stop)
	served, expired := runSyncPass(t, cacher)
	hook.fn.Store(nil)
	if !stopped || served != 1 || expired != 0 {
		t.Fatalf("stopped=%v served=%d expired=%d, want the member served by the pass that saw the Stop", stopped, served, expired)
	}
	runSyncPass(t, cacher)
	if st := snapshot(t, cacher, w); st.inSet {
		t.Errorf("expected the stopped watcher out of the set, got %+v", st)
	}
	if got := readAll(t, w, 5*time.Second); !got.closed || len(got.errors) != 0 {
		t.Errorf("expected a clean close, got closed=%v errors=%d", got.closed, len(got.errors))
	}
}

// TestStallResumeNoBookmarkWhileUnsynced: the watcher has room, so only the
// unsynced check keeps the bookmark out. A bookmark above the missed event would
// let the client resume past an event it never got.
func TestStallResumeNoBookmarkWhileUnsynced(t *testing.T) {
	cacher, clk := newStallResumeCacher(t)
	w := stallResumeWatch(t, cacher, "ns", 100)
	add := func(rv uint64) { stallResumeAddPods(t, cacher, "ns", rv, rv) }
	next := wedgeWatchers(t, cacher, []*cacheWatcher{w}, add, 101)
	missed := next - 1
	readAll(t, w, 100*time.Millisecond)
	waitInputDrained(t, cacher, w)
	dispatcherDo(t, cacher, func(syncPassFunc) {
		ev := &watchCacheEvent{Type: watch.Bookmark, Object: &example.Pod{}, ResourceVersion: missed}
		if err := (storage.APIObjectVersioner{}).UpdateObject(ev.Object, missed); err != nil {
			t.Error(err)
		}
		// The step expires the watcher's bookmark bucket, so the dispatch selects it.
		clk.Step(2 * time.Minute)
		cacher.dispatchEvent(ev)
		if len(w.input) != 0 || w.position != missed-1 || !w.unsynced {
			t.Errorf("a bookmark reached an unsynced watcher: len(input)=%d position=%d unsynced=%v", len(w.input), w.position, w.unsynced)
		}
	})
}

// TestStallResumeWrapCohortSkipsServedMembers: hi leads the first cohort, lo
// leads the second one after the wrap, and its scan passes hi's position. hi must
// get no duplicates, and the scan stops once lo, the only active member, blocks.
func TestStallResumeWrapCohortSkipsServedMembers(t *testing.T) {
	cacher, _ := newStallResumeCacher(t)
	pinHistoryCapacity(t, cacher, 10000)
	const end = 1000
	stallResumeAddPods(t, cacher, "ns", 101, end)
	waitDispatched(t, cacher)
	lo := detachedWatcher(cacher, 250, namespacedName{})
	hi := detachedWatcher(cacher, 1000, namespacedName{})
	registerUnsynced(t, cacher, []*cacheWatcher{lo, hi}, func(i int) uint64 { return []uint64{100, 300}[i] })
	dispatcherDo(t, cacher, func(runPass syncPassFunc) {
		cacher.stall.syncCursor = 300
		if served, expired := runPass(); served != 2 || expired != 0 {
			t.Errorf("served=%d expired=%d, want both served", served, expired)
		}
		if hi.unsynced || hi.position != end {
			t.Errorf("hi: unsynced=%v position=%d, want synced at %d", hi.unsynced, hi.position, end)
		}
		if !lo.unsynced || lo.position != 350 {
			t.Errorf("lo: unsynced=%v position=%d, want blocked at 350", lo.unsynced, lo.position)
		}
		if cacher.stall.syncCursor != 351 {
			t.Errorf("cursor %d, want 351: the scan stops once lo, the only active member, is blocked", cacher.stall.syncCursor)
		}
	})
	assertExactSequence(t, drainInput(hi), 301, end)
	assertExactSequence(t, drainInput(lo), 101, 350)
}

// TestStallResumePositionOnMiss: a scoped watcher accepts nothing for a long
// time, then misses one event. The position is the missed RV minus one. With the
// last accepted RV it would be below the history and get a 410 at its first stall.
func TestStallResumePositionOnMiss(t *testing.T) {
	cacher, _ := newStallResumeCacher(t)
	pinHistoryCapacity(t, cacher, 100)
	w := stallResumeWatch(t, cacher, "ns-a", 100)
	ws := []*cacheWatcher{w}
	addA := func(rv uint64) { stallResumeAddPods(t, cacher, "ns-a", rv, rv) }
	rv := fillWatchers(t, cacher, ws, addA, 101)
	stallResumeAddPods(t, cacher, "ns-b", rv, rv+299)
	missed := rv + 300
	addA(missed)
	waitDispatcherState(t, cacher, ws, "unsynced", func(w *cacheWatcher) bool { return w.unsynced })
	if st := snapshot(t, cacher, w); st.position != missed-1 {
		t.Fatalf("position %d, want %d, one below the missed event", st.position, missed-1)
	}
	if _, expired := runSyncPass(t, cacher); expired != 0 {
		t.Fatalf("the watcher expired, but the history still holds the only event it missed")
	}
	got := resyncWhileReading(t, cacher, ws, missed)[0]
	assertExactSequence(t, got.rvs[:len(got.rvs)-1], 101, rv-1)
	if got.rvs[len(got.rvs)-1] != missed {
		t.Fatalf("last event %d, want %d", got.rvs[len(got.rvs)-1], missed)
	}
}

// TestStallResumeTriggerMemberOverTwoPasses: the maps of the last pass must not
// leak into the next one, or a trigger indexed watcher gets duplicates.
func TestStallResumeTriggerMemberOverTwoPasses(t *testing.T) {
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
	for rv := uint64(101); rv <= 106; rv++ {
		pod := &example.Pod{Name: fmt.Sprintf("pod-%d", rv), Namespace: "ns", ResourceVersion: strconv.FormatUint(rv, 10), Spec: example.PodSpec{NodeName: "node-1"}}
		if err := cacher.watchCache.Add(pod); err != nil {
			t.Fatal(err)
		}
	}
	waitDispatched(t, cacher)
	w := detachedWatcher(cacher, 2, namespacedName{})
	node := "node-1"
	w.selector.indexValue = &node
	registerUnsynced(t, cacher, []*cacheWatcher{w}, func(int) uint64 { return 100 })
	var got []uint64
	for range 3 {
		runSyncPass(t, cacher)
		got = append(got, drainInput(w)...)
	}
	assertExactSequence(t, got, 101, 106)
}

// TestStallResumeTriggerFanout: a trigger indexed watcher on a cacher with no
// trigger function gets every event, like the live path.
func TestStallResumeTriggerFanout(t *testing.T) {
	cacher, _ := newStallResumeCacher(t)
	stallResumeAddPods(t, cacher, "ns", 101, 104)
	waitDispatched(t, cacher)
	w := detachedWatcher(cacher, 2, namespacedName{})
	node := "node-1"
	w.selector.indexValue = &node
	registerUnsynced(t, cacher, []*cacheWatcher{w}, func(int) uint64 { return 100 })
	var got []uint64
	for range 2 {
		runSyncPass(t, cacher)
		got = append(got, drainInput(w)...)
	}
	assertExactSequence(t, got, 101, 104)
}

// TestStallResumePassDurationDrivesBusyPeriod: the measured pass duration
// reaches nextPassPeriod.
func TestStallResumePassDurationDrivesBusyPeriod(t *testing.T) {
	hook := &indexerHook{}
	cacher, clk := newStallResumeCacher(t, hook.config)
	stallResumeAddPods(t, cacher, "ns", 101, 110)
	waitDispatched(t, cacher)
	w := detachedWatcher(cacher, 1, namespacedName{})
	registerUnsynced(t, cacher, []*cacheWatcher{w}, func(int) uint64 { return 100 })
	stepped := false
	step := func(runtime.Object) {
		if !stepped {
			stepped = true
			clk.Step(500 * time.Microsecond)
		}
	}
	hook.fn.Store(&step)
	runSyncPass(t, cacher)
	hook.fn.Store(nil)
	if period, _ := schedulerState(t, cacher); period != syncPassBusyDutyFactor*500*time.Microsecond {
		t.Fatalf("period %v after a busy pass of 500µs, want %v", period, syncPassBusyDutyFactor*500*time.Microsecond)
	}
}

// TestStallResumeIdleBackoffHoldsUnderTraffic: event traffic must not end the
// idle period, or a wedged watcher costs a pass every 10 ms for ever.
func TestStallResumeIdleBackoffHoldsUnderTraffic(t *testing.T) {
	cacher, clk := newStallResumeCacher(t)
	w := stallResumeWatch(t, cacher, "ns-a", 100)
	next := wedgeWatchers(t, cacher, []*cacheWatcher{w}, func(rv uint64) { stallResumeAddPods(t, cacher, "ns-a", rv, rv) }, 101)
	runSyncPass(t, cacher)
	if period, _ := schedulerState(t, cacher); period != syncPassIdlePeriod {
		t.Fatalf("period %v after a pass that served nothing, want idle", period)
	}
	// An event that stalls nobody, with the clock still.
	stallResumeAddPods(t, cacher, "ns-b", next, next)
	waitDispatched(t, cacher)
	if period, _ := schedulerState(t, cacher); period != syncPassIdlePeriod {
		t.Fatalf("period %v after an event that stalled nobody, want idle", period)
	}
	dispatcherDo(t, cacher, func(syncPassFunc) {
		clk.Step(syncPassIdlePeriod)
		cacher.syncAfterDispatch()
		if !cacher.stall.lastPass.Equal(clk.Now()) {
			t.Errorf("no inline pass after the idle period")
		}
		if cacher.stall.passPeriod != syncPassIdlePeriod {
			t.Errorf("period %v after an inline pass that served nothing, want idle", cacher.stall.passPeriod)
		}
	})
}

func TestStallResumeNoPassWhileEverybodyIsSynced(t *testing.T) {
	cacher, clk := newStallResumeCacher(t)
	stallResumeWatch(t, cacher, "ns", 100)
	clk.Step(500 * time.Millisecond)
	stallResumeAddPods(t, cacher, "ns", 101, 105)
	waitDispatched(t, cacher)
	dispatcherDo(t, cacher, func(syncPassFunc) {
		if !cacher.stall.lastPass.IsZero() || cacher.stall.passArmed {
			t.Errorf("a pass ran or the timer is armed with nothing unsynced: lastPass=%v armed=%v", cacher.stall.lastPass, cacher.stall.passArmed)
		}
	})
}

// TestStallResumeTrafficDoesNotPostponeTheTimer: traffic slower than the period
// must not push the timer pass out.
func TestStallResumeTrafficDoesNotPostponeTheTimer(t *testing.T) {
	cacher, clk := newStallResumeCacher(t)
	w := stallResumeWatch(t, cacher, "ns-a", 100)
	next := wedgeWatchers(t, cacher, []*cacheWatcher{w}, func(rv uint64) { stallResumeAddPods(t, cacher, "ns-a", rv, rv) }, 101)
	clk.Step(syncPassPeriod / 2)
	stallResumeAddPods(t, cacher, "ns-b", next, next)
	waitDispatched(t, cacher)
	clk.Step(syncPassPeriod / 2)
	if err := wait.PollUntilContextTimeout(context.Background(), time.Millisecond, 2*time.Second, true, func(context.Context) (bool, error) {
		var ran bool
		dispatcherDo(t, cacher, func(syncPassFunc) { ran = cacher.stall.lastPass.Equal(clk.Now()) })
		return ran, nil
	}); err != nil {
		t.Fatalf("the timer did not fire one period after it was armed")
	}
}

// TestStallResumeSecondStallHistogramSample: the histogram sample of a second
// stall counts only the events of that stall.
func TestStallResumeSecondStallHistogramSample(t *testing.T) {
	cacher, _ := newStallResumeCacher(t)
	w := stallResumeWatch(t, cacher, "ns", 100)
	ws := []*cacheWatcher{w}
	add := func(rv uint64) { stallResumeAddPods(t, cacher, "ns", rv, rv) }
	next := wedgeWatchers(t, cacher, ws, add, 101)
	stallResumeAddPods(t, cacher, "ns", next, next+49)
	resyncWhileReading(t, cacher, ws, next+49)
	before := readStallResumeCounters(t)
	next = wedgeWatchers(t, cacher, ws, add, next+50)
	resyncWhileReading(t, cacher, ws, next-1)
	after := readStallResumeCounters(t)
	if after.roundSamples-before.roundSamples != 1 || after.roundSum-before.roundSum != 1 {
		t.Fatalf("second stall: %d samples summing to %v, want one sample of 1", after.roundSamples-before.roundSamples, after.roundSum-before.roundSum)
	}
}

// TestStallResumePassMarksTheTimeline: an event a pass delivers carries the
// enqueue time, and no dispatch start, so it stays out of cacher_queue_latency.
func TestStallResumePassMarksTheTimeline(t *testing.T) {
	cacher, clk := newStallResumeCacher(t)
	stallResumeAddPods(t, cacher, "ns", 101, 101)
	waitDispatched(t, cacher)
	w := detachedWatcher(cacher, 1, namespacedName{})
	registerUnsynced(t, cacher, []*cacheWatcher{w}, func(int) uint64 { return 100 })
	clk.Step(time.Second)
	runSyncPass(t, cacher)
	item := <-w.input
	if got := item.event.timeline[metrics.PointWatcherEnqueued]; !got.Equal(clk.Now()) {
		t.Errorf("enqueue time %v, want the pass time %v", got, clk.Now())
	}
	if got := item.event.timeline[metrics.PointDispatchStarted]; !got.IsZero() {
		t.Errorf("a pass must not mark the dispatch start, got %v", got)
	}
}

func TestStallResumeInvalidatedInitialIntervalReason(t *testing.T) {
	setStallResumeGate(t, true)
	before := readStallResumeCounters(t)
	testCacheIntervalInvalidationStopsWatch(t, true)
	after := readStallResumeCounters(t)
	if after.expiredInitial-before.expiredInitial != 1 || after.expired != before.expired {
		t.Fatalf("want terminated{resource_expired_initial} +1 only, got initial %v expired %v", after.expiredInitial-before.expiredInitial, after.expired-before.expired)
	}
}

func TestStallResumeExpiredMessageCarriesThePosition(t *testing.T) {
	cacher, _ := newStallResumeCacher(t)
	pinHistoryCapacity(t, cacher, 100)
	w := stallResumeWatch(t, cacher, "ns", 100)
	add := func(rv uint64) { stallResumeAddPods(t, cacher, "ns", rv, rv) }
	next := wedgeWatchers(t, cacher, []*cacheWatcher{w}, add, 101)
	stallResumeAddPods(t, cacher, "ns", next, next+199)
	runSyncPass(t, cacher)
	got := readUntil(t, w, ^uint64(0), 15*time.Second)
	want := fmt.Sprintf("too old resource version: %d", next-2)
	if len(got.errors) != 1 || got.errors[0].Message != want {
		t.Fatalf("errors %+v, want one with the message %q", got.errors, want)
	}
}

// TestStallResumeStopWhileThe410IsBlocked: the client reads just enough for the
// goroutine to drain input and block on the 410 with result full, then stops.
// The goroutine must exit and the 410 must not be counted.
func TestStallResumeStopWhileThe410IsBlocked(t *testing.T) {
	cacher, _ := newStallResumeCacher(t)
	pinHistoryCapacity(t, cacher, 100)
	w := stallResumeWatch(t, cacher, "ns", 100)
	add := func(rv uint64) { stallResumeAddPods(t, cacher, "ns", rv, rv) }
	next := wedgeWatchers(t, cacher, []*cacheWatcher{w}, add, 101)
	stallResumeAddPods(t, cacher, "ns", next, next+199)
	waitDispatched(t, cacher)
	var nonDrainingStops atomic.Int32
	forget := w.forget
	w.forget = func(drain bool) {
		forget(drain)
		if !drain {
			nonDrainingStops.Add(1)
		}
	}
	if _, expired := runSyncPass(t, cacher); expired != 1 {
		t.Fatalf("expected the watcher to expire, got %d", expired)
	}
	// The watcher holds cap(result) + 1 + cap(input) events. After 1 +
	// cap(input) reads result is full again, input is empty and closed, and
	// the goroutine holds the 410.
	for range 1 + cap(w.input) {
		<-w.ResultChan()
	}
	waitDispatcherState(t, cacher, []*cacheWatcher{w}, "blocked on the 410", func(w *cacheWatcher) bool {
		return len(w.input) == 0 && len(w.result) == cap(w.result)
	})
	before := readStallResumeCounters(t)
	w.Stop()
	// One forget(false) from the client, one from the goroutine's exit.
	if err := wait.PollUntilContextTimeout(context.Background(), time.Millisecond, 5*time.Second, true, func(context.Context) (bool, error) {
		return nonDrainingStops.Load() > 1, nil
	}); err != nil {
		t.Fatalf("the watcher goroutine is still blocked on the 410 with nobody reading")
	}
	if after := readStallResumeCounters(t); after.expired != before.expired {
		t.Errorf("a 410 nobody got must not be counted")
	}
}

// TestStallResumeContextCancelWhileUnsynced: the stream closes with no error
// event and the next pass drops the watcher.
func TestStallResumeContextCancelWhileUnsynced(t *testing.T) {
	cacher, _ := newStallResumeCacher(t)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	w := scopedWatch(t, cacher, ctx, "/pods", storage.Everything)
	add := func(rv uint64) { stallResumeAddPods(t, cacher, "ns", rv, rv) }
	wedgeWatchers(t, cacher, []*cacheWatcher{w}, add, 101)
	cancel()
	got := readAll(t, w, 5*time.Second)
	if !got.closed || len(got.errors) != 0 {
		t.Fatalf("expected a clean close, got closed=%v errors=%d", got.closed, len(got.errors))
	}
	assertContiguous(t, got.rvs)
	if served, expired := runSyncPass(t, cacher); served != 0 || expired != 0 {
		t.Fatalf("served=%d expired=%d, want the cancelled watcher dropped", served, expired)
	}
	if st := snapshot(t, cacher, w); st.inSet {
		t.Fatalf("expected the cancelled watcher out of the set, got %+v", st)
	}
}

// TestStallResumeSlowReaderKeepsShortPeriod: a reader slower than the busy
// period blocks a pass, and the next pass finds its input still full. That pass
// must not go idle at once: 100 ms between passes caps a 10 slot watcher at 100
// events/s, so it would never resync. Only a pass after a normal period does.
func TestStallResumeSlowReaderKeepsShortPeriod(t *testing.T) {
	cacher, _ := newStallResumeCacher(t)
	w := stallResumeWatch(t, cacher, "ns", 100)
	add := func(rv uint64) { stallResumeAddPods(t, cacher, "ns", rv, rv) }
	next := wedgeWatchers(t, cacher, []*cacheWatcher{w}, add, 101)
	stallResumeAddPods(t, cacher, "ns", next, next+50)
	<-w.ResultChan()
	waitDispatcherState(t, cacher, []*cacheWatcher{w}, "one slot free", func(w *cacheWatcher) bool {
		return len(w.input) == cap(w.input)-1
	})
	for i, want := range []struct {
		served int
		period time.Duration
	}{{1, syncPassBusyPeriod}, {0, syncPassPeriod}, {0, syncPassIdlePeriod}} {
		served, _ := runSyncPass(t, cacher)
		if period, _ := schedulerState(t, cacher); served != want.served || period != want.period {
			t.Fatalf("pass %d: served=%d period=%v, want %d and %v", i+1, served, period, want.served, want.period)
		}
	}
}
