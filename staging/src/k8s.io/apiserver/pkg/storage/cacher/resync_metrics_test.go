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
	"strings"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/apis/example"
	"k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/cacher/metrics"
	cachertesting "k8s.io/apiserver/pkg/storage/cacher/testing"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	compbasemetrics "k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/testutil"
	"k8s.io/utils/clock"
	testingclock "k8s.io/utils/clock/testing"
)

// newResyncMetricsRegistry returns a registry with the resync metrics, which
// Register skips while the gate is off. It must run before the test creates
// its Cachers, because a Cacher keeps the series that it resolved when it was
// created.
func newResyncMetricsRegistry(t *testing.T) compbasemetrics.KubeRegistry {
	registry := compbasemetrics.NewKubeRegistry()
	registry.MustRegister(metrics.StalledWatchers, metrics.WatcherStalls, metrics.WatcherSkippedEvents)
	registry.Reset()
	t.Cleanup(registry.Reset)
	return registry
}

// resyncMetricsTest drives one Cacher.
type resyncMetricsTest struct {
	*Cacher
	t        *testing.T
	registry compbasemetrics.KubeRegistry
	rv       uint64
}

func newResyncMetricsTest(t *testing.T, registry compbasemetrics.KubeRegistry, c clock.WithTicker) *resyncMetricsTest {
	cacher, _, err := newTestCacherWithoutSyncing(&cachertesting.MockStorage{}, c)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(cacher.Stop)
	if err := cacher.Wait(context.Background()); err != nil {
		t.Fatal(err)
	}
	// 100 is the resourceVersion of the mock's initial list.
	return &resyncMetricsTest{Cacher: cacher, t: t, registry: registry, rv: 100}
}

func (c *resyncMetricsTest) options() storage.ListOptions {
	return storage.ListOptions{ResourceVersion: fmt.Sprint(c.rv), Predicate: storage.Everything, Recursive: true}
}

func (c *resyncMetricsTest) open() watch.Interface {
	c.t.Helper()
	w, err := c.Watch(context.Background(), "/pods/ns", c.options())
	if err != nil {
		c.t.Fatal(err)
	}
	return w
}

func (c *resyncMetricsTest) add(n int) {
	c.t.Helper()
	for range n {
		c.rv++
		pod := &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: fmt.Sprintf("pod-%d", c.rv), Namespace: "ns", ResourceVersion: fmt.Sprint(c.rv)}}
		if err := c.watchCache.Add(pod); err != nil {
			c.t.Fatal(err)
		}
	}
}

// openStalled returns watches that all stall.
func (c *resyncMetricsTest) openStalled(n int) []watch.Interface {
	c.t.Helper()
	watches := make([]watch.Interface, n)
	for i := range watches {
		watches[i] = c.open()
	}
	// More than a watcher's input and result channels hold.
	c.add(100)
	return watches
}

// read reads w until it delivers upTo or ends.
func (c *resyncMetricsTest) read(w watch.Interface, upTo uint64) {
	c.t.Helper()
	timeout := time.After(wait.ForeverTestTimeout)
	for {
		select {
		case event, ok := <-w.ResultChan():
			if !ok {
				return
			}
			object := event.Object
			if cacheable, isCacheable := object.(runtime.CacheableObject); isCacheable {
				object = cacheable.GetObject()
			}
			if pod, isPod := object.(*example.Pod); isPod && pod.ResourceVersion == fmt.Sprint(upTo) {
				return
			}
		case <-timeout:
			c.t.Fatalf("timed out waiting for %d", upTo)
		}
	}
}

// expectMetrics waits until the time series of the resource have these values.
func (c *resyncMetricsTest) expectMetrics(stalledWatchers, stalls, skippedEvents int) {
	c.t.Helper()
	want := fmt.Sprintf(`
# HELP apiserver_watch_cache_stalled_watchers [ALPHA] Number of watchers that fell behind and are served from the watch cache history until they catch up, broken by resource type.
# TYPE apiserver_watch_cache_stalled_watchers gauge
apiserver_watch_cache_stalled_watchers{group="",resource="pods"} %d
# HELP apiserver_watch_cache_watcher_stalls_total [ALPHA] Counter of watchers that fell behind because their buffer was full, broken by resource type.
# TYPE apiserver_watch_cache_watcher_stalls_total counter
apiserver_watch_cache_watcher_stalls_total{group="",resource="pods"} %d
# HELP apiserver_watch_cache_watcher_skipped_events_total [ALPHA] Counter of events not dispatched to a watcher because their resourceVersion is not newer than the last one dispatched to it, broken by resource type.
# TYPE apiserver_watch_cache_watcher_skipped_events_total counter
apiserver_watch_cache_watcher_skipped_events_total{group="",resource="pods"} %d
`, stalledWatchers, stalls, skippedEvents)
	var err error
	if pollErr := wait.PollUntilContextTimeout(context.Background(), time.Millisecond, wait.ForeverTestTimeout, true, func(context.Context) (bool, error) {
		err = testutil.GatherAndCompare(c.registry, strings.NewReader(want),
			"apiserver_watch_cache_stalled_watchers",
			"apiserver_watch_cache_watcher_stalls_total",
			"apiserver_watch_cache_watcher_skipped_events_total")
		return err == nil, nil
	}); pollErr != nil {
		c.t.Fatal(err)
	}
}

func TestStalledWatchersMetrics(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.WatchCacheStallResume, true)

	// Every case leaves no stalled watcher.
	testCases := []struct {
		name    string
		recover func(c *resyncMetricsTest, w watch.Interface)
	}{
		{
			name: "caught up",
			recover: func(c *resyncMetricsTest, w watch.Interface) {
				c.read(w, c.rv)
			},
		},
		{
			name: "stopped by the client",
			recover: func(c *resyncMetricsTest, w watch.Interface) {
				w.Stop()
			},
		},
		{
			name: "history too old",
			recover: func(c *resyncMetricsTest, w watch.Interface) {
				c.watchCache.Lock()
				capacity := c.watchCache.history.capacity
				c.watchCache.history.upperBoundCapacity = capacity
				c.watchCache.Unlock()
				c.add(2 * capacity)
				c.read(w, c.rv)
			},
		},
		{
			name: "watch cache reinitialized",
			recover: func(c *resyncMetricsTest, w watch.Interface) {
				c.terminateAllWatchers()
			},
		},
		{
			name: "cacher stopped",
			recover: func(c *resyncMetricsTest, w watch.Interface) {
				c.Stop()
			},
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			c := newResyncMetricsTest(t, newResyncMetricsRegistry(t), clock.RealClock{})
			w := c.openStalled(1)[0]
			defer w.Stop()
			c.expectMetrics(1, 1, 0)
			tc.recover(c, w)
			c.expectMetrics(0, 1, 0)
		})
	}
}

// openUnsynced opens a watch on a Cacher whose clock does not tick, so it stays
// unsynced. The watch is sent to the channel once it is synced or canceled.
func (c *resyncMetricsTest) openUnsynced(ctx context.Context) <-chan watch.Interface {
	c.t.Helper()
	opened := make(chan watch.Interface, 1)
	go func() {
		w, err := c.Watch(ctx, "/pods/ns", c.options())
		if err != nil && ctx.Err() == nil {
			c.t.Error(err)
		}
		opened <- w
	}()
	c.waitUnsynced(1)
	return opened
}

func (c *resyncMetricsTest) waitUnsynced(want int) {
	c.t.Helper()
	if err := wait.PollUntilContextTimeout(context.Background(), time.Millisecond, wait.ForeverTestTimeout, true, func(context.Context) (bool, error) {
		c.RLock()
		defer c.RUnlock()
		return len(c.unsyncedWatchers) == want, nil
	}); err != nil {
		c.t.Fatalf("waiting for %d unsynced watchers: %v", want, err)
	}
}

func TestStalledWatchersMetricIgnoresNewWatchers(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.WatchCacheStallResume, true)
	c := newResyncMetricsTest(t, newResyncMetricsRegistry(t), testingclock.NewFakeClock(time.Now()))
	ctx, cancel := context.WithCancel(context.Background())
	opened := c.openUnsynced(ctx)
	c.expectMetrics(0, 0, 0)
	cancel()
	<-opened
	c.waitUnsynced(0)
	c.expectMetrics(0, 0, 0)
}

// A watcher that is stopped while its event is dispatched must not be counted.
func TestStalledWatchersMetricStopDuringDispatch(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.WatchCacheStallResume, true)
	stops := map[string]func(c *resyncMetricsTest, w watch.Interface){
		"client":           func(_ *resyncMetricsTest, w watch.Interface) { w.Stop() },
		"reinitialization": func(c *resyncMetricsTest, _ watch.Interface) { c.terminateAllWatchers() },
	}
	for name, stop := range stops {
		t.Run(name, func(t *testing.T) {
			// The clock does not tick, so the test is the only dispatcher.
			c := newResyncMetricsTest(t, newResyncMetricsRegistry(t), testingclock.NewFakeClock(time.Now()))
			opened := c.openUnsynced(context.Background())
			c.resyncWatchers()
			w := <-opened
			if w == nil {
				t.Fatal("the watch failed")
			}
			defer w.Stop()

			c.startDispatching(&watchCacheEvent{ObjFields: map[string]string{"metadata.namespace": "ns"}, ResourceVersion: c.rv + 1})
			if len(c.watchersBuffer) != 1 {
				t.Fatalf("got %d watchers, want 1", len(c.watchersBuffer))
			}
			c.blockedWatchers = append(c.blockedWatchers, c.watchersBuffer...)
			stop(c, w)
			c.finishDispatching()
			c.expectMetrics(0, 0, 0)
		})
	}
}

// Cachers of one resource, for example the served versions of a CRD, share a series.
func TestStalledWatchersMetricSharedByCachers(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.WatchCacheStallResume, true)
	registry := newResyncMetricsRegistry(t)
	a, b := newResyncMetricsTest(t, registry, clock.RealClock{}), newResyncMetricsTest(t, registry, clock.RealClock{})
	a.openStalled(2)
	a.expectMetrics(2, 2, 0)
	b.openStalled(3)
	a.expectMetrics(5, 5, 0)
	b.Stop()
	a.expectMetrics(2, 5, 0)
	a.Stop()
	a.expectMetrics(0, 5, 0)
}

func TestWatcherSkippedEventsMetric(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.WatchCacheStallResume, true)
	c := newResyncMetricsTest(t, newResyncMetricsRegistry(t), clock.RealClock{})
	w := c.open()
	defer w.Stop()
	c.add(1)
	c.read(w, c.rv)
	c.expectMetrics(0, 0, 0)

	c.processEvent(&watchCacheEvent{
		Type:            watch.Added,
		Object:          &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod", Namespace: "ns"}},
		ObjFields:       map[string]string{"metadata.namespace": "ns", "metadata.name": "pod"},
		Key:             "/pods/ns/pod",
		ResourceVersion: c.rv,
	})
	// The dispatcher handles events in order, so it is done with the one above.
	c.add(1)
	c.read(w, c.rv)
	c.expectMetrics(0, 0, 1)
}
