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
	"bytes"
	"io"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/cacher/metrics"
	compbasemetrics "k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/testutil"

	cachertesting "k8s.io/apiserver/pkg/storage/cacher/testing"
)

var testGroupResource = schema.GroupResource{Resource: "pods"}

func allowAllFilter(string, labels.Set, fields.Set, runtime.Object) bool { return true }

// registerForTest makes the given metrics live for the duration of a test.
// component-base metrics are no-ops until registered somewhere.
func registerForTest(t *testing.T, collectors ...compbasemetrics.Registerable) {
	t.Helper()
	registry := compbasemetrics.NewKubeRegistry()
	for _, collector := range collectors {
		if err := registry.Register(collector); err != nil {
			t.Fatalf("failed to register metric: %v", err)
		}
	}
}

func newTestWatcher(chanSize int, scope string) *cacheWatcher {
	w := newCacheWatcher(chanSize, allowAllFilter, func(bool) {}, storage.APIObjectVersioner{},
		time.Now(), false, testGroupResource, metrics.NewNoopWatcherMetricsObservers(), nil, "test")
	w.scope = scope
	return w
}

// A watcher is closed with labels describing which buffer backed up, how broad
// the watch was, and the channel size it was created with. Those three facts
// previously only existed in a V(1) log line.
func TestTerminatedWatcherIsLabelled(t *testing.T) {
	metrics.TerminatedWatchersDetailed.Reset()
	t.Cleanup(metrics.TerminatedWatchersDetailed.Reset)
	registerForTest(t, metrics.TerminatedWatchersDetailed)

	// chanSize 1, and both buffers are filled, so the kill is result_full.
	w := newTestWatcher(1, watchScopeCluster)
	w.input <- &watchCacheEvent{Object: &v1.Pod{}, ResourceVersion: 1}
	w.result <- *w.convertToWatchEvent(&watchCacheEvent{Object: &v1.Pod{}, ResourceVersion: 1})

	// A nil timer means add() closes the watcher immediately.
	if w.add(&watchCacheEvent{Object: &v1.Pod{}, ResourceVersion: 2}, nil) {
		t.Fatal("expected add to fail against a full input channel")
	}

	count, err := testutil.GetCounterMetricValue(metrics.TerminatedWatchersDetailed.WithLabelValues(
		testGroupResource.Group, testGroupResource.Resource, "result_full", watchScopeCluster, "10"))
	if err != nil {
		t.Fatalf("failed to read metric: %v", err)
	}
	if count != 1 {
		t.Errorf("expected one termination labelled result_full/cluster/10, got %v", count)
	}
}

// The reason label distinguishes a serve loop that was not consuming from one
// that was: the second is the endpointslice burst signature, where the input
// channel is full while the result channel sits empty.
func TestTerminatedWatcherReasonReflectsResultChannel(t *testing.T) {
	metrics.TerminatedWatchersDetailed.Reset()
	t.Cleanup(metrics.TerminatedWatchersDetailed.Reset)
	registerForTest(t, metrics.TerminatedWatchersDetailed)

	w := newTestWatcher(1, watchScopeNamespace)
	w.input <- &watchCacheEvent{Object: &v1.Pod{}, ResourceVersion: 1}
	// result deliberately left empty.

	if w.add(&watchCacheEvent{Object: &v1.Pod{}, ResourceVersion: 2}, nil) {
		t.Fatal("expected add to fail against a full input channel")
	}

	count, err := testutil.GetCounterMetricValue(metrics.TerminatedWatchersDetailed.WithLabelValues(
		testGroupResource.Group, testGroupResource.Resource, "result_empty", watchScopeNamespace, "10"))
	if err != nil {
		t.Fatalf("failed to read metric: %v", err)
	}
	if count != 1 {
		t.Errorf("expected one termination labelled result_empty/namespace, got %v", count)
	}
}

func TestWatchScopeFor(t *testing.T) {
	testCases := []struct {
		name  string
		scope namespacedName
		want  string
	}{
		{name: "unscoped", scope: namespacedName{}, want: watchScopeCluster},
		{name: "namespaced", scope: namespacedName{namespace: "ns"}, want: watchScopeNamespace},
		{name: "single object", scope: namespacedName{namespace: "ns", name: "n"}, want: watchScopeResource},
		{name: "cluster scoped single object", scope: namespacedName{name: "n"}, want: watchScopeResource},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			if got := watchScopeFor(tc.scope); got != tc.want {
				t.Errorf("watchScopeFor(%+v) = %q, want %q", tc.scope, got, tc.want)
			}
		})
	}
}

// Backlog and conversion cost are sampled, so the observation count must track
// the sample rate rather than the delivery count.
func TestBacklogObservationsAreSampled(t *testing.T) {
	for _, m := range []*compbasemetrics.HistogramVec{
		metrics.WatchInputBacklog, metrics.WatchResultBacklog, metrics.WatchEventConversionDuration,
	} {
		m.Reset()
		t.Cleanup(m.Reset)
	}
	registerForTest(t, metrics.WatchInputBacklog, metrics.WatchResultBacklog, metrics.WatchEventConversionDuration)

	const deliveries = watchBacklogSampleRate * 3
	w := newTestWatcher(deliveries, watchScopeCluster)
	for i := range deliveries {
		w.sendWatchCacheEvent(&watchCacheEvent{Object: &v1.Pod{}, ResourceVersion: uint64(i + 1)})
	}

	for name, m := range map[string]*compbasemetrics.HistogramVec{
		"input backlog":       metrics.WatchInputBacklog,
		"result backlog":      metrics.WatchResultBacklog,
		"conversion duration": metrics.WatchEventConversionDuration,
	} {
		count, err := testutil.GetHistogramMetricCount(m.WithLabelValues(testGroupResource.Group, testGroupResource.Resource))
		if err != nil {
			t.Fatalf("failed to read %s: %v", name, err)
		}
		if count != 3 {
			t.Errorf("%s: expected 3 samples from %d deliveries at 1/%d, got %d",
				name, deliveries, watchBacklogSampleRate, count)
		}
	}
}

// The result backlog is observed before the handoff, so it reports the depth
// the event arrived into.
func TestResultBacklogReportsDepthOnArrival(t *testing.T) {
	metrics.WatchResultBacklog.Reset()
	t.Cleanup(metrics.WatchResultBacklog.Reset)
	registerForTest(t, metrics.WatchResultBacklog)

	w := newTestWatcher(watchBacklogSampleRate*2, watchScopeCluster)
	// Nothing consumes w.result, so the depth grows with every delivery and the
	// sampled observation lands at sampleRate-1 events already queued.
	for i := range watchBacklogSampleRate {
		w.sendWatchCacheEvent(&watchCacheEvent{Object: &v1.Pod{}, ResourceVersion: uint64(i + 1)})
	}

	sum, err := testutil.GetHistogramMetricValue(metrics.WatchResultBacklog.WithLabelValues(
		testGroupResource.Group, testGroupResource.Resource))
	if err != nil {
		t.Fatalf("failed to read metric: %v", err)
	}
	if want := float64(watchBacklogSampleRate - 1); sum != want {
		t.Errorf("expected the single sample to observe a depth of %v, got %v", want, sum)
	}
}

// A cachingObject serializes once per event; every further delivery of the same
// event reuses it. That ratio is the whole reason serialization is 35% of the
// pods serve path and 2% cluster-wide.
func TestSerializationCacheHitRate(t *testing.T) {
	metrics.SerializationCacheTotal.Reset()
	t.Cleanup(metrics.SerializationCacheTotal.Reset)
	registerForTest(t, metrics.SerializationCacheTotal)

	object, err := newCachingObject(&v1.Pod{})
	if err != nil {
		t.Fatalf("failed to build caching object: %v", err)
	}
	object.cacheObservers = metrics.NewSerializationCacheObservers(testGroupResource)

	const deliveries = 5
	encode := func(_ runtime.Object, w io.Writer) error {
		_, err := w.Write([]byte("serialized"))
		return err
	}
	for range deliveries {
		if err := object.CacheEncode(runtime.Identifier("test"), encode, &bytes.Buffer{}); err != nil {
			t.Fatalf("CacheEncode failed: %v", err)
		}
	}

	read := func(result string) float64 {
		t.Helper()
		value, err := testutil.GetCounterMetricValue(metrics.SerializationCacheTotal.WithLabelValues(
			testGroupResource.Group, testGroupResource.Resource, result))
		if err != nil {
			t.Fatalf("failed to read metric: %v", err)
		}
		return value
	}
	if got := read("miss"); got != 1 {
		t.Errorf("expected exactly one serialization, got %v", got)
	}
	if got := read("hit"); got != deliveries-1 {
		t.Errorf("expected %d reuses, got %v", deliveries-1, got)
	}
}

// One dispatch shares a single timer across every blocked watcher, so after the
// timer fires the rest are closed with no grace. Distinguishing that from many
// independently slow watchers is the point of these two metrics.
func TestDispatchAmplification(t *testing.T) {
	metrics.TerminatedWatchersPerDispatch.Reset()
	metrics.WatchersTerminatedWithoutGrace.Reset()
	t.Cleanup(metrics.TerminatedWatchersPerDispatch.Reset)
	t.Cleanup(metrics.WatchersTerminatedWithoutGrace.Reset)
	registerForTest(t, metrics.TerminatedWatchersPerDispatch, metrics.WatchersTerminatedWithoutGrace)

	backingStorage := &cachertesting.MockStorage{}
	cacher, _, err := newTestCacher(backingStorage)
	if err != nil {
		t.Fatalf("failed to create cacher: %v", err)
	}
	defer cacher.Stop()

	// Three watchers, all with a full input channel and no reader, registered
	// so that a single dispatch sees all three as blocked.
	const watchers = 3
	for i := range watchers {
		w := newTestWatcher(1, watchScopeCluster)
		w.input <- &watchCacheEvent{Object: &v1.Pod{}, ResourceVersion: 1}
		cacher.watchers.addWatcher(w, i, namespacedName{}, "", false)
	}

	// An already-exhausted budget means the timer fires immediately, so every
	// blocked watcher after the first is closed without grace.
	cacher.dispatchTimeoutBudget = &exhaustedBudget{}
	cacher.dispatchEvent(&watchCacheEvent{
		Object:          &v1.Pod{},
		ResourceVersion: 2,
		ObjFields:       fields.Set{},
	})

	perDispatch, err := testutil.GetHistogramMetricCount(metrics.TerminatedWatchersPerDispatch.WithLabelValues(
		cacher.groupResource.Group, cacher.groupResource.Resource))
	if err != nil {
		t.Fatalf("failed to read metric: %v", err)
	}
	if perDispatch != 1 {
		t.Errorf("expected a single dispatch to be recorded, got %d", perDispatch)
	}
	killed, err := testutil.GetHistogramMetricValue(metrics.TerminatedWatchersPerDispatch.WithLabelValues(
		cacher.groupResource.Group, cacher.groupResource.Resource))
	if err != nil {
		t.Fatalf("failed to read metric: %v", err)
	}
	if killed != watchers {
		t.Errorf("expected one dispatch to report %d terminations, got %v", watchers, killed)
	}

	// The first kill consumes the shared timer; the rest get nothing.
	withoutGrace, err := testutil.GetCounterMetricValue(metrics.WatchersTerminatedWithoutGrace.WithLabelValues(
		cacher.groupResource.Group, cacher.groupResource.Resource))
	if err != nil {
		t.Fatalf("failed to read metric: %v", err)
	}
	if withoutGrace != watchers-1 {
		t.Errorf("expected %d terminations without grace, got %v", watchers-1, withoutGrace)
	}
}

// exhaustedBudget grants no grace at all, so the shared timer fires on the
// first blocked watcher.
type exhaustedBudget struct{}

func (*exhaustedBudget) takeAvailable() time.Duration { return 0 }
func (*exhaustedBudget) returnUnused(time.Duration)   {}
