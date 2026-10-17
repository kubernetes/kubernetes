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
	"bytes"
	"context"
	"fmt"
	goruntime "runtime"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/apis/example"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/cacher/metrics"
	etcd3testing "k8s.io/apiserver/pkg/storage/etcd3/testing"
	k8smetrics "k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/testutil"
	"k8s.io/utils/clock"
)

// probeCountingStorage counts the list requests made by the object size probe.
type probeCountingStorage struct {
	storage.Interface
	probeLists atomic.Int64
}

func (s *probeCountingStorage) GetList(ctx context.Context, key string, opts storage.ListOptions, listObj runtime.Object) error {
	if opts.Predicate.Limit == objectSizeProbePageSize {
		s.probeLists.Add(1)
	}
	return s.Interface.GetList(ctx, key, opts, listObj)
}

func newObjectSizeBudgetTestStorage(t *testing.T) *probeCountingStorage {
	t.Helper()
	server, etcdStorage := newEtcdTestStorage(t, etcd3testing.PathPrefix())
	t.Cleanup(func() { server.Terminate(t) })
	return &probeCountingStorage{Interface: etcdStorage}
}

func objectSizeBudgetTestConfig(s storage.Interface, budget int64) Config {
	return Config{
		Storage:             s,
		Versioner:           storage.APIObjectVersioner{},
		GroupResource:       schema.GroupResource{Resource: "pods"},
		EventsHistoryWindow: DefaultEventFreshDuration,
		ResourcePrefix:      "/pods/",
		KeyFunc: func(obj runtime.Object) (string, error) {
			return storage.NamespaceKeyFunc("/pods/", obj)
		},
		GetAttrsFunc: GetPodAttrs,
		NewFunc:      newPod,
		NewListFunc:  newPodList,
		Codec:        examplev1ProtoCodec,
		Clock:        clock.RealClock{},

		MaxAverageObjectSizeBytes: budget,
	}
}

func newObjectSizeBudgetTestCacher(t *testing.T, s storage.Interface, budget int64) (*Cacher, *CacheDelegator) {
	t.Helper()
	cacher, err := NewCacherFromConfig(objectSizeBudgetTestConfig(s, budget))
	if err != nil {
		t.Fatal(err)
	}
	delegator := NewCacheDelegator(cacher, s)
	t.Cleanup(func() {
		delegator.Stop()
		cacher.Stop()
	})
	return cacher, delegator
}

// createPaddedPods creates count pods whose names start with namePrefix, each
// padded with an annotation of size bytes.
func createPaddedPods(ctx context.Context, t *testing.T, s storage.Interface, namePrefix string, count, size int) {
	t.Helper()
	for i := 0; i < count; i++ {
		pod := &example.Pod{ObjectMeta: metav1.ObjectMeta{
			Namespace:   "ns",
			Name:        fmt.Sprintf("%s-%05d", namePrefix, i),
			Annotations: map[string]string{"padding": strings.Repeat("x", size)},
		}}
		if err := s.Create(ctx, computePodKey(pod), pod, &example.Pod{}, 0); err != nil {
			t.Fatal(err)
		}
	}
}

func waitForBypass(ctx context.Context, t *testing.T, cacher *Cacher) {
	t.Helper()
	err := wait.PollUntilContextTimeout(ctx, 10*time.Millisecond, wait.ForeverTestTimeout, true, func(context.Context) (bool, error) {
		return cacher.isBypassed(), nil
	})
	if err != nil {
		t.Fatalf("resource was not taken out of the watch cache: %v", err)
	}
}

func cachedObjectCount(c *Cacher) int {
	c.watchCache.RLock()
	defer c.watchCache.RUnlock()
	return len(c.watchCache.storage.List())
}

func TestProbeObjectSize(t *testing.T) {
	testCases := []struct {
		name     string
		populate func(ctx context.Context, t *testing.T, s storage.Interface)
		budget   int64

		wantExceeded bool
		wantExamined int64
		// maxExamined, if set, is checked instead of wantExamined.
		maxExamined int64
		maxRequests int64
	}{
		{
			name:         "empty resource",
			populate:     func(context.Context, *testing.T, storage.Interface) {},
			budget:       1000,
			wantExamined: 0,
			maxRequests:  1,
		},
		{
			name: "small objects are measured in full",
			populate: func(ctx context.Context, t *testing.T, s storage.Interface) {
				createPaddedPods(ctx, t, s, "small", 120, 100)
			},
			budget:       10000,
			wantExamined: 120,
			maxRequests:  3,
		},
		{
			name: "large objects stop the probe as soon as the budget is proven exceeded",
			populate: func(ctx context.Context, t *testing.T, s storage.Interface) {
				createPaddedPods(ctx, t, s, "large", 400, 5000)
			},
			budget:       1000,
			wantExceeded: true,
			maxExamined:  100,
			maxRequests:  2,
		},
		{
			name: "a large first page does not decide for a resource whose average is small",
			populate: func(ctx context.Context, t *testing.T, s storage.Interface) {
				createPaddedPods(ctx, t, s, "a-large", 60, 20000)
				createPaddedPods(ctx, t, s, "b-small", 500, 100)
			},
			budget:       5000,
			wantExceeded: false,
			wantExamined: 560,
			maxRequests:  12,
		},
		{
			name: "large objects beyond the probe limit are left to the periodic check",
			populate: func(ctx context.Context, t *testing.T, s storage.Interface) {
				createPaddedPods(ctx, t, s, "a-small", 1000, 100)
				createPaddedPods(ctx, t, s, "b-large", 150, 20000)
			},
			budget:       2000,
			wantExceeded: false,
			wantExamined: objectSizeProbeMaxObjects,
			maxRequests:  objectSizeProbeMaxObjects / objectSizeProbePageSize,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			s := newObjectSizeBudgetTestStorage(t)
			tc.populate(ctx, t, s)

			c := &Cacher{
				storage:              s,
				resourcePrefix:       "/pods/",
				newListFunc:          newPodList,
				codec:                examplev1ProtoCodec,
				maxAverageObjectSize: tc.budget,
			}
			res, err := c.probeObjectSize(ctx)
			if err != nil {
				t.Fatal(err)
			}
			t.Logf("result: %+v, requests: %d", res, s.probeLists.Load())

			if res.exceeded != tc.wantExceeded {
				t.Errorf("exceeded = %v, want %v", res.exceeded, tc.wantExceeded)
			}
			if tc.maxExamined > 0 {
				if res.examined > tc.maxExamined {
					t.Errorf("examined %d objects, want at most %d", res.examined, tc.maxExamined)
				}
			} else if res.examined != tc.wantExamined {
				t.Errorf("examined %d objects, want %d", res.examined, tc.wantExamined)
			}
			if got := s.probeLists.Load(); got > tc.maxRequests {
				t.Errorf("made %d list requests, want at most %d", got, tc.maxRequests)
			}
		})
	}
}

// TestObjectSizeBudgetBypassesBeforePopulating checks that a resource which
// already exceeds the budget is never loaded into the watch cache, and is
// served from storage instead.
func TestObjectSizeBudgetBypassesBeforePopulating(t *testing.T) {
	// Register only this metric, in a registry of our own: registering all cacher
	// metrics would make them record for the rest of the test binary, and throw
	// off tests that count observations.
	registry := k8smetrics.NewKubeRegistry()
	if err := registry.Register(metrics.WatchCacheObjectSizeBudgetExceeded); err != nil {
		t.Fatal(err)
	}
	// Once registered, the gauge keeps series recorded by other tests.
	metrics.WatchCacheObjectSizeBudgetExceeded.Reset()
	ctx := context.Background()
	s := newObjectSizeBudgetTestStorage(t)
	createPaddedPods(ctx, t, s, "large", 30, 5000)

	cacher, delegator := newObjectSizeBudgetTestCacher(t, s, 1000)
	waitForBypass(ctx, t, cacher)

	if rv := cacher.watchCache.getListResourceVersion(); rv != 0 {
		t.Errorf("watch cache was populated up to resource version %d", rv)
	}
	if n := cachedObjectCount(cacher); n != 0 {
		t.Errorf("watch cache holds %d objects, want 0", n)
	}
	verifyServedFromStorage(ctx, t, delegator, 30)

	if err := testutil.GatherAndCompare(registry, strings.NewReader(`
# HELP apiserver_watch_cache_object_size_budget_exceeded [ALPHA] Set to 1 for resources served without watch cache because their average object size exceeds --watch-cache-max-average-object-size, broken by resource type.
# TYPE apiserver_watch_cache_object_size_budget_exceeded gauge
apiserver_watch_cache_object_size_budget_exceeded{group="",resource="pods"} 1
`), "apiserver_watch_cache_object_size_budget_exceeded"); err != nil {
		t.Error(err)
	}
}

// TestObjectSizeBudgetKeepsSmallResourceCached checks that a resource within
// the budget is cached exactly as it would be without one.
func TestObjectSizeBudgetKeepsSmallResourceCached(t *testing.T) {
	ctx := context.Background()
	s := newObjectSizeBudgetTestStorage(t)
	createPaddedPods(ctx, t, s, "small", 30, 100)

	cacher, delegator := newObjectSizeBudgetTestCacher(t, s, 10000)
	if err := cacher.Wait(ctx); err != nil {
		t.Fatal(err)
	}
	if cacher.isBypassed() {
		t.Fatal("resource within budget was taken out of the watch cache")
	}
	if n := cachedObjectCount(cacher); n != 30 {
		t.Errorf("watch cache holds %d objects, want 30", n)
	}
	if got := s.probeLists.Load(); got != 1 {
		t.Errorf("probe made %d list requests, want 1", got)
	}
	if err := delegator.ReadinessCheck(); err != nil {
		t.Errorf("ReadinessCheck() = %v", err)
	}
}

// TestObjectSizeBudgetDisabled checks that without a budget no probing takes
// place.
func TestObjectSizeBudgetDisabled(t *testing.T) {
	ctx := context.Background()
	s := newObjectSizeBudgetTestStorage(t)
	createPaddedPods(ctx, t, s, "large", 30, 5000)

	cacher, _ := newObjectSizeBudgetTestCacher(t, s, 0)
	if err := cacher.Wait(ctx); err != nil {
		t.Fatal(err)
	}
	if cacher.isBypassed() {
		t.Fatal("resource was taken out of the watch cache without a budget")
	}
	if got := s.probeLists.Load(); got != 0 {
		t.Errorf("probe made %d list requests, want 0", got)
	}
}

// TestObjectSizeBudgetBypassesGrownResource checks that a resource which
// exceeds the budget only after its watch cache was populated is taken out of
// it: existing watches are closed, the cached objects are released, and
// requests are served from storage.
func TestObjectSizeBudgetBypassesGrownResource(t *testing.T) {
	original := objectSizeCheckPeriod
	objectSizeCheckPeriod = 50 * time.Millisecond
	t.Cleanup(func() { objectSizeCheckPeriod = original })

	ctx := context.Background()
	s := newObjectSizeBudgetTestStorage(t)
	cacher, delegator := newObjectSizeBudgetTestCacher(t, s, 1000)
	if err := cacher.Wait(ctx); err != nil {
		t.Fatal(err)
	}
	if cacher.isBypassed() {
		t.Fatal("empty resource was taken out of the watch cache")
	}

	rv, err := s.GetCurrentResourceVersion(ctx)
	if err != nil {
		t.Fatal(err)
	}
	w, err := delegator.Watch(ctx, "/pods/", storage.ListOptions{
		ResourceVersion: fmt.Sprint(rv),
		Predicate:       storage.Everything,
		Recursive:       true,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer w.Stop()

	createPaddedPods(ctx, t, s, "large", 30, 5000)
	waitForBypass(ctx, t, cacher)

	// The watch served by the cache must end, so that its client re-establishes it.
	timeout := time.After(wait.ForeverTestTimeout)
drain:
	for {
		select {
		case _, ok := <-w.ResultChan():
			if !ok {
				break drain
			}
		case <-timeout:
			t.Fatal("watch served from the watch cache was not closed")
		}
	}

	err = wait.PollUntilContextTimeout(ctx, 10*time.Millisecond, wait.ForeverTestTimeout, true, func(context.Context) (bool, error) {
		return cachedObjectCount(cacher) == 0, nil
	})
	if err != nil {
		t.Errorf("watch cache still holds %d objects", cachedObjectCount(cacher))
	}

	verifyServedFromStorage(ctx, t, delegator, 30)

	stats, err := delegator.Stats(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if stats.ObjectCount != 31 {
		t.Errorf("Stats().ObjectCount = %d, want 31", stats.ObjectCount)
	}

	// Writes still work, without the cache's suggestion of the current object.
	key := "/pods/ns/large-00000"
	var updated example.Pod
	err = delegator.GuaranteedUpdate(ctx, key, &updated, false, nil, storage.SimpleUpdate(func(obj runtime.Object) (runtime.Object, error) {
		pod := obj.(*example.Pod)
		pod.Labels = map[string]string{"updated": "true"}
		return pod, nil
	}), nil)
	if err != nil {
		t.Fatalf("GuaranteedUpdate: %v", err)
	}
	if err := delegator.Delete(ctx, key, &example.Pod{}, nil, storage.ValidateAllObjectFunc, nil, storage.DeleteOptions{}); err != nil {
		t.Fatalf("Delete: %v", err)
	}
}

// verifyServedFromStorage checks that GET, LIST and WATCH requests that would
// normally be served from the watch cache succeed. It creates one more pod.
func verifyServedFromStorage(ctx context.Context, t *testing.T, delegator *CacheDelegator, wantPods int) {
	t.Helper()

	if err := delegator.ReadinessCheck(); err != nil {
		t.Errorf("ReadinessCheck() = %v, want nil", err)
	}

	list := &example.PodList{}
	err := delegator.GetList(ctx, "/pods/", storage.ListOptions{
		ResourceVersion:      "0",
		ResourceVersionMatch: metav1.ResourceVersionMatchNotOlderThan,
		Predicate:            storage.Everything,
		Recursive:            true,
	}, list)
	if err != nil {
		t.Fatalf("GetList: %v", err)
	}
	if len(list.Items) != wantPods {
		t.Errorf("GetList returned %d pods, want %d", len(list.Items), wantPods)
	}

	pod := &example.Pod{}
	if err := delegator.Get(ctx, "/pods/ns/large-00001", storage.GetOptions{ResourceVersion: "0"}, pod); err != nil {
		t.Fatalf("Get: %v", err)
	}

	w, err := delegator.Watch(ctx, "/pods/", storage.ListOptions{
		ResourceVersion: list.ResourceVersion,
		Predicate:       storage.Everything,
		Recursive:       true,
	})
	if err != nil {
		t.Fatalf("Watch: %v", err)
	}
	defer w.Stop()
	newPod := &example.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: "ns", Name: "after-bypass"}}
	if err := delegator.Create(ctx, computePodKey(newPod), newPod, &example.Pod{}, 0); err != nil {
		t.Fatal(err)
	}
	select {
	case event := <-w.ResultChan():
		if event.Type != watch.Added || event.Object.(*example.Pod).Name != "after-bypass" {
			t.Errorf("unexpected watch event: %v %#v", event.Type, event.Object)
		}
	case <-time.After(wait.ForeverTestTimeout):
		t.Error("watch did not deliver an event")
	}
}

// TestCheckObjectSize checks that the periodic check, limited to a few objects
// per call, finds large objects wherever they are in the cache, and does not
// take a resource whose average object size is within budget out of it.
func TestCheckObjectSize(t *testing.T) {
	original := objectSizeCheckMaxObjects
	objectSizeCheckMaxObjects = 10
	t.Cleanup(func() { objectSizeCheckMaxObjects = original })

	testCases := []struct {
		name     string
		populate func(ctx context.Context, t *testing.T, s storage.Interface)
		budget   int64
		checks   int
		// bypassOnCheck is the call expected to take the resource out of the
		// watch cache, or zero if none is.
		bypassOnCheck int
	}{
		{
			name: "within budget over several passes",
			populate: func(ctx context.Context, t *testing.T, s storage.Interface) {
				createPaddedPods(ctx, t, s, "small", 25, 100)
			},
			budget: 2000,
			checks: 8,
		},
		{
			name: "large objects between small ones are proven over budget as soon as they are measured",
			populate: func(ctx context.Context, t *testing.T, s storage.Interface) {
				createPaddedPods(ctx, t, s, "a-small", 15, 100)
				createPaddedPods(ctx, t, s, "b-large", 5, 20000)
				createPaddedPods(ctx, t, s, "c-small", 15, 100)
			},
			budget:        2000,
			checks:        8,
			bypassOnCheck: 2,
		},
		{
			name: "objects slightly over budget are found over a complete pass",
			populate: func(ctx context.Context, t *testing.T, s storage.Interface) {
				createPaddedPods(ctx, t, s, "medium", 30, 2500)
			},
			budget:        2000,
			checks:        8,
			bypassOnCheck: 3,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			s := newObjectSizeBudgetTestStorage(t)
			tc.populate(ctx, t, s)

			// Without a budget there is no monitor running concurrently.
			cacher, _ := newObjectSizeBudgetTestCacher(t, s, 0)
			if err := cacher.Wait(ctx); err != nil {
				t.Fatal(err)
			}
			cacher.maxAverageObjectSize = tc.budget

			for i := 1; i <= tc.checks; i++ {
				bypassed := cacher.checkObjectSize()
				if want := i == tc.bypassOnCheck; bypassed != want {
					t.Fatalf("check %d: bypassed = %v, want %v", i, bypassed, want)
				}
				if bypassed {
					break
				}
			}
			if cacher.isBypassed() != (tc.bypassOnCheck > 0) {
				t.Errorf("isBypassed() = %v", cacher.isBypassed())
			}
		})
	}
}

// TestProcessEventAfterStop checks that the reflector cannot block forever
// handing events to a stopped cacher. That would keep Stop from returning, and
// a cacher taken out of the watch cache at runtime from releasing its memory.
func TestProcessEventAfterStop(t *testing.T) {
	ctx := context.Background()
	s := newObjectSizeBudgetTestStorage(t)
	cacher, _ := newObjectSizeBudgetTestCacher(t, s, 0)
	if err := cacher.Wait(ctx); err != nil {
		t.Fatal(err)
	}
	cacher.Stop()

	done := make(chan struct{})
	go func() {
		defer close(done)
		for i := 0; i < 2*cap(cacher.incoming); i++ {
			cacher.processEvent(&watchCacheEvent{Type: watch.Added, Object: &example.Pod{}})
		}
	}()
	select {
	case <-done:
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatal("processEvent blocked after the cacher was stopped")
	}
}

// TestObjectSizeBudgetReadRacingRelease checks that a GET or LIST that was
// routed to the watch cache before the resource was taken out of it, but that
// reads the cache only after its contents were released, is served from storage
// instead of finding nothing.
func TestObjectSizeBudgetReadRacingRelease(t *testing.T) {
	tests := []struct {
		name string
		// frame is the function in which the request reads the watch cache.
		frame string
		// read returns the number of pods the request found.
		read func(ctx context.Context, delegator *CacheDelegator) (int, error)
	}{
		{
			name:  "Get",
			frame: "(*watchCache).WaitUntilFreshAndGet(",
			read: func(ctx context.Context, delegator *CacheDelegator) (int, error) {
				pod := &example.Pod{}
				if err := delegator.Get(ctx, "/pods/ns/pod-00001", storage.GetOptions{ResourceVersion: "0"}, pod); err != nil {
					return 0, err
				}
				if pod.Name != "pod-00001" {
					return 0, fmt.Errorf("got pod %q", pod.Name)
				}
				return 1, nil
			},
		},
		{
			name:  "GetList",
			frame: "(*watchCache).WaitUntilFreshAndGetList(",
			read: func(ctx context.Context, delegator *CacheDelegator) (int, error) {
				list := &example.PodList{}
				err := delegator.GetList(ctx, "/pods/", storage.ListOptions{
					ResourceVersion: "0",
					Predicate:       storage.Everything,
					Recursive:       true,
				}, list)
				return len(list.Items), err
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			s := newObjectSizeBudgetTestStorage(t)
			createPaddedPods(ctx, t, s, "pod", 3, 10)
			// No budget: the resource is taken out of the watch cache by hand.
			cacher, delegator := newObjectSizeBudgetTestCacher(t, s, 0)
			if err := cacher.Wait(ctx); err != nil {
				t.Fatal(err)
			}
			want, err := tc.read(ctx, delegator)
			if err != nil {
				t.Fatal(err)
			}

			// Hold the request right before it reads the cache, and meanwhile
			// take the resource out of the watch cache and release its contents,
			// as the periodic check does.
			type result struct {
				n   int
				err error
			}
			done := make(chan result, 1)
			cacher.watchCache.Lock()
			go func() {
				n, err := tc.read(ctx, delegator)
				done <- result{n, err}
			}()
			if err := waitForGoroutineIn(tc.frame); err != nil {
				cacher.watchCache.Unlock()
				t.Fatalf("request did not reach the watch cache: %v", err)
			}
			cacher.bypassForObjectSize("test", 0, 0)
			cacher.watchCache.releaseLocked()
			cacher.watchCache.Unlock()

			select {
			case r := <-done:
				if r.err != nil || r.n != want {
					t.Errorf("request racing the release found %d pods, err %v; want %d pods", r.n, r.err, want)
				}
			case <-time.After(wait.ForeverTestTimeout):
				t.Fatal("request did not complete")
			}
		})
	}
}

// waitForGoroutineIn waits until some goroutine is executing function fn.
func waitForGoroutineIn(fn string) error {
	return wait.PollUntilContextTimeout(context.Background(), time.Millisecond, wait.ForeverTestTimeout, true, func(context.Context) (bool, error) {
		buf := make([]byte, 1<<20)
		for {
			n := goruntime.Stack(buf, true)
			if n < len(buf) {
				return bytes.Contains(buf[:n], []byte(fn)), nil
			}
			buf = make([]byte, 2*len(buf))
		}
	})
}
