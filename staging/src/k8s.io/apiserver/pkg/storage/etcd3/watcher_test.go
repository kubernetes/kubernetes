/*
Copyright 2016 The Kubernetes Authors.

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

package etcd3

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	clientv3 "go.etcd.io/etcd/client/v3"
	grpccodes "google.golang.org/grpc/codes"
	grpcstatus "google.golang.org/grpc/status"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/apis/example"
	"k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/etcd3/metrics"
	"k8s.io/apiserver/pkg/storage/etcd3/testserver"
	etcdfeature "k8s.io/apiserver/pkg/storage/feature"
	storagetesting "k8s.io/apiserver/pkg/storage/testing"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
	"k8s.io/utils/ptr"
)

func TestWatch(t *testing.T) {
	t.Run("Watch", func(t *testing.T) {
		ctx, store, _ := testSetup(t)
		storagetesting.RunTestWatch(ctx, t, store)
	})
	t.Run("ClusterScopedWatch", func(t *testing.T) {
		ctx, store, _ := testSetup(t)
		storagetesting.RunTestClusterScopedWatch(ctx, t, store)
	})
	t.Run("NamespaceScopedWatch", func(t *testing.T) {
		ctx, store, _ := testSetup(t)
		storagetesting.RunTestNamespaceScopedWatch(ctx, t, store)
	})
	t.Run("DeleteTriggerWatch", func(t *testing.T) {
		ctx, store, _ := testSetup(t)
		storagetesting.RunTestDeleteTriggerWatch(ctx, t, store)
	})
	t.Run("WatchFromZero", func(t *testing.T) {
		ctx, store, client := testSetup(t)
		storagetesting.RunTestWatchFromZero(ctx, t, store, compactStorage(store, client.Client))
	})
	t.Run("WatchFromNonZero", func(t *testing.T) {
		ctx, store, _ := testSetup(t)
		storagetesting.RunTestWatchFromNonZero(ctx, t, store)
	})
	t.Run("DelayedWatchDelivery", func(t *testing.T) {
		ctx, store, _ := testSetup(t)
		storagetesting.RunTestDelayedWatchDelivery(ctx, t, store)
	})
	t.Run("WatchError", func(t *testing.T) {
		ctx, store, _ := testSetup(t)
		storagetesting.RunTestWatchError(ctx, t, &storeWithPrefixTransformer{store})
	})
	t.Run("WatchContextCancel", func(t *testing.T) {
		ctx, store, _ := testSetup(t)
		storagetesting.RunTestWatchContextCancel(ctx, t, store)
	})
	t.Run("WatcherTimeout", func(t *testing.T) {
		ctx, store, _ := testSetup(t)
		storagetesting.RunTestWatcherTimeout(ctx, t, store)
	})
	t.Run("WatchDeleteEventObjectHaveLatestRV", func(t *testing.T) {
		ctx, store, _ := testSetup(t)
		storagetesting.RunTestWatchDeleteEventObjectHaveLatestRV(ctx, t, store)
	})
	t.Run("WatchInitializationSignal", func(t *testing.T) {
		ctx, store, _ := testSetup(t)
		storagetesting.RunTestWatchInitializationSignal(ctx, t, store)
	})
	t.Run("ProgressNotify", func(t *testing.T) {
		clusterConfig := testserver.NewTestConfig(t)
		clusterConfig.WatchProgressNotifyInterval = time.Second
		ctx, store, client := testSetup(t, withClientConfig(clusterConfig))

		storagetesting.RunOptionalTestProgressNotify(ctx, t, store, increaseRVFunc(client.Client))
	})
	t.Run("WatchWithUnsafeDelete", func(t *testing.T) {
		featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.AllowUnsafeMalformedObjectDeletion, true)
		ctx, store, _ := testSetup(t)
		storagetesting.RunTestWatchWithUnsafeDelete(ctx, t, &storeWithCorruptedTransformer{store})
	})
	t.Run("WatchDispatchBookmarkEvents", func(t *testing.T) {
		clusterConfig := testserver.NewTestConfig(t)
		clusterConfig.WatchProgressNotifyInterval = time.Second
		ctx, store, _ := testSetup(t, withClientConfig(clusterConfig))

		storagetesting.RunTestWatchDispatchBookmarkEvents(ctx, t, store, false)
	})
	t.Run("SendInitialEventsBackwardCompatibility", func(t *testing.T) {
		ctx, store, _ := testSetup(t)
		storagetesting.RunSendInitialEventsBackwardCompatibility(ctx, t, store)
	})
	for _, rangeStream := range []bool{false, true} {
		t.Run(fmt.Sprintf("RangeStream=%v", rangeStream), func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.EtcdRangeStream, rangeStream)
			t.Run("WatchSemantics", func(t *testing.T) {
				ctx, store, _ := testSetup(t)
				storagetesting.RunWatchSemantics(ctx, t, store)
			})
			t.Run("WatchSemanticsWithConcurrentDecode", func(t *testing.T) {
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ConcurrentWatchObjectDecode, true)
				ctx, store, _ := testSetup(t)
				storagetesting.RunWatchSemantics(ctx, t, store)
			})
			t.Run("WatchSemanticInitialEventsExtended", func(t *testing.T) {
				ctx, store, _ := testSetup(t)
				storagetesting.RunWatchSemanticInitialEventsExtended(ctx, t, store)
			})
			t.Run("WatchListMatchSingle", func(t *testing.T) {
				ctx, store, _ := testSetup(t)
				storagetesting.RunWatchListMatchSingle(ctx, t, store)
			})
		})
	}
	t.Run("WatchErrorEventIsBlockingFurtherEvent", func(t *testing.T) {
		ctx, store, _ := testSetup(t)
		storagetesting.RunWatchErrorIsBlockingFurtherEvents(ctx, t, &storeWithPrefixTransformer{store})
	})
}

// =======================================================================
// Implementation-specific tests are following.
// The following tests are exercising the details of the implementation
// not the actual user-facing contract of storage interface.
// As such, they may focus e.g. on non-functional aspects like performance
// impact.
// =======================================================================

// TestWatchErrorIncorrectConfiguration checks if an error
// will be returned when the storage hasn't been properly
// initialised for watch requests
func TestWatchErrorIncorrectConfiguration(t *testing.T) {
	scenarios := []struct {
		name            string
		setupFn         func(opts *setupOptions)
		requestOpts     storage.ListOptions
		enableWatchList bool
		expectedErr     error
	}{
		{
			name:        "no newFunc provided",
			setupFn:     func(opts *setupOptions) { opts.newFunc = nil },
			requestOpts: storage.ListOptions{ProgressNotify: true},
			expectedErr: apierrors.NewInternalError(errors.New("progressNotify for watch is unsupported by the etcd storage because no newFunc was provided")),
		},
	}
	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			if scenario.enableWatchList {
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.WatchList, true)
			}
			origCtx, store, _ := testSetup(t, scenario.setupFn)
			ctx, cancel := context.WithCancel(origCtx)
			defer cancel()

			w, err := store.watcher.Watch(ctx, "/abc", 0, scenario.requestOpts)
			if err == nil {
				t.Fatalf("expected an error but got none")
			}
			if w != nil {
				t.Fatalf("didn't expect a watcher because the test assumes incorrect store initialisation")
			}
			if err.Error() != scenario.expectedErr.Error() {
				t.Fatalf("unexpected err = %v, expected = %v", err, scenario.expectedErr)
			}
		})
	}
}

func TestTooLargeResourceVersionErrorForWatchList(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.WatchList, true)
	origCtx, store, _ := testSetup(t)
	ctx, cancel := context.WithCancel(origCtx)
	defer cancel()
	requestOpts := storage.ListOptions{
		SendInitialEvents: ptr.To(true),
		Recursive:         true,
		Predicate: storage.SelectionPredicate{
			Field:               fields.Everything(),
			Label:               labels.Everything(),
			AllowWatchBookmarks: true,
		},
	}
	var expectedErr *apierrors.StatusError
	if !errors.As(storage.NewTooLargeResourceVersionError(uint64(102), 1, 0), &expectedErr) {
		t.Fatalf("Unable to convert NewTooLargeResourceVersionError to apierrors.StatusError")
	}

	w, err := store.watcher.Watch(ctx, "/abc/", int64(102), requestOpts)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Stop()

	actualEvent := <-w.ResultChan()
	if actualEvent.Type != watch.Error {
		t.Fatalf("Unexpected type of the event: %v, expected: %v", actualEvent.Type, watch.Error)
	}
	actualErr, ok := actualEvent.Object.(*metav1.Status)
	if !ok {
		t.Fatalf("Expected *apierrors.StatusError, got: %#v", actualEvent.Object)
	}

	if actualErr.Details.RetryAfterSeconds <= 0 {
		t.Fatalf("RetryAfterSeconds must be > 0, actual value: %v", actualErr.Details.RetryAfterSeconds)
	}
	// rewrite the Details as it contains retry seconds
	// and validate the whole struct
	expectedErr.ErrStatus.Details = actualErr.Details
	if diff := cmp.Diff(*actualErr, expectedErr.ErrStatus); diff != "" {
		t.Fatalf("Unexpected error returned, diff: %v", diff)
	}
}

func TestWatchChanSync(t *testing.T) {
	modes := []struct {
		name        string
		rangeStream bool
	}{
		{name: "Paginated"},
		{name: "RangeStream", rangeStream: true},
	}

	testCases := []struct {
		name             string
		watchKey         string
		watcherMaxLimit  int64
		expectEventCount int
		expectGetCount   int
	}{
		{
			name:            "None of the current objects match watchKey: sync with empty page",
			watchKey:        "/pods/test/",
			watcherMaxLimit: 1,
			expectGetCount:  1,
		},
		{
			name:             "The number of current objects is less than defaultWatcherMaxLimit: sync with one page",
			watchKey:         "/pods/",
			watcherMaxLimit:  3,
			expectEventCount: 2,
			expectGetCount:   1,
		},
		{
			name:             "a new item added to etcd before returning a second page is not returned: sync with two page",
			watchKey:         "/pods/",
			watcherMaxLimit:  1,
			expectEventCount: 2,
			expectGetCount:   2,
		},
	}

	for _, mode := range modes {
		for _, testCase := range testCases {
			t.Run(mode.name+"/"+testCase.name, func(t *testing.T) {
				orig := defaultWatcherMaxLimit
				defer func() { defaultWatcherMaxLimit = orig }()
				defaultWatcherMaxLimit = testCase.watcherMaxLimit

				origCtx, store, _ := testSetup(t)
				initList, err := initStoreData(origCtx, store)
				if err != nil {
					t.Fatal(err)
				}

				kvWrapper := newEtcdClientKVWrapper(store.client.KV)
				kvWrapper.getReactors = append(kvWrapper.getReactors, func() {
					barThird := &example.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: "third", Name: "bar"}}
					podKey := fmt.Sprintf("/pods/%s/%s", barThird.Namespace, barThird.Name)
					storedObj := &example.Pod{}

					err := store.Create(context.Background(), podKey, barThird, storedObj, 0)
					if err != nil {
						t.Errorf("failed to create object: %v", err)
					}
				})

				store.client.KV = kvWrapper

				w := store.watcher.createWatchChan(
					origCtx,
					testCase.watchKey,
					0,
					true,
					false,
					storage.Everything)

				sync := w.syncPaginated
				if mode.rangeStream {
					sync = w.syncStreamRecursive
				}
				if err := sync(); err != nil {
					t.Fatal(err)
				}

				if w.initialRev <= 0 {
					t.Errorf("expected initialRev to be set, got %d", w.initialRev)
				}

				// close incomingEventChan so we can read incomingEventChan non-blocking
				close(w.incomingEventChan)

				eventsReceived := 0
				for event := range w.incomingEventChan {
					eventsReceived++
					storagetesting.ExpectContains(t, "incorrect list pods", initList, event.key)
				}

				if eventsReceived != testCase.expectEventCount {
					t.Errorf("Unexpected number of events: %v, expected: %v", eventsReceived, testCase.expectEventCount)
				}

				if mode.rangeStream {
					if kvWrapper.getStreamCallCounter != 1 {
						t.Errorf("Unexpected called times of client.KV.GetStream() : %v, expected: 1", kvWrapper.getStreamCallCounter)
					}
				} else if kvWrapper.getCallCounter != testCase.expectGetCount {
					t.Errorf("Unexpected called times of client.KV.Get() : %v, expected: %v", kvWrapper.getCallCounter, testCase.expectGetCount)
				}
			})
		}
	}
}

// TestWatchChanSyncStreamMatchesPaginated verifies syncStreamRecursive queues the same
// key/value/revision set as syncPaginated for the same etcd state.
func TestWatchChanSyncStreamMatchesPaginated(t *testing.T) {
	origCtx, store, _ := testSetup(t)

	want := map[string]struct{}{}
	for i := range 20 {
		pod := &example.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: fmt.Sprintf("ns-%d", i%3), Name: fmt.Sprintf("pod-%d", i)}}
		key := fmt.Sprintf("/pods/%s/%s", pod.Namespace, pod.Name)
		if err := store.Create(origCtx, key, pod, &example.Pod{}, 0); err != nil {
			t.Fatalf("failed to create object: %v", err)
		}
		want[key] = struct{}{}
	}

	stream := drainSync(t, store, origCtx, func(wc *watchChan) error { return wc.syncStreamRecursive() })
	paginated := drainSync(t, store, origCtx, func(wc *watchChan) error { return wc.syncPaginated() })

	if len(stream) != len(want) {
		t.Errorf("syncStreamRecursive queued %d events, expected %d", len(stream), len(want))
	}
	if diff := cmp.Diff(paginated, stream, cmp.AllowUnexported(event{})); diff != "" {
		t.Errorf("syncStreamRecursive and syncPaginated queued different events (-paginated +stream):\n%s", diff)
	}
}

func resetFeatureSupportCheckerDuringTest(t *testing.T) {
	t.Helper()
	orig := etcdfeature.DefaultFeatureSupportChecker
	etcdfeature.DefaultFeatureSupportChecker = etcdfeature.NewDefaultFeatureSupportChecker()
	t.Cleanup(func() { etcdfeature.DefaultFeatureSupportChecker = orig })
}

func TestWatchChanSyncStreamFallsBackToPaginated(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.EtcdRangeStream, true)
	resetFeatureSupportCheckerDuringTest(t)

	origCtx, store, _ := testSetup(t)
	initList, err := initStoreData(origCtx, store)
	if err != nil {
		t.Fatal(err)
	}

	kvWrapper := newEtcdClientKVWrapper(store.client.KV)
	kvWrapper.streamUnimplemented = true
	store.client.KV = kvWrapper

	w := store.watcher.createWatchChan(origCtx, "/pods/", 0, true, false, storage.Everything)

	if err := w.sync(); err != nil {
		t.Fatalf("sync failed: %v", err)
	}

	if kvWrapper.getStreamCallCounter != 1 {
		t.Errorf("expected GetStream to be called once, got %d", kvWrapper.getStreamCallCounter)
	}
	if w.initialRev <= 0 {
		t.Errorf("expected initialRev to be set by the paginated fallback, got %d", w.initialRev)
	}
	if etcdfeature.DefaultFeatureSupportChecker.Supports(storage.RangeStream) {
		t.Error("expected RangeStream to be marked unsupported after the Unimplemented fallback")
	}

	close(w.incomingEventChan)
	eventsReceived := 0
	for event := range w.incomingEventChan {
		eventsReceived++
		storagetesting.ExpectContains(t, "incorrect list pods", initList, event.key)
	}
	if eventsReceived != len(initList) {
		t.Errorf("Unexpected number of events: %v, expected: %v", eventsReceived, len(initList))
	}
}

func drainSync(t *testing.T, store *store, ctx context.Context, sync func(*watchChan) error) map[string]*event {
	t.Helper()
	wc := store.watcher.createWatchChan(ctx, "/pods/", 0, true, false, storage.Everything)
	if err := sync(wc); err != nil {
		t.Fatalf("sync failed: %v", err)
	}
	close(wc.incomingEventChan)
	out := map[string]*event{}
	for e := range wc.incomingEventChan {
		out[e.key] = e
	}
	return out
}

func TestWatchChanSyncStreamCompactionError(t *testing.T) {
	metrics.Register()
	ctx, store, _ := testSetup(t)

	r1, err := store.client.KV.Put(ctx, "/pods/a", "v1")
	if err != nil {
		t.Fatal(err)
	}
	r2, err := store.client.KV.Put(ctx, "/pods/b", "v2")
	if err != nil {
		t.Fatal(err)
	}
	// Compacting at r2's revision removes r1's, so a read pinned to r1 fails.
	if _, err := store.client.KV.Compact(ctx, r2.Header.Revision, clientv3.WithCompactPhysical()); err != nil {
		t.Fatal(err)
	}

	kv := newEtcdClientKVWrapper(store.client.KV)
	kv.streamRev = r1.Header.Revision
	store.client.KV = kv

	legacyregistry.Reset()
	t.Cleanup(legacyregistry.Reset)

	wc := store.watcher.createWatchChan(ctx, "/pods/", 0, true, false, storage.Everything)
	if err := wc.syncStreamRecursive(); !apierrors.IsResourceExpired(err) {
		t.Fatalf("expected ResourceExpired from a compacted revision, got %T %v", err, err)
	}

	expected := `# HELP etcd_request_errors_total [ALPHA] Etcd failed request counts for each operation and object type.
# TYPE etcd_request_errors_total counter
etcd_request_errors_total{group="",operation="listStream",resource="pods"} 1
`
	if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(expected), "etcd_request_errors_total"); err != nil {
		t.Error(err)
	}
}

func TestWatchChanSyncStreamMetrics(t *testing.T) {
	metrics.Register()

	t.Run("success records a listStream request", func(t *testing.T) {
		ctx, store, _ := testSetup(t)
		pod := &example.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: "ns", Name: "pod-1"}}
		if err := store.Create(ctx, "/pods/ns/pod-1", pod, &example.Pod{}, 0); err != nil {
			t.Fatal(err)
		}

		legacyregistry.Reset()
		t.Cleanup(legacyregistry.Reset)

		wc := store.watcher.createWatchChan(ctx, "/pods/", 0, true, false, storage.Everything)
		if err := wc.syncStreamRecursive(); err != nil {
			t.Fatal(err)
		}

		expected := `# HELP etcd_requests_total [ALPHA] Etcd request counts for each operation and object type.
# TYPE etcd_requests_total counter
etcd_requests_total{group="",operation="listStream",resource="pods"} 1
`
		if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(expected), "etcd_requests_total"); err != nil {
			t.Error(err)
		}
	})

	t.Run("unimplemented is not recorded", func(t *testing.T) {
		ctx, store, _ := testSetup(t)
		kv := newEtcdClientKVWrapper(store.client.KV)
		kv.streamUnimplemented = true
		store.client.KV = kv

		legacyregistry.Reset()
		t.Cleanup(legacyregistry.Reset)

		wc := store.watcher.createWatchChan(ctx, "/pods/", 0, true, false, storage.Everything)
		err := wc.syncStreamRecursive()
		if grpcstatus.Code(err) != grpccodes.Unimplemented {
			t.Fatalf("expected Unimplemented error, got %v", err)
		}

		if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(""), "etcd_requests_total", "etcd_request_errors_total"); err != nil {
			t.Error(err)
		}
	})
}

// NOTE: it's not thread-safe
type etcdClientKVWrapper struct {
	clientv3.KV
	// keeps track of the number of times Get method is called
	getCallCounter int
	// keeps track of the number of times GetStream method is called
	getStreamCallCounter int
	// when true, GetStream returns a gRPC Unimplemented error
	streamUnimplemented bool
	// when nonzero, GetStream pins the stream to this revision
	streamRev int64
	// getReactors is called after the etcd KV's get function is executed.
	getReactors []func()
}

func newEtcdClientKVWrapper(kv clientv3.KV) *etcdClientKVWrapper {
	return &etcdClientKVWrapper{
		KV:             kv,
		getCallCounter: 0,
	}
}

func (ecw *etcdClientKVWrapper) GetStream(ctx context.Context, key string, opts ...clientv3.OpOption) (clientv3.GetStreamChan, error) {
	ecw.getStreamCallCounter++
	if ecw.streamUnimplemented {
		return nil, grpcstatus.Error(grpccodes.Unimplemented, "RangeStream is unimplemented")
	}
	if ecw.streamRev != 0 {
		opts = append(opts, clientv3.WithRev(ecw.streamRev))
	}
	return ecw.KV.GetStream(ctx, key, opts...)
}

func (ecw *etcdClientKVWrapper) Get(ctx context.Context, key string, opts ...clientv3.OpOption) (*clientv3.GetResponse, error) {
	resp, err := ecw.KV.Get(ctx, key, opts...)
	ecw.getCallCounter++
	if err != nil {
		return nil, err
	}

	if len(ecw.getReactors) > 0 {
		reactor := ecw.getReactors[0]
		ecw.getReactors = ecw.getReactors[1:]
		reactor()
	}

	return resp, nil
}

func initStoreData(ctx context.Context, store storage.Interface) ([]interface{}, error) {
	barFirst := &example.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: "first", Name: "bar"}}
	barSecond := &example.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: "second", Name: "bar"}}

	preset := []struct {
		key       string
		obj       *example.Pod
		storedObj *example.Pod
	}{
		{
			key: fmt.Sprintf("/pods/%s/%s", barFirst.Namespace, barFirst.Name),
			obj: barFirst,
		},
		{
			key: fmt.Sprintf("/pods/%s/%s", barSecond.Namespace, barSecond.Name),
			obj: barSecond,
		},
	}

	for i, ps := range preset {
		preset[i].storedObj = &example.Pod{}
		err := store.Create(ctx, ps.key, ps.obj, preset[i].storedObj, 0)
		if err != nil {
			return nil, fmt.Errorf("failed to create object: %w", err)
		}
	}

	var created []interface{}
	for _, item := range preset {
		created = append(created, item.key)
	}
	return created, nil
}
