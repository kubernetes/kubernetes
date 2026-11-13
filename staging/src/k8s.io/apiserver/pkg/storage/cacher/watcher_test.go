/*
Copyright 2015 The Kubernetes Authors.

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

package cacher_test

import (
	"context"
	"fmt"
	"strconv"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/apis/example"
	"k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/cacher"
	"k8s.io/apiserver/pkg/storage/cacher/testing/setup"
	storagetesting "k8s.io/apiserver/pkg/storage/testing"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	testingclock "k8s.io/utils/clock/testing"
)

func TestWatch(t *testing.T) {
	for _, tc := range []struct {
		watchCacheStallResume bool
		watchWithoutPrevKV    bool
	}{
		{watchCacheStallResume: false, watchWithoutPrevKV: true},
		{watchCacheStallResume: false, watchWithoutPrevKV: false},
		{watchCacheStallResume: true, watchWithoutPrevKV: true},
	} {
		t.Run(fmt.Sprintf("WatchCacheStallResume=%v/WatchWithoutPrevKV=%v", tc.watchCacheStallResume, tc.watchWithoutPrevKV), func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.WatchCacheStallResume, tc.watchCacheStallResume)
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.WatchFromStorageWithoutPrevKV, tc.watchWithoutPrevKV)
			testWatch(t, tc.watchCacheStallResume)
		})
	}
}

func testWatch(t *testing.T, watchCacheStallResume bool) {
	t.Run("Watch", func(t *testing.T) {
		t.Parallel()
		ctx, cacher, terminate := setup.Setup(t)
		t.Cleanup(terminate)
		storagetesting.RunTestWatch(ctx, t, cacher)
	})
	t.Run("WatchFromZero", func(t *testing.T) {
		t.Parallel()
		ctx, delegator, server, terminate := setup.SetupWithEtcdServer(t)
		t.Cleanup(terminate)
		storagetesting.RunTestWatchFromZero(ctx, t, delegator, cacher.CompactWatch(delegator, server.V3Client.Client))
	})
	t.Run("DeleteTriggerWatch", func(t *testing.T) {
		t.Parallel()
		ctx, cacher, terminate := setup.Setup(t)
		t.Cleanup(terminate)
		storagetesting.RunTestDeleteTriggerWatch(ctx, t, cacher)
	})
	t.Run("WatchFromNonZero", func(t *testing.T) {
		t.Parallel()
		ctx, cacher, terminate := setup.Setup(t)
		t.Cleanup(terminate)
		storagetesting.RunTestWatchFromNonZero(ctx, t, cacher)
	})
	t.Run("DelayedWatchDelivery", func(t *testing.T) {
		t.Parallel()
		ctx, cacher, terminate := setup.Setup(t)
		t.Cleanup(terminate)
		storagetesting.RunTestDelayedWatchDelivery(ctx, t, cacher)
	})
	t.Run("WatchError", func(t *testing.T) {
		// TODO(#109831): Enable use of this test and run it.
	})
	t.Run("WatchContextCancel", func(t *testing.T) {
		t.Parallel()
		ctx, cacher, terminate := setup.Setup(t)
		t.Cleanup(terminate)
		storagetesting.RunTestWatchContextCancel(ctx, t, cacher)
	})
	t.Run("WatcherTimeout", func(t *testing.T) {
		t.Parallel()
		ctx, cacher, terminate := setup.Setup(t)
		t.Cleanup(terminate)
		storagetesting.RunTestWatcherTimeout(ctx, t, cacher)
	})
	t.Run("WatchDeleteEventObjectHaveLatestRV", func(t *testing.T) {
		t.Parallel()
		ctx, cacher, terminate := setup.Setup(t)
		t.Cleanup(terminate)
		storagetesting.RunTestWatchDeleteEventObjectHaveLatestRV(ctx, t, cacher, true, false, nil)
	})
	t.Run("WatchInitializationSignal", func(t *testing.T) {
		t.Parallel()
		ctx, cacher, terminate := setup.Setup(t)
		t.Cleanup(terminate)
		storagetesting.RunTestWatchInitializationSignal(ctx, t, cacher)
	})
	t.Run("ClusterScopedWatch", func(t *testing.T) {
		t.Parallel()
		ctx, cacher, terminate := setup.Setup(t, setup.WithClusterScopedKeyFunc, setup.WithNodeNameAndNamespaceIndex)
		t.Cleanup(terminate)
		storagetesting.RunTestClusterScopedWatch(ctx, t, cacher)
	})
	t.Run("NamespaceScopedWatch", func(t *testing.T) {
		t.Parallel()
		ctx, cacher, terminate := setup.Setup(t, setup.WithNodeNameAndNamespaceIndex)
		t.Cleanup(terminate)
		storagetesting.RunTestNamespaceScopedWatch(ctx, t, cacher)
	})
	t.Run("WatchDispatchBookmarkEvents", func(t *testing.T) {
		t.Parallel()
		ctx, cacher, terminate := setup.Setup(t)
		t.Cleanup(terminate)
		storagetesting.RunTestWatchDispatchBookmarkEvents(ctx, t, cacher, true)
	})
	t.Run("WatchBookmarksWithCorrectResourceVersion", func(t *testing.T) {
		t.Parallel()
		ctx, cacher, terminate := setup.Setup(t)
		t.Cleanup(terminate)
		storagetesting.RunTestOptionalWatchBookmarksWithCorrectResourceVersion(ctx, t, cacher)
	})
	t.Run("SendInitialEventsBackwardCompatibility", func(t *testing.T) {
		t.Parallel()
		ctx, store, terminate := setup.Setup(t)
		t.Cleanup(terminate)
		storagetesting.RunSendInitialEventsBackwardCompatibility(ctx, t, store)
	})
	t.Run("WatchResync", func(t *testing.T) {
		t.Parallel()
		fakeClock := testingclock.NewFakeClock(time.Now())
		ctx, delegator, server, terminate := setup.SetupWithEtcdServer(t, setup.WithClock(fakeClock), setup.WithNodeNameAndNamespaceIndex)
		t.Cleanup(terminate)
		store := &setup.CreateWrapper{CacheDelegator: delegator}
		resync := func() {
			time.Sleep(50 * time.Millisecond)
			fakeClock.Step(100 * time.Millisecond)
			time.Sleep(50 * time.Millisecond)
			fakeClock.Step(2 * time.Second)
			time.Sleep(50 * time.Millisecond)
		}
		triggerBookmarks := func() {
			time.Sleep(50 * time.Millisecond)
			fakeClock.Step(cacher.DefaultBookmarkFrequency + time.Second)
			time.Sleep(50 * time.Millisecond)
		}
		storagetesting.RunTestWatchResync(ctx, t, store, true, watchCacheStallResume, cacher.CompactWatch(delegator, server.V3Client.Client), resync, triggerBookmarks)
	})
	// These tests override WatchList and must finish before the parallel subtests start.
	t.Run("WatchSemantics", func(t *testing.T) {
		store, terminate := setup.SetupWithEtcdAndCreateWrapper(t)
		t.Cleanup(terminate)
		storagetesting.RunWatchSemantics(context.TODO(), t, store)
	})
	t.Run("WatchSemanticInitialEventsExtended", func(t *testing.T) {
		store, terminate := setup.SetupWithEtcdAndCreateWrapper(t)
		t.Cleanup(terminate)
		storagetesting.RunWatchSemanticInitialEventsExtended(context.TODO(), t, store)
	})
	t.Run("WatchListMatchSingle", func(t *testing.T) {
		store, terminate := setup.SetupWithEtcdAndCreateWrapper(t)
		t.Cleanup(terminate)
		storagetesting.RunWatchListMatchSingle(context.TODO(), t, store)
	})
}

func TestWatchResyncSingleTick(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.WatchCacheStallResume, true)
	fakeClock := testingclock.NewFakeClock(time.Now())
	ctx, delegator, _, terminate := setup.SetupWithEtcdServer(t, setup.WithClock(fakeClock), setup.WithMaxResyncEvents(5))
	t.Cleanup(terminate)
	store := &setup.CreateWrapper{CacheDelegator: delegator}

	rv, err := store.GetCurrentResourceVersion(ctx)
	require.NoError(t, err)

	var (
		w, otherWatch      watch.Interface
		watchErr, otherErr error
		watchesOpened      sync.WaitGroup
	)
	opts := storage.ListOptions{ResourceVersion: strconv.FormatUint(rv, 10), Predicate: storage.Everything, Recursive: true}
	watchesOpened.Add(2)
	go func() {
		defer watchesOpened.Done()
		w, watchErr = store.Watch(ctx, "/pods/ns", opts)
	}()
	go func() {
		defer watchesOpened.Done()
		otherWatch, otherErr = store.Watch(ctx, "/pods/other-ns", opts)
	}()
	time.Sleep(50 * time.Millisecond)
	fakeClock.Step(100 * time.Millisecond)
	watchesOpened.Wait()
	require.NoError(t, watchErr)
	require.NoError(t, otherErr)
	defer w.Stop()
	defer otherWatch.Stop()

	const stallCount = 35
	stallEvents := make([]watch.Event, stallCount)
	for i := range stallCount {
		pod := &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: fmt.Sprintf("stall-%02d", i), Namespace: "ns"}}
		require.NoError(t, store.Create(ctx, fmt.Sprintf("/pods/ns/%s", pod.Name), pod, pod, 0))
		stallEvents[i] = watch.Event{Type: watch.Added, Object: pod}
	}
	for i := range 12 {
		pod := &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: fmt.Sprintf("other-%02d", i), Namespace: "other-ns"}}
		require.NoError(t, store.Create(ctx, fmt.Sprintf("/pods/other-ns/%s", pod.Name), pod, pod, 0))
		ev := <-otherWatch.ResultChan()
		require.Equal(t, watch.Added, ev.Type)
	}

	done := make(chan struct{})
	go func() {
		defer close(done)
		storagetesting.ExpectWatchEventsExact(stallEvents).Check(t, w)
	}()
	time.Sleep(50 * time.Millisecond)
	fakeClock.Step(100 * time.Millisecond)
	time.Sleep(50 * time.Millisecond)
	select {
	case <-done:
		t.Fatal("expected single resync tick not to finish catchup")
	default:
	}
	for range 10 {
		fakeClock.Step(100 * time.Millisecond)
		time.Sleep(20 * time.Millisecond)
	}
	<-done

	livePod := &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "live", Namespace: "ns"}}
	require.NoError(t, store.Create(ctx, "/pods/ns/live", livePod, livePod, 0))
	storagetesting.ExpectWatchEventsExact([]watch.Event{{Type: watch.Added, Object: livePod}}).Check(t, w)
}
