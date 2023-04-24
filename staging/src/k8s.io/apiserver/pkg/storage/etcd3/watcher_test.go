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
	"fmt"
	"sync"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/etcd3/testserver"
	storagetesting "k8s.io/apiserver/pkg/storage/testing"
)

func TestWatch(t *testing.T) {
	ctx, store, _ := testSetup(t)
	storagetesting.RunTestWatch(ctx, t, store)
}

func TestClusterScopedWatch(t *testing.T) {
	ctx, store, _ := testSetup(t)
	storagetesting.RunTestClusterScopedWatch(ctx, t, store)
}

func TestNamespaceScopedWatch(t *testing.T) {
	ctx, store, _ := testSetup(t)
	storagetesting.RunTestNamespaceScopedWatch(ctx, t, store)
}

func TestDeleteTriggerWatch(t *testing.T) {
	ctx, store, _ := testSetup(t)
	storagetesting.RunTestDeleteTriggerWatch(ctx, t, store)
}

func TestWatchFromZero(t *testing.T) {
	ctx, store, client := testSetup(t)
	storagetesting.RunTestWatchFromZero(ctx, t, store, compactStorage(client))
}

// TestWatchFromNoneZero tests that
// - watch from non-0 should just watch changes after given version
func TestWatchFromNoneZero(t *testing.T) {
	ctx, store, _ := testSetup(t)
	storagetesting.RunTestWatchFromNoneZero(ctx, t, store)
}

func TestDelayedWatchDelivery(t *testing.T) {
	ctx, store, _ := testSetup(t)
	storagetesting.RunTestDelayedWatchDelivery(ctx, t, store)
}

func TestWatchError(t *testing.T) {
	ctx, store, _ := testSetup(t)
	storagetesting.RunTestWatchError(ctx, t, &storeWithPrefixTransformer{store})
}

func TestWatchContextCancel(t *testing.T) {
	ctx, store, _ := testSetup(t)
	storagetesting.RunTestWatchContextCancel(ctx, t, store)
}

func TestWatcherTimeout(t *testing.T) {
	ctx, store, _ := testSetup(t)
	storagetesting.RunTestWatcherTimeout(ctx, t, store)
}

func TestWatchDeleteEventObjectHaveLatestRV(t *testing.T) {
	ctx, store, _ := testSetup(t)
	storagetesting.RunTestWatchDeleteEventObjectHaveLatestRV(ctx, t, store)
}

func TestWatchInitializationSignal(t *testing.T) {
	ctx, store, _ := testSetup(t)
	storagetesting.RunTestWatchInitializationSignal(ctx, t, store)
}

func TestProgressNotify(t *testing.T) {
	clusterConfig := testserver.NewTestConfig(t)
	clusterConfig.ExperimentalWatchProgressNotifyInterval = time.Second
	ctx, store, _ := testSetup(t, withClientConfig(clusterConfig))

	storagetesting.RunOptionalTestProgressNotify(ctx, t, store)
}

// =======================================================================
// Implementation-specific tests are following.
// The following tests are exercising the details of the implementation
// not the actual user-facing contract of storage interface.
// As such, they may focus e.g. on non-functional aspects like performance
// impact.
// =======================================================================

func TestWatchErrResultNotBlockAfterCancel(t *testing.T) {
	origCtx, store, _ := testSetup(t)
	ctx, cancel := context.WithCancel(origCtx)
	w := store.watcher.createWatchChan(ctx, "/abc", 0, false, false, newTestTransformer(), storage.Everything)
	// make resultChan and errChan blocking to ensure ordering.
	w.resultChan = make(chan watch.Event)
	w.errChan = make(chan error)
	// The event flow goes like:
	// - first we send an error, it should block on resultChan.
	// - Then we cancel ctx. The blocking on resultChan should be freed up
	//   and run() goroutine should return.
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		w.run()
		wg.Done()
	}()
	w.errChan <- fmt.Errorf("some error")
	cancel()
	wg.Wait()
}
