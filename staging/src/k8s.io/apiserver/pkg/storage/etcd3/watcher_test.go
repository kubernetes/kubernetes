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

	storagetesting "k8s.io/apiserver/pkg/storage/testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/apis/example"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/etcd3/testserver"
)

func TestWatch(t *testing.T) {
	ctx, store, _ := testSetup(t)
	storagetesting.RunTestWatch(ctx, t, store)
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

func TestWatchError(t *testing.T) {
	ctx, store, _ := testSetup(t)
	storagetesting.RunTestWatchError(ctx, t, &storeWithPrefixTransformer{store})
}

func TestWatchContextCancel(t *testing.T) {
	ctx, store, _ := testSetup(t)
	storagetesting.RunTestWatchContextCancel(ctx, t, store)
}

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

func TestWatchDeleteEventObjectHaveLatestRV(t *testing.T) {
	ctx, store, _ := testSetup(t)

	key, storedObj := storagetesting.TestPropagateStore(ctx, t, store, &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}})

	watchCtx, _ := context.WithTimeout(ctx, wait.ForeverTestTimeout)
	w, err := store.Watch(watchCtx, key, storage.ListOptions{ResourceVersion: storedObj.ResourceVersion, Predicate: storage.Everything})
	if err != nil {
		t.Fatalf("Watch failed: %v", err)
	}

	deletedObj := &example.Pod{}
	if err := store.Delete(ctx, key, deletedObj, &storage.Preconditions{}, storage.ValidateAllObjectFunc, nil); err != nil {
		t.Fatalf("Delete failed: %v", err)
	}

	// Verify that ResourceVersion has changed on deletion.
	if storedObj.ResourceVersion == deletedObj.ResourceVersion {
		t.Fatalf("ResourceVersion didn't changed on deletion: %s", deletedObj.ResourceVersion)
	}

	select {
	case event := <-w.ResultChan():
		watchedDeleteObj := event.Object.(*example.Pod)
		if e, a := deletedObj.ResourceVersion, watchedDeleteObj.ResourceVersion; e != a {
			t.Errorf("Unexpected resource version: %v, expected %v", a, e)
		}
	}
}

func TestWatchInitializationSignal(t *testing.T) {
	ctx, store, _ := testSetup(t)
	storagetesting.RunTestWatchInitializationSignal(ctx, t, store)
}

func TestProgressNotify(t *testing.T) {
	clusterConfig := testserver.NewTestConfig(t)
	clusterConfig.ExperimentalWatchProgressNotifyInterval = time.Second
	ctx, store, _ := testSetup(t, withClientConfig(clusterConfig))

	key := "/somekey"
	input := &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "name"}}
	out := &example.Pod{}
	if err := store.Create(ctx, key, input, out, 0); err != nil {
		t.Fatalf("Create failed: %v", err)
	}
	validateResourceVersion := storagetesting.ResourceVersionNotOlderThan(out.ResourceVersion)

	opts := storage.ListOptions{
		ResourceVersion: out.ResourceVersion,
		Predicate:       storage.Everything,
		ProgressNotify:  true,
	}
	w, err := store.Watch(ctx, key, opts)
	if err != nil {
		t.Fatalf("Watch failed: %v", err)
	}

	// when we send a bookmark event, the client expects the event to contain an
	// object of the correct type, but with no fields set other than the resourceVersion
	storagetesting.TestCheckResultFunc(t, watch.Bookmark, w, func(object runtime.Object) error {
		// first, check that we have the correct resource version
		obj, ok := object.(metav1.Object)
		if !ok {
			return fmt.Errorf("got %T, not metav1.Object", object)
		}
		if err := validateResourceVersion(obj.GetResourceVersion()); err != nil {
			return err
		}

		// then, check that we have the right type and content
		pod, ok := object.(*example.Pod)
		if !ok {
			return fmt.Errorf("got %T, not *example.Pod", object)
		}
		pod.ResourceVersion = ""
		storagetesting.ExpectNoDiff(t, "bookmark event should contain an object with no fields set other than resourceVersion", newPod(), pod)
		return nil
	})
}
