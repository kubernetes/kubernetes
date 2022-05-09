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

	"go.etcd.io/etcd/api/v3/mvccpb"
	clientv3 "go.etcd.io/etcd/client/v3"
	storagetesting "k8s.io/apiserver/pkg/storage/testing"

	"k8s.io/apimachinery/pkg/api/apitesting"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/apis/example"
	examplev1 "k8s.io/apiserver/pkg/apis/example/v1"
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

// TestWatchFromZero tests that
// - watch from 0 should sync up and grab the object added before
// - watch from 0 is able to return events for objects whose previous version has been compacted
func TestWatchFromZero(t *testing.T) {
	ctx, store, client := testSetup(t)
	key, storedObj := storagetesting.TestPropogateStore(ctx, t, store, &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "ns"}})

	w, err := store.Watch(ctx, key, storage.ListOptions{ResourceVersion: "0", Predicate: storage.Everything})
	if err != nil {
		t.Fatalf("Watch failed: %v", err)
	}
	storagetesting.TestCheckResult(t, watch.Added, w, storedObj)
	w.Stop()

	// Update
	out := &example.Pod{}
	err = store.GuaranteedUpdate(ctx, key, out, true, nil, storage.SimpleUpdate(
		func(runtime.Object) (runtime.Object, error) {
			return &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "ns", Annotations: map[string]string{"a": "1"}}}, nil
		}), nil)
	if err != nil {
		t.Fatalf("GuaranteedUpdate failed: %v", err)
	}

	// Make sure when we watch from 0 we receive an ADDED event
	w, err = store.Watch(ctx, key, storage.ListOptions{ResourceVersion: "0", Predicate: storage.Everything})
	if err != nil {
		t.Fatalf("Watch failed: %v", err)
	}
	storagetesting.TestCheckResult(t, watch.Added, w, out)
	w.Stop()

	// Update again
	out = &example.Pod{}
	err = store.GuaranteedUpdate(ctx, key, out, true, nil, storage.SimpleUpdate(
		func(runtime.Object) (runtime.Object, error) {
			return &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "ns"}}, nil
		}), nil)
	if err != nil {
		t.Fatalf("GuaranteedUpdate failed: %v", err)
	}

	// Compact previous versions
	revToCompact, err := store.versioner.ParseResourceVersion(out.ResourceVersion)
	if err != nil {
		t.Fatalf("Error converting %q to an int: %v", storedObj.ResourceVersion, err)
	}
	_, err = client.Compact(ctx, int64(revToCompact), clientv3.WithCompactPhysical())
	if err != nil {
		t.Fatalf("Error compacting: %v", err)
	}

	// Make sure we can still watch from 0 and receive an ADDED event
	w, err = store.Watch(ctx, key, storage.ListOptions{ResourceVersion: "0", Predicate: storage.Everything})
	if err != nil {
		t.Fatalf("Watch failed: %v", err)
	}
	storagetesting.TestCheckResult(t, watch.Added, w, out)
}

// TestWatchFromNoneZero tests that
// - watch from non-0 should just watch changes after given version
func TestWatchFromNoneZero(t *testing.T) {
	ctx, store, _ := testSetup(t)
	storagetesting.RunTestWatchFromNoneZero(ctx, t, store)
}

func TestWatchError(t *testing.T) {
	// this codec fails on decodes, which will bubble up so we can verify the behavior
	invalidCodec := &testCodec{apitesting.TestCodec(codecs, examplev1.SchemeGroupVersion)}
	ctx, invalidStore, client := testSetup(t, withCodec(invalidCodec))
	w, err := invalidStore.Watch(ctx, "/abc", storage.ListOptions{ResourceVersion: "0", Predicate: storage.Everything})
	if err != nil {
		t.Fatalf("Watch failed: %v", err)
	}
	codec := apitesting.TestCodec(codecs, examplev1.SchemeGroupVersion)
	_, validStore, _ := testSetup(t, withCodec(codec), withClient(client))
	if err := validStore.GuaranteedUpdate(ctx, "/abc", &example.Pod{}, true, nil, storage.SimpleUpdate(
		func(runtime.Object) (runtime.Object, error) {
			return &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}}, nil
		}), nil); err != nil {
		t.Fatalf("GuaranteedUpdate failed: %v", err)
	}
	storagetesting.TestCheckEventType(t, watch.Error, w)
}

func TestWatchContextCancel(t *testing.T) {
	ctx, store, _ := testSetup(t)
	storagetesting.RunTestWatchContextCancel(ctx, t, store)
}

func TestWatchErrResultNotBlockAfterCancel(t *testing.T) {
	origCtx, store, _ := testSetup(t)
	ctx, cancel := context.WithCancel(origCtx)
	w := store.watcher.createWatchChan(ctx, "/abc", 0, false, false, storage.Everything)
	// make resutlChan and errChan blocking to ensure ordering.
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
	ctx, store, client := testSetup(t)
	key, storedObj := storagetesting.TestPropogateStore(ctx, t, store, &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}})

	w, err := store.Watch(ctx, key, storage.ListOptions{ResourceVersion: storedObj.ResourceVersion, Predicate: storage.Everything})
	if err != nil {
		t.Fatalf("Watch failed: %v", err)
	}
	rv, err := APIObjectVersioner{}.ObjectResourceVersion(storedObj)
	if err != nil {
		t.Fatalf("failed to parse resourceVersion on stored object: %v", err)
	}
	etcdW := client.Watch(ctx, key, clientv3.WithRev(int64(rv)))

	if err := store.Delete(ctx, key, &example.Pod{}, &storage.Preconditions{}, storage.ValidateAllObjectFunc, nil); err != nil {
		t.Fatalf("Delete failed: %v", err)
	}

	var e watch.Event
	watchCtx, _ := context.WithTimeout(ctx, wait.ForeverTestTimeout)
	select {
	case e = <-w.ResultChan():
	case <-watchCtx.Done():
		t.Fatalf("timed out waiting for watch event")
	}
	deletedRV, err := deletedRevision(watchCtx, etcdW)
	if err != nil {
		t.Fatalf("did not see delete event in raw watch: %v", err)
	}
	watchedDeleteObj := e.Object.(*example.Pod)

	watchedDeleteRev, err := store.versioner.ParseResourceVersion(watchedDeleteObj.ResourceVersion)
	if err != nil {
		t.Fatalf("ParseWatchResourceVersion failed: %v", err)
	}
	if int64(watchedDeleteRev) != deletedRV {
		t.Errorf("Object from delete event have version: %v, should be the same as etcd delete's mod rev: %d",
			watchedDeleteRev, deletedRV)
	}
}

func deletedRevision(ctx context.Context, watch <-chan clientv3.WatchResponse) (int64, error) {
	for {
		select {
		case <-ctx.Done():
			return 0, ctx.Err()
		case wres := <-watch:
			for _, evt := range wres.Events {
				if evt.Type == mvccpb.DELETE && evt.Kv != nil {
					return evt.Kv.ModRevision, nil
				}
			}
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
	validateResourceVersion := resourceVersionNotOlderThan(out.ResourceVersion)

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

type testCodec struct {
	runtime.Codec
}

func (c *testCodec) Decode(data []byte, defaults *schema.GroupVersionKind, into runtime.Object) (runtime.Object, *schema.GroupVersionKind, error) {
	return nil, nil, errTestingDecode
}

// resourceVersionNotOlderThan returns a function to validate resource versions. Resource versions
// referring to points in logical time before the sentinel generate an error. All logical times as
// new as the sentinel or newer generate no error.
func resourceVersionNotOlderThan(sentinel string) func(string) error {
	return func(resourceVersion string) error {
		objectVersioner := APIObjectVersioner{}
		actualRV, err := objectVersioner.ParseResourceVersion(resourceVersion)
		if err != nil {
			return err
		}
		expectedRV, err := objectVersioner.ParseResourceVersion(sentinel)
		if err != nil {
			return err
		}
		if actualRV < expectedRV {
			return fmt.Errorf("expected a resourceVersion no smaller than than %d, but got %d", expectedRV, actualRV)
		}
		return nil
	}
}
