/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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
	"errors"
	"fmt"
	"reflect"
	"testing"
	"time"

	"sync"

	"github.com/coreos/etcd/integration"
	"golang.org/x/net/context"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/storage"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/pkg/watch"
)

func TestWatch(t *testing.T) {
	testWatch(t, false)
}

func TestWatchList(t *testing.T) {
	testWatch(t, true)
}

// It tests that
// - first occurrence of objects should notify Add event
// -
func testWatch(t *testing.T, recursive bool) {
	ctx, store, cluster := testSetup(t)
	defer cluster.Terminate(t)

	podFoo := &api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo"}}
	podBar := &api.Pod{ObjectMeta: api.ObjectMeta{Name: "bar"}}

	tests := []struct {
		key        string
		filter     storage.FilterFunc
		watchTests []*testWatchStruct
	}{{ // create a key
		key:        "/somekey-1",
		watchTests: []*testWatchStruct{{podFoo, true, watch.Added}},
		filter:     storage.Everything,
	}, { // create a key but obj gets filtered
		key:        "/somekey-2",
		watchTests: []*testWatchStruct{{podFoo, false, ""}},
		filter:     func(runtime.Object) bool { return false },
	}, { // create a key but obj gets filtered. Then update it with unfiltered obj
		key:        "/somekey-3",
		watchTests: []*testWatchStruct{{podFoo, false, ""}, {podBar, true, watch.Added}},
		filter: func(obj runtime.Object) bool {
			pod := obj.(*api.Pod)
			return pod.Name == "bar"
		},
	}, { // update
		key:        "/somekey-4",
		watchTests: []*testWatchStruct{{podFoo, true, watch.Added}, {podBar, true, watch.Modified}},
		filter:     storage.Everything,
	}, { // delete because of being filtered
		key:        "/somekey-5",
		watchTests: []*testWatchStruct{{podFoo, true, watch.Added}, {podBar, true, watch.Deleted}},
		filter: func(obj runtime.Object) bool {
			pod := obj.(*api.Pod)
			return pod.Name != "bar"
		},
	}}
	for i, tt := range tests {
		w, err := store.watch(ctx, tt.key, "0", tt.filter, recursive)
		if err != nil {
			t.Fatalf("Watch failed: %v", err)
		}
		for _, watchTest := range tt.watchTests {
			out := &api.Pod{}
			key := tt.key
			if recursive {
				key = key + "/item"
			}
			err := store.GuaranteedUpdate(ctx, key, out, true, nil, storage.SimpleUpdate(
				func(runtime.Object) (runtime.Object, error) {
					return watchTest.obj, nil
				}))
			if err != nil {
				t.Fatalf("GuaranteedUpdate failed: %v", err)
			}
			if watchTest.expectEvent {
				testCheckResult(t, i, watchTest.watchType, w, nil)
			}
		}
		w.Stop()
		testCheckStop(t, i, w)
	}
}

func TestDeleteTriggerWatch(t *testing.T) {
	ctx, store, cluster := testSetup(t)
	defer cluster.Terminate(t)
	key, storedObj := testPropogateStore(t, store, ctx, &api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo"}})
	w, err := store.Watch(ctx, key, storedObj.ResourceVersion, storage.Everything)
	if err != nil {
		t.Fatalf("Watch failed: %v", err)
	}
	if err := store.Delete(ctx, key, &api.Pod{}, nil); err != nil {
		t.Fatalf("Delete failed: %v", err)
	}
	testCheckResult(t, 0, watch.Deleted, w, storedObj)
}

// TestWatchSync tests that
// - watch from 0 should sync up and grab the object added before
// - watch from non-0 should just watch changes after given version
func TestWatchFromZeroAndNoneZero(t *testing.T) {
	ctx, store, cluster := testSetup(t)
	defer cluster.Terminate(t)
	key, storedObj := testPropogateStore(t, store, ctx, &api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo"}})

	w, err := store.Watch(ctx, key, "0", storage.Everything)
	if err != nil {
		t.Fatalf("Watch failed: %v", err)
	}
	testCheckResult(t, 0, watch.Added, w, storedObj)
	w.Stop()
	testCheckStop(t, 0, w)

	w, err = store.Watch(ctx, key, storedObj.ResourceVersion, storage.Everything)
	if err != nil {
		t.Fatalf("Watch failed: %v", err)
	}
	out := &api.Pod{}
	store.GuaranteedUpdate(ctx, key, out, true, nil, storage.SimpleUpdate(
		func(runtime.Object) (runtime.Object, error) {
			return &api.Pod{ObjectMeta: api.ObjectMeta{Name: "bar"}}, err
		}))
	testCheckResult(t, 0, watch.Modified, w, out)
}

func TestWatchError(t *testing.T) {
	cluster := integration.NewClusterV3(t, &integration.ClusterConfig{Size: 1})
	defer cluster.Terminate(t)
	invalidStore := newStore(cluster.RandClient(), &testCodec{testapi.Default.Codec()}, "")
	ctx := context.Background()
	w, err := invalidStore.Watch(ctx, "/abc", "0", storage.Everything)
	if err != nil {
		t.Fatalf("Watch failed: %v", err)
	}
	validStore := newStore(cluster.RandClient(), testapi.Default.Codec(), "")
	validStore.GuaranteedUpdate(ctx, "/abc", &api.Pod{}, true, nil, storage.SimpleUpdate(
		func(runtime.Object) (runtime.Object, error) {
			return &api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo"}}, nil
		}))
	testCheckResult(t, 0, watch.Error, w, nil)
}

func TestWatchContextCancel(t *testing.T) {
	ctx, store, cluster := testSetup(t)
	defer cluster.Terminate(t)
	canceledCtx, cancel := context.WithCancel(ctx)
	cancel()
	w := store.watcher.createWatchChan(canceledCtx, "/abc", 0, false, storage.Everything)
	// When we do a client.Get with a canceled context, it will return error.
	// Nonetheless, when we try to send it over internal errChan, we should detect
	// it's context canceled and not send it.
	err := w.sync()
	w.ctx = ctx
	w.sendError(err)
	select {
	case err := <-w.errChan:
		t.Errorf("cancelling context shouldn't return any error. Err: %v", err)
	default:
	}
}

func TestWatchErrResultNotBlockAfterCancel(t *testing.T) {
	origCtx, store, cluster := testSetup(t)
	defer cluster.Terminate(t)
	ctx, cancel := context.WithCancel(origCtx)
	w := store.watcher.createWatchChan(ctx, "/abc", 0, false, storage.Everything)
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

type testWatchStruct struct {
	obj         *api.Pod
	expectEvent bool
	watchType   watch.EventType
}

type testCodec struct {
	runtime.Codec
}

func (c *testCodec) Decode(data []byte, defaults *unversioned.GroupVersionKind, into runtime.Object) (runtime.Object, *unversioned.GroupVersionKind, error) {
	return nil, nil, errors.New("Expected decoding failure")
}

func testCheckResult(t *testing.T, i int, expectEventType watch.EventType, w watch.Interface, expectObj *api.Pod) {
	select {
	case res := <-w.ResultChan():
		if res.Type != expectEventType {
			t.Errorf("#%d: event type want=%v, get=%v", i, expectEventType, res.Type)
			return
		}
		if expectObj != nil && !reflect.DeepEqual(expectObj, res.Object) {
			t.Errorf("#%d: obj want=\n%#v\nget=\n%#v", i, expectObj, res.Object)
		}
	case <-time.After(wait.ForeverTestTimeout):
		t.Errorf("#%d: time out after waiting %v on ResultChan", i, wait.ForeverTestTimeout)
	}
}

func testCheckStop(t *testing.T, i int, w watch.Interface) {
	select {
	case e, ok := <-w.ResultChan():
		if ok {
			var obj string
			switch e.Object.(type) {
			case *api.Pod:
				obj = e.Object.(*api.Pod).Name
			case *unversioned.Status:
				obj = e.Object.(*unversioned.Status).Message
			}
			t.Errorf("#%d: ResultChan should have been closed. Event: %s. Object: %s", i, e.Type, obj)
		}
	case <-time.After(wait.ForeverTestTimeout):
		t.Errorf("#%d: time out after waiting 1s on ResultChan", i)
	}
}
