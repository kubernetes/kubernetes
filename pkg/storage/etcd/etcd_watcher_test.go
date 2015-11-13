/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package etcd

import (
	rt "runtime"
	"testing"

	"github.com/coreos/go-etcd/etcd"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/storage"
	"k8s.io/kubernetes/pkg/storage/etcd/etcdtest"
	etcdtesting "k8s.io/kubernetes/pkg/storage/etcd/testing"
	"k8s.io/kubernetes/pkg/watch"

	"golang.org/x/net/context"
)

var versioner = APIObjectVersioner{}

// Implements etcdCache interface as empty methods (i.e. does not cache any objects)
type fakeEtcdCache struct{}

func (f *fakeEtcdCache) getFromCache(index uint64, filter storage.FilterFunc) (runtime.Object, bool) {
	return nil, false
}

func (f *fakeEtcdCache) addToCache(index uint64, obj runtime.Object) {
}

var _ etcdCache = &fakeEtcdCache{}

func TestWatchInterpretations(t *testing.T) {
	codec := testapi.Default.Codec()
	// Declare some pods to make the test cases compact.
	podFoo := &api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo"}}
	podBar := &api.Pod{ObjectMeta: api.ObjectMeta{Name: "bar"}}
	podBaz := &api.Pod{ObjectMeta: api.ObjectMeta{Name: "baz"}}
	firstLetterIsB := func(obj runtime.Object) bool {
		return obj.(*api.Pod).Name[0] == 'b'
	}

	// All of these test cases will be run with the firstLetterIsB FilterFunc.
	table := map[string]struct {
		actions       []string // Run this test item for every action here.
		prevNodeValue string
		nodeValue     string
		expectEmit    bool
		expectType    watch.EventType
		expectObject  runtime.Object
	}{
		"create": {
			actions:      []string{"create", "get"},
			nodeValue:    runtime.EncodeOrDie(codec, podBar),
			expectEmit:   true,
			expectType:   watch.Added,
			expectObject: podBar,
		},
		"create but filter blocks": {
			actions:    []string{"create", "get"},
			nodeValue:  runtime.EncodeOrDie(codec, podFoo),
			expectEmit: false,
		},
		"delete": {
			actions:       []string{"delete"},
			prevNodeValue: runtime.EncodeOrDie(codec, podBar),
			expectEmit:    true,
			expectType:    watch.Deleted,
			expectObject:  podBar,
		},
		"delete but filter blocks": {
			actions:    []string{"delete"},
			nodeValue:  runtime.EncodeOrDie(codec, podFoo),
			expectEmit: false,
		},
		"modify appears to create 1": {
			actions:      []string{"set", "compareAndSwap"},
			nodeValue:    runtime.EncodeOrDie(codec, podBar),
			expectEmit:   true,
			expectType:   watch.Added,
			expectObject: podBar,
		},
		"modify appears to create 2": {
			actions:       []string{"set", "compareAndSwap"},
			prevNodeValue: runtime.EncodeOrDie(codec, podFoo),
			nodeValue:     runtime.EncodeOrDie(codec, podBar),
			expectEmit:    true,
			expectType:    watch.Added,
			expectObject:  podBar,
		},
		"modify appears to delete": {
			actions:       []string{"set", "compareAndSwap"},
			prevNodeValue: runtime.EncodeOrDie(codec, podBar),
			nodeValue:     runtime.EncodeOrDie(codec, podFoo),
			expectEmit:    true,
			expectType:    watch.Deleted,
			expectObject:  podBar, // Should return last state that passed the filter!
		},
		"modify modifies": {
			actions:       []string{"set", "compareAndSwap"},
			prevNodeValue: runtime.EncodeOrDie(codec, podBar),
			nodeValue:     runtime.EncodeOrDie(codec, podBaz),
			expectEmit:    true,
			expectType:    watch.Modified,
			expectObject:  podBaz,
		},
		"modify ignores": {
			actions:    []string{"set", "compareAndSwap"},
			nodeValue:  runtime.EncodeOrDie(codec, podFoo),
			expectEmit: false,
		},
	}

	for name, item := range table {
		for _, action := range item.actions {
			w := newEtcdWatcher(true, nil, firstLetterIsB, codec, versioner, nil, &fakeEtcdCache{})
			emitCalled := false
			w.emit = func(event watch.Event) {
				emitCalled = true
				if !item.expectEmit {
					return
				}
				if e, a := item.expectType, event.Type; e != a {
					t.Errorf("'%v - %v': expected %v, got %v", name, action, e, a)
				}
				if e, a := item.expectObject, event.Object; !api.Semantic.DeepDerivative(e, a) {
					t.Errorf("'%v - %v': expected %v, got %v", name, action, e, a)
				}
			}

			var n, pn *etcd.Node
			if item.nodeValue != "" {
				n = &etcd.Node{Value: item.nodeValue}
			}
			if item.prevNodeValue != "" {
				pn = &etcd.Node{Value: item.prevNodeValue}
			}

			w.sendResult(&etcd.Response{
				Action:   action,
				Node:     n,
				PrevNode: pn,
			})

			if e, a := item.expectEmit, emitCalled; e != a {
				t.Errorf("'%v - %v': expected %v, got %v", name, action, e, a)
			}
			w.Stop()
		}
	}
}

func TestWatchInterpretation_ResponseNotSet(t *testing.T) {
	w := newEtcdWatcher(false, nil, storage.Everything, codec, versioner, nil, &fakeEtcdCache{})
	w.emit = func(e watch.Event) {
		t.Errorf("Unexpected emit: %v", e)
	}

	w.sendResult(&etcd.Response{
		Action: "update",
	})
	w.Stop()
}

func TestWatchInterpretation_ResponseNoNode(t *testing.T) {
	actions := []string{"create", "set", "compareAndSwap", "delete"}
	for _, action := range actions {
		w := newEtcdWatcher(false, nil, storage.Everything, codec, versioner, nil, &fakeEtcdCache{})
		w.emit = func(e watch.Event) {
			t.Errorf("Unexpected emit: %v", e)
		}
		w.sendResult(&etcd.Response{
			Action: action,
		})
		w.Stop()
	}
}

func TestWatchInterpretation_ResponseBadData(t *testing.T) {
	actions := []string{"create", "set", "compareAndSwap", "delete"}
	for _, action := range actions {
		w := newEtcdWatcher(false, nil, storage.Everything, codec, versioner, nil, &fakeEtcdCache{})
		w.emit = func(e watch.Event) {
			t.Errorf("Unexpected emit: %v", e)
		}
		w.sendResult(&etcd.Response{
			Action: action,
			Node: &etcd.Node{
				Value: "foobar",
			},
		})
		w.sendResult(&etcd.Response{
			Action: action,
			PrevNode: &etcd.Node{
				Value: "foobar",
			},
		})
		w.Stop()
	}
}

/* TODO: So believe it or not... but this test is flakey with the go-etcd client library
 * which I'm surprised by.  Apprently you can close the client that is performing the watch
 * and the watch *never returns.*  I would like to still keep this test here and re-enable
 * with the new 2.2+ client library.
func TestWatchEtcdError(t *testing.T) {
	codec := testapi.Default.Codec()
	server := etcdtesting.NewEtcdTestClientServer(t)
	h := newEtcdHelper(server.Client, codec, etcdtest.PathPrefix())
	watching, err := h.Watch(context.TODO(), "/some/key", 4, storage.Everything)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	server.Terminate(t)

	got := <-watching.ResultChan()
	if got.Type != watch.Error {
		t.Fatalf("Unexpected non-error")
	}
	watching.Stop()
} */

func TestWatch(t *testing.T) {
	codec := testapi.Default.Codec()
	server := etcdtesting.NewEtcdTestClientServer(t)
	defer server.Terminate(t)
	key := "/some/key"
	h := newEtcdHelper(server.Client, codec, etcdtest.PathPrefix())

	watching, err := h.Watch(context.TODO(), key, 0, storage.Everything)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	// Test normal case
	pod := &api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo"}}
	returnObj := &api.Pod{}
	err = h.Set(context.TODO(), key, pod, returnObj, 0)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	event := <-watching.ResultChan()
	if e, a := watch.Added, event.Type; e != a {
		t.Errorf("Expected %v, got %v", e, a)
	}
	if e, a := pod, event.Object; !api.Semantic.DeepDerivative(e, a) {
		t.Errorf("Expected %v, got %v", e, a)
	}

	watching.Stop()

	if _, open := <-watching.ResultChan(); open {
		t.Errorf("An injected error did not cause a graceful shutdown")
	}
}

func emptySubsets() []api.EndpointSubset {
	return []api.EndpointSubset{}
}

func makeSubsets(ip string, port int) []api.EndpointSubset {
	return []api.EndpointSubset{{
		Addresses: []api.EndpointAddress{{IP: ip}},
		Ports:     []api.EndpointPort{{Port: port}},
	}}
}

func TestWatchEtcdState(t *testing.T) {
	codec := testapi.Default.Codec()
	key := etcdtest.AddPrefix("/somekey/foo")
	server := etcdtesting.NewEtcdTestClientServer(t)
	defer server.Terminate(t)

	h := newEtcdHelper(server.Client, codec, etcdtest.PathPrefix())
	watching, err := h.Watch(context.TODO(), key, 0, storage.Everything)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	endpoint := &api.Endpoints{
		ObjectMeta: api.ObjectMeta{Name: "foo"},
		Subsets:    emptySubsets(),
	}

	err = h.Set(context.TODO(), key, endpoint, endpoint, 0)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	event := <-watching.ResultChan()
	if event.Type != watch.Added {
		t.Errorf("Unexpected event %#v", event)
	}

	subset := makeSubsets("127.0.0.1", 9000)
	endpoint.Subsets = subset

	// CAS the previous value
	err = h.Set(context.TODO(), key, endpoint, endpoint, 0)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	event = <-watching.ResultChan()
	if event.Type != watch.Modified {
		t.Errorf("Unexpected event %#v", event)
	}

	if e, a := endpoint, event.Object; !api.Semantic.DeepDerivative(e, a) {
		t.Errorf("%s: expected %v, got %v", e, a)
	}

	watching.Stop()
}

func TestWatchFromZeroIndex(t *testing.T) {
	codec := testapi.Default.Codec()
	pod := &api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo"}}

	key := etcdtest.AddPrefix("/somekey/foo")
	server := etcdtesting.NewEtcdTestClientServer(t)
	defer server.Terminate(t)

	h := newEtcdHelper(server.Client, codec, etcdtest.PathPrefix())

	// set before the watch and verify events
	err := h.Set(context.TODO(), key, pod, pod, 0)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	// check for concatenation on watch event with CAS
	pod.Name = "bar"
	err = h.Set(context.TODO(), key, pod, pod, 0)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	watching, err := h.Watch(context.TODO(), key, 0, storage.Everything)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	// marked as modified b/c of concatenation
	event := <-watching.ResultChan()
	if event.Type != watch.Modified {
		t.Errorf("Unexpected event %#v", event)
	}

	err = h.Set(context.TODO(), key, pod, pod, 0)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	event = <-watching.ResultChan()
	if event.Type != watch.Modified {
		t.Errorf("Unexpected event %#v", event)
	}

	if e, a := pod, event.Object; !api.Semantic.DeepDerivative(e, a) {
		t.Errorf("%s: expected %v, got %v", e, a)
	}

	watching.Stop()
}

func TestWatchListFromZeroIndex(t *testing.T) {
	codec := testapi.Default.Codec()
	key := etcdtest.AddPrefix("/some/key")
	server := etcdtesting.NewEtcdTestClientServer(t)
	defer server.Terminate(t)
	h := newEtcdHelper(server.Client, codec, key)

	watching, err := h.WatchList(context.TODO(), key, 0, storage.Everything)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	// creates key/foo which should trigger the WatchList for "key"
	pod := &api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo"}}
	err = h.Create(context.TODO(), pod.Name, pod, pod, 0)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	event, _ := <-watching.ResultChan()
	if event.Type != watch.Added {
		t.Errorf("Unexpected event %#v", event)
	}

	if e, a := pod, event.Object; !api.Semantic.DeepDerivative(e, a) {
		t.Errorf("%s: expected %v, got %v", e, a)
	}

	watching.Stop()
}

func TestWatchListIgnoresRootKey(t *testing.T) {
	codec := testapi.Default.Codec()
	pod := &api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo"}}
	key := etcdtest.AddPrefix("/some/key")
	server := etcdtesting.NewEtcdTestClientServer(t)
	defer server.Terminate(t)
	h := newEtcdHelper(server.Client, codec, key)

	watching, err := h.WatchList(context.TODO(), key, 0, storage.Everything)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	// creates key/foo which should trigger the WatchList for "key"
	err = h.Create(context.TODO(), key, pod, pod, 0)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	// force context switch to ensure watches would catch and notify.
	rt.Gosched()

	select {
	case event, _ := <-watching.ResultChan():
		t.Fatalf("Unexpected event: %#v", event)
	default:
		// fall through, expected behavior
	}

	watching.Stop()
}

func TestWatchPurposefulShutdown(t *testing.T) {
	server := etcdtesting.NewEtcdTestClientServer(t)
	defer server.Terminate(t)
	key := "/some/key"
	h := newEtcdHelper(server.Client, codec, etcdtest.PathPrefix())

	// Test purposeful shutdown
	watching, err := h.Watch(context.TODO(), key, 0, storage.Everything)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	watching.Stop()
	rt.Gosched()

	if _, open := <-watching.ResultChan(); open {
		t.Errorf("Channel should be closed")
	}
}
