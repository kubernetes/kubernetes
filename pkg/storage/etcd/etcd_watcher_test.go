/*
Copyright 2014 The Kubernetes Authors.

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

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/storage"
	"k8s.io/kubernetes/pkg/storage/etcd/etcdtest"
	etcdtesting "k8s.io/kubernetes/pkg/storage/etcd/testing"

	etcd "github.com/coreos/etcd/client"
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

	// All of these test cases will be run with the firstLetterIsB Filter.
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
	firstLetterIsB := func(obj runtime.Object) bool {
		return obj.(*api.Pod).Name[0] == 'b'
	}
	for name, item := range table {
		for _, action := range item.actions {
			w := newEtcdWatcher(true, false, nil, firstLetterIsB, codec, versioner, nil, &fakeEtcdCache{})
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
	_, codec := testScheme(t)
	w := newEtcdWatcher(false, false, nil, storage.SimpleFilter(storage.Everything), codec, versioner, nil, &fakeEtcdCache{})
	w.emit = func(e watch.Event) {
		t.Errorf("Unexpected emit: %v", e)
	}

	w.sendResult(&etcd.Response{
		Action: "update",
	})
	w.Stop()
}

func TestWatchInterpretation_ResponseNoNode(t *testing.T) {
	_, codec := testScheme(t)
	actions := []string{"create", "set", "compareAndSwap", "delete"}
	for _, action := range actions {
		w := newEtcdWatcher(false, false, nil, storage.SimpleFilter(storage.Everything), codec, versioner, nil, &fakeEtcdCache{})
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
	_, codec := testScheme(t)
	actions := []string{"create", "set", "compareAndSwap", "delete"}
	for _, action := range actions {
		w := newEtcdWatcher(false, false, nil, storage.SimpleFilter(storage.Everything), codec, versioner, nil, &fakeEtcdCache{})
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

func TestSendResultDeleteEventHaveLatestIndex(t *testing.T) {
	codec := testapi.Default.Codec()
	filter := func(obj runtime.Object) bool {
		return obj.(*api.Pod).Name != "bar"
	}
	w := newEtcdWatcher(false, false, nil, filter, codec, versioner, nil, &fakeEtcdCache{})

	eventChan := make(chan watch.Event, 1)
	w.emit = func(e watch.Event) {
		eventChan <- e
	}

	fooPod := &api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo"}}
	barPod := &api.Pod{ObjectMeta: api.ObjectMeta{Name: "bar"}}
	fooBytes, err := runtime.Encode(codec, fooPod)
	if err != nil {
		t.Fatalf("Encode failed: %v", err)
	}
	barBytes, err := runtime.Encode(codec, barPod)
	if err != nil {
		t.Fatalf("Encode failed: %v", err)
	}

	tests := []struct {
		response *etcd.Response
		expRV    string
	}{{ // Delete event
		response: &etcd.Response{
			Action: EtcdDelete,
			Node: &etcd.Node{
				ModifiedIndex: 2,
			},
			PrevNode: &etcd.Node{
				Value:         string(fooBytes),
				ModifiedIndex: 1,
			},
		},
		expRV: "2",
	}, { // Modify event with uninterested data
		response: &etcd.Response{
			Action: EtcdSet,
			Node: &etcd.Node{
				Value:         string(barBytes),
				ModifiedIndex: 2,
			},
			PrevNode: &etcd.Node{
				Value:         string(fooBytes),
				ModifiedIndex: 1,
			},
		},
		expRV: "2",
	}}

	for i, tt := range tests {
		w.sendResult(tt.response)
		ev := <-eventChan
		if ev.Type != watch.Deleted {
			t.Errorf("#%d: event type want=Deleted, get=%s", i, ev.Type)
			return
		}
		rv := ev.Object.(*api.Pod).ResourceVersion
		if rv != tt.expRV {
			t.Errorf("#%d: resource version want=%s, get=%s", i, tt.expRV, rv)
		}
	}
	w.Stop()
}

func TestWatch(t *testing.T) {
	codec := testapi.Default.Codec()
	server := etcdtesting.NewEtcdTestClientServer(t)
	defer server.Terminate(t)
	key := "/some/key"
	h := newEtcdHelper(server.Client, codec, etcdtest.PathPrefix())

	watching, err := h.Watch(context.TODO(), key, "0", storage.Everything)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	// watching is explicitly closed below.

	// Test normal case
	pod := &api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo"}}
	returnObj := &api.Pod{}
	err = h.Create(context.TODO(), key, pod, returnObj, 0)
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

	// There is a race in etcdWatcher so that after calling Stop() one of
	// two things can happen:
	// - ResultChan() may be closed (triggered by closing userStop channel)
	// - an Error "context cancelled" may be emitted (triggered by cancelling request
	//   to etcd and putting that error to etcdError channel)
	// We need to be prepared for both here.
	event, open := <-watching.ResultChan()
	if open && event.Type != watch.Error {
		t.Errorf("Unexpected event from stopped watcher: %#v", event)
	}
}

func emptySubsets() []api.EndpointSubset {
	return []api.EndpointSubset{}
}

func makeSubsets(ip string, port int) []api.EndpointSubset {
	return []api.EndpointSubset{{
		Addresses: []api.EndpointAddress{{IP: ip}},
		Ports:     []api.EndpointPort{{Port: int32(port)}},
	}}
}

func TestWatchEtcdState(t *testing.T) {
	codec := testapi.Default.Codec()
	key := "/somekey/foo"
	server := etcdtesting.NewEtcdTestClientServer(t)
	defer server.Terminate(t)

	h := newEtcdHelper(server.Client, codec, etcdtest.PathPrefix())
	watching, err := h.Watch(context.TODO(), key, "0", storage.Everything)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	defer watching.Stop()

	endpoint := &api.Endpoints{
		ObjectMeta: api.ObjectMeta{Name: "foo"},
		Subsets:    emptySubsets(),
	}

	err = h.Create(context.TODO(), key, endpoint, endpoint, 0)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	event := <-watching.ResultChan()
	if event.Type != watch.Added {
		t.Errorf("Unexpected event %#v", event)
	}

	subset := makeSubsets("127.0.0.1", 9000)
	endpoint.Subsets = subset
	endpoint.ResourceVersion = ""

	// CAS the previous value
	updateFn := func(input runtime.Object, res storage.ResponseMeta) (runtime.Object, *uint64, error) {
		newObj, err := api.Scheme.DeepCopy(endpoint)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
			return nil, nil, err
		}
		return newObj.(*api.Endpoints), nil, nil
	}
	err = h.GuaranteedUpdate(context.TODO(), key, &api.Endpoints{}, false, nil, updateFn)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	event = <-watching.ResultChan()
	if event.Type != watch.Modified {
		t.Errorf("Unexpected event %#v", event)
	}

	if e, a := endpoint, event.Object; !api.Semantic.DeepDerivative(e, a) {
		t.Errorf("Unexpected error: expected %#v, got %#v", e, a)
	}
}

func TestWatchFromZeroIndex(t *testing.T) {
	codec := testapi.Default.Codec()
	pod := &api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo"}}

	key := "/somekey/foo"
	server := etcdtesting.NewEtcdTestClientServer(t)
	defer server.Terminate(t)

	h := newEtcdHelper(server.Client, codec, etcdtest.PathPrefix())

	// set before the watch and verify events
	err := h.Create(context.TODO(), key, pod, pod, 0)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	pod.ResourceVersion = ""

	watching, err := h.Watch(context.TODO(), key, "0", storage.Everything)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	// The create trigger ADDED event when watching from 0
	event := <-watching.ResultChan()
	watching.Stop()
	if event.Type != watch.Added {
		t.Errorf("Unexpected event %#v", event)
	}

	// check for concatenation on watch event with CAS
	updateFn := func(input runtime.Object, res storage.ResponseMeta) (runtime.Object, *uint64, error) {
		pod := input.(*api.Pod)
		pod.Name = "bar"
		return pod, nil, nil
	}
	err = h.GuaranteedUpdate(context.TODO(), key, &api.Pod{}, false, nil, updateFn)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	watching, err = h.Watch(context.TODO(), key, "0", storage.Everything)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	defer watching.Stop()

	// because we watch from 0, first event that we receive will always be ADDED
	event = <-watching.ResultChan()
	if event.Type != watch.Added {
		t.Errorf("Unexpected event %#v", event)
	}

	pod.Name = "baz"
	updateFn = func(input runtime.Object, res storage.ResponseMeta) (runtime.Object, *uint64, error) {
		pod := input.(*api.Pod)
		pod.Name = "baz"
		return pod, nil, nil
	}
	err = h.GuaranteedUpdate(context.TODO(), key, &api.Pod{}, false, nil, updateFn)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	event = <-watching.ResultChan()
	if event.Type != watch.Modified {
		t.Errorf("Unexpected event %#v", event)
	}

	if e, a := pod, event.Object; a == nil || !api.Semantic.DeepDerivative(e, a) {
		t.Errorf("Unexpected error: expected %#v, got %#v", e, a)
	}
}

func TestWatchListFromZeroIndex(t *testing.T) {
	codec := testapi.Default.Codec()
	prefix := "/some/key"
	server := etcdtesting.NewEtcdTestClientServer(t)
	defer server.Terminate(t)
	h := newEtcdHelper(server.Client, codec, prefix)

	watching, err := h.WatchList(context.TODO(), "/", "0", storage.Everything)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	defer watching.Stop()

	// creates foo which should trigger the WatchList for "/"
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
		t.Errorf("Unexpected error: expected %v, got %v", e, a)
	}
}

func TestWatchListIgnoresRootKey(t *testing.T) {
	codec := testapi.Default.Codec()
	pod := &api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo"}}
	key := "/some/key"
	server := etcdtesting.NewEtcdTestClientServer(t)
	defer server.Terminate(t)
	h := newEtcdHelper(server.Client, codec, key)

	watching, err := h.WatchList(context.TODO(), key, "0", storage.Everything)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	defer watching.Stop()

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
}

func TestWatchPurposefulShutdown(t *testing.T) {
	_, codec := testScheme(t)
	server := etcdtesting.NewEtcdTestClientServer(t)
	defer server.Terminate(t)
	key := "/some/key"
	h := newEtcdHelper(server.Client, codec, etcdtest.PathPrefix())

	// Test purposeful shutdown
	watching, err := h.Watch(context.TODO(), key, "0", storage.Everything)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	watching.Stop()
	rt.Gosched()

	// There is a race in etcdWatcher so that after calling Stop() one of
	// two things can happen:
	// - ResultChan() may be closed (triggered by closing userStop channel)
	// - an Error "context cancelled" may be emitted (triggered by cancelling request
	//   to etcd and putting that error to etcdError channel)
	// We need to be prepared for both here.
	event, open := <-watching.ResultChan()
	if open && event.Type != watch.Error {
		t.Errorf("Unexpected event from stopped watcher: %#v", event)
	}
}
