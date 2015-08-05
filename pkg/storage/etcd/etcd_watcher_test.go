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
	"fmt"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/latest"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/storage"
	"k8s.io/kubernetes/pkg/tools"
	"k8s.io/kubernetes/pkg/tools/etcdtest"
	"k8s.io/kubernetes/pkg/watch"
	"github.com/coreos/go-etcd/etcd"
)

var versioner = APIObjectVersioner{}

// Implements etcdCache interface as empty methods (i.e. does not cache any objects)
type fakeEtcdCache struct{}

func (f *fakeEtcdCache) getFromCache(index uint64) (runtime.Object, bool) {
	return nil, false
}

func (f *fakeEtcdCache) addToCache(index uint64, obj runtime.Object) {
}

var _ etcdCache = &fakeEtcdCache{}

func TestWatchInterpretations(t *testing.T) {
	codec := latest.Codec
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

func TestWatchEtcdError(t *testing.T) {
	codec := latest.Codec
	fakeClient := tools.NewFakeEtcdClient(t)
	fakeClient.ExpectNotFoundGet("/some/key")
	fakeClient.WatchImmediateError = fmt.Errorf("immediate error")
	h := newEtcdHelper(fakeClient, codec, etcdtest.PathPrefix())

	watching, err := h.Watch("/some/key", 4, storage.Everything)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	defer watching.Stop()

	got := <-watching.ResultChan()
	if got.Type != watch.Error {
		t.Fatalf("Unexpected non-error")
	}
	status, ok := got.Object.(*api.Status)
	if !ok {
		t.Fatalf("Unexpected non-error object type")
	}
	if status.Message != "immediate error" {
		t.Errorf("Unexpected wrong error")
	}
	if status.Status != api.StatusFailure {
		t.Errorf("Unexpected wrong error status")
	}
}

func TestWatch(t *testing.T) {
	codec := latest.Codec
	fakeClient := tools.NewFakeEtcdClient(t)
	key := "/some/key"
	prefixedKey := etcdtest.AddPrefix(key)
	fakeClient.ExpectNotFoundGet(prefixedKey)
	h := newEtcdHelper(fakeClient, codec, etcdtest.PathPrefix())

	watching, err := h.Watch(key, 0, storage.Everything)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	fakeClient.WaitForWatchCompletion()
	// when server returns not found, the watch index starts at the next value (1)
	if fakeClient.WatchIndex != 1 {
		t.Errorf("Expected client to be at index %d, got %#v", 1, fakeClient)
	}

	// Test normal case
	pod := &api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo"}}
	podBytes, _ := codec.Encode(pod)
	fakeClient.WatchResponse <- &etcd.Response{
		Action: "set",
		Node: &etcd.Node{
			Value: string(podBytes),
		},
	}

	event := <-watching.ResultChan()
	if e, a := watch.Added, event.Type; e != a {
		t.Errorf("Expected %v, got %v", e, a)
	}
	if e, a := pod, event.Object; !api.Semantic.DeepDerivative(e, a) {
		t.Errorf("Expected %v, got %v", e, a)
	}

	// Test error case
	fakeClient.WatchInjectError <- fmt.Errorf("Injected error")

	if errEvent, ok := <-watching.ResultChan(); !ok {
		t.Errorf("no error result?")
	} else {
		if e, a := watch.Error, errEvent.Type; e != a {
			t.Errorf("Expected %v, got %v", e, a)
		}
		if e, a := "Injected error", errEvent.Object.(*api.Status).Message; e != a {
			t.Errorf("Expected %v, got %v", e, a)
		}
	}

	// Did everything shut down?
	if _, open := <-fakeClient.WatchResponse; open {
		t.Errorf("An injected error did not cause a graceful shutdown")
	}
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
	codec := latest.Codec
	baseKey := "/somekey/foo"
	prefixedKey := etcdtest.AddPrefix(baseKey)
	type T struct {
		Type      watch.EventType
		Endpoints []api.EndpointSubset
	}
	testCases := map[string]struct {
		Initial   map[string]tools.EtcdResponseWithError
		Responses []*etcd.Response
		From      uint64
		Expected  []*T
	}{
		"from not found": {
			Initial: map[string]tools.EtcdResponseWithError{},
			Responses: []*etcd.Response{
				{
					Action: "create",
					Node: &etcd.Node{
						Value: string(runtime.EncodeOrDie(codec, &api.Endpoints{
							ObjectMeta: api.ObjectMeta{Name: "foo"},
							Subsets:    emptySubsets(),
						})),
					},
				},
			},
			From: 1,
			Expected: []*T{
				{watch.Added, nil},
			},
		},
		"from version 1": {
			Responses: []*etcd.Response{
				{
					Action: "compareAndSwap",
					Node: &etcd.Node{
						Value: string(runtime.EncodeOrDie(codec, &api.Endpoints{
							ObjectMeta: api.ObjectMeta{Name: "foo"},
							Subsets:    makeSubsets("127.0.0.1", 9000),
						})),
						CreatedIndex:  1,
						ModifiedIndex: 2,
					},
					PrevNode: &etcd.Node{
						Value: string(runtime.EncodeOrDie(codec, &api.Endpoints{
							ObjectMeta: api.ObjectMeta{Name: "foo"},
							Subsets:    emptySubsets(),
						})),
						CreatedIndex:  1,
						ModifiedIndex: 1,
					},
				},
			},
			From: 1,
			Expected: []*T{
				{watch.Modified, makeSubsets("127.0.0.1", 9000)},
			},
		},
		"from initial state": {
			Initial: map[string]tools.EtcdResponseWithError{
				prefixedKey: {
					R: &etcd.Response{
						Action: "get",
						Node: &etcd.Node{
							Value: string(runtime.EncodeOrDie(codec, &api.Endpoints{
								ObjectMeta: api.ObjectMeta{Name: "foo"},
								Subsets:    emptySubsets(),
							})),
							CreatedIndex:  1,
							ModifiedIndex: 1,
						},
						EtcdIndex: 1,
					},
				},
			},
			Responses: []*etcd.Response{
				nil,
				{
					Action: "compareAndSwap",
					Node: &etcd.Node{
						Value: string(runtime.EncodeOrDie(codec, &api.Endpoints{
							ObjectMeta: api.ObjectMeta{Name: "foo"},
							Subsets:    makeSubsets("127.0.0.1", 9000),
						})),
						CreatedIndex:  1,
						ModifiedIndex: 2,
					},
					PrevNode: &etcd.Node{
						Value: string(runtime.EncodeOrDie(codec, &api.Endpoints{
							ObjectMeta: api.ObjectMeta{Name: "foo"},
							Subsets:    emptySubsets(),
						})),
						CreatedIndex:  1,
						ModifiedIndex: 1,
					},
				},
			},
			Expected: []*T{
				{watch.Added, nil},
				{watch.Modified, makeSubsets("127.0.0.1", 9000)},
			},
		},
	}

	for k, testCase := range testCases {
		fakeClient := tools.NewFakeEtcdClient(t)
		for key, value := range testCase.Initial {
			fakeClient.Data[key] = value
		}

		h := newEtcdHelper(fakeClient, codec, etcdtest.PathPrefix())
		watching, err := h.Watch(baseKey, testCase.From, storage.Everything)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}

		fakeClient.WaitForWatchCompletion()

		t.Logf("Testing %v", k)
		for i := range testCase.Responses {
			if testCase.Responses[i] != nil {
				fakeClient.WatchResponse <- testCase.Responses[i]
			}
			event := <-watching.ResultChan()
			if e, a := testCase.Expected[i].Type, event.Type; e != a {
				t.Errorf("%s: expected type %v, got %v", k, e, a)
				break
			}
			if e, a := testCase.Expected[i].Endpoints, event.Object.(*api.Endpoints).Subsets; !api.Semantic.DeepDerivative(e, a) {
				t.Errorf("%s: expected type %v, got %v", k, e, a)
				break
			}
		}
		watching.Stop()
	}
}

func TestWatchFromZeroIndex(t *testing.T) {
	codec := latest.Codec
	pod := &api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo"}}

	testCases := map[string]struct {
		Response        tools.EtcdResponseWithError
		ExpectedVersion string
		ExpectedType    watch.EventType
	}{
		"get value created": {
			tools.EtcdResponseWithError{
				R: &etcd.Response{
					Node: &etcd.Node{
						Value:         runtime.EncodeOrDie(codec, pod),
						CreatedIndex:  1,
						ModifiedIndex: 1,
					},
					Action:    "get",
					EtcdIndex: 2,
				},
			},
			"1",
			watch.Added,
		},
		"get value modified": {
			tools.EtcdResponseWithError{
				R: &etcd.Response{
					Node: &etcd.Node{
						Value:         runtime.EncodeOrDie(codec, pod),
						CreatedIndex:  1,
						ModifiedIndex: 2,
					},
					Action:    "get",
					EtcdIndex: 3,
				},
			},
			"2",
			watch.Modified,
		},
	}

	for k, testCase := range testCases {
		fakeClient := tools.NewFakeEtcdClient(t)
		key := "/some/key"
		prefixedKey := etcdtest.AddPrefix(key)
		fakeClient.Data[prefixedKey] = testCase.Response
		h := newEtcdHelper(fakeClient, codec, etcdtest.PathPrefix())

		watching, err := h.Watch(key, 0, storage.Everything)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}

		fakeClient.WaitForWatchCompletion()
		if e, a := testCase.Response.R.EtcdIndex+1, fakeClient.WatchIndex; e != a {
			t.Errorf("%s: expected watch index to be %d, got %d", k, e, a)
		}

		// the existing node is detected and the index set
		event := <-watching.ResultChan()
		if e, a := testCase.ExpectedType, event.Type; e != a {
			t.Errorf("%s: expected %v, got %v", k, e, a)
		}
		actualPod, ok := event.Object.(*api.Pod)
		if !ok {
			t.Fatalf("%s: expected a pod, got %#v", k, event.Object)
		}
		if actualPod.ResourceVersion != testCase.ExpectedVersion {
			t.Errorf("%s: expected pod with resource version %v, Got %#v", k, testCase.ExpectedVersion, actualPod)
		}
		pod.ResourceVersion = testCase.ExpectedVersion
		if e, a := pod, event.Object; !api.Semantic.DeepDerivative(e, a) {
			t.Errorf("%s: expected %v, got %v", k, e, a)
		}
		watching.Stop()
	}
}

func TestWatchListFromZeroIndex(t *testing.T) {
	codec := latest.Codec
	pod := &api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo"}}
	key := "/some/key"
	prefixedKey := etcdtest.AddPrefix(key)
	fakeClient := tools.NewFakeEtcdClient(t)
	fakeClient.Data[prefixedKey] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Dir: true,
				Nodes: etcd.Nodes{
					&etcd.Node{
						Value:         runtime.EncodeOrDie(codec, pod),
						CreatedIndex:  1,
						ModifiedIndex: 1,
						Nodes:         etcd.Nodes{},
					},
					&etcd.Node{
						Value:         runtime.EncodeOrDie(codec, pod),
						CreatedIndex:  2,
						ModifiedIndex: 2,
						Nodes:         etcd.Nodes{},
					},
				},
			},
			Action:    "get",
			EtcdIndex: 3,
		},
	}
	h := newEtcdHelper(fakeClient, codec, etcdtest.PathPrefix())

	watching, err := h.WatchList(key, 0, storage.Everything)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	// the existing node is detected and the index set
	event, open := <-watching.ResultChan()
	if !open {
		t.Fatalf("unexpected channel close")
	}
	for i := 0; i < 2; i++ {
		if e, a := watch.Added, event.Type; e != a {
			t.Errorf("Expected %v, got %v", e, a)
		}
		actualPod, ok := event.Object.(*api.Pod)
		if !ok {
			t.Fatalf("expected a pod, got %#v", event.Object)
		}
		if actualPod.ResourceVersion != "1" {
			t.Errorf("Expected pod with resource version %d, Got %#v", 1, actualPod)
		}
		pod.ResourceVersion = "1"
		if e, a := pod, event.Object; !api.Semantic.DeepDerivative(e, a) {
			t.Errorf("Expected %v, got %v", e, a)
		}
	}

	fakeClient.WaitForWatchCompletion()
	watching.Stop()
}

func TestWatchListIgnoresRootKey(t *testing.T) {
	codec := latest.Codec
	pod := &api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo"}}
	key := "/some/key"
	prefixedKey := etcdtest.AddPrefix(key)

	fakeClient := tools.NewFakeEtcdClient(t)
	h := newEtcdHelper(fakeClient, codec, etcdtest.PathPrefix())

	watching, err := h.WatchList(key, 1, storage.Everything)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	fakeClient.WaitForWatchCompletion()

	// This is the root directory of the watch, which happens to have a value encoded
	fakeClient.WatchResponse <- &etcd.Response{
		Action: "delete",
		PrevNode: &etcd.Node{
			Key:           prefixedKey,
			Value:         runtime.EncodeOrDie(codec, pod),
			CreatedIndex:  1,
			ModifiedIndex: 1,
		},
	}
	// Delete of the parent directory of a key is an event that a list watch would receive,
	// but will have no value so the decode will fail.
	fakeClient.WatchResponse <- &etcd.Response{
		Action: "delete",
		PrevNode: &etcd.Node{
			Key:           prefixedKey,
			Value:         "",
			CreatedIndex:  1,
			ModifiedIndex: 1,
		},
	}
	close(fakeClient.WatchStop)

	// the existing node is detected and the index set
	_, open := <-watching.ResultChan()
	if open {
		t.Fatalf("unexpected channel open")
	}

	watching.Stop()
}

func TestWatchFromNotFound(t *testing.T) {
	fakeClient := tools.NewFakeEtcdClient(t)
	key := "/some/key"
	prefixedKey := etcdtest.AddPrefix(key)
	fakeClient.Data[prefixedKey] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: nil,
		},
		E: &etcd.EtcdError{
			Index:     2,
			ErrorCode: 100,
		},
	}
	h := newEtcdHelper(fakeClient, codec, etcdtest.PathPrefix())

	watching, err := h.Watch(key, 0, storage.Everything)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	fakeClient.WaitForWatchCompletion()
	if fakeClient.WatchIndex != 3 {
		t.Errorf("Expected client to wait for %d, got %#v", 3, fakeClient)
	}

	watching.Stop()
}

func TestWatchFromOtherError(t *testing.T) {
	fakeClient := tools.NewFakeEtcdClient(t)
	key := "/some/key"
	prefixedKey := etcdtest.AddPrefix(key)
	fakeClient.Data[prefixedKey] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: nil,
		},
		E: &etcd.EtcdError{
			Index:     2,
			ErrorCode: 101,
		},
	}
	h := newEtcdHelper(fakeClient, codec, etcdtest.PathPrefix())

	watching, err := h.Watch(key, 0, storage.Everything)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	errEvent := <-watching.ResultChan()
	if e, a := watch.Error, errEvent.Type; e != a {
		t.Errorf("Expected %v, got %v", e, a)
	}
	if e, a := "101:  () [2]", errEvent.Object.(*api.Status).Message; e != a {
		t.Errorf("Expected %v, got %v", e, a)
	}

	select {
	case _, ok := <-watching.ResultChan():
		if ok {
			t.Fatalf("expected result channel to be closed")
		}
	case <-time.After(1 * time.Second):
		t.Fatalf("watch should have closed channel: %#v", watching)
	}

	if fakeClient.WatchResponse != nil || fakeClient.WatchIndex != 0 {
		t.Fatalf("Watch should not have been invoked: %#v", fakeClient)
	}
}

func TestWatchPurposefulShutdown(t *testing.T) {
	fakeClient := tools.NewFakeEtcdClient(t)

	h := newEtcdHelper(fakeClient, codec, etcdtest.PathPrefix())
	key := "/some/key"
	prefixedKey := etcdtest.AddPrefix(key)
	fakeClient.ExpectNotFoundGet(prefixedKey)

	// Test purposeful shutdown
	watching, err := h.Watch(key, 0, storage.Everything)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	fakeClient.WaitForWatchCompletion()
	watching.Stop()

	// Did everything shut down?
	if _, open := <-fakeClient.WatchResponse; open {
		t.Errorf("A stop did not cause a graceful shutdown")
	}
	if _, open := <-watching.ResultChan(); open {
		t.Errorf("An injected error did not cause a graceful shutdown")
	}
}
