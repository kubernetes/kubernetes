/*
Copyright 2014 Google Inc. All rights reserved.

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

package tools

import (
	"fmt"
	"reflect"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
	"github.com/coreos/go-etcd/etcd"
)

func TestWatchInterpretations(t *testing.T) {
	// Declare some pods to make the test cases compact.
	podFoo := &api.Pod{JSONBase: api.JSONBase{ID: "foo"}}
	podBar := &api.Pod{JSONBase: api.JSONBase{ID: "bar"}}
	podBaz := &api.Pod{JSONBase: api.JSONBase{ID: "baz"}}
	firstLetterIsB := func(obj interface{}) bool {
		return obj.(*api.Pod).ID[0] == 'b'
	}

	// All of these test cases will be run with the firstLetterIsB FilterFunc.
	table := map[string]struct {
		actions       []string // Run this test item for every action here.
		prevNodeValue string
		nodeValue     string
		expectEmit    bool
		expectType    watch.EventType
		expectObject  interface{}
	}{
		"create": {
			actions:      []string{"create", "get"},
			nodeValue:    api.EncodeOrDie(podBar),
			expectEmit:   true,
			expectType:   watch.Added,
			expectObject: podBar,
		},
		"create but filter blocks": {
			actions:    []string{"create", "get"},
			nodeValue:  api.EncodeOrDie(podFoo),
			expectEmit: false,
		},
		"delete": {
			actions:       []string{"delete"},
			prevNodeValue: api.EncodeOrDie(podBar),
			expectEmit:    true,
			expectType:    watch.Deleted,
			expectObject:  podBar,
		},
		"delete but filter blocks": {
			actions:    []string{"delete"},
			nodeValue:  api.EncodeOrDie(podFoo),
			expectEmit: false,
		},
		"modify appears to create 1": {
			actions:      []string{"set", "compareAndSwap"},
			nodeValue:    api.EncodeOrDie(podBar),
			expectEmit:   true,
			expectType:   watch.Added,
			expectObject: podBar,
		},
		"modify appears to create 2": {
			actions:       []string{"set", "compareAndSwap"},
			prevNodeValue: api.EncodeOrDie(podFoo),
			nodeValue:     api.EncodeOrDie(podBar),
			expectEmit:    true,
			expectType:    watch.Added,
			expectObject:  podBar,
		},
		"modify appears to delete": {
			actions:       []string{"set", "compareAndSwap"},
			prevNodeValue: api.EncodeOrDie(podBar),
			nodeValue:     api.EncodeOrDie(podFoo),
			expectEmit:    true,
			expectType:    watch.Deleted,
			expectObject:  podBar, // Should return last state that passed the filter!
		},
		"modify modifies": {
			actions:       []string{"set", "compareAndSwap"},
			prevNodeValue: api.EncodeOrDie(podBar),
			nodeValue:     api.EncodeOrDie(podBaz),
			expectEmit:    true,
			expectType:    watch.Modified,
			expectObject:  podBaz,
		},
		"modify ignores": {
			actions:    []string{"set", "compareAndSwap"},
			nodeValue:  api.EncodeOrDie(podFoo),
			expectEmit: false,
		},
	}

	for name, item := range table {
		for _, action := range item.actions {
			w := newEtcdWatcher(true, firstLetterIsB, codec, versioner, nil)
			emitCalled := false
			w.emit = func(event watch.Event) {
				emitCalled = true
				if !item.expectEmit {
					return
				}
				if e, a := item.expectType, event.Type; e != a {
					t.Errorf("'%v - %v': expected %v, got %v", name, action, e, a)
				}
				if e, a := item.expectObject, event.Object; !reflect.DeepEqual(e, a) {
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
	w := newEtcdWatcher(false, Everything, codec, versioner, nil)
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
		w := newEtcdWatcher(false, Everything, codec, versioner, nil)
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
		w := newEtcdWatcher(false, Everything, codec, versioner, nil)
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

func TestWatch(t *testing.T) {
	fakeClient := NewFakeEtcdClient(t)
	fakeClient.expectNotFoundGetSet["/some/key"] = struct{}{}
	h := EtcdHelper{fakeClient, codec, versioner}

	watching, err := h.Watch("/some/key", 0)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	fakeClient.WaitForWatchCompletion()
	// when server returns not found, the watch index starts at the next value (1)
	if fakeClient.WatchIndex != 1 {
		t.Errorf("Expected client to be at index %d, got %#v", 1, fakeClient)
	}

	// Test normal case
	pod := &api.Pod{JSONBase: api.JSONBase{ID: "foo"}}
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
	if e, a := pod, event.Object; !reflect.DeepEqual(e, a) {
		t.Errorf("Expected %v, got %v", e, a)
	}

	// Test error case
	fakeClient.WatchInjectError <- fmt.Errorf("Injected error")

	// Did everything shut down?
	if _, open := <-fakeClient.WatchResponse; open {
		t.Errorf("An injected error did not cause a graceful shutdown")
	}
	if _, open := <-watching.ResultChan(); open {
		t.Errorf("An injected error did not cause a graceful shutdown")
	}
}

func TestWatchEtcdState(t *testing.T) {
	type T struct {
		Type      watch.EventType
		Endpoints []string
	}
	testCases := map[string]struct {
		Initial   map[string]EtcdResponseWithError
		Responses []*etcd.Response
		From      uint64
		Expected  []*T
	}{
		"from not found": {
			Initial: map[string]EtcdResponseWithError{},
			Responses: []*etcd.Response{
				{
					Action: "create",
					Node: &etcd.Node{
						Value: string(api.EncodeOrDie(&api.Endpoints{JSONBase: api.JSONBase{ID: "foo"}, Endpoints: []string{}})),
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
						Value:         string(api.EncodeOrDie(&api.Endpoints{JSONBase: api.JSONBase{ID: "foo"}, Endpoints: []string{"127.0.0.1:9000"}})),
						CreatedIndex:  1,
						ModifiedIndex: 2,
					},
					PrevNode: &etcd.Node{
						Value:         string(api.EncodeOrDie(&api.Endpoints{JSONBase: api.JSONBase{ID: "foo"}, Endpoints: []string{}})),
						CreatedIndex:  1,
						ModifiedIndex: 1,
					},
				},
			},
			From: 1,
			Expected: []*T{
				{watch.Modified, []string{"127.0.0.1:9000"}},
			},
		},
		"from initial state": {
			Initial: map[string]EtcdResponseWithError{
				"/somekey/foo": {
					R: &etcd.Response{
						Action: "get",
						Node: &etcd.Node{
							Value:         string(api.EncodeOrDie(&api.Endpoints{JSONBase: api.JSONBase{ID: "foo"}, Endpoints: []string{}})),
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
						Value:         string(api.EncodeOrDie(&api.Endpoints{JSONBase: api.JSONBase{ID: "foo"}, Endpoints: []string{"127.0.0.1:9000"}})),
						CreatedIndex:  1,
						ModifiedIndex: 2,
					},
					PrevNode: &etcd.Node{
						Value:         string(api.EncodeOrDie(&api.Endpoints{JSONBase: api.JSONBase{ID: "foo"}, Endpoints: []string{}})),
						CreatedIndex:  1,
						ModifiedIndex: 1,
					},
				},
			},
			Expected: []*T{
				{watch.Added, nil},
				{watch.Modified, []string{"127.0.0.1:9000"}},
			},
		},
	}

	for k, testCase := range testCases {
		fakeClient := NewFakeEtcdClient(t)
		for key, value := range testCase.Initial {
			fakeClient.Data[key] = value
		}
		h := EtcdHelper{fakeClient, codec, versioner}
		watching, err := h.Watch("/somekey/foo", testCase.From)
		if err != nil {
			t.Errorf("%s: unexpected error: %v", k, err)
			continue
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
			if e, a := testCase.Expected[i].Endpoints, event.Object.(*api.Endpoints).Endpoints; !reflect.DeepEqual(e, a) {
				t.Errorf("%s: expected type %v, got %v", k, e, a)
				break
			}
		}
		watching.Stop()
	}
}

func TestWatchFromZeroIndex(t *testing.T) {
	pod := &api.Pod{JSONBase: api.JSONBase{ID: "foo"}}

	testCases := map[string]struct {
		Response        EtcdResponseWithError
		ExpectedVersion uint64
		ExpectedType    watch.EventType
	}{
		"get value created": {
			EtcdResponseWithError{
				R: &etcd.Response{
					Node: &etcd.Node{
						Value:         api.EncodeOrDie(pod),
						CreatedIndex:  1,
						ModifiedIndex: 1,
					},
					Action:    "get",
					EtcdIndex: 2,
				},
			},
			1,
			watch.Added,
		},
		"get value modified": {
			EtcdResponseWithError{
				R: &etcd.Response{
					Node: &etcd.Node{
						Value:         api.EncodeOrDie(pod),
						CreatedIndex:  1,
						ModifiedIndex: 2,
					},
					Action:    "get",
					EtcdIndex: 3,
				},
			},
			2,
			watch.Modified,
		},
	}

	for k, testCase := range testCases {
		fakeClient := NewFakeEtcdClient(t)
		fakeClient.Data["/some/key"] = testCase.Response
		h := EtcdHelper{fakeClient, codec, versioner}

		watching, err := h.Watch("/some/key", 0)
		if err != nil {
			t.Fatalf("%s: unexpected error: %v", k, err)
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
			t.Errorf("%s: expected pod with resource version %d, Got %#v", k, testCase.ExpectedVersion, actualPod)
		}
		pod.ResourceVersion = testCase.ExpectedVersion
		if e, a := pod, event.Object; !reflect.DeepEqual(e, a) {
			t.Errorf("%s: expected %v, got %v", k, e, a)
		}
		watching.Stop()
	}
}

func TestWatchListFromZeroIndex(t *testing.T) {
	pod := &api.Pod{JSONBase: api.JSONBase{ID: "foo"}}

	fakeClient := NewFakeEtcdClient(t)
	fakeClient.Data["/some/key"] = EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Dir: true,
				Nodes: etcd.Nodes{
					&etcd.Node{
						Value:         api.EncodeOrDie(pod),
						CreatedIndex:  1,
						ModifiedIndex: 1,
						Nodes:         etcd.Nodes{},
					},
					&etcd.Node{
						Value:         api.EncodeOrDie(pod),
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
	h := EtcdHelper{fakeClient, codec, versioner}

	watching, err := h.WatchList("/some/key", 0, Everything)
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
		if actualPod.ResourceVersion != 1 {
			t.Errorf("Expected pod with resource version %d, Got %#v", 1, actualPod)
		}
		pod.ResourceVersion = 1
		if e, a := pod, event.Object; !reflect.DeepEqual(e, a) {
			t.Errorf("Expected %v, got %v", e, a)
		}
	}

	fakeClient.WaitForWatchCompletion()
	watching.Stop()
}

func TestWatchFromNotFound(t *testing.T) {
	fakeClient := NewFakeEtcdClient(t)
	fakeClient.Data["/some/key"] = EtcdResponseWithError{
		R: &etcd.Response{
			Node: nil,
		},
		E: &etcd.EtcdError{
			Index:     2,
			ErrorCode: 100,
		},
	}
	h := EtcdHelper{fakeClient, codec, versioner}

	watching, err := h.Watch("/some/key", 0)
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
	fakeClient := NewFakeEtcdClient(t)
	fakeClient.Data["/some/key"] = EtcdResponseWithError{
		R: &etcd.Response{
			Node: nil,
		},
		E: &etcd.EtcdError{
			Index:     2,
			ErrorCode: 101,
		},
	}
	h := EtcdHelper{fakeClient, codec, versioner}

	watching, err := h.Watch("/some/key", 0)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
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
	fakeClient := NewFakeEtcdClient(t)
	h := EtcdHelper{fakeClient, codec, versioner}
	fakeClient.expectNotFoundGetSet["/some/key"] = struct{}{}

	// Test purposeful shutdown
	watching, err := h.Watch("/some/key", 0)
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
