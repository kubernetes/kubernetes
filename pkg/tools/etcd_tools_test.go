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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
	"github.com/coreos/go-etcd/etcd"
)

type fakeEtcdGetSet struct {
	get func(key string, sort, recursive bool) (*etcd.Response, error)
	set func(key, value string, ttl uint64) (*etcd.Response, error)
}

func TestIsNotFoundErr(t *testing.T) {
	try := func(err error, isNotFound bool) {
		if IsEtcdNotFound(err) != isNotFound {
			t.Errorf("Expected %#v to return %v, but it did not", err, isNotFound)
		}
	}
	try(EtcdErrorNotFound, true)
	try(&etcd.EtcdError{ErrorCode: 101}, false)
	try(nil, false)
	try(fmt.Errorf("some other kind of error"), false)
}

type testMarshalType struct {
	ID string `json:"id"`
}

func TestExtractList(t *testing.T) {
	fakeClient := MakeFakeEtcdClient(t)
	fakeClient.Data["/some/key"] = EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Nodes: []*etcd.Node{
					{
						Value: `{"id":"foo"}`,
					},
					{
						Value: `{"id":"bar"}`,
					},
					{
						Value: `{"id":"baz"}`,
					},
				},
			},
		},
	}
	expect := []testMarshalType{
		{"foo"},
		{"bar"},
		{"baz"},
	}
	var got []testMarshalType
	helper := EtcdHelper{fakeClient}
	err := helper.ExtractList("/some/key", &got)
	if err != nil {
		t.Errorf("Unexpected error %#v", err)
	}
	if !reflect.DeepEqual(got, expect) {
		t.Errorf("Wanted %#v, got %#v", expect, got)
	}
}

func TestExtractObj(t *testing.T) {
	fakeClient := MakeFakeEtcdClient(t)
	expect := testMarshalType{ID: "foo"}
	fakeClient.Set("/some/key", util.MakeJSONString(expect), 0)
	helper := EtcdHelper{fakeClient}
	var got testMarshalType
	err := helper.ExtractObj("/some/key", &got, false)
	if err != nil {
		t.Errorf("Unexpected error %#v", err)
	}
	if !reflect.DeepEqual(got, expect) {
		t.Errorf("Wanted %#v, got %#v", expect, got)
	}
}

func TestExtractObjNotFoundErr(t *testing.T) {
	fakeClient := MakeFakeEtcdClient(t)
	fakeClient.Data["/some/key"] = EtcdResponseWithError{
		R: &etcd.Response{
			Node: nil,
		},
		E: &etcd.EtcdError{
			ErrorCode: 100,
		},
	}
	fakeClient.Data["/some/key2"] = EtcdResponseWithError{
		R: &etcd.Response{
			Node: nil,
		},
	}
	fakeClient.Data["/some/key3"] = EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Value: "",
			},
		},
	}
	helper := EtcdHelper{fakeClient}
	try := func(key string) {
		var got testMarshalType
		err := helper.ExtractObj(key, &got, false)
		if err == nil {
			t.Errorf("%s: wanted error but didn't get one", key)
		}
		err = helper.ExtractObj(key, &got, true)
		if err != nil {
			t.Errorf("%s: didn't want error but got %#v", key, err)
		}
	}

	try("/some/key")
	try("/some/key2")
	try("/some/key3")
}

func TestSetObj(t *testing.T) {
	obj := testMarshalType{ID: "foo"}
	fakeClient := MakeFakeEtcdClient(t)
	helper := EtcdHelper{fakeClient}
	err := helper.SetObj("/some/key", obj)
	if err != nil {
		t.Errorf("Unexpected error %#v", err)
	}
	expect := util.MakeJSONString(obj)
	got := fakeClient.Data["/some/key"].R.Node.Value
	if expect != got {
		t.Errorf("Wanted %v, got %v", expect, got)
	}
}

func TestWatchInterpretation_ListAdd(t *testing.T) {
	called := false
	w := newEtcdWatcher(true, func(interface{}) bool {
		called = true
		return true
	})
	pod := &api.Pod{JSONBase: api.JSONBase{ID: "foo"}}
	podBytes, _ := api.Encode(pod)

	go w.sendResult(&etcd.Response{
		Action: "set",
		Node: &etcd.Node{
			Nodes: etcd.Nodes{
				{
					Value: string(podBytes),
				},
			},
		},
	})

	got := <-w.outgoing
	if e, a := watch.Added, got.Type; e != a {
		t.Errorf("Expected %v, got %v", e, a)
	}
	if e, a := pod, got.Object; !reflect.DeepEqual(e, a) {
		t.Errorf("Expected %v, got %v", e, a)
	}
	if !called {
		t.Errorf("filter never called")
	}
}

func TestWatchInterpretation_ListDelete(t *testing.T) {
	called := false
	w := newEtcdWatcher(true, func(interface{}) bool {
		called = true
		return true
	})
	pod := &api.Pod{JSONBase: api.JSONBase{ID: "foo"}}
	podBytes, _ := api.Encode(pod)

	go w.sendResult(&etcd.Response{
		Action: "delete",
		PrevNode: &etcd.Node{
			Nodes: etcd.Nodes{
				{
					Value: string(podBytes),
				},
			},
		},
	})

	got := <-w.outgoing
	if e, a := watch.Deleted, got.Type; e != a {
		t.Errorf("Expected %v, got %v", e, a)
	}
	if e, a := pod, got.Object; !reflect.DeepEqual(e, a) {
		t.Errorf("Expected %v, got %v", e, a)
	}
	if !called {
		t.Errorf("filter never called")
	}
}

func TestWatchInterpretation_SingleAdd(t *testing.T) {
	w := newEtcdWatcher(false, func(interface{}) bool {
		t.Errorf("unexpected filter call")
		return true
	})
	pod := &api.Pod{JSONBase: api.JSONBase{ID: "foo"}}
	podBytes, _ := api.Encode(pod)

	go w.sendResult(&etcd.Response{
		Action: "set",
		Node: &etcd.Node{
			Value: string(podBytes),
		},
	})

	got := <-w.outgoing
	if e, a := watch.Added, got.Type; e != a {
		t.Errorf("Expected %v, got %v", e, a)
	}
	if e, a := pod, got.Object; !reflect.DeepEqual(e, a) {
		t.Errorf("Expected %v, got %v", e, a)
	}
}

func TestWatchInterpretation_SingleDelete(t *testing.T) {
	w := newEtcdWatcher(false, func(interface{}) bool {
		t.Errorf("unexpected filter call")
		return true
	})
	pod := &api.Pod{JSONBase: api.JSONBase{ID: "foo"}}
	podBytes, _ := api.Encode(pod)

	go w.sendResult(&etcd.Response{
		Action: "delete",
		PrevNode: &etcd.Node{
			Value: string(podBytes),
		},
	})

	got := <-w.outgoing
	if e, a := watch.Deleted, got.Type; e != a {
		t.Errorf("Expected %v, got %v", e, a)
	}
	if e, a := pod, got.Object; !reflect.DeepEqual(e, a) {
		t.Errorf("Expected %v, got %v", e, a)
	}
}

func TestWatch(t *testing.T) {
	fakeEtcd := MakeFakeEtcdClient(t)
	h := EtcdHelper{fakeEtcd}

	watching, err := h.Watch("/some/key")
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	fakeEtcd.WaitForWatchCompletion()

	// Test normal case
	pod := &api.Pod{JSONBase: api.JSONBase{ID: "foo"}}
	podBytes, _ := api.Encode(pod)
	fakeEtcd.WatchResponse <- &etcd.Response{
		Action: "set",
		Node: &etcd.Node{
			Value: string(podBytes),
		},
	}

	select {
	case event := <-watching.ResultChan():
		if e, a := watch.Added, event.Type; e != a {
			t.Errorf("Expected %v, got %v", e, a)
		}
		if e, a := pod, event.Object; !reflect.DeepEqual(e, a) {
			t.Errorf("Expected %v, got %v", e, a)
		}
	case <-time.After(10 * time.Millisecond):
		t.Errorf("Expected 1 call but got 0")
	}

	// Test error case
	fakeEtcd.WatchInjectError <- fmt.Errorf("Injected error")

	// Did everything shut down?
	if _, open := <-fakeEtcd.WatchResponse; open {
		t.Errorf("An injected error did not cause a graceful shutdown")
	}
	if _, open := <-watching.ResultChan(); open {
		t.Errorf("An injected error did not cause a graceful shutdown")
	}
}

func TestWatchPurposefulShutdown(t *testing.T) {
	fakeEtcd := MakeFakeEtcdClient(t)
	h := EtcdHelper{fakeEtcd}

	// Test purposeful shutdown
	watching, err := h.Watch("/some/key")
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	fakeEtcd.WaitForWatchCompletion()
	watching.Stop()

	// Did everything shut down?
	if _, open := <-fakeEtcd.WatchResponse; open {
		t.Errorf("A stop did not cause a graceful shutdown")
	}
	if _, open := <-watching.ResultChan(); open {
		t.Errorf("An injected error did not cause a graceful shutdown")
	}
}
