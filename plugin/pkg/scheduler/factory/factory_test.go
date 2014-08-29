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

package factory

import (
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/cache"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

func TestCreate(t *testing.T) {
	handler := util.FakeHandler{
		StatusCode:   500,
		ResponseBody: "",
		T:            t,
	}
	server := httptest.NewServer(&handler)
	client := client.NewOrDie(server.URL, nil)
	factory := ConfigFactory{client}
	factory.Create()
}

func TestCreateWatches(t *testing.T) {
	factory := ConfigFactory{nil}
	table := []struct {
		rv           uint64
		location     string
		watchFactory func(rv uint64) (watch.Interface, error)
	}{
		// Minion watch
		{
			rv:           0,
			location:     "/api/v1beta1/watch/minions?resourceVersion=0",
			watchFactory: factory.createMinionWatch,
		}, {
			rv:           42,
			location:     "/api/v1beta1/watch/minions?resourceVersion=42",
			watchFactory: factory.createMinionWatch,
		},
		// Assigned pod watches
		{
			rv:           0,
			location:     "/api/v1beta1/watch/pods?fields=DesiredState.Host!%3D&resourceVersion=0",
			watchFactory: factory.createAssignedPodWatch,
		}, {
			rv:           42,
			location:     "/api/v1beta1/watch/pods?fields=DesiredState.Host!%3D&resourceVersion=42",
			watchFactory: factory.createAssignedPodWatch,
		},
		// Unassigned pod watches
		{
			rv:           0,
			location:     "/api/v1beta1/watch/pods?fields=DesiredState.Host%3D&resourceVersion=0",
			watchFactory: factory.createUnassignedPodWatch,
		}, {
			rv:           42,
			location:     "/api/v1beta1/watch/pods?fields=DesiredState.Host%3D&resourceVersion=42",
			watchFactory: factory.createUnassignedPodWatch,
		},
	}

	for _, item := range table {
		handler := util.FakeHandler{
			StatusCode:   500,
			ResponseBody: "",
			T:            t,
		}
		server := httptest.NewServer(&handler)
		factory.Client = client.NewOrDie(server.URL, nil)
		// This test merely tests that the correct request is made.
		item.watchFactory(item.rv)
		handler.ValidateRequest(t, item.location, "GET", nil)
	}
}

func TestPollMinions(t *testing.T) {
	table := []struct {
		minions []api.Minion
	}{
		{
			minions: []api.Minion{
				{JSONBase: api.JSONBase{ID: "foo"}},
				{JSONBase: api.JSONBase{ID: "bar"}},
			},
		},
	}

	for _, item := range table {
		ml := &api.MinionList{Items: item.minions}
		handler := util.FakeHandler{
			StatusCode:   200,
			ResponseBody: api.EncodeOrDie(ml),
			T:            t,
		}
		mux := http.NewServeMux()
		// FakeHandler musn't be sent requests other than the one you want to test.
		mux.Handle("/api/v1beta1/minions", &handler)
		server := httptest.NewServer(mux)
		client := client.NewOrDie(server.URL, nil)
		cf := ConfigFactory{client}

		ce, err := cf.pollMinions()
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
			continue
		}
		handler.ValidateRequest(t, "/api/v1beta1/minions", "GET", nil)

		if e, a := len(item.minions), ce.Len(); e != a {
			t.Errorf("Expected %v, got %v", e, a)
		}
	}
}

func TestDefaultErrorFunc(t *testing.T) {
	testPod := &api.Pod{JSONBase: api.JSONBase{ID: "foo"}}
	handler := util.FakeHandler{
		StatusCode:   200,
		ResponseBody: api.EncodeOrDie(testPod),
		T:            t,
	}
	mux := http.NewServeMux()
	// FakeHandler musn't be sent requests other than the one you want to test.
	mux.Handle("/api/v1beta1/pods/foo", &handler)
	server := httptest.NewServer(mux)
	factory := ConfigFactory{client.NewOrDie(server.URL, nil)}
	queue := cache.NewFIFO()
	errFunc := factory.makeDefaultErrorFunc(queue)

	errFunc(testPod, nil)
	for {
		// This is a terrible way to do this but I plan on replacing this
		// whole error handling system in the future. The test will time
		// out if something doesn't work.
		time.Sleep(10 * time.Millisecond)
		got, exists := queue.Get("foo")
		if !exists {
			continue
		}
		handler.ValidateRequest(t, "/api/v1beta1/pods/foo", "GET", nil)
		if e, a := testPod, got; !reflect.DeepEqual(e, a) {
			t.Errorf("Expected %v, got %v", e, a)
		}
		break
	}
}

func TestStoreToMinionLister(t *testing.T) {
	store := cache.NewStore()
	ids := util.NewStringSet("foo", "bar", "baz")
	for id := range ids {
		store.Add(id, &api.Minion{JSONBase: api.JSONBase{ID: id}})
	}
	sml := storeToMinionLister{store}

	got, err := sml.List()
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if !ids.HasAll(got...) || len(got) != len(ids) {
		t.Errorf("Expected %v, got %v", ids, got)
	}
}

func TestStoreToPodLister(t *testing.T) {
	store := cache.NewStore()
	ids := []string{"foo", "bar", "baz"}
	for _, id := range ids {
		store.Add(id, &api.Pod{
			JSONBase: api.JSONBase{ID: id},
			Labels:   map[string]string{"name": id},
		})
	}
	spl := storeToPodLister{store}

	for _, id := range ids {
		got, err := spl.ListPods(labels.Set{"name": id}.AsSelector())
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
			continue
		}
		if e, a := 1, len(got); e != a {
			t.Errorf("Expected %v, got %v", e, a)
			continue
		}
		if e, a := id, got[0].ID; e != a {
			t.Errorf("Expected %v, got %v", e, a)
			continue
		}
	}
}

func TestMinionEnumerator(t *testing.T) {
	testList := &api.MinionList{
		Items: []api.Minion{
			{JSONBase: api.JSONBase{ID: "foo"}},
			{JSONBase: api.JSONBase{ID: "bar"}},
			{JSONBase: api.JSONBase{ID: "baz"}},
		},
	}
	me := minionEnumerator{testList}

	if e, a := 3, me.Len(); e != a {
		t.Fatalf("expected %v, got %v", e, a)
	}
	for i := range testList.Items {
		gotID, gotObj := me.Get(i)
		if e, a := testList.Items[i].ID, gotID; e != a {
			t.Errorf("Expected %v, got %v", e, a)
		}
		if e, a := &testList.Items[i], gotObj; !reflect.DeepEqual(e, a) {
			t.Errorf("Expected %#v, got %v#", e, a)
		}
	}
}

func TestBind(t *testing.T) {
	table := []struct {
		binding *api.Binding
	}{
		{binding: &api.Binding{PodID: "foo", Host: "foohost.kubernetes.mydomain.com"}},
	}

	for _, item := range table {
		handler := util.FakeHandler{
			StatusCode:   200,
			ResponseBody: "",
			T:            t,
		}
		server := httptest.NewServer(&handler)
		client := client.NewOrDie(server.URL, nil)
		b := binder{client}

		if err := b.Bind(item.binding); err != nil {
			t.Errorf("Unexpected error: %v", err)
			continue
		}
		expectedBody := api.EncodeOrDie(item.binding)
		handler.ValidateRequest(t, "/api/v1beta1/bindings", "POST", &expectedBody)
	}
}
