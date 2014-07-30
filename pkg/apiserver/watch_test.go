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

package apiserver

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"reflect"
	"testing"

	"code.google.com/p/go.net/websocket"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

var watchTestTable = []struct {
	t   watch.EventType
	obj interface{}
}{
	{watch.Added, &Simple{Name: "A Name"}},
	{watch.Modified, &Simple{Name: "Another Name"}},
	{watch.Deleted, &Simple{Name: "Another Name"}},
}

func TestWatchWebsocket(t *testing.T) {
	simpleStorage := &SimpleRESTStorage{}
	handler := New(map[string]RESTStorage{
		"foo": simpleStorage,
	}, "/prefix/version")
	server := httptest.NewServer(handler)

	dest, _ := url.Parse(server.URL)
	dest.Scheme = "ws" // Required by websocket, though the server never sees it.
	dest.Path = "/prefix/version/watch/foo"
	dest.RawQuery = "id=myID"

	ws, err := websocket.Dial(dest.String(), "", "http://localhost")
	expectNoError(t, err)

	if a, e := simpleStorage.requestedID, "myID"; a != e {
		t.Fatalf("Expected %v, got %v", e, a)
	}

	try := func(action watch.EventType, object interface{}) {
		// Send
		simpleStorage.fakeWatch.Action(action, object)
		// Test receive
		var got api.WatchEvent
		err := websocket.JSON.Receive(ws, &got)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		if got.Type != action {
			t.Errorf("Unexpected type: %v", got.Type)
		}
		if e, a := object, got.Object.Object; !reflect.DeepEqual(e, a) {
			t.Errorf("Expected %v, got %v", e, a)
		}
	}

	for _, item := range watchTestTable {
		try(item.t, item.obj)
	}
	simpleStorage.fakeWatch.Stop()

	var got api.WatchEvent
	err = websocket.JSON.Receive(ws, &got)
	if err == nil {
		t.Errorf("Unexpected non-error")
	}
}

func TestWatchHTTP(t *testing.T) {
	simpleStorage := &SimpleRESTStorage{}
	handler := New(map[string]RESTStorage{
		"foo": simpleStorage,
	}, "/prefix/version")
	server := httptest.NewServer(handler)
	client := http.Client{}

	dest, _ := url.Parse(server.URL)
	dest.Path = "/prefix/version/watch/foo"
	dest.RawQuery = "id=myID"

	request, err := http.NewRequest("GET", dest.String(), nil)
	expectNoError(t, err)
	response, err := client.Do(request)
	expectNoError(t, err)
	if response.StatusCode != http.StatusOK {
		t.Errorf("Unexpected response %#v", response)
	}

	if a, e := simpleStorage.requestedID, "myID"; a != e {
		t.Fatalf("Expected %v, got %v", e, a)
	}

	decoder := json.NewDecoder(response.Body)

	try := func(action watch.EventType, object interface{}) {
		// Send
		simpleStorage.fakeWatch.Action(action, object)
		// Test receive
		var got api.WatchEvent
		err := decoder.Decode(&got)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		if got.Type != action {
			t.Errorf("Unexpected type: %v", got.Type)
		}
		if e, a := object, got.Object.Object; !reflect.DeepEqual(e, a) {
			t.Errorf("Expected %v, got %v", e, a)
		}
	}

	for _, item := range watchTestTable {
		try(item.t, item.obj)
	}
	simpleStorage.fakeWatch.Stop()

	var got api.WatchEvent
	err = decoder.Decode(&got)
	if err == nil {
		t.Errorf("Unexpected non-error")
	}
}
