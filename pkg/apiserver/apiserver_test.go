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
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"net/url"
	"reflect"
	"sync"
	"testing"
	"time"

	"code.google.com/p/go.net/websocket"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

func init() {
	api.AddKnownTypes(Simple{}, SimpleList{})
}

// TODO: This doesn't reduce typing enough to make it worth the less readable errors. Remove.
func expectNoError(t *testing.T, err error) {
	if err != nil {
		t.Errorf("Unexpected error: %#v", err)
	}
}

type Simple struct {
	api.JSONBase `yaml:",inline" json:",inline"`
	Name         string `yaml:"name,omitempty" json:"name,omitempty"`
}

type SimpleList struct {
	api.JSONBase `yaml:",inline" json:",inline"`
	Items        []Simple `yaml:"items,omitempty" json:"items,omitempty"`
}

type SimpleRESTStorage struct {
	errors  map[string]error
	list    []Simple
	item    Simple
	deleted string
	updated Simple
	created Simple

	// Valid if WatchAll or WatchSingle is called
	fakeWatch *watch.FakeWatcher

	// Set if WatchSingle is called
	requestedID string

	// If non-nil, called inside the WorkFunc when answering update, delete, create.
	// obj recieves the original input to the update, delete, or create call.
	injectedFunction func(obj interface{}) (returnObj interface{}, err error)
}

func (storage *SimpleRESTStorage) List(labels.Selector) (interface{}, error) {
	result := &SimpleList{
		Items: storage.list,
	}
	return result, storage.errors["list"]
}

func (storage *SimpleRESTStorage) Get(id string) (interface{}, error) {
	return storage.item, storage.errors["get"]
}

func (storage *SimpleRESTStorage) Delete(id string) (<-chan interface{}, error) {
	storage.deleted = id
	if err := storage.errors["delete"]; err != nil {
		return nil, err
	}
	return MakeAsync(func() (interface{}, error) {
		if storage.injectedFunction != nil {
			return storage.injectedFunction(id)
		}
		return api.Status{Status: api.StatusSuccess}, nil
	}), nil
}

func (storage *SimpleRESTStorage) Extract(body []byte) (interface{}, error) {
	var item Simple
	api.DecodeInto(body, &item)
	return item, storage.errors["extract"]
}

func (storage *SimpleRESTStorage) Create(obj interface{}) (<-chan interface{}, error) {
	storage.created = obj.(Simple)
	if err := storage.errors["create"]; err != nil {
		return nil, err
	}
	return MakeAsync(func() (interface{}, error) {
		if storage.injectedFunction != nil {
			return storage.injectedFunction(obj)
		}
		return obj, nil
	}), nil
}

func (storage *SimpleRESTStorage) Update(obj interface{}) (<-chan interface{}, error) {
	storage.updated = obj.(Simple)
	if err := storage.errors["update"]; err != nil {
		return nil, err
	}
	return MakeAsync(func() (interface{}, error) {
		if storage.injectedFunction != nil {
			return storage.injectedFunction(obj)
		}
		return obj, nil
	}), nil
}

// Implement ResourceWatcher.
func (storage *SimpleRESTStorage) WatchAll() (watch.Interface, error) {
	if err := storage.errors["watchAll"]; err != nil {
		return nil, err
	}
	storage.fakeWatch = watch.NewFake()
	return storage.fakeWatch, nil
}

// Implement ResourceWatcher.
func (storage *SimpleRESTStorage) WatchSingle(id string) (watch.Interface, error) {
	storage.requestedID = id
	if err := storage.errors["watchSingle"]; err != nil {
		return nil, err
	}
	storage.fakeWatch = watch.NewFake()
	return storage.fakeWatch, nil
}

func extractBody(response *http.Response, object interface{}) (string, error) {
	defer response.Body.Close()
	body, err := ioutil.ReadAll(response.Body)
	if err != nil {
		return string(body), err
	}
	err = api.DecodeInto(body, object)
	return string(body), err
}

func TestSimpleList(t *testing.T) {
	storage := map[string]RESTStorage{}
	simpleStorage := SimpleRESTStorage{}
	storage["simple"] = &simpleStorage
	handler := New(storage, "/prefix/version")
	server := httptest.NewServer(handler)

	resp, err := http.Get(server.URL + "/prefix/version/simple")
	expectNoError(t, err)

	if resp.StatusCode != http.StatusOK {
		t.Errorf("Unexpected status: %d, Expected: %d, %#v", resp.StatusCode, http.StatusOK, resp)
	}
}

func TestErrorList(t *testing.T) {
	storage := map[string]RESTStorage{}
	simpleStorage := SimpleRESTStorage{
		errors: map[string]error{"list": fmt.Errorf("test Error")},
	}
	storage["simple"] = &simpleStorage
	handler := New(storage, "/prefix/version")
	server := httptest.NewServer(handler)

	resp, err := http.Get(server.URL + "/prefix/version/simple")
	expectNoError(t, err)

	if resp.StatusCode != http.StatusInternalServerError {
		t.Errorf("Unexpected status: %d, Expected: %d, %#v", resp.StatusCode, http.StatusOK, resp)
	}
}

func TestNonEmptyList(t *testing.T) {
	storage := map[string]RESTStorage{}
	simpleStorage := SimpleRESTStorage{
		list: []Simple{
			{
				Name: "foo",
			},
		},
	}
	storage["simple"] = &simpleStorage
	handler := New(storage, "/prefix/version")
	server := httptest.NewServer(handler)

	resp, err := http.Get(server.URL + "/prefix/version/simple")
	expectNoError(t, err)

	if resp.StatusCode != http.StatusOK {
		t.Errorf("Unexpected status: %d, Expected: %d, %#v", resp.StatusCode, http.StatusOK, resp)
	}

	var listOut SimpleList
	body, err := extractBody(resp, &listOut)
	expectNoError(t, err)
	if len(listOut.Items) != 1 {
		t.Errorf("Unexpected response: %#v", listOut)
		return
	}
	if listOut.Items[0].Name != simpleStorage.list[0].Name {
		t.Errorf("Unexpected data: %#v, %s", listOut.Items[0], string(body))
	}
}

func TestGet(t *testing.T) {
	storage := map[string]RESTStorage{}
	simpleStorage := SimpleRESTStorage{
		item: Simple{
			Name: "foo",
		},
	}
	storage["simple"] = &simpleStorage
	handler := New(storage, "/prefix/version")
	server := httptest.NewServer(handler)

	resp, err := http.Get(server.URL + "/prefix/version/simple/id")
	var itemOut Simple
	body, err := extractBody(resp, &itemOut)
	expectNoError(t, err)
	if itemOut.Name != simpleStorage.item.Name {
		t.Errorf("Unexpected data: %#v, expected %#v (%s)", itemOut, simpleStorage.item, string(body))
	}
}

func TestGetMissing(t *testing.T) {
	storage := map[string]RESTStorage{}
	simpleStorage := SimpleRESTStorage{
		errors: map[string]error{"get": NewNotFoundErr("simple", "id")},
	}
	storage["simple"] = &simpleStorage
	handler := New(storage, "/prefix/version")
	server := httptest.NewServer(handler)

	resp, err := http.Get(server.URL + "/prefix/version/simple/id")
	expectNoError(t, err)
	if resp.StatusCode != http.StatusNotFound {
		t.Errorf("Unexpected response %#v", resp)
	}
}

func TestDelete(t *testing.T) {
	storage := map[string]RESTStorage{}
	simpleStorage := SimpleRESTStorage{}
	ID := "id"
	storage["simple"] = &simpleStorage
	handler := New(storage, "/prefix/version")
	server := httptest.NewServer(handler)

	client := http.Client{}
	request, err := http.NewRequest("DELETE", server.URL+"/prefix/version/simple/"+ID, nil)
	_, err = client.Do(request)
	expectNoError(t, err)
	if simpleStorage.deleted != ID {
		t.Errorf("Unexpected delete: %s, expected %s (%s)", simpleStorage.deleted, ID)
	}
}

func TestDeleteMissing(t *testing.T) {
	storage := map[string]RESTStorage{}
	ID := "id"
	simpleStorage := SimpleRESTStorage{
		errors: map[string]error{"delete": NewNotFoundErr("simple", ID)},
	}
	storage["simple"] = &simpleStorage
	handler := New(storage, "/prefix/version")
	server := httptest.NewServer(handler)

	client := http.Client{}
	request, err := http.NewRequest("DELETE", server.URL+"/prefix/version/simple/"+ID, nil)
	response, err := client.Do(request)
	expectNoError(t, err)
	if response.StatusCode != http.StatusNotFound {
		t.Errorf("Unexpected response %#v", response)
	}
}

func TestUpdate(t *testing.T) {
	storage := map[string]RESTStorage{}
	simpleStorage := SimpleRESTStorage{}
	ID := "id"
	storage["simple"] = &simpleStorage
	handler := New(storage, "/prefix/version")
	server := httptest.NewServer(handler)

	item := Simple{
		Name: "bar",
	}
	body, err := api.Encode(item)
	expectNoError(t, err)
	client := http.Client{}
	request, err := http.NewRequest("PUT", server.URL+"/prefix/version/simple/"+ID, bytes.NewReader(body))
	_, err = client.Do(request)
	expectNoError(t, err)
	if simpleStorage.updated.Name != item.Name {
		t.Errorf("Unexpected update value %#v, expected %#v.", simpleStorage.updated, item)
	}
}

func TestUpdateMissing(t *testing.T) {
	storage := map[string]RESTStorage{}
	ID := "id"
	simpleStorage := SimpleRESTStorage{
		errors: map[string]error{"update": NewNotFoundErr("simple", ID)},
	}
	storage["simple"] = &simpleStorage
	handler := New(storage, "/prefix/version")
	server := httptest.NewServer(handler)

	item := Simple{
		Name: "bar",
	}
	body, err := api.Encode(item)
	expectNoError(t, err)
	client := http.Client{}
	request, err := http.NewRequest("PUT", server.URL+"/prefix/version/simple/"+ID, bytes.NewReader(body))
	response, err := client.Do(request)
	expectNoError(t, err)
	if response.StatusCode != http.StatusNotFound {
		t.Errorf("Unexpected response %#v", response)
	}
}

func TestBadPath(t *testing.T) {
	handler := New(map[string]RESTStorage{}, "/prefix/version")
	server := httptest.NewServer(handler)
	client := http.Client{}

	request, err := http.NewRequest("GET", server.URL+"/foobar", nil)
	expectNoError(t, err)
	response, err := client.Do(request)
	expectNoError(t, err)
	if response.StatusCode != http.StatusNotFound {
		t.Errorf("Unexpected response %#v", response)
	}
}

func TestMissingPath(t *testing.T) {
	handler := New(map[string]RESTStorage{}, "/prefix/version")
	server := httptest.NewServer(handler)
	client := http.Client{}

	request, err := http.NewRequest("GET", server.URL+"/prefix/version", nil)
	expectNoError(t, err)
	response, err := client.Do(request)
	expectNoError(t, err)
	if response.StatusCode != http.StatusNotFound {
		t.Errorf("Unexpected response %#v", response)
	}
}

func TestMissingStorage(t *testing.T) {
	handler := New(map[string]RESTStorage{
		"foo": &SimpleRESTStorage{},
	}, "/prefix/version")
	server := httptest.NewServer(handler)
	client := http.Client{}

	request, err := http.NewRequest("GET", server.URL+"/prefix/version/foobar", nil)
	expectNoError(t, err)
	response, err := client.Do(request)
	expectNoError(t, err)
	if response.StatusCode != http.StatusNotFound {
		t.Errorf("Unexpected response %#v", response)
	}
}

func TestCreate(t *testing.T) {
	simpleStorage := &SimpleRESTStorage{}
	handler := New(map[string]RESTStorage{
		"foo": simpleStorage,
	}, "/prefix/version")
	server := httptest.NewServer(handler)
	client := http.Client{}

	simple := Simple{Name: "foo"}
	data, _ := api.Encode(simple)
	request, err := http.NewRequest("POST", server.URL+"/prefix/version/foo", bytes.NewBuffer(data))
	expectNoError(t, err)
	response, err := client.Do(request)
	expectNoError(t, err)
	if response.StatusCode != http.StatusAccepted {
		t.Errorf("Unexpected response %#v", response)
	}

	var itemOut api.Status
	body, err := extractBody(response, &itemOut)
	expectNoError(t, err)
	if itemOut.Status != api.StatusWorking || itemOut.Details == "" {
		t.Errorf("Unexpected status: %#v (%s)", itemOut, string(body))
	}
}

func TestCreateNotFound(t *testing.T) {
	simpleStorage := &SimpleRESTStorage{
		// storage.Create can fail with not found error in theory.
		// See https://github.com/GoogleCloudPlatform/kubernetes/pull/486#discussion_r15037092.
		errors: map[string]error{"create": NewNotFoundErr("simple", "id")},
	}
	handler := New(map[string]RESTStorage{
		"foo": simpleStorage,
	}, "/prefix/version")
	server := httptest.NewServer(handler)
	client := http.Client{}

	simple := Simple{Name: "foo"}
	data, _ := api.Encode(simple)
	request, err := http.NewRequest("POST", server.URL+"/prefix/version/foo", bytes.NewBuffer(data))
	expectNoError(t, err)
	response, err := client.Do(request)
	expectNoError(t, err)
	if response.StatusCode != http.StatusNotFound {
		t.Errorf("Unexpected response %#v", response)
	}
}

func TestParseTimeout(t *testing.T) {
	if d := parseTimeout(""); d != 30*time.Second {
		t.Errorf("blank timeout produces %v", d)
	}
	if d := parseTimeout("not a timeout"); d != 30*time.Second {
		t.Errorf("bad timeout produces %v", d)
	}
	if d := parseTimeout("10s"); d != 10*time.Second {
		t.Errorf("10s timeout produced: %v", d)
	}
}

func TestSyncCreate(t *testing.T) {
	storage := SimpleRESTStorage{
		injectedFunction: func(obj interface{}) (interface{}, error) {
			time.Sleep(200 * time.Millisecond)
			return obj, nil
		},
	}
	handler := New(map[string]RESTStorage{
		"foo": &storage,
	}, "/prefix/version")
	server := httptest.NewServer(handler)
	client := http.Client{}

	simple := Simple{Name: "foo"}
	data, _ := api.Encode(simple)
	request, err := http.NewRequest("POST", server.URL+"/prefix/version/foo?sync=true", bytes.NewBuffer(data))
	expectNoError(t, err)
	wg := sync.WaitGroup{}
	wg.Add(1)
	var response *http.Response
	go func() {
		response, err = client.Do(request)
		wg.Done()
	}()
	wg.Wait()
	expectNoError(t, err)
	var itemOut Simple
	body, err := extractBody(response, &itemOut)
	expectNoError(t, err)
	if !reflect.DeepEqual(itemOut, simple) {
		t.Errorf("Unexpected data: %#v, expected %#v (%s)", itemOut, simple, string(body))
	}
	if response.StatusCode != http.StatusOK {
		t.Errorf("Unexpected status: %d, Expected: %d, %#v", response.StatusCode, http.StatusOK, response)
	}
}

func TestSyncCreateTimeout(t *testing.T) {
	storage := SimpleRESTStorage{
		injectedFunction: func(obj interface{}) (interface{}, error) {
			time.Sleep(400 * time.Millisecond)
			return obj, nil
		},
	}
	handler := New(map[string]RESTStorage{
		"foo": &storage,
	}, "/prefix/version")
	server := httptest.NewServer(handler)
	client := http.Client{}

	simple := Simple{Name: "foo"}
	data, _ := api.Encode(simple)
	request, err := http.NewRequest("POST", server.URL+"/prefix/version/foo?sync=true&timeout=200ms", bytes.NewBuffer(data))
	expectNoError(t, err)
	wg := sync.WaitGroup{}
	wg.Add(1)
	var response *http.Response
	go func() {
		response, err = client.Do(request)
		wg.Done()
	}()
	wg.Wait()
	expectNoError(t, err)
	var itemOut api.Status
	_, err = extractBody(response, &itemOut)
	expectNoError(t, err)
	if itemOut.Status != api.StatusWorking || itemOut.Details == "" {
		t.Errorf("Unexpected status %#v", itemOut)
	}
	if response.StatusCode != http.StatusAccepted {
		t.Errorf("Unexpected status: %d, Expected: %d, %#v", response.StatusCode, 202, response)
	}
}

func TestOpGet(t *testing.T) {
	simpleStorage := &SimpleRESTStorage{}
	handler := New(map[string]RESTStorage{
		"foo": simpleStorage,
	}, "/prefix/version")
	server := httptest.NewServer(handler)
	client := http.Client{}

	simple := Simple{Name: "foo"}
	data, _ := api.Encode(simple)
	request, err := http.NewRequest("POST", server.URL+"/prefix/version/foo", bytes.NewBuffer(data))
	expectNoError(t, err)
	response, err := client.Do(request)
	expectNoError(t, err)
	if response.StatusCode != http.StatusAccepted {
		t.Errorf("Unexpected response %#v", response)
	}

	var itemOut api.Status
	body, err := extractBody(response, &itemOut)
	expectNoError(t, err)
	if itemOut.Status != api.StatusWorking || itemOut.Details == "" {
		t.Errorf("Unexpected status: %#v (%s)", itemOut, string(body))
	}

	req2, err := http.NewRequest("GET", server.URL+"/prefix/version/operations/"+itemOut.Details, nil)
	expectNoError(t, err)
	_, err = client.Do(req2)
	expectNoError(t, err)
	if response.StatusCode != http.StatusAccepted {
		t.Errorf("Unexpected response %#v", response)
	}
}

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
		apiObj, err := api.Decode(got.EmbeddedObject)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		if !reflect.DeepEqual(object, apiObj) {
			t.Errorf("Expected %v, got %v", object, apiObj)
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
		apiObj, err := api.Decode(got.EmbeddedObject)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		if !reflect.DeepEqual(object, apiObj) {
			t.Errorf("Expected %v, got %v", object, apiObj)
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
