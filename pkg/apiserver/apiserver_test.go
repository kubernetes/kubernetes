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
	"errors"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/version"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

func convert(obj interface{}) (interface{}, error) {
	return obj, nil
}

func init() {
	api.AddKnownTypes("", Simple{}, SimpleList{})
	api.AddKnownTypes("v1beta1", Simple{}, SimpleList{})
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
	// obj receives the original input to the update, delete, or create call.
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

func TestNotFound(t *testing.T) {
	type T struct {
		Method string
		Path   string
	}
	cases := map[string]T{
		"PATCH method":                 T{"PATCH", "/prefix/version/foo"},
		"GET long prefix":              T{"GET", "/prefix/"},
		"GET missing storage":          T{"GET", "/prefix/version/blah"},
		"GET with extra segment":       T{"GET", "/prefix/version/foo/bar/baz"},
		"POST with extra segment":      T{"POST", "/prefix/version/foo/bar"},
		"DELETE without extra segment": T{"DELETE", "/prefix/version/foo"},
		"DELETE with extra segment":    T{"DELETE", "/prefix/version/foo/bar/baz"},
		"PUT without extra segment":    T{"PUT", "/prefix/version/foo"},
		"PUT with extra segment":       T{"PUT", "/prefix/version/foo/bar/baz"},
		"watch missing storage":        T{"GET", "/prefix/version/watch/"},
		"watch with bad method":        T{"POST", "/prefix/version/watch/foo/bar"},
	}
	handler := New(map[string]RESTStorage{
		"foo": &SimpleRESTStorage{},
	}, "/prefix/version")
	server := httptest.NewServer(handler)
	client := http.Client{}
	for k, v := range cases {
		request, err := http.NewRequest(v.Method, server.URL+v.Path, nil)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}

		response, err := client.Do(request)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}

		if response.StatusCode != http.StatusNotFound {
			t.Errorf("Expected %d for %s (%s), Got %#v", http.StatusNotFound, v, k, response)
		}
	}
}

func TestVersion(t *testing.T) {
	handler := New(map[string]RESTStorage{}, "/prefix/version")
	server := httptest.NewServer(handler)
	client := http.Client{}

	request, err := http.NewRequest("GET", server.URL+"/version", nil)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	response, err := client.Do(request)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	var info version.Info
	err = json.NewDecoder(response.Body).Decode(&info)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if !reflect.DeepEqual(version.Get(), info) {
		t.Errorf("Expected %#v, Got %#v", version.Get(), info)
	}
}

func TestSimpleList(t *testing.T) {
	storage := map[string]RESTStorage{}
	simpleStorage := SimpleRESTStorage{}
	storage["simple"] = &simpleStorage
	handler := New(storage, "/prefix/version")
	server := httptest.NewServer(handler)

	resp, err := http.Get(server.URL + "/prefix/version/simple")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

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
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if resp.StatusCode != http.StatusInternalServerError {
		t.Errorf("Unexpected status: %d, Expected: %d, %#v", resp.StatusCode, http.StatusOK, resp)
	}
}

func TestNonEmptyList(t *testing.T) {
	storage := map[string]RESTStorage{}
	simpleStorage := SimpleRESTStorage{
		list: []Simple{
			{
				JSONBase: api.JSONBase{Kind: "Simple"},
				Name:     "foo",
			},
		},
	}
	storage["simple"] = &simpleStorage
	handler := New(storage, "/prefix/version")
	server := httptest.NewServer(handler)

	resp, err := http.Get(server.URL + "/prefix/version/simple")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if resp.StatusCode != http.StatusOK {
		t.Errorf("Unexpected status: %d, Expected: %d, %#v", resp.StatusCode, http.StatusOK, resp)
	}

	var listOut SimpleList
	body, err := extractBody(resp, &listOut)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

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
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

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
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

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
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if simpleStorage.deleted != ID {
		t.Errorf("Unexpected delete: %s, expected %s", simpleStorage.deleted, ID)
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
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

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
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	client := http.Client{}
	request, err := http.NewRequest("PUT", server.URL+"/prefix/version/simple/"+ID, bytes.NewReader(body))
	_, err = client.Do(request)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

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
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	client := http.Client{}
	request, err := http.NewRequest("PUT", server.URL+"/prefix/version/simple/"+ID, bytes.NewReader(body))
	response, err := client.Do(request)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if response.StatusCode != http.StatusNotFound {
		t.Errorf("Unexpected response %#v", response)
	}
}

func TestBadPath(t *testing.T) {
	handler := New(map[string]RESTStorage{}, "/prefix/version")
	server := httptest.NewServer(handler)
	client := http.Client{}

	request, err := http.NewRequest("GET", server.URL+"/foobar", nil)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	response, err := client.Do(request)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if response.StatusCode != http.StatusNotFound {
		t.Errorf("Unexpected response %#v", response)
	}
}

func TestCreate(t *testing.T) {
	simpleStorage := &SimpleRESTStorage{}
	handler := New(map[string]RESTStorage{
		"foo": simpleStorage,
	}, "/prefix/version")
	handler.asyncOpWait = 0
	server := httptest.NewServer(handler)
	client := http.Client{}

	simple := Simple{
		Name: "foo",
	}
	data, _ := api.Encode(simple)
	request, err := http.NewRequest("POST", server.URL+"/prefix/version/foo", bytes.NewBuffer(data))
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	response, err := client.Do(request)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if response.StatusCode != http.StatusAccepted {
		t.Errorf("Unexpected response %#v", response)
	}

	var itemOut api.Status
	body, err := extractBody(response, &itemOut)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if itemOut.Status != api.StatusWorking || itemOut.Details == nil || itemOut.Details.ID == "" {
		t.Errorf("Unexpected status: %#v (%s)", itemOut, string(body))
	}
}

func TestCreateNotFound(t *testing.T) {
	handler := New(map[string]RESTStorage{
		"simple": &SimpleRESTStorage{
			// storage.Create can fail with not found error in theory.
			// See https://github.com/GoogleCloudPlatform/kubernetes/pull/486#discussion_r15037092.
			errors: map[string]error{"create": NewNotFoundErr("simple", "id")},
		},
	}, "/prefix/version")
	server := httptest.NewServer(handler)
	client := http.Client{}

	simple := Simple{Name: "foo"}
	data, _ := api.Encode(simple)
	request, err := http.NewRequest("POST", server.URL+"/prefix/version/simple", bytes.NewBuffer(data))
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	response, err := client.Do(request)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

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
			time.Sleep(5 * time.Millisecond)
			return obj, nil
		},
	}
	handler := New(map[string]RESTStorage{
		"foo": &storage,
	}, "/prefix/version")
	server := httptest.NewServer(handler)
	client := http.Client{}

	simple := Simple{
		Name: "foo",
	}
	data, _ := api.Encode(simple)
	request, err := http.NewRequest("POST", server.URL+"/prefix/version/foo?sync=true", bytes.NewBuffer(data))
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	wg := sync.WaitGroup{}
	wg.Add(1)
	var response *http.Response
	go func() {
		response, err = client.Do(request)
		wg.Done()
	}()
	wg.Wait()
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	var itemOut Simple
	body, err := extractBody(response, &itemOut)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if !reflect.DeepEqual(itemOut, simple) {
		t.Errorf("Unexpected data: %#v, expected %#v (%s)", itemOut, simple, string(body))
	}
	if response.StatusCode != http.StatusOK {
		t.Errorf("Unexpected status: %d, Expected: %d, %#v", response.StatusCode, http.StatusOK, response)
	}
}

func expectApiStatus(t *testing.T, method, url string, data []byte, code int) *api.Status {
	client := http.Client{}
	request, err := http.NewRequest(method, url, bytes.NewBuffer(data))
	if err != nil {
		t.Fatalf("unexpected error %#v", err)
		return nil
	}
	response, err := client.Do(request)
	if err != nil {
		t.Fatalf("unexpected error %#v", err)
		return nil
	}
	var status api.Status
	_, err = extractBody(response, &status)
	if err != nil {
		t.Fatalf("unexpected error %#v", err)
		return nil
	}
	if code != response.StatusCode {
		t.Fatalf("Expected %d, Got %d", code, response.StatusCode)
	}
	return &status
}

func TestAsyncDelayReturnsError(t *testing.T) {
	storage := SimpleRESTStorage{
		injectedFunction: func(obj interface{}) (interface{}, error) {
			return nil, errors.New("error")
		},
	}
	handler := New(map[string]RESTStorage{"foo": &storage}, "/prefix/version")
	handler.asyncOpWait = time.Millisecond / 2
	server := httptest.NewServer(handler)

	status := expectApiStatus(t, "DELETE", fmt.Sprintf("%s/prefix/version/foo/bar", server.URL), nil, http.StatusInternalServerError)
	if status.Status != api.StatusFailure || status.Message != "error" || status.Details != nil {
		t.Errorf("Unexpected status %#v", status)
	}
}

func TestAsyncCreateError(t *testing.T) {
	ch := make(chan struct{})
	storage := SimpleRESTStorage{
		injectedFunction: func(obj interface{}) (interface{}, error) {
			<-ch
			return nil, errors.New("error")
		},
	}
	handler := New(map[string]RESTStorage{"foo": &storage}, "/prefix/version")
	handler.asyncOpWait = 0
	server := httptest.NewServer(handler)

	simple := Simple{Name: "foo"}
	data, _ := api.Encode(simple)

	status := expectApiStatus(t, "POST", fmt.Sprintf("%s/prefix/version/foo", server.URL), data, http.StatusAccepted)
	if status.Status != api.StatusWorking || status.Details == nil || status.Details.ID == "" {
		t.Errorf("Unexpected status %#v", status)
	}

	otherStatus := expectApiStatus(t, "GET", fmt.Sprintf("%s/prefix/version/operations/%s", server.URL, status.Details.ID), []byte{}, http.StatusAccepted)
	if !reflect.DeepEqual(status, otherStatus) {
		t.Errorf("Expected %#v, Got %#v", status, otherStatus)
	}

	ch <- struct{}{}
	time.Sleep(time.Millisecond)

	finalStatus := expectApiStatus(t, "GET", fmt.Sprintf("%s/prefix/version/operations/%s?after=1", server.URL, status.Details.ID), []byte{}, http.StatusOK)
	expectedStatus := &api.Status{
		Code:    http.StatusInternalServerError,
		Message: "error",
		Status:  api.StatusFailure,
	}
	if !reflect.DeepEqual(expectedStatus, finalStatus) {
		t.Errorf("Expected %#v, Got %#v", expectedStatus, finalStatus)
	}
}

func TestWriteJSONDecodeError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		type T struct {
			Value string
		}
		writeJSON(http.StatusOK, &T{"Undecodable"}, w)
	}))
	client := http.Client{}
	resp, err := client.Get(server.URL)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if resp.StatusCode != http.StatusInternalServerError {
		t.Errorf("unexpected status code %d", resp.StatusCode)
	}
}

type marshalError struct {
	err error
}

func (m *marshalError) MarshalJSON() ([]byte, error) {
	return []byte{}, m.err
}

func TestWriteRAWJSONMarshalError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		writeRawJSON(http.StatusOK, &marshalError{errors.New("Undecodable")}, w)
	}))
	client := http.Client{}
	resp, err := client.Get(server.URL)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if resp.StatusCode != http.StatusInternalServerError {
		t.Errorf("unexpected status code %d", resp.StatusCode)
	}
}

func TestSyncCreateTimeout(t *testing.T) {
	storage := SimpleRESTStorage{
		injectedFunction: func(obj interface{}) (interface{}, error) {
			time.Sleep(5 * time.Millisecond)
			return obj, nil
		},
	}
	handler := New(map[string]RESTStorage{
		"foo": &storage,
	}, "/prefix/version")
	server := httptest.NewServer(handler)

	simple := Simple{Name: "foo"}
	data, _ := api.Encode(simple)
	itemOut := expectApiStatus(t, "POST", server.URL+"/prefix/version/foo?sync=true&timeout=4ms", data, http.StatusAccepted)
	if itemOut.Status != api.StatusWorking || itemOut.Details == nil || itemOut.Details.ID == "" {
		t.Errorf("Unexpected status %#v", itemOut)
	}
}
