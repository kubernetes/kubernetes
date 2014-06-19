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
	"reflect"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
)

// TODO: This doesn't reduce typing enough to make it worth the less readable errors. Remove.
func expectNoError(t *testing.T, err error) {
	if err != nil {
		t.Errorf("Unexpected error: %#v", err)
	}
}

type Simple struct {
	Name string
}

type SimpleList struct {
	Items []Simple
}

type SimpleRESTStorage struct {
	err     error
	list    []Simple
	item    Simple
	deleted string
	updated Simple
}

func (storage *SimpleRESTStorage) List(labels.Selector) (interface{}, error) {
	result := SimpleList{
		Items: storage.list,
	}
	return result, storage.err
}

func (storage *SimpleRESTStorage) Get(id string) (interface{}, error) {
	return storage.item, storage.err
}

func (storage *SimpleRESTStorage) Delete(id string) error {
	storage.deleted = id
	return storage.err
}

func (storage *SimpleRESTStorage) Extract(body string) (interface{}, error) {
	var item Simple
	json.Unmarshal([]byte(body), &item)
	return item, storage.err
}

func (storage *SimpleRESTStorage) Create(interface{}) error {
	return storage.err
}

func (storage *SimpleRESTStorage) Update(object interface{}) error {
	storage.updated = object.(Simple)
	return storage.err
}

func extractBody(response *http.Response, object interface{}) (string, error) {
	defer response.Body.Close()
	body, err := ioutil.ReadAll(response.Body)
	if err != nil {
		return string(body), err
	}
	err = json.Unmarshal(body, object)
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

	if resp.StatusCode != 200 {
		t.Errorf("Unexpected status: %d, Expected: %d, %#v", resp.StatusCode, 200, resp)
	}
}

func TestErrorList(t *testing.T) {
	storage := map[string]RESTStorage{}
	simpleStorage := SimpleRESTStorage{
		err: fmt.Errorf("test Error"),
	}
	storage["simple"] = &simpleStorage
	handler := New(storage, "/prefix/version")
	server := httptest.NewServer(handler)

	resp, err := http.Get(server.URL + "/prefix/version/simple")
	expectNoError(t, err)

	if resp.StatusCode != 500 {
		t.Errorf("Unexpected status: %d, Expected: %d, %#v", resp.StatusCode, 200, resp)
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

	if resp.StatusCode != 200 {
		t.Errorf("Unexpected status: %d, Expected: %d, %#v", resp.StatusCode, 200, resp)
	}

	var listOut SimpleList
	body, err := extractBody(resp, &listOut)
	if len(listOut.Items) != 1 {
		t.Errorf("Unexpected response: %#v", listOut)
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
	body, err := json.Marshal(item)
	expectNoError(t, err)
	client := http.Client{}
	request, err := http.NewRequest("PUT", server.URL+"/prefix/version/simple/"+ID, bytes.NewReader(body))
	_, err = client.Do(request)
	expectNoError(t, err)
	if simpleStorage.updated.Name != item.Name {
		t.Errorf("Unexpected update value %#v, expected %#v.", simpleStorage.updated, item)
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
	if response.StatusCode != 404 {
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
	if response.StatusCode != 404 {
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
	if response.StatusCode != 404 {
		t.Errorf("Unexpected response %#v", response)
	}
}

func TestCreate(t *testing.T) {
	handler := New(map[string]RESTStorage{
		"foo": &SimpleRESTStorage{},
	}, "/prefix/version")
	server := httptest.NewServer(handler)
	client := http.Client{}

	simple := Simple{Name: "foo"}
	data, _ := json.Marshal(simple)
	request, err := http.NewRequest("POST", server.URL+"/prefix/version/foo", bytes.NewBuffer(data))
	expectNoError(t, err)
	response, err := client.Do(request)
	expectNoError(t, err)
	if response.StatusCode != 200 {
		t.Errorf("Unexpected response %#v", response)
	}

	var itemOut Simple
	body, err := extractBody(response, &itemOut)
	expectNoError(t, err)
	if !reflect.DeepEqual(itemOut, simple) {
		t.Errorf("Unexpected data: %#v, expected %#v (%s)", itemOut, simple, string(body))
	}
}
