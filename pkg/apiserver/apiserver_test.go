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
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	apierrs "github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/latest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/testapi"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/version"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

func convert(obj runtime.Object) (runtime.Object, error) {
	return obj, nil
}

var codec = testapi.Codec()
var selfLinker = latest.SelfLinker

func init() {
	api.Scheme.AddKnownTypes("", &Simple{}, &SimpleList{})
	api.Scheme.AddKnownTypes(testapi.Version(), &Simple{}, &SimpleList{})
}

type Simple struct {
	api.TypeMeta   `yaml:",inline" json:",inline"`
	api.ObjectMeta `yaml:"metadata" json:"metadata"`
	Other          string `yaml:"other,omitempty" json:"other,omitempty"`
}

func (*Simple) IsAnAPIObject() {}

type SimpleList struct {
	api.TypeMeta `yaml:",inline" json:",inline"`
	api.ListMeta `yaml:"metadata,inline" json:"metadata,inline"`
	Items        []Simple `yaml:"items,omitempty" json:"items,omitempty"`
}

func (*SimpleList) IsAnAPIObject() {}

func TestSimpleSetupRight(t *testing.T) {
	s := &Simple{ObjectMeta: api.ObjectMeta{Name: "aName"}}
	wire, err := codec.Encode(s)
	if err != nil {
		t.Fatal(err)
	}
	s2, err := codec.Decode(wire)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(s, s2) {
		t.Fatalf("encode/decode broken:\n%#v\n%#v\n", s, s2)
	}
}

type SimpleRESTStorage struct {
	errors  map[string]error
	list    []Simple
	item    Simple
	deleted string
	updated *Simple
	created *Simple

	// These are set when Watch is called
	fakeWatch                  *watch.FakeWatcher
	requestedLabelSelector     labels.Selector
	requestedFieldSelector     labels.Selector
	requestedResourceVersion   string
	requestedResourceNamespace string

	// The id requested, and location to return for ResourceLocation
	requestedResourceLocationID string
	resourceLocation            string
	expectedResourceNamespace   string

	// If non-nil, called inside the WorkFunc when answering update, delete, create.
	// obj receives the original input to the update, delete, or create call.
	injectedFunction func(obj runtime.Object) (returnObj runtime.Object, err error)
}

func (storage *SimpleRESTStorage) List(ctx api.Context, label, field labels.Selector) (runtime.Object, error) {
	result := &SimpleList{
		Items: storage.list,
	}
	return result, storage.errors["list"]
}

func (storage *SimpleRESTStorage) Get(ctx api.Context, id string) (runtime.Object, error) {
	return api.Scheme.CopyOrDie(&storage.item), storage.errors["get"]
}

func (storage *SimpleRESTStorage) Delete(ctx api.Context, id string) (<-chan RESTResult, error) {
	storage.deleted = id
	if err := storage.errors["delete"]; err != nil {
		return nil, err
	}
	return MakeAsync(func() (runtime.Object, error) {
		if storage.injectedFunction != nil {
			return storage.injectedFunction(&Simple{ObjectMeta: api.ObjectMeta{Name: id}})
		}
		return &api.Status{Status: api.StatusSuccess}, nil
	}), nil
}

func (storage *SimpleRESTStorage) New() runtime.Object {
	return &Simple{}
}

func (storage *SimpleRESTStorage) Create(ctx api.Context, obj runtime.Object) (<-chan RESTResult, error) {
	storage.created = obj.(*Simple)
	if err := storage.errors["create"]; err != nil {
		return nil, err
	}
	return MakeAsync(func() (runtime.Object, error) {
		if storage.injectedFunction != nil {
			return storage.injectedFunction(obj)
		}
		return obj, nil
	}), nil
}

func (storage *SimpleRESTStorage) Update(ctx api.Context, obj runtime.Object) (<-chan RESTResult, error) {
	storage.updated = obj.(*Simple)
	if err := storage.errors["update"]; err != nil {
		return nil, err
	}
	return MakeAsync(func() (runtime.Object, error) {
		if storage.injectedFunction != nil {
			return storage.injectedFunction(obj)
		}
		return obj, nil
	}), nil
}

// Implement ResourceWatcher.
func (storage *SimpleRESTStorage) Watch(ctx api.Context, label, field labels.Selector, resourceVersion string) (watch.Interface, error) {
	storage.requestedLabelSelector = label
	storage.requestedFieldSelector = field
	storage.requestedResourceVersion = resourceVersion
	storage.requestedResourceNamespace = api.Namespace(ctx)
	if err := storage.errors["watch"]; err != nil {
		return nil, err
	}
	storage.fakeWatch = watch.NewFake()
	return storage.fakeWatch, nil
}

// Implement Redirector.
func (storage *SimpleRESTStorage) ResourceLocation(ctx api.Context, id string) (string, error) {
	// validate that the namespace context on the request matches the expected input
	storage.requestedResourceNamespace = api.Namespace(ctx)
	if storage.expectedResourceNamespace != storage.requestedResourceNamespace {
		return "", fmt.Errorf("Expected request namespace %s, but got namespace %s", storage.expectedResourceNamespace, storage.requestedResourceNamespace)
	}
	storage.requestedResourceLocationID = id
	if err := storage.errors["resourceLocation"]; err != nil {
		return "", err
	}
	return storage.resourceLocation, nil
}

func extractBody(response *http.Response, object runtime.Object) (string, error) {
	defer response.Body.Close()
	body, err := ioutil.ReadAll(response.Body)
	if err != nil {
		return string(body), err
	}
	err = codec.DecodeInto(body, object)
	return string(body), err
}

func TestNotFound(t *testing.T) {
	type T struct {
		Method string
		Path   string
	}
	cases := map[string]T{
		"PATCH method":                 {"PATCH", "/prefix/version/foo"},
		"GET long prefix":              {"GET", "/prefix/"},
		"GET missing storage":          {"GET", "/prefix/version/blah"},
		"GET with extra segment":       {"GET", "/prefix/version/foo/bar/baz"},
		"POST with extra segment":      {"POST", "/prefix/version/foo/bar"},
		"DELETE without extra segment": {"DELETE", "/prefix/version/foo"},
		"DELETE with extra segment":    {"DELETE", "/prefix/version/foo/bar/baz"},
		"PUT without extra segment":    {"PUT", "/prefix/version/foo"},
		"PUT with extra segment":       {"PUT", "/prefix/version/foo/bar/baz"},
		"watch missing storage":        {"GET", "/prefix/version/watch/"},
		"watch with bad method":        {"POST", "/prefix/version/watch/foo/bar"},
	}
	handler := Handle(map[string]RESTStorage{
		"foo": &SimpleRESTStorage{},
	}, codec, "/prefix/version", selfLinker)
	server := httptest.NewServer(handler)
	defer server.Close()
	client := http.Client{}
	for k, v := range cases {
		request, err := http.NewRequest(v.Method, server.URL+v.Path, nil)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
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
	handler := Handle(map[string]RESTStorage{}, codec, "/prefix/version", selfLinker)
	server := httptest.NewServer(handler)
	defer server.Close()
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
	selfLinker := &setTestSelfLinker{
		t:           t,
		expectedSet: "/prefix/version/simple",
	}
	handler := Handle(storage, codec, "/prefix/version", selfLinker)
	server := httptest.NewServer(handler)
	defer server.Close()

	resp, err := http.Get(server.URL + "/prefix/version/simple")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if resp.StatusCode != http.StatusOK {
		t.Errorf("Unexpected status: %d, Expected: %d, %#v", resp.StatusCode, http.StatusOK, resp)
	}
	if !selfLinker.called {
		t.Errorf("Never set self link")
	}
}

func TestErrorList(t *testing.T) {
	storage := map[string]RESTStorage{}
	simpleStorage := SimpleRESTStorage{
		errors: map[string]error{"list": fmt.Errorf("test Error")},
	}
	storage["simple"] = &simpleStorage
	handler := Handle(storage, codec, "/prefix/version", selfLinker)
	server := httptest.NewServer(handler)
	defer server.Close()

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
				TypeMeta: api.TypeMeta{Kind: "Simple"},
				Other:    "foo",
			},
		},
	}
	storage["simple"] = &simpleStorage
	handler := Handle(storage, codec, "/prefix/version", selfLinker)
	server := httptest.NewServer(handler)
	defer server.Close()

	resp, err := http.Get(server.URL + "/prefix/version/simple")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if resp.StatusCode != http.StatusOK {
		t.Errorf("Unexpected status: %d, Expected: %d, %#v", resp.StatusCode, http.StatusOK, resp)
		body, _ := ioutil.ReadAll(resp.Body)
		t.Logf("Data: %s", string(body))
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
	if listOut.Items[0].Other != simpleStorage.list[0].Other {
		t.Errorf("Unexpected data: %#v, %s", listOut.Items[0], string(body))
	}
}

func TestGet(t *testing.T) {
	storage := map[string]RESTStorage{}
	simpleStorage := SimpleRESTStorage{
		item: Simple{
			Other: "foo",
		},
	}
	selfLinker := &setTestSelfLinker{
		t:           t,
		expectedSet: "/prefix/version/simple/id",
	}
	storage["simple"] = &simpleStorage
	handler := Handle(storage, codec, "/prefix/version", selfLinker)
	server := httptest.NewServer(handler)
	defer server.Close()

	resp, err := http.Get(server.URL + "/prefix/version/simple/id")
	var itemOut Simple
	body, err := extractBody(resp, &itemOut)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if itemOut.Name != simpleStorage.item.Name {
		t.Errorf("Unexpected data: %#v, expected %#v (%s)", itemOut, simpleStorage.item, string(body))
	}
	if !selfLinker.called {
		t.Errorf("Never set self link")
	}
}

func TestGetMissing(t *testing.T) {
	storage := map[string]RESTStorage{}
	simpleStorage := SimpleRESTStorage{
		errors: map[string]error{"get": apierrs.NewNotFound("simple", "id")},
	}
	storage["simple"] = &simpleStorage
	handler := Handle(storage, codec, "/prefix/version", selfLinker)
	server := httptest.NewServer(handler)
	defer server.Close()

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
	handler := Handle(storage, codec, "/prefix/version", selfLinker)
	server := httptest.NewServer(handler)
	defer server.Close()

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
		errors: map[string]error{"delete": apierrs.NewNotFound("simple", ID)},
	}
	storage["simple"] = &simpleStorage
	handler := Handle(storage, codec, "/prefix/version", selfLinker)
	server := httptest.NewServer(handler)
	defer server.Close()

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
	selfLinker := &setTestSelfLinker{
		t:           t,
		expectedSet: "/prefix/version/simple/" + ID,
	}
	handler := Handle(storage, codec, "/prefix/version", selfLinker)
	server := httptest.NewServer(handler)
	defer server.Close()

	item := &Simple{
		Other: "bar",
	}
	body, err := codec.Encode(item)
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
	if !selfLinker.called {
		t.Errorf("Never set self link")
	}
}

func TestUpdateMissing(t *testing.T) {
	storage := map[string]RESTStorage{}
	ID := "id"
	simpleStorage := SimpleRESTStorage{
		errors: map[string]error{"update": apierrs.NewNotFound("simple", ID)},
	}
	storage["simple"] = &simpleStorage
	handler := Handle(storage, codec, "/prefix/version", selfLinker)
	server := httptest.NewServer(handler)
	defer server.Close()

	item := &Simple{
		Other: "bar",
	}
	body, err := codec.Encode(item)
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

func TestCreate(t *testing.T) {
	wait := sync.WaitGroup{}
	wait.Add(1)
	simpleStorage := &SimpleRESTStorage{
		injectedFunction: func(obj runtime.Object) (returnObj runtime.Object, err error) {
			wait.Wait()
			return &Simple{}, nil
		},
	}
	handler := Handle(map[string]RESTStorage{
		"foo": simpleStorage,
	}, codec, "/prefix/version", selfLinker)
	handler.(*defaultAPIServer).group.handler.asyncOpWait = 0
	server := httptest.NewServer(handler)
	defer server.Close()
	client := http.Client{}

	simple := &Simple{
		Other: "foo",
	}
	data, _ := codec.Encode(simple)
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
	wait.Done()
}

func TestCreateNotFound(t *testing.T) {
	handler := Handle(map[string]RESTStorage{
		"simple": &SimpleRESTStorage{
			// storage.Create can fail with not found error in theory.
			// See https://github.com/GoogleCloudPlatform/kubernetes/pull/486#discussion_r15037092.
			errors: map[string]error{"create": apierrs.NewNotFound("simple", "id")},
		},
	}, codec, "/prefix/version", selfLinker)
	server := httptest.NewServer(handler)
	defer server.Close()
	client := http.Client{}

	simple := &Simple{Other: "foo"}
	data, _ := codec.Encode(simple)
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

type setTestSelfLinker struct {
	t           *testing.T
	expectedSet string
	name        string
	called      bool
}

func (s *setTestSelfLinker) Name(runtime.Object) (string, error)   { return s.name, nil }
func (*setTestSelfLinker) SelfLink(runtime.Object) (string, error) { return "", nil }
func (s *setTestSelfLinker) SetSelfLink(obj runtime.Object, selfLink string) error {
	if e, a := s.expectedSet, selfLink; e != a {
		s.t.Errorf("expected '%v', got '%v'", e, a)
	}
	s.called = true
	return nil
}

func TestSyncCreate(t *testing.T) {
	storage := SimpleRESTStorage{
		injectedFunction: func(obj runtime.Object) (runtime.Object, error) {
			time.Sleep(5 * time.Millisecond)
			return obj, nil
		},
	}
	selfLinker := &setTestSelfLinker{
		t:           t,
		name:        "bar",
		expectedSet: "/prefix/version/foo/bar",
	}
	handler := Handle(map[string]RESTStorage{
		"foo": &storage,
	}, codec, "/prefix/version", selfLinker)
	server := httptest.NewServer(handler)
	defer server.Close()
	client := http.Client{}

	simple := &Simple{
		Other: "bar",
	}
	data, _ := codec.Encode(simple)
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

	if !reflect.DeepEqual(&itemOut, simple) {
		t.Errorf("Unexpected data: %#v, expected %#v (%s)", itemOut, simple, string(body))
	}
	if response.StatusCode != http.StatusOK {
		t.Errorf("Unexpected status: %d, Expected: %d, %#v", response.StatusCode, http.StatusOK, response)
	}
	if !selfLinker.called {
		t.Errorf("Never set self link")
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
		t.Fatalf("unexpected error on %s %s: %v", method, url, err)
		return nil
	}
	var status api.Status
	_, err = extractBody(response, &status)
	if err != nil {
		t.Fatalf("unexpected error on %s %s: %v", method, url, err)
		return nil
	}
	if code != response.StatusCode {
		t.Fatalf("Expected %s %s to return %d, Got %d", method, url, code, response.StatusCode)
	}
	return &status
}

func TestAsyncDelayReturnsError(t *testing.T) {
	storage := SimpleRESTStorage{
		injectedFunction: func(obj runtime.Object) (runtime.Object, error) {
			return nil, apierrs.NewAlreadyExists("foo", "bar")
		},
	}
	handler := Handle(map[string]RESTStorage{"foo": &storage}, codec, "/prefix/version", selfLinker)
	handler.(*defaultAPIServer).group.handler.asyncOpWait = time.Millisecond / 2
	server := httptest.NewServer(handler)
	defer server.Close()

	status := expectApiStatus(t, "DELETE", fmt.Sprintf("%s/prefix/version/foo/bar", server.URL), nil, http.StatusConflict)
	if status.Status != api.StatusFailure || status.Message == "" || status.Details == nil || status.Reason != api.StatusReasonAlreadyExists {
		t.Errorf("Unexpected status %#v", status)
	}
}

func TestAsyncCreateError(t *testing.T) {
	ch := make(chan struct{})
	storage := SimpleRESTStorage{
		injectedFunction: func(obj runtime.Object) (runtime.Object, error) {
			<-ch
			return nil, apierrs.NewAlreadyExists("foo", "bar")
		},
	}
	selfLinker := &setTestSelfLinker{
		t:           t,
		name:        "bar",
		expectedSet: "/prefix/version/foo/bar",
	}
	handler := Handle(map[string]RESTStorage{"foo": &storage}, codec, "/prefix/version", selfLinker)
	handler.(*defaultAPIServer).group.handler.asyncOpWait = 0
	server := httptest.NewServer(handler)
	defer server.Close()

	simple := &Simple{Other: "bar"}
	data, _ := codec.Encode(simple)

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
	expectedErr := apierrs.NewAlreadyExists("foo", "bar")
	expectedStatus := &api.Status{
		Status:  api.StatusFailure,
		Code:    http.StatusConflict,
		Reason:  "AlreadyExists",
		Message: expectedErr.Error(),
		Details: &api.StatusDetails{
			Kind: "foo",
			ID:   "bar",
		},
	}
	if !reflect.DeepEqual(expectedStatus, finalStatus) {
		t.Errorf("Expected %#v, Got %#v", expectedStatus, finalStatus)
		if finalStatus.Details != nil {
			t.Logf("Details %#v, Got %#v", *expectedStatus.Details, *finalStatus.Details)
		}
	}
	if !selfLinker.called {
		t.Errorf("Never set self link")
	}
}

type UnregisteredAPIObject struct {
	Value string
}

func (*UnregisteredAPIObject) IsAnAPIObject() {}

func TestWriteJSONDecodeError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		writeJSON(http.StatusOK, latest.Codec, &UnregisteredAPIObject{"Undecodable"}, w)
	}))
	defer server.Close()
	status := expectApiStatus(t, "GET", server.URL, nil, http.StatusInternalServerError)
	if status.Reason != api.StatusReasonUnknown {
		t.Errorf("unexpected reason %#v", status)
	}
	if !strings.Contains(status.Message, "type apiserver.UnregisteredAPIObject is not registered") {
		t.Errorf("unexpected message %#v", status)
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
	defer server.Close()
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
	testOver := make(chan struct{})
	defer close(testOver)
	storage := SimpleRESTStorage{
		injectedFunction: func(obj runtime.Object) (runtime.Object, error) {
			// Eliminate flakes by ensuring the create operation takes longer than this test.
			<-testOver
			return obj, nil
		},
	}
	handler := Handle(map[string]RESTStorage{
		"foo": &storage,
	}, codec, "/prefix/version", selfLinker)
	server := httptest.NewServer(handler)
	defer server.Close()

	simple := &Simple{Other: "foo"}
	data, _ := codec.Encode(simple)
	itemOut := expectApiStatus(t, "POST", server.URL+"/prefix/version/foo?sync=true&timeout=4ms", data, http.StatusAccepted)
	if itemOut.Status != api.StatusWorking || itemOut.Details == nil || itemOut.Details.ID == "" {
		t.Errorf("Unexpected status %#v", itemOut)
	}
}

func TestCORSAllowedOrigins(t *testing.T) {
	table := []struct {
		allowedOrigins util.StringList
		origin         string
		allowed        bool
	}{
		{[]string{}, "example.com", false},
		{[]string{"example.com"}, "example.com", true},
		{[]string{"example.com"}, "not-allowed.com", false},
		{[]string{"not-matching.com", "example.com"}, "example.com", true},
		{[]string{".*"}, "example.com", true},
	}

	for _, item := range table {
		allowedOriginRegexps, err := util.CompileRegexps(item.allowedOrigins)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}

		handler := CORS(
			Handle(map[string]RESTStorage{}, codec, "/prefix/version", selfLinker),
			allowedOriginRegexps, nil, nil, "true",
		)
		server := httptest.NewServer(handler)
		defer server.Close()
		client := http.Client{}

		request, err := http.NewRequest("GET", server.URL+"/version", nil)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		request.Header.Set("Origin", item.origin)

		response, err := client.Do(request)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}

		if item.allowed {
			if !reflect.DeepEqual(item.origin, response.Header.Get("Access-Control-Allow-Origin")) {
				t.Errorf("Expected %#v, Got %#v", item.origin, response.Header.Get("Access-Control-Allow-Origin"))
			}

			if response.Header.Get("Access-Control-Allow-Credentials") == "" {
				t.Errorf("Expected Access-Control-Allow-Credentials header to be set")
			}

			if response.Header.Get("Access-Control-Allow-Headers") == "" {
				t.Errorf("Expected Access-Control-Allow-Headers header to be set")
			}

			if response.Header.Get("Access-Control-Allow-Methods") == "" {
				t.Errorf("Expected Access-Control-Allow-Methods header to be set")
			}
		} else {
			if response.Header.Get("Access-Control-Allow-Origin") != "" {
				t.Errorf("Expected Access-Control-Allow-Origin header to not be set")
			}

			if response.Header.Get("Access-Control-Allow-Credentials") != "" {
				t.Errorf("Expected Access-Control-Allow-Credentials header to not be set")
			}

			if response.Header.Get("Access-Control-Allow-Headers") != "" {
				t.Errorf("Expected Access-Control-Allow-Headers header to not be set")
			}

			if response.Header.Get("Access-Control-Allow-Methods") != "" {
				t.Errorf("Expected Access-Control-Allow-Methods header to not be set")
			}
		}
	}
}
