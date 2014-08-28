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

package client

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

func TestDoRequestNewWay(t *testing.T) {
	reqBody := "request body"
	expectedObj := &api.Service{Port: 12345}
	expectedBody, _ := api.Encode(expectedObj)
	fakeHandler := util.FakeHandler{
		StatusCode:   200,
		ResponseBody: string(expectedBody),
		T:            t,
	}
	testServer := httptest.NewServer(&fakeHandler)
	auth := AuthInfo{User: "user", Password: "pass"}
	c := NewOrDie(testServer.URL, &auth)
	obj, err := c.Verb("POST").
		Path("foo/bar").
		Path("baz").
		ParseSelectorParam("labels", "name=foo").
		Timeout(time.Second).
		Body([]byte(reqBody)).
		Do().Get()
	if err != nil {
		t.Errorf("Unexpected error: %v %#v", err, err)
		return
	}
	if obj == nil {
		t.Error("nil obj")
	} else if !reflect.DeepEqual(obj, expectedObj) {
		t.Errorf("Expected: %#v, got %#v", expectedObj, obj)
	}
	fakeHandler.ValidateRequest(t, "/api/v1beta1/foo/bar/baz?labels=name%3Dfoo", "POST", &reqBody)
	if fakeHandler.RequestReceived.Header["Authorization"] == nil {
		t.Errorf("Request is missing authorization header: %#v", *fakeHandler.RequestReceived)
	}
}

func TestDoRequestNewWayReader(t *testing.T) {
	reqObj := &api.Pod{JSONBase: api.JSONBase{ID: "foo"}}
	reqBodyExpected, _ := api.Encode(reqObj)
	expectedObj := &api.Service{Port: 12345}
	expectedBody, _ := api.Encode(expectedObj)
	fakeHandler := util.FakeHandler{
		StatusCode:   200,
		ResponseBody: string(expectedBody),
		T:            t,
	}
	testServer := httptest.NewServer(&fakeHandler)
	auth := AuthInfo{User: "user", Password: "pass"}
	c := NewOrDie(testServer.URL, &auth)
	obj, err := c.Verb("POST").
		Path("foo/bar").
		Path("baz").
		SelectorParam("labels", labels.Set{"name": "foo"}.AsSelector()).
		Sync(true).
		Timeout(time.Second).
		Body(bytes.NewBuffer(reqBodyExpected)).
		Do().Get()
	if err != nil {
		t.Errorf("Unexpected error: %v %#v", err, err)
		return
	}
	if obj == nil {
		t.Error("nil obj")
	} else if !reflect.DeepEqual(obj, expectedObj) {
		t.Errorf("Expected: %#v, got %#v", expectedObj, obj)
	}
	tmpStr := string(reqBodyExpected)
	fakeHandler.ValidateRequest(t, "/api/v1beta1/foo/bar/baz?labels=name%3Dfoo&sync=true&timeout=1s", "POST", &tmpStr)
	if fakeHandler.RequestReceived.Header["Authorization"] == nil {
		t.Errorf("Request is missing authorization header: %#v", *fakeHandler.RequestReceived)
	}
}

func TestDoRequestNewWayObj(t *testing.T) {
	reqObj := &api.Pod{JSONBase: api.JSONBase{ID: "foo"}}
	reqBodyExpected, _ := api.Encode(reqObj)
	expectedObj := &api.Service{Port: 12345}
	expectedBody, _ := api.Encode(expectedObj)
	fakeHandler := util.FakeHandler{
		StatusCode:   200,
		ResponseBody: string(expectedBody),
		T:            t,
	}
	testServer := httptest.NewServer(&fakeHandler)
	auth := AuthInfo{User: "user", Password: "pass"}
	c := NewOrDie(testServer.URL, &auth)
	obj, err := c.Verb("POST").
		Path("foo/bar").
		Path("baz").
		SelectorParam("labels", labels.Set{"name": "foo"}.AsSelector()).
		Timeout(time.Second).
		Body(reqObj).
		Do().Get()
	if err != nil {
		t.Errorf("Unexpected error: %v %#v", err, err)
		return
	}
	if obj == nil {
		t.Error("nil obj")
	} else if !reflect.DeepEqual(obj, expectedObj) {
		t.Errorf("Expected: %#v, got %#v", expectedObj, obj)
	}
	tmpStr := string(reqBodyExpected)
	fakeHandler.ValidateRequest(t, "/api/v1beta1/foo/bar/baz?labels=name%3Dfoo", "POST", &tmpStr)
	if fakeHandler.RequestReceived.Header["Authorization"] == nil {
		t.Errorf("Request is missing authorization header: %#v", *fakeHandler.RequestReceived)
	}
}

func TestDoRequestNewWayFile(t *testing.T) {
	reqObj := &api.Pod{JSONBase: api.JSONBase{ID: "foo"}}
	reqBodyExpected, err := api.Encode(reqObj)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	file, err := ioutil.TempFile("", "foo")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	_, err = file.Write(reqBodyExpected)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	expectedObj := &api.Service{Port: 12345}
	expectedBody, _ := api.Encode(expectedObj)
	fakeHandler := util.FakeHandler{
		StatusCode:   200,
		ResponseBody: string(expectedBody),
		T:            t,
	}
	testServer := httptest.NewServer(&fakeHandler)
	auth := AuthInfo{User: "user", Password: "pass"}
	c := NewOrDie(testServer.URL, &auth)
	obj, err := c.Verb("POST").
		Path("foo/bar").
		Path("baz").
		ParseSelectorParam("labels", "name=foo").
		Timeout(time.Second).
		Body(file.Name()).
		Do().Get()
	if err != nil {
		t.Errorf("Unexpected error: %v %#v", err, err)
		return
	}
	if obj == nil {
		t.Error("nil obj")
	} else if !reflect.DeepEqual(obj, expectedObj) {
		t.Errorf("Expected: %#v, got %#v", expectedObj, obj)
	}
	tmpStr := string(reqBodyExpected)
	fakeHandler.ValidateRequest(t, "/api/v1beta1/foo/bar/baz?labels=name%3Dfoo", "POST", &tmpStr)
	if fakeHandler.RequestReceived.Header["Authorization"] == nil {
		t.Errorf("Request is missing authorization header: %#v", *fakeHandler.RequestReceived)
	}
}

func TestVerbs(t *testing.T) {
	c := NewOrDie("localhost", nil)
	if r := c.Post(); r.verb != "POST" {
		t.Errorf("Post verb is wrong")
	}
	if r := c.Put(); r.verb != "PUT" {
		t.Errorf("Put verb is wrong")
	}
	if r := c.Get(); r.verb != "GET" {
		t.Errorf("Get verb is wrong")
	}
	if r := c.Delete(); r.verb != "DELETE" {
		t.Errorf("Delete verb is wrong")
	}
}

func TestAbsPath(t *testing.T) {
	expectedPath := "/bar/foo"
	c := NewOrDie("localhost", nil)
	r := c.Post().Path("/foo").AbsPath(expectedPath)
	if r.path != expectedPath {
		t.Errorf("unexpected path: %s, expected %s", r.path, expectedPath)
	}
}

func TestSync(t *testing.T) {
	c := NewOrDie("localhost", nil)
	r := c.Get()
	if r.sync {
		t.Errorf("sync has wrong default")
	}
	r.Sync(false)
	if r.sync {
		t.Errorf("'Sync' doesn't work")
	}
	r.Sync(true)
	if !r.sync {
		t.Errorf("'Sync' doesn't work")
	}
}

func TestUintParam(t *testing.T) {
	table := []struct {
		name      string
		testVal   uint64
		expectStr string
	}{
		{"foo", 31415, "http://localhost?foo=31415"},
		{"bar", 42, "http://localhost?bar=42"},
		{"baz", 0, "http://localhost?baz=0"},
	}

	for _, item := range table {
		c := NewOrDie("localhost", nil)
		r := c.Get().AbsPath("").UintParam(item.name, item.testVal)
		if e, a := item.expectStr, r.finalURL(); e != a {
			t.Errorf("expected %v, got %v", e, a)
		}
	}
}

func TestUnacceptableParamNames(t *testing.T) {
	table := []struct {
		name          string
		testVal       string
		expectSuccess bool
	}{
		{"sync", "foo", false},
		{"timeout", "42", false},
	}

	for _, item := range table {
		c := NewOrDie("localhost", nil)
		r := c.Get().setParam(item.name, item.testVal)
		if e, a := item.expectSuccess, r.err == nil; e != a {
			t.Errorf("expected %v, got %v (%v)", e, a, r.err)
		}
	}
}

func TestSetPollPeriod(t *testing.T) {
	c := NewOrDie("localhost", nil)
	r := c.Get()
	if r.pollPeriod == 0 {
		t.Errorf("polling should be on by default")
	}
	r.PollPeriod(time.Hour)
	if r.pollPeriod != time.Hour {
		t.Errorf("'PollPeriod' doesn't work")
	}
}

func TestPolling(t *testing.T) {
	objects := []interface{}{
		&api.Status{Status: api.StatusWorking, Details: &api.StatusDetails{ID: "1234"}},
		&api.Status{Status: api.StatusWorking, Details: &api.StatusDetails{ID: "1234"}},
		&api.Status{Status: api.StatusWorking, Details: &api.StatusDetails{ID: "1234"}},
		&api.Status{Status: api.StatusWorking, Details: &api.StatusDetails{ID: "1234"}},
		&api.Status{Status: api.StatusSuccess},
	}

	callNumber := 0
	testServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		data, err := api.Encode(objects[callNumber])
		if err != nil {
			t.Errorf("Unexpected encode error")
		}
		callNumber++
		w.Write(data)
	}))

	auth := AuthInfo{User: "user", Password: "pass"}
	c := NewOrDie(testServer.URL, &auth)

	trials := []func(){
		func() {
			// Check that we do indeed poll when asked to.
			obj, err := c.Get().PollPeriod(5 * time.Millisecond).Do().Get()
			if err != nil {
				t.Errorf("Unexpected error: %v %#v", err, err)
				return
			}
			if s, ok := obj.(*api.Status); !ok || s.Status != api.StatusSuccess {
				t.Errorf("Unexpected return object: %#v", obj)
				return
			}
			if callNumber != len(objects) {
				t.Errorf("Unexpected number of calls: %v", callNumber)
			}
		},
		func() {
			// Check that we don't poll when asked not to.
			obj, err := c.Get().PollPeriod(0).Do().Get()
			if err == nil {
				t.Errorf("Unexpected non error: %v", obj)
				return
			}
			if se, ok := err.(*StatusErr); !ok || se.Status.Status != api.StatusWorking {
				t.Errorf("Unexpected kind of error: %#v", err)
				return
			}
			if callNumber != 1 {
				t.Errorf("Unexpected number of calls: %v", callNumber)
			}
		},
	}
	for _, f := range trials {
		callNumber = 0
		f()
	}
}

func authFromReq(r *http.Request) (*AuthInfo, bool) {
	auth, ok := r.Header["Authorization"]
	if !ok {
		return nil, false
	}

	encoded := strings.Split(auth[0], " ")
	if len(encoded) != 2 || encoded[0] != "Basic" {
		return nil, false
	}

	decoded, err := base64.StdEncoding.DecodeString(encoded[1])
	if err != nil {
		return nil, false
	}
	parts := strings.Split(string(decoded), ":")
	if len(parts) != 2 {
		return nil, false
	}
	return &AuthInfo{User: parts[0], Password: parts[1]}, true
}

// checkAuth sets errors if the auth found in r doesn't match the expectation.
// TODO: Move to util, test in more places.
func checkAuth(t *testing.T, expect AuthInfo, r *http.Request) {
	foundAuth, found := authFromReq(r)
	if !found {
		t.Errorf("no auth found")
	} else if e, a := expect, *foundAuth; !reflect.DeepEqual(e, a) {
		t.Fatalf("Wrong basic auth: wanted %#v, got %#v", e, a)
	}
}

func TestWatch(t *testing.T) {
	var table = []struct {
		t   watch.EventType
		obj interface{}
	}{
		{watch.Added, &api.Pod{JSONBase: api.JSONBase{ID: "first"}}},
		{watch.Modified, &api.Pod{JSONBase: api.JSONBase{ID: "second"}}},
		{watch.Deleted, &api.Pod{JSONBase: api.JSONBase{ID: "third"}}},
	}

	auth := AuthInfo{User: "user", Password: "pass"}
	testServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		checkAuth(t, auth, r)
		flusher, ok := w.(http.Flusher)
		if !ok {
			panic("need flusher!")
		}

		w.Header().Set("Transfer-Encoding", "chunked")
		w.WriteHeader(http.StatusOK)
		flusher.Flush()

		encoder := json.NewEncoder(w)
		for _, item := range table {
			encoder.Encode(&api.WatchEvent{item.t, api.APIObject{item.obj}})
			flusher.Flush()
		}
	}))

	s, err := New(testServer.URL, &auth)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	watching, err := s.Get().Path("path/to/watch/thing").Watch()
	if err != nil {
		t.Fatalf("Unexpected error")
	}

	for _, item := range table {
		got, ok := <-watching.ResultChan()
		if !ok {
			t.Fatalf("Unexpected early close")
		}
		if e, a := item.t, got.Type; e != a {
			t.Errorf("Expected %v, got %v", e, a)
		}
		if e, a := item.obj, got.Object; !reflect.DeepEqual(e, a) {
			t.Errorf("Expected %v, got %v", e, a)
		}
	}

	_, ok := <-watching.ResultChan()
	if ok {
		t.Fatal("Unexpected non-close")
	}
}
