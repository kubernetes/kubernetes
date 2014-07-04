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
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
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
	testServer := httptest.NewTLSServer(&fakeHandler)
	auth := AuthInfo{User: "user", Password: "pass"}
	s := New(testServer.URL, &auth)
	obj, err := s.Verb("POST").
		Path("foo/bar").
		Path("baz").
		ParseSelector("name=foo").
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
	fakeHandler.ValidateRequest(t, "/api/v1beta1/foo/bar/baz", "POST", &reqBody)
	if fakeHandler.RequestReceived.URL.RawQuery != "labels=name%3Dfoo" {
		t.Errorf("Unexpected query: %v", fakeHandler.RequestReceived.URL.RawQuery)
	}
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
	testServer := httptest.NewTLSServer(&fakeHandler)
	auth := AuthInfo{User: "user", Password: "pass"}
	s := New(testServer.URL, &auth)
	obj, err := s.Verb("POST").
		Path("foo/bar").
		Path("baz").
		Selector(labels.Set{"name": "foo"}.AsSelector()).
		Sync(false).
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
	fakeHandler.ValidateRequest(t, "/api/v1beta1/foo/bar/baz", "POST", &tmpStr)
	if fakeHandler.RequestReceived.URL.RawQuery != "labels=name%3Dfoo" {
		t.Errorf("Unexpected query: %v", fakeHandler.RequestReceived.URL.RawQuery)
	}
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
	testServer := httptest.NewTLSServer(&fakeHandler)
	auth := AuthInfo{User: "user", Password: "pass"}
	s := New(testServer.URL, &auth)
	obj, err := s.Verb("POST").
		Path("foo/bar").
		Path("baz").
		Selector(labels.Set{"name": "foo"}.AsSelector()).
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
	fakeHandler.ValidateRequest(t, "/api/v1beta1/foo/bar/baz", "POST", &tmpStr)
	if fakeHandler.RequestReceived.URL.RawQuery != "labels=name%3Dfoo" {
		t.Errorf("Unexpected query: %v", fakeHandler.RequestReceived.URL.RawQuery)
	}
	if fakeHandler.RequestReceived.Header["Authorization"] == nil {
		t.Errorf("Request is missing authorization header: %#v", *fakeHandler.RequestReceived)
	}
}

func TestDoRequestNewWayFile(t *testing.T) {
	reqObj := &api.Pod{JSONBase: api.JSONBase{ID: "foo"}}
	reqBodyExpected, err := api.Encode(reqObj)
	expectNoError(t, err)
	file, err := ioutil.TempFile("", "foo")
	expectNoError(t, err)
	_, err = file.Write(reqBodyExpected)
	expectNoError(t, err)

	expectedObj := &api.Service{Port: 12345}
	expectedBody, _ := api.Encode(expectedObj)
	fakeHandler := util.FakeHandler{
		StatusCode:   200,
		ResponseBody: string(expectedBody),
		T:            t,
	}
	testServer := httptest.NewTLSServer(&fakeHandler)
	auth := AuthInfo{User: "user", Password: "pass"}
	s := New(testServer.URL, &auth)
	obj, err := s.Verb("POST").
		Path("foo/bar").
		Path("baz").
		ParseSelector("name=foo").
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
	fakeHandler.ValidateRequest(t, "/api/v1beta1/foo/bar/baz", "POST", &tmpStr)
	if fakeHandler.RequestReceived.URL.RawQuery != "labels=name%3Dfoo" {
		t.Errorf("Unexpected query: %v", fakeHandler.RequestReceived.URL.RawQuery)
	}
	if fakeHandler.RequestReceived.Header["Authorization"] == nil {
		t.Errorf("Request is missing authorization header: %#v", *fakeHandler.RequestReceived)
	}
}

func TestVerbs(t *testing.T) {
	c := New("", nil)
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
	c := New("", nil)
	r := c.Post().Path("/foo").AbsPath(expectedPath)
	if r.path != expectedPath {
		t.Errorf("unexpected path: %s, expected %s", r.path, expectedPath)
	}
}

func TestSync(t *testing.T) {
	c := New("", nil)
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

func TestSetPollPeriod(t *testing.T) {
	c := New("", nil)
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
		&api.Status{Status: api.StatusWorking, Details: "1234"},
		&api.Status{Status: api.StatusWorking, Details: "1234"},
		&api.Status{Status: api.StatusWorking, Details: "1234"},
		&api.Status{Status: api.StatusWorking, Details: "1234"},
		&api.Status{Status: api.StatusSuccess},
	}

	callNumber := 0
	testServer := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		data, err := api.Encode(objects[callNumber])
		if err != nil {
			t.Errorf("Unexpected encode error")
		}
		callNumber++
		w.Write(data)
	}))

	auth := AuthInfo{User: "user", Password: "pass"}
	s := New(testServer.URL, &auth)

	trials := []func(){
		func() {
			// Check that we do indeed poll when asked to.
			obj, err := s.Get().PollPeriod(5 * time.Millisecond).Do().Get()
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
			obj, err := s.Get().PollPeriod(0).Do().Get()
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
