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
	"io/ioutil"
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
	if fakeHandler.RequestReceived.URL.RawQuery != "labels=name%3Dfoo&timeout=1s" {
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
	if fakeHandler.RequestReceived.URL.RawQuery != "labels=name%3Dfoo&timeout=1s" {
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
	if fakeHandler.RequestReceived.URL.RawQuery != "labels=name%3Dfoo&timeout=1s" {
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
