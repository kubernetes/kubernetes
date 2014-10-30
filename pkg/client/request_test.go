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
	"errors"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/latest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/testapi"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1beta1"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1beta2"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
	watchjson "github.com/GoogleCloudPlatform/kubernetes/pkg/watch/json"
)

func skipPolling(name string) (*Request, bool) {
	return nil, false
}

func TestRequestWithErrorWontChange(t *testing.T) {
	original := Request{err: errors.New("test")}
	r := original
	changed := r.Param("foo", "bar").
		SelectorParam("labels", labels.Set{"a": "b"}.AsSelector()).
		UintParam("uint", 1).
		AbsPath("/abs").
		Path("test").
		ParseSelectorParam("foo", "a=b").
		Namespace("new").
		NoPoll().
		Body("foo").
		Poller(skipPolling).
		Timeout(time.Millisecond).
		Sync(true)
	if changed != &r {
		t.Errorf("returned request should point to the same object")
	}
	if !reflect.DeepEqual(&original, changed) {
		t.Errorf("expected %#v, got %#v", &original, changed)
	}
}

func TestRequestParseSelectorParam(t *testing.T) {
	r := (&Request{}).ParseSelectorParam("foo", "a")
	if r.err == nil || r.params != nil {
		t.Errorf("should have set err and left params nil: %#v", r)
	}
}

func TestRequestParam(t *testing.T) {
	r := (&Request{}).Param("foo", "a")
	if !reflect.DeepEqual(map[string]string{"foo": "a"}, r.params) {
		t.Errorf("should have set a param: %#v", r)
	}
}

type NotAnAPIObject struct{}

func (NotAnAPIObject) IsAnAPIObject() {}

func TestRequestBody(t *testing.T) {
	// test unknown type
	r := (&Request{}).Body([]string{"test"})
	if r.err == nil || r.body != nil {
		t.Errorf("should have set err and left body nil: %#v", r)
	}

	// test error set when failing to read file
	f, err := ioutil.TempFile("", "test")
	if err != nil {
		t.Fatalf("unable to create temp file")
	}
	os.Remove(f.Name())
	r = (&Request{}).Body(f.Name())
	if r.err == nil || r.body != nil {
		t.Errorf("should have set err and left body nil: %#v", r)
	}

	// test unencodable api object
	r = (&Request{codec: latest.Codec}).Body(&NotAnAPIObject{})
	if r.err == nil || r.body != nil {
		t.Errorf("should have set err and left body nil: %#v", r)
	}
}

func TestResultIntoWithErrReturnsErr(t *testing.T) {
	res := Result{err: errors.New("test")}
	if err := res.Into(&api.Pod{}); err != res.err {
		t.Errorf("should have returned exact error from result")
	}
}

func TestTransformResponse(t *testing.T) {
	invalid := []byte("aaaaa")
	uri, _ := url.Parse("http://localhost")
	testCases := []struct {
		Response *http.Response
		Data     []byte
		Created  bool
		Error    bool
	}{
		{Response: &http.Response{StatusCode: 200}, Data: []byte{}},
		{Response: &http.Response{StatusCode: 201}, Data: []byte{}, Created: true},
		{Response: &http.Response{StatusCode: 199}, Error: true},
		{Response: &http.Response{StatusCode: 500}, Error: true},
		{Response: &http.Response{StatusCode: 200, Body: ioutil.NopCloser(bytes.NewReader(invalid))}, Data: invalid},
		{Response: &http.Response{StatusCode: 200, Body: ioutil.NopCloser(bytes.NewReader(invalid))}, Data: invalid},
	}
	for i, test := range testCases {
		r := NewRequest(nil, "", uri, testapi.Codec())
		if test.Response.Body == nil {
			test.Response.Body = ioutil.NopCloser(bytes.NewReader([]byte{}))
		}
		response, created, err := r.transformResponse(test.Response, &http.Request{})
		hasErr := err != nil
		if hasErr != test.Error {
			t.Errorf("%d: unexpected error: %f %v", i, test.Error, err)
		}
		if !(test.Data == nil && response == nil) && !reflect.DeepEqual(test.Data, response) {
			t.Errorf("%d: unexpected response: %#v %#v", i, test.Data, response)
		}
		if test.Created != created {
			t.Errorf("%d: expected created %f, got %f", i, test.Created, created)
		}
	}
}

type clientFunc func(req *http.Request) (*http.Response, error)

func (f clientFunc) Do(req *http.Request) (*http.Response, error) {
	return f(req)
}

func TestRequestWatch(t *testing.T) {
	testCases := []struct {
		Request *Request
		Err     bool
	}{
		{
			Request: &Request{err: errors.New("bail")},
			Err:     true,
		},
		{
			Request: &Request{baseURL: &url.URL{}, path: "%"},
			Err:     true,
		},
		{
			Request: &Request{
				client: clientFunc(func(req *http.Request) (*http.Response, error) {
					return nil, errors.New("err")
				}),
				baseURL: &url.URL{},
			},
			Err: true,
		},
		{
			Request: &Request{
				client: clientFunc(func(req *http.Request) (*http.Response, error) {
					return &http.Response{StatusCode: http.StatusForbidden}, nil
				}),
				baseURL: &url.URL{},
			},
			Err: true,
		},
	}
	for i, testCase := range testCases {
		watch, err := testCase.Request.Watch()
		hasErr := err != nil
		if hasErr != testCase.Err {
			t.Errorf("%d: expected %f, got %f: %v", i, testCase.Err, hasErr, err)
		}
		if hasErr && watch != nil {
			t.Errorf("%d: watch should be nil when error is returned", i)
		}
	}
}

func TestRequestStream(t *testing.T) {
	testCases := []struct {
		Request *Request
		Err     bool
	}{
		{
			Request: &Request{err: errors.New("bail")},
			Err:     true,
		},
		{
			Request: &Request{baseURL: &url.URL{}, path: "%"},
			Err:     true,
		},
		{
			Request: &Request{
				client: clientFunc(func(req *http.Request) (*http.Response, error) {
					return nil, errors.New("err")
				}),
				baseURL: &url.URL{},
			},
			Err: true,
		},
	}
	for i, testCase := range testCases {
		body, err := testCase.Request.Stream()
		hasErr := err != nil
		if hasErr != testCase.Err {
			t.Errorf("%d: expected %f, got %f: %v", i, testCase.Err, hasErr, err)
		}
		if hasErr && body != nil {
			t.Errorf("%d: body should be nil when error is returned", i)
		}
	}
}

func TestRequestDo(t *testing.T) {
	testCases := []struct {
		Request *Request
		Err     bool
	}{
		{
			Request: &Request{err: errors.New("bail")},
			Err:     true,
		},
		{
			Request: &Request{baseURL: &url.URL{}, path: "%"},
			Err:     true,
		},
		{
			Request: &Request{
				client: clientFunc(func(req *http.Request) (*http.Response, error) {
					return nil, errors.New("err")
				}),
				baseURL: &url.URL{},
			},
			Err: true,
		},
	}
	for i, testCase := range testCases {
		body, err := testCase.Request.Do().Raw()
		hasErr := err != nil
		if hasErr != testCase.Err {
			t.Errorf("%d: expected %f, got %f: %v", i, testCase.Err, hasErr, err)
		}
		if hasErr && body != nil {
			t.Errorf("%d: body should be nil when error is returned", i)
		}
	}
}

func TestDoRequestNewWay(t *testing.T) {
	reqBody := "request body"
	expectedObj := &api.Service{Spec: api.ServiceSpec{Port: 12345}}
	expectedBody, _ := v1beta2.Codec.Encode(expectedObj)
	fakeHandler := util.FakeHandler{
		StatusCode:   200,
		ResponseBody: string(expectedBody),
		T:            t,
	}
	testServer := httptest.NewServer(&fakeHandler)
	defer testServer.Close()
	c := NewOrDie(&Config{Host: testServer.URL, Version: "v1beta2", Username: "user", Password: "pass"})
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
	fakeHandler.ValidateRequest(t, "/api/v1beta2/foo/bar/baz?labels=name%3Dfoo", "POST", &reqBody)
	if fakeHandler.RequestReceived.Header["Authorization"] == nil {
		t.Errorf("Request is missing authorization header: %#v", *fakeHandler.RequestReceived)
	}
}

func TestDoRequestNewWayReader(t *testing.T) {
	reqObj := &api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo"}}
	reqBodyExpected, _ := v1beta1.Codec.Encode(reqObj)
	expectedObj := &api.Service{Spec: api.ServiceSpec{Port: 12345}}
	expectedBody, _ := v1beta1.Codec.Encode(expectedObj)
	fakeHandler := util.FakeHandler{
		StatusCode:   200,
		ResponseBody: string(expectedBody),
		T:            t,
	}
	testServer := httptest.NewServer(&fakeHandler)
	c := NewOrDie(&Config{Host: testServer.URL, Version: "v1beta1", Username: "user", Password: "pass"})
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
	reqObj := &api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo"}}
	reqBodyExpected, _ := v1beta2.Codec.Encode(reqObj)
	expectedObj := &api.Service{Spec: api.ServiceSpec{Port: 12345}}
	expectedBody, _ := v1beta2.Codec.Encode(expectedObj)
	fakeHandler := util.FakeHandler{
		StatusCode:   200,
		ResponseBody: string(expectedBody),
		T:            t,
	}
	testServer := httptest.NewServer(&fakeHandler)
	c := NewOrDie(&Config{Host: testServer.URL, Version: "v1beta2", Username: "user", Password: "pass"})
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
	fakeHandler.ValidateRequest(t, "/api/v1beta2/foo/bar/baz?labels=name%3Dfoo", "POST", &tmpStr)
	if fakeHandler.RequestReceived.Header["Authorization"] == nil {
		t.Errorf("Request is missing authorization header: %#v", *fakeHandler.RequestReceived)
	}
}

func TestDoRequestNewWayFile(t *testing.T) {
	reqObj := &api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo"}}
	reqBodyExpected, err := v1beta1.Codec.Encode(reqObj)
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

	expectedObj := &api.Service{Spec: api.ServiceSpec{Port: 12345}}
	expectedBody, _ := v1beta1.Codec.Encode(expectedObj)
	fakeHandler := util.FakeHandler{
		StatusCode:   200,
		ResponseBody: string(expectedBody),
		T:            t,
	}
	testServer := httptest.NewServer(&fakeHandler)
	c := NewOrDie(&Config{Host: testServer.URL, Version: "v1beta1", Username: "user", Password: "pass"})
	wasCreated := true
	obj, err := c.Verb("POST").
		Path("foo/bar").
		Path("baz").
		ParseSelectorParam("labels", "name=foo").
		Timeout(time.Second).
		Body(file.Name()).
		Do().WasCreated(&wasCreated).Get()
	if err != nil {
		t.Errorf("Unexpected error: %v %#v", err, err)
		return
	}
	if obj == nil {
		t.Error("nil obj")
	} else if !reflect.DeepEqual(obj, expectedObj) {
		t.Errorf("Expected: %#v, got %#v", expectedObj, obj)
	}
	if wasCreated {
		t.Errorf("expected object was not created")
	}
	tmpStr := string(reqBodyExpected)
	fakeHandler.ValidateRequest(t, "/api/v1beta1/foo/bar/baz?labels=name%3Dfoo", "POST", &tmpStr)
	if fakeHandler.RequestReceived.Header["Authorization"] == nil {
		t.Errorf("Request is missing authorization header: %#v", *fakeHandler.RequestReceived)
	}
}

func TestWasCreated(t *testing.T) {
	reqObj := &api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo"}}
	reqBodyExpected, err := v1beta1.Codec.Encode(reqObj)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	expectedObj := &api.Service{Spec: api.ServiceSpec{Port: 12345}}
	expectedBody, _ := v1beta1.Codec.Encode(expectedObj)
	fakeHandler := util.FakeHandler{
		StatusCode:   201,
		ResponseBody: string(expectedBody),
		T:            t,
	}
	testServer := httptest.NewServer(&fakeHandler)
	c := NewOrDie(&Config{Host: testServer.URL, Version: "v1beta1", Username: "user", Password: "pass"})
	wasCreated := false
	obj, err := c.Verb("PUT").
		Path("foo/bar").
		Path("baz").
		ParseSelectorParam("labels", "name=foo").
		Timeout(time.Second).
		Body(reqBodyExpected).
		Do().WasCreated(&wasCreated).Get()
	if err != nil {
		t.Errorf("Unexpected error: %v %#v", err, err)
		return
	}
	if obj == nil {
		t.Error("nil obj")
	} else if !reflect.DeepEqual(obj, expectedObj) {
		t.Errorf("Expected: %#v, got %#v", expectedObj, obj)
	}
	if !wasCreated {
		t.Errorf("Expected object was created")
	}

	tmpStr := string(reqBodyExpected)
	fakeHandler.ValidateRequest(t, "/api/v1beta1/foo/bar/baz?labels=name%3Dfoo", "PUT", &tmpStr)
	if fakeHandler.RequestReceived.Header["Authorization"] == nil {
		t.Errorf("Request is missing authorization header: %#v", *fakeHandler.RequestReceived)
	}
}

func TestVerbs(t *testing.T) {
	c := NewOrDie(&Config{})
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
	c := NewOrDie(&Config{})
	r := c.Post().Path("/foo").AbsPath(expectedPath)
	if r.path != expectedPath {
		t.Errorf("unexpected path: %s, expected %s", r.path, expectedPath)
	}
}

func TestSync(t *testing.T) {
	c := NewOrDie(&Config{})
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
		c := NewOrDie(&Config{})
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
		c := NewOrDie(&Config{})
		r := c.Get().setParam(item.name, item.testVal)
		if e, a := item.expectSuccess, r.err == nil; e != a {
			t.Errorf("expected %v, got %v (%v)", e, a, r.err)
		}
	}
}

func TestBody(t *testing.T) {
	const data = "test payload"

	f, err := ioutil.TempFile("", "test_body")
	if err != nil {
		t.Fatalf("TempFile error: %v", err)
	}
	if _, err := f.WriteString(data); err != nil {
		t.Fatalf("TempFile.WriteString error: %v", err)
	}
	f.Close()

	c := NewOrDie(&Config{})
	tests := []interface{}{[]byte(data), f.Name(), strings.NewReader(data)}
	for i, tt := range tests {
		r := c.Post().Body(tt)
		if r.err != nil {
			t.Errorf("%d: r.Body(%#v) error: %v", i, tt, r.err)
			continue
		}
		buf := make([]byte, len(data))
		if _, err := r.body.Read(buf); err != nil {
			t.Errorf("%d: r.body.Read error: %v", i, err)
			continue
		}
		body := string(buf)
		if body != data {
			t.Errorf("%d: r.body = %q; want %q", i, body, data)
		}
	}
}

func TestSetPoller(t *testing.T) {
	c := NewOrDie(&Config{})
	r := c.Get()
	if c.PollPeriod == 0 {
		t.Errorf("polling should be on by default")
	}
	if r.poller == nil {
		t.Errorf("polling should be on by default")
	}
	r.NoPoll()
	if r.poller != nil {
		t.Errorf("'NoPoll' doesn't work")
	}
}

func TestPolling(t *testing.T) {
	objects := []runtime.Object{
		&api.Status{Status: api.StatusWorking, Details: &api.StatusDetails{ID: "1234"}},
		&api.Status{Status: api.StatusWorking, Details: &api.StatusDetails{ID: "1234"}},
		&api.Status{Status: api.StatusWorking, Details: &api.StatusDetails{ID: "1234"}},
		&api.Status{Status: api.StatusWorking, Details: &api.StatusDetails{ID: "1234"}},
		&api.Status{Status: api.StatusSuccess},
	}

	callNumber := 0
	testServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if callNumber == 0 {
			if r.URL.Path != "/api/v1beta1/" {
				t.Fatalf("unexpected request URL path %s", r.URL.Path)
			}
		} else {
			if r.URL.Path != "/api/v1beta1/operations/1234" {
				t.Fatalf("unexpected request URL path %s", r.URL.Path)
			}
		}
		t.Logf("About to write %d", callNumber)
		data, err := v1beta1.Codec.Encode(objects[callNumber])
		if err != nil {
			t.Errorf("Unexpected encode error")
		}
		callNumber++
		w.Write(data)
	}))

	c := NewOrDie(&Config{Host: testServer.URL, Version: "v1beta1", Username: "user", Password: "pass"})
	c.PollPeriod = 1 * time.Millisecond
	trials := []func(){
		func() {
			// Check that we do indeed poll when asked to.
			obj, err := c.Get().Do().Get()
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
			obj, err := c.Get().NoPoll().Do().Get()
			if err == nil {
				t.Errorf("Unexpected non error: %v", obj)
				return
			}
			if se, ok := err.(APIStatus); !ok || se.Status().Status != api.StatusWorking {
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

func authFromReq(r *http.Request) (*Config, bool) {
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
	return &Config{Username: parts[0], Password: parts[1]}, true
}

// checkAuth sets errors if the auth found in r doesn't match the expectation.
// TODO: Move to util, test in more places.
func checkAuth(t *testing.T, expect *Config, r *http.Request) {
	foundAuth, found := authFromReq(r)
	if !found {
		t.Errorf("no auth found")
	} else if e, a := expect, foundAuth; !reflect.DeepEqual(e, a) {
		t.Fatalf("Wrong basic auth: wanted %#v, got %#v", e, a)
	}
}

func TestWatch(t *testing.T) {
	var table = []struct {
		t   watch.EventType
		obj runtime.Object
	}{
		{watch.Added, &api.Pod{ObjectMeta: api.ObjectMeta{Name: "first"}}},
		{watch.Modified, &api.Pod{ObjectMeta: api.ObjectMeta{Name: "second"}}},
		{watch.Deleted, &api.Pod{ObjectMeta: api.ObjectMeta{Name: "last"}}},
	}

	auth := &Config{Username: "user", Password: "pass"}
	testServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		checkAuth(t, auth, r)
		flusher, ok := w.(http.Flusher)
		if !ok {
			panic("need flusher!")
		}

		w.Header().Set("Transfer-Encoding", "chunked")
		w.WriteHeader(http.StatusOK)
		flusher.Flush()

		encoder := watchjson.NewEncoder(w, latest.Codec)
		for _, item := range table {
			if err := encoder.Encode(&watch.Event{item.t, item.obj}); err != nil {
				panic(err)
			}
			flusher.Flush()
		}
	}))

	s, err := New(&Config{
		Host:     testServer.URL,
		Version:  "v1beta1",
		Username: "user",
		Password: "pass",
	})
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

func TestStream(t *testing.T) {
	auth := &Config{Username: "user", Password: "pass"}
	expectedBody := "expected body"

	testServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		checkAuth(t, auth, r)
		flusher, ok := w.(http.Flusher)
		if !ok {
			panic("need flusher!")
		}
		w.Header().Set("Transfer-Encoding", "chunked")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(expectedBody))
		flusher.Flush()
	}))

	s, err := New(&Config{
		Host:     testServer.URL,
		Version:  "v1beta1",
		Username: "user",
		Password: "pass",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	readCloser, err := s.Get().Path("path/to/stream/thing").Stream()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	defer readCloser.Close()
	buf := new(bytes.Buffer)
	buf.ReadFrom(readCloser)
	resultBody := buf.String()

	if expectedBody != resultBody {
		t.Errorf("Expected %s, got %s", expectedBody, resultBody)
	}
}
