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
	"net/http"
	"net/http/httptest"
	"net/url"
	"reflect"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/latest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1beta1"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1beta2"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

func TestChecksCodec(t *testing.T) {
	testCases := map[string]struct {
		Err    bool
		Prefix string
		Codec  runtime.Codec
	}{
		"v1beta1": {false, "/v1beta1/", v1beta1.Codec},
		"":        {false, "/v1beta1/", v1beta1.Codec},
		"v1beta2": {false, "/v1beta2/", v1beta2.Codec},
		"v1beta3": {true, "", nil},
	}
	for version, expected := range testCases {
		client, err := RESTClientFor(&Config{Host: "127.0.0.1", Version: version})
		switch {
		case err == nil && expected.Err:
			t.Errorf("expected error but was nil")
			continue
		case err != nil && !expected.Err:
			t.Errorf("unexpected error %v", err)
			continue
		case err != nil:
			continue
		}
		if e, a := expected.Prefix, client.baseURL.Path; e != a {
			t.Errorf("expected %#v, got %#v", e, a)
		}
		if e, a := expected.Codec, client.Codec; e != a {
			t.Errorf("expected %#v, got %#v", e, a)
		}
	}
}

func TestValidatesHostParameter(t *testing.T) {
	testCases := []struct {
		Host   string
		Prefix string

		URL string
		Err bool
	}{
		{"127.0.0.1", "", "http://127.0.0.1/v1beta1/", false},
		{"127.0.0.1:8080", "", "http://127.0.0.1:8080/v1beta1/", false},
		{"foo.bar.com", "", "http://foo.bar.com/v1beta1/", false},
		{"http://host/prefix", "", "http://host/prefix/v1beta1/", false},
		{"http://host", "", "http://host/v1beta1/", false},
		{"http://host", "/", "http://host/v1beta1/", false},
		{"http://host", "/other", "http://host/other/v1beta1/", false},
		{"host/server", "", "", true},
	}
	for i, testCase := range testCases {
		c, err := RESTClientFor(&Config{Host: testCase.Host, Prefix: testCase.Prefix, Version: "v1beta1"})
		switch {
		case err == nil && testCase.Err:
			t.Errorf("expected error but was nil")
			continue
		case err != nil && !testCase.Err:
			t.Errorf("unexpected error %v", err)
			continue
		case err != nil:
			continue
		}
		if e, a := testCase.URL, c.baseURL.String(); e != a {
			t.Errorf("%d: expected host %s, got %s", i, e, a)
			continue
		}
	}
}

func TestDoRequest(t *testing.T) {
	invalid := "aaaaa"
	uri, _ := url.Parse("http://localhost")
	testClients := []testClient{
		{Request: testRequest{Method: "GET", Path: "good"}, Response: Response{StatusCode: 200}},
		{Request: testRequest{Method: "GET", Path: "bad%ZZ"}, Error: true},
		{Client: &Client{&RESTClient{baseURL: uri}}, Request: testRequest{Method: "GET", Path: "nocertificate"}, Error: true},
		{Request: testRequest{Method: "GET", Path: "error"}, Response: Response{StatusCode: 500}, Error: true},
		{Request: testRequest{Method: "POST", Path: "faildecode"}, Response: Response{StatusCode: 200, RawBody: &invalid}},
		{Request: testRequest{Method: "GET", Path: "failread"}, Response: Response{StatusCode: 200, RawBody: &invalid}},
	}
	for _, c := range testClients {
		client := c.Setup()
		prefix := *client.baseURL
		prefix.Path += c.Request.Path
		request := &http.Request{
			Method: c.Request.Method,
			Header: make(http.Header),
			URL:    &prefix,
		}
		response, err := client.doRequest(request)
		//t.Logf("dorequest: %#v\n%#v\n%v", request.URL, response, err)
		c.ValidateRaw(t, response, err)
	}
}

func TestDoRequestBearer(t *testing.T) {
	status := &api.Status{Status: api.StatusWorking}
	expectedBody, _ := latest.Codec.Encode(status)
	fakeHandler := util.FakeHandler{
		StatusCode:   202,
		ResponseBody: string(expectedBody),
		T:            t,
	}
	testServer := httptest.NewServer(&fakeHandler)
	request, _ := http.NewRequest("GET", testServer.URL+"/foo/bar", nil)
	c, err := RESTClientFor(&Config{Host: testServer.URL, BearerToken: "test"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	c.doRequest(request)
	if fakeHandler.RequestReceived.Header.Get("Authorization") != "Bearer test" {
		t.Errorf("Request is missing authorization header: %#v", *request)
	}
}

func TestDoRequestAccepted(t *testing.T) {
	status := &api.Status{Status: api.StatusWorking}
	expectedBody, _ := latest.Codec.Encode(status)
	fakeHandler := util.FakeHandler{
		StatusCode:   202,
		ResponseBody: string(expectedBody),
		T:            t,
	}
	testServer := httptest.NewServer(&fakeHandler)
	request, _ := http.NewRequest("GET", testServer.URL+"/foo/bar", nil)
	c, err := RESTClientFor(&Config{Host: testServer.URL, Username: "test"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	body, err := c.doRequest(request)
	if fakeHandler.RequestReceived.Header["Authorization"] == nil {
		t.Errorf("Request is missing authorization header: %#v", *request)
	}
	if err == nil {
		t.Error("Unexpected non-error")
		return
	}
	se, ok := err.(APIStatus)
	if !ok {
		t.Errorf("Unexpected kind of error: %#v", err)
		return
	}
	if !reflect.DeepEqual(se.Status(), *status) {
		t.Errorf("Unexpected status: %#v %#v", se.Status(), status)
	}
	if body != nil {
		t.Errorf("Expected nil body, but saw: '%s'", body)
	}
	fakeHandler.ValidateRequest(t, "/foo/bar", "GET", nil)
}

func TestDoRequestAcceptedSuccess(t *testing.T) {
	status := &api.Status{Status: api.StatusSuccess}
	expectedBody, _ := latest.Codec.Encode(status)
	fakeHandler := util.FakeHandler{
		StatusCode:   202,
		ResponseBody: string(expectedBody),
		T:            t,
	}
	testServer := httptest.NewServer(&fakeHandler)
	request, _ := http.NewRequest("GET", testServer.URL+"/foo/bar", nil)
	c, err := RESTClientFor(&Config{Host: testServer.URL, Username: "user", Password: "pass"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	body, err := c.doRequest(request)
	if fakeHandler.RequestReceived.Header["Authorization"] == nil {
		t.Errorf("Request is missing authorization header: %#v", *request)
	}
	if err != nil {
		t.Errorf("Unexpected error %#v", err)
	}
	statusOut, err := latest.Codec.Decode(body)
	if err != nil {
		t.Errorf("Unexpected error %#v", err)
	}
	if !reflect.DeepEqual(status, statusOut) {
		t.Errorf("Unexpected mis-match. Expected %#v.  Saw %#v", status, statusOut)
	}
	fakeHandler.ValidateRequest(t, "/foo/bar", "GET", nil)
}

func TestDoRequestFailed(t *testing.T) {
	status := &api.Status{Status: api.StatusFailure, Reason: api.StatusReasonInvalid, Details: &api.StatusDetails{ID: "test", Kind: "test"}}
	expectedBody, _ := latest.Codec.Encode(status)
	fakeHandler := util.FakeHandler{
		StatusCode:   404,
		ResponseBody: string(expectedBody),
		T:            t,
	}
	testServer := httptest.NewServer(&fakeHandler)
	request, _ := http.NewRequest("GET", testServer.URL+"/foo/bar", nil)
	c, err := RESTClientFor(&Config{Host: testServer.URL})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	body, err := c.doRequest(request)
	if err == nil || body != nil {
		t.Errorf("unexpected non-error: %#v", body)
	}
	ss, ok := err.(APIStatus)
	if !ok {
		t.Errorf("unexpected error type %v", err)
	}
	actual := ss.Status()
	if !reflect.DeepEqual(status, &actual) {
		t.Errorf("Unexpected mis-match. Expected %#v.  Saw %#v", status, actual)
	}
}
