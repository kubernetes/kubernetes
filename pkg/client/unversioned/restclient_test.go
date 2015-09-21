/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package unversioned

import (
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util"
)

func TestSetsCodec(t *testing.T) {
	testCases := map[string]struct {
		Err    bool
		Prefix string
		Codec  runtime.Codec
	}{
		testapi.Default.Version(): {false, "/api/" + testapi.Default.Version() + "/", testapi.Default.Codec()},
		"invalidVersion":          {true, "", nil},
	}
	for version, expected := range testCases {
		client, err := New(&Config{Host: "127.0.0.1", Version: version})
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
		if e, a := expected.Prefix, client.RESTClient.baseURL.Path; e != a {
			t.Errorf("expected %#v, got %#v", e, a)
		}
		if e, a := expected.Codec, client.RESTClient.Codec; e != a {
			t.Errorf("expected %#v, got %#v", e, a)
		}
	}
}

func TestRESTClientRequires(t *testing.T) {
	if _, err := RESTClientFor(&Config{Host: "127.0.0.1", Version: "", Codec: testapi.Default.Codec()}); err == nil {
		t.Errorf("unexpected non-error")
	}
	if _, err := RESTClientFor(&Config{Host: "127.0.0.1", Version: testapi.Default.Version()}); err == nil {
		t.Errorf("unexpected non-error")
	}
	if _, err := RESTClientFor(&Config{Host: "127.0.0.1", Version: testapi.Default.Version(), Codec: testapi.Default.Codec()}); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestValidatesHostParameter(t *testing.T) {
	testCases := []struct {
		Host   string
		Prefix string

		URL string
		Err bool
	}{
		{"127.0.0.1", "", "http://127.0.0.1/" + testapi.Default.Version() + "/", false},
		{"127.0.0.1:8080", "", "http://127.0.0.1:8080/" + testapi.Default.Version() + "/", false},
		{"foo.bar.com", "", "http://foo.bar.com/" + testapi.Default.Version() + "/", false},
		{"http://host/prefix", "", "http://host/prefix/" + testapi.Default.Version() + "/", false},
		{"http://host", "", "http://host/" + testapi.Default.Version() + "/", false},
		{"http://host", "/", "http://host/" + testapi.Default.Version() + "/", false},
		{"http://host", "/other", "http://host/other/" + testapi.Default.Version() + "/", false},
		{"host/server", "", "", true},
	}
	for i, testCase := range testCases {
		c, err := RESTClientFor(&Config{Host: testCase.Host, Prefix: testCase.Prefix, Version: testapi.Default.Version(), Codec: testapi.Default.Codec()})
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

func TestDoRequestBearer(t *testing.T) {
	status := &unversioned.Status{Status: unversioned.StatusFailure}
	expectedBody, _ := testapi.Default.Codec().Encode(status)
	fakeHandler := util.FakeHandler{
		StatusCode:   400,
		ResponseBody: string(expectedBody),
		T:            t,
	}
	testServer := httptest.NewServer(&fakeHandler)
	defer testServer.Close()
	request, _ := http.NewRequest("GET", testServer.URL, nil)
	c, err := RESTClientFor(&Config{
		Host:        testServer.URL,
		Version:     testapi.Default.Version(),
		Codec:       testapi.Default.Codec(),
		BearerToken: "test",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	err = c.Get().Do().Error()
	if err == nil {
		t.Fatalf("unexpected non-error: %v", err)
	}
	if fakeHandler.RequestReceived.Header.Get("Authorization") != "Bearer test" {
		t.Errorf("Request is missing authorization header: %#v", *request)
	}
}

func TestDoRequestWithoutPassword(t *testing.T) {
	status := &unversioned.Status{Status: unversioned.StatusFailure}
	expectedBody, _ := testapi.Default.Codec().Encode(status)
	fakeHandler := util.FakeHandler{
		StatusCode:   400,
		ResponseBody: string(expectedBody),
		T:            t,
	}
	testServer := httptest.NewServer(&fakeHandler)
	defer testServer.Close()
	c, err := RESTClientFor(&Config{
		Host:     testServer.URL,
		Version:  testapi.Default.Version(),
		Codec:    testapi.Default.Codec(),
		Username: "test",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	body, err := c.Get().Prefix("test").Do().Raw()
	if err == nil {
		t.Fatalf("Unexpected non-error")
	}
	if fakeHandler.RequestReceived.Header["Authorization"] == nil {
		t.Errorf("Request is missing authorization header: %#v", fakeHandler.RequestReceived)
	}
	se, ok := err.(APIStatus)
	if !ok {
		t.Fatalf("Unexpected kind of error: %#v", err)
	}
	if !reflect.DeepEqual(se.Status(), *status) {
		t.Errorf("Unexpected status: %#v %#v", se.Status(), status)
	}
	if body != nil {
		t.Errorf("Expected nil body, but saw: '%s'", string(body))
	}
	fakeHandler.ValidateRequest(t, "/"+testapi.Default.Version()+"/test", "GET", nil)
}

func TestDoRequestSuccess(t *testing.T) {
	status := &unversioned.Status{Status: unversioned.StatusSuccess}
	expectedBody, _ := testapi.Default.Codec().Encode(status)
	fakeHandler := util.FakeHandler{
		StatusCode:   200,
		ResponseBody: string(expectedBody),
		T:            t,
	}
	testServer := httptest.NewServer(&fakeHandler)
	defer testServer.Close()
	c, err := RESTClientFor(&Config{
		Host:     testServer.URL,
		Version:  testapi.Default.Version(),
		Codec:    testapi.Default.Codec(),
		Username: "user",
		Password: "pass",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	body, err := c.Get().Prefix("test").Do().Raw()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if fakeHandler.RequestReceived.Header["Authorization"] == nil {
		t.Errorf("Request is missing authorization header: %#v", fakeHandler.RequestReceived)
	}
	statusOut, err := testapi.Default.Codec().Decode(body)
	if err != nil {
		t.Errorf("Unexpected error %#v", err)
	}
	if !reflect.DeepEqual(status, statusOut) {
		t.Errorf("Unexpected mis-match. Expected %#v.  Saw %#v", status, statusOut)
	}
	fakeHandler.ValidateRequest(t, "/"+testapi.Default.Version()+"/test", "GET", nil)
}

func TestDoRequestFailed(t *testing.T) {
	status := &unversioned.Status{
		Code:    http.StatusNotFound,
		Status:  unversioned.StatusFailure,
		Reason:  unversioned.StatusReasonNotFound,
		Message: " \"\" not found",
		Details: &unversioned.StatusDetails{},
	}
	expectedBody, _ := testapi.Default.Codec().Encode(status)
	fakeHandler := util.FakeHandler{
		StatusCode:   404,
		ResponseBody: string(expectedBody),
		T:            t,
	}
	testServer := httptest.NewServer(&fakeHandler)
	defer testServer.Close()
	c, err := RESTClientFor(&Config{
		Host:    testServer.URL,
		Version: testapi.Default.Version(),
		Codec:   testapi.Default.Codec(),
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	body, err := c.Get().Do().Raw()
	if err == nil || body != nil {
		t.Errorf("unexpected non-error: %#v", body)
	}
	ss, ok := err.(APIStatus)
	if !ok {
		t.Errorf("unexpected error type %v", err)
	}
	actual := ss.Status()
	if !reflect.DeepEqual(status, &actual) {
		t.Errorf("Unexpected mis-match: %s", util.ObjectDiff(status, &actual))
	}
}

func TestDoRequestCreated(t *testing.T) {
	status := &unversioned.Status{Status: unversioned.StatusSuccess}
	expectedBody, _ := testapi.Default.Codec().Encode(status)
	fakeHandler := util.FakeHandler{
		StatusCode:   201,
		ResponseBody: string(expectedBody),
		T:            t,
	}
	testServer := httptest.NewServer(&fakeHandler)
	defer testServer.Close()
	c, err := RESTClientFor(&Config{
		Host:     testServer.URL,
		Version:  testapi.Default.Version(),
		Codec:    testapi.Default.Codec(),
		Username: "user",
		Password: "pass",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	created := false
	body, err := c.Get().Prefix("test").Do().WasCreated(&created).Raw()
	if err != nil {
		t.Errorf("Unexpected error %#v", err)
	}
	if !created {
		t.Errorf("Expected object to be created")
	}
	statusOut, err := testapi.Default.Codec().Decode(body)
	if err != nil {
		t.Errorf("Unexpected error %#v", err)
	}
	if !reflect.DeepEqual(status, statusOut) {
		t.Errorf("Unexpected mis-match. Expected %#v.  Saw %#v", status, statusOut)
	}
	fakeHandler.ValidateRequest(t, "/"+testapi.Default.Version()+"/test", "GET", nil)
}
