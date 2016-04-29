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

package restclient

import (
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"reflect"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/diff"
	utiltesting "k8s.io/kubernetes/pkg/util/testing"
)

func TestDoRequestSuccess(t *testing.T) {
	status := &unversioned.Status{Status: unversioned.StatusSuccess}
	expectedBody, _ := runtime.Encode(testapi.Default.Codec(), status)
	fakeHandler := utiltesting.FakeHandler{
		StatusCode:   200,
		ResponseBody: string(expectedBody),
		T:            t,
	}
	testServer := httptest.NewServer(&fakeHandler)
	defer testServer.Close()
	c, err := RESTClientFor(&Config{
		Host: testServer.URL,
		ContentConfig: ContentConfig{
			GroupVersion:         testapi.Default.GroupVersion(),
			NegotiatedSerializer: testapi.NegotiatedSerializer,
		},
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
	statusOut, err := runtime.Decode(testapi.Default.Codec(), body)
	if err != nil {
		t.Errorf("Unexpected error %#v", err)
	}
	if !reflect.DeepEqual(status, statusOut) {
		t.Errorf("Unexpected mis-match. Expected %#v.  Saw %#v", status, statusOut)
	}
	fakeHandler.ValidateRequest(t, "/"+testapi.Default.GroupVersion().String()+"/test", "GET", nil)
}

func TestDoRequestFailed(t *testing.T) {
	status := &unversioned.Status{
		Code:    http.StatusNotFound,
		Status:  unversioned.StatusFailure,
		Reason:  unversioned.StatusReasonNotFound,
		Message: " \"\" not found",
		Details: &unversioned.StatusDetails{},
	}
	expectedBody, _ := runtime.Encode(testapi.Default.Codec(), status)
	fakeHandler := utiltesting.FakeHandler{
		StatusCode:   404,
		ResponseBody: string(expectedBody),
		T:            t,
	}
	testServer := httptest.NewServer(&fakeHandler)
	defer testServer.Close()
	c, err := RESTClientFor(&Config{
		Host: testServer.URL,
		ContentConfig: ContentConfig{
			GroupVersion:         testapi.Default.GroupVersion(),
			NegotiatedSerializer: testapi.NegotiatedSerializer,
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	body, err := c.Get().Do().Raw()
	if err == nil || body != nil {
		t.Errorf("unexpected non-error: %#v", body)
	}
	ss, ok := err.(errors.APIStatus)
	if !ok {
		t.Errorf("unexpected error type %v", err)
	}
	actual := ss.Status()
	expected := *status
	// The decoder will apply the default Version and Kind to the Status.
	expected.APIVersion = "v1"
	expected.Kind = "Status"
	if !reflect.DeepEqual(&expected, &actual) {
		t.Errorf("Unexpected mis-match: %s", diff.ObjectDiff(status, &actual))
	}
}

func TestDoRequestCreated(t *testing.T) {
	status := &unversioned.Status{Status: unversioned.StatusSuccess}
	expectedBody, _ := runtime.Encode(testapi.Default.Codec(), status)
	fakeHandler := utiltesting.FakeHandler{
		StatusCode:   201,
		ResponseBody: string(expectedBody),
		T:            t,
	}
	testServer := httptest.NewServer(&fakeHandler)
	defer testServer.Close()
	c, err := RESTClientFor(&Config{
		Host: testServer.URL,
		ContentConfig: ContentConfig{
			GroupVersion:         testapi.Default.GroupVersion(),
			NegotiatedSerializer: testapi.NegotiatedSerializer,
		},
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
	statusOut, err := runtime.Decode(testapi.Default.Codec(), body)
	if err != nil {
		t.Errorf("Unexpected error %#v", err)
	}
	if !reflect.DeepEqual(status, statusOut) {
		t.Errorf("Unexpected mis-match. Expected %#v.  Saw %#v", status, statusOut)
	}
	fakeHandler.ValidateRequest(t, "/"+testapi.Default.GroupVersion().String()+"/test", "GET", nil)
}

func TestCreateBackoffManager(t *testing.T) {

	theUrl, _ := url.Parse("http://localhost")

	// 1 second base backoff + duration of 2 seconds -> exponential backoff for requests.
	os.Setenv(envBackoffBase, "1")
	os.Setenv(envBackoffDuration, "2")
	backoff := readExpBackoffConfig()
	backoff.UpdateBackoff(theUrl, nil, 500)
	backoff.UpdateBackoff(theUrl, nil, 500)
	if backoff.CalculateBackoff(theUrl)/time.Second != 2 {
		t.Errorf("Backoff env not working.")
	}

	// 0 duration -> no backoff.
	os.Setenv(envBackoffBase, "1")
	os.Setenv(envBackoffDuration, "0")
	backoff.UpdateBackoff(theUrl, nil, 500)
	backoff.UpdateBackoff(theUrl, nil, 500)
	backoff = readExpBackoffConfig()
	if backoff.CalculateBackoff(theUrl)/time.Second != 0 {
		t.Errorf("Zero backoff duration, but backoff still occuring.")
	}

	// No env -> No backoff.
	os.Setenv(envBackoffBase, "")
	os.Setenv(envBackoffDuration, "")
	backoff = readExpBackoffConfig()
	backoff.UpdateBackoff(theUrl, nil, 500)
	backoff.UpdateBackoff(theUrl, nil, 500)
	if backoff.CalculateBackoff(theUrl)/time.Second != 0 {
		t.Errorf("Backoff should have been 0.")
	}

}
