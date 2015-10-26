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

package apiserver

import (
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"reflect"
	"regexp"
	"strings"
	"sync"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/testapi"
)

type fakeRL bool

func (fakeRL) Stop()             {}
func (f fakeRL) CanAccept() bool { return bool(f) }
func (f fakeRL) Accept()         {}

func expectHTTP(url string, code int, t *testing.T) {
	r, err := http.Get(url)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
		return
	}
	if r.StatusCode != code {
		t.Errorf("unexpected response: %v", r.StatusCode)
	}
}

func getPath(resource, namespace, name string) string {
	return testapi.Default.ResourcePath(resource, namespace, name)
}

func pathWithPrefix(prefix, resource, namespace, name string) string {
	return testapi.Default.ResourcePathWithPrefix(prefix, resource, namespace, name)
}

func TestMaxInFlight(t *testing.T) {
	const Iterations = 3
	block := sync.WaitGroup{}
	block.Add(1)
	oneFinished := sync.WaitGroup{}
	oneFinished.Add(1)
	var once sync.Once
	sem := make(chan bool, Iterations)

	re := regexp.MustCompile("[.*\\/watch][^\\/proxy.*]")

	// Calls verifies that the server is actually blocked up before running the rest of the test
	calls := &sync.WaitGroup{}
	calls.Add(Iterations * 3)

	server := httptest.NewServer(MaxInFlightLimit(sem, re, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.Contains(r.URL.Path, "dontwait") {
			return
		}
		if calls != nil {
			calls.Done()
		}
		block.Wait()
	})))
	defer server.Close()

	// These should hang, but not affect accounting.
	for i := 0; i < Iterations; i++ {
		// These should hang waiting on block...
		go func() {
			expectHTTP(server.URL+"/foo/bar/watch", http.StatusOK, t)
			once.Do(oneFinished.Done)
		}()
	}

	for i := 0; i < Iterations; i++ {
		// These should hang waiting on block...
		go func() {
			expectHTTP(server.URL+"/proxy/foo/bar", http.StatusOK, t)
			once.Do(oneFinished.Done)
		}()
	}
	expectHTTP(server.URL+"/dontwait", http.StatusOK, t)

	for i := 0; i < Iterations; i++ {
		// These should hang waiting on block...
		go func() {
			expectHTTP(server.URL, http.StatusOK, t)
			once.Do(oneFinished.Done)
		}()
	}
	calls.Wait()
	calls = nil

	// Do this multiple times to show that it rate limit rejected requests don't block.
	for i := 0; i < 2; i++ {
		expectHTTP(server.URL, errors.StatusTooManyRequests, t)
	}

	// Validate that non-accounted URLs still work
	expectHTTP(server.URL+"/dontwait/watch", http.StatusOK, t)

	block.Done()

	// Show that we recover from being blocked up.
	// However, we should until at least one of the requests really finishes.
	oneFinished.Wait()
	expectHTTP(server.URL, http.StatusOK, t)
}

func TestReadOnly(t *testing.T) {
	server := httptest.NewServer(ReadOnly(http.HandlerFunc(
		func(w http.ResponseWriter, req *http.Request) {
			if req.Method != "GET" {
				t.Errorf("Unexpected call: %v", req.Method)
			}
		},
	)))
	defer server.Close()
	for _, verb := range []string{"GET", "POST", "PUT", "DELETE", "CREATE"} {
		req, err := http.NewRequest(verb, server.URL, nil)
		if err != nil {
			t.Fatalf("Couldn't make request: %v", err)
		}
		http.DefaultClient.Do(req)
	}
}

func TestTimeout(t *testing.T) {
	sendResponse := make(chan struct{}, 1)
	writeErrors := make(chan error, 1)
	timeout := make(chan time.Time, 1)
	resp := "test response"
	timeoutResp := "test timeout"

	ts := httptest.NewServer(TimeoutHandler(http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			<-sendResponse
			_, err := w.Write([]byte(resp))
			writeErrors <- err
		}),
		func(*http.Request) (<-chan time.Time, string) {
			return timeout, timeoutResp
		}))
	defer ts.Close()

	// No timeouts
	sendResponse <- struct{}{}
	res, err := http.Get(ts.URL)
	if err != nil {
		t.Error(err)
	}
	if res.StatusCode != http.StatusOK {
		t.Errorf("got res.StatusCode %d; expected %d", res.StatusCode, http.StatusOK)
	}
	body, _ := ioutil.ReadAll(res.Body)
	if string(body) != resp {
		t.Errorf("got body %q; expected %q", string(body), resp)
	}
	if err := <-writeErrors; err != nil {
		t.Errorf("got unexpected Write error on first request: %v", err)
	}

	// Times out
	timeout <- time.Time{}
	res, err = http.Get(ts.URL)
	if err != nil {
		t.Error(err)
	}
	if res.StatusCode != http.StatusGatewayTimeout {
		t.Errorf("got res.StatusCode %d; expected %d", res.StatusCode, http.StatusServiceUnavailable)
	}
	body, _ = ioutil.ReadAll(res.Body)
	if string(body) != timeoutResp {
		t.Errorf("got body %q; expected %q", string(body), timeoutResp)
	}

	// Now try to send a response
	sendResponse <- struct{}{}
	if err := <-writeErrors; err != http.ErrHandlerTimeout {
		t.Errorf("got Write error of %v; expected %v", err, http.ErrHandlerTimeout)
	}
}

func TestGetAPIRequestInfo(t *testing.T) {
	successCases := []struct {
		method              string
		url                 string
		expectedVerb        string
		expectedAPIPrefix   string
		expectedAPIGroup    string
		expectedAPIVersion  string
		expectedNamespace   string
		expectedResource    string
		expectedSubresource string
		expectedName        string
		expectedParts       []string
	}{

		// resource paths
		{"GET", "/api/v1/namespaces", "list", "api", "", "v1", "", "namespaces", "", "", []string{"namespaces"}},
		{"GET", "/api/v1/namespaces/other", "get", "api", "", "v1", "other", "namespaces", "", "other", []string{"namespaces", "other"}},

		{"GET", "/api/v1/namespaces/other/pods", "list", "api", "", "v1", "other", "pods", "", "", []string{"pods"}},
		{"GET", "/api/v1/namespaces/other/pods/foo", "get", "api", "", "v1", "other", "pods", "", "foo", []string{"pods", "foo"}},
		{"GET", "/api/v1/pods", "list", "api", "", "v1", api.NamespaceAll, "pods", "", "", []string{"pods"}},
		{"GET", "/api/v1/namespaces/other/pods/foo", "get", "api", "", "v1", "other", "pods", "", "foo", []string{"pods", "foo"}},
		{"GET", "/api/v1/namespaces/other/pods", "list", "api", "", "v1", "other", "pods", "", "", []string{"pods"}},

		// special verbs
		{"GET", "/api/v1/proxy/namespaces/other/pods/foo", "proxy", "api", "", "v1", "other", "pods", "", "foo", []string{"pods", "foo"}},
		{"GET", "/api/v1/redirect/namespaces/other/pods/foo", "redirect", "api", "", "v1", "other", "pods", "", "foo", []string{"pods", "foo"}},
		{"GET", "/api/v1/watch/pods", "watch", "api", "", "v1", api.NamespaceAll, "pods", "", "", []string{"pods"}},
		{"GET", "/api/v1/watch/namespaces/other/pods", "watch", "api", "", "v1", "other", "pods", "", "", []string{"pods"}},

		// subresource identification
		{"GET", "/api/v1/namespaces/other/pods/foo/status", "get", "api", "", "v1", "other", "pods", "status", "foo", []string{"pods", "foo", "status"}},
		{"PUT", "/api/v1/namespaces/other/finalize", "update", "api", "", "v1", "other", "finalize", "", "", []string{"finalize"}},

		// verb identification
		{"PATCH", "/api/v1/namespaces/other/pods/foo", "patch", "api", "", "v1", "other", "pods", "", "foo", []string{"pods", "foo"}},
		{"DELETE", "/api/v1/namespaces/other/pods/foo", "delete", "api", "", "v1", "other", "pods", "", "foo", []string{"pods", "foo"}},
		{"POST", "/api/v1/namespaces/other/pods", "create", "api", "", "v1", "other", "pods", "", "", []string{"pods"}},

		// api group identification
		{"POST", "/apis/extensions/v1/namespaces/other/pods", "create", "api", "extensions", "v1", "other", "pods", "", "", []string{"pods"}},

		// api version identification
		{"POST", "/apis/extensions/v1beta3/namespaces/other/pods", "create", "api", "extensions", "v1beta3", "other", "pods", "", "", []string{"pods"}},
	}

	requestInfoResolver := newTestRequestInfoResolver()

	for _, successCase := range successCases {
		req, _ := http.NewRequest(successCase.method, successCase.url, nil)

		apiRequestInfo, err := requestInfoResolver.GetRequestInfo(req)
		if err != nil {
			t.Errorf("Unexpected error for url: %s %v", successCase.url, err)
		}
		if !apiRequestInfo.IsResourceRequest {
			t.Errorf("Expected resource request")
		}
		if successCase.expectedVerb != apiRequestInfo.Verb {
			t.Errorf("Unexpected verb for url: %s, expected: %s, actual: %s", successCase.url, successCase.expectedVerb, apiRequestInfo.Verb)
		}
		if successCase.expectedAPIVersion != apiRequestInfo.APIVersion {
			t.Errorf("Unexpected apiVersion for url: %s, expected: %s, actual: %s", successCase.url, successCase.expectedAPIVersion, apiRequestInfo.APIVersion)
		}
		if successCase.expectedNamespace != apiRequestInfo.Namespace {
			t.Errorf("Unexpected namespace for url: %s, expected: %s, actual: %s", successCase.url, successCase.expectedNamespace, apiRequestInfo.Namespace)
		}
		if successCase.expectedResource != apiRequestInfo.Resource {
			t.Errorf("Unexpected resource for url: %s, expected: %s, actual: %s", successCase.url, successCase.expectedResource, apiRequestInfo.Resource)
		}
		if successCase.expectedSubresource != apiRequestInfo.Subresource {
			t.Errorf("Unexpected resource for url: %s, expected: %s, actual: %s", successCase.url, successCase.expectedSubresource, apiRequestInfo.Subresource)
		}
		if successCase.expectedName != apiRequestInfo.Name {
			t.Errorf("Unexpected name for url: %s, expected: %s, actual: %s", successCase.url, successCase.expectedName, apiRequestInfo.Name)
		}
		if !reflect.DeepEqual(successCase.expectedParts, apiRequestInfo.Parts) {
			t.Errorf("Unexpected parts for url: %s, expected: %v, actual: %v", successCase.url, successCase.expectedParts, apiRequestInfo.Parts)
		}
	}

	errorCases := map[string]string{
		"no resource path":            "/",
		"just apiversion":             "/api/version/",
		"just prefix, group, version": "/apis/group/version/",
		"apiversion with no resource": "/api/version/",
		"bad prefix":                  "/badprefix/version/resource",
		"missing api group":           "/apis/version/resource",
	}
	for k, v := range errorCases {
		req, err := http.NewRequest("GET", v, nil)
		if err != nil {
			t.Errorf("Unexpected error %v", err)
		}
		apiRequestInfo, err := requestInfoResolver.GetRequestInfo(req)
		if err != nil {
			t.Errorf("%s: Unexpected error %v", k, err)
		}
		if apiRequestInfo.IsResourceRequest {
			t.Errorf("%s: expected non-resource request", k)
		}
	}
}

func TestGetNonAPIRequestInfo(t *testing.T) {
	tests := map[string]struct {
		url      string
		expected bool
	}{
		"simple groupless":  {"/api/version/resource", true},
		"simple group":      {"/apis/group/version/resource/name/subresource", true},
		"more steps":        {"/api/version/resource/name/subresource", true},
		"group list":        {"/apis/extensions/v1beta1/job", true},
		"group get":         {"/apis/extensions/v1beta1/job/foo", true},
		"group subresource": {"/apis/extensions/v1beta1/job/foo/scale", true},

		"bad root":                     {"/not-api/version/resource", false},
		"group without enough steps":   {"/apis/extensions/v1beta1", false},
		"group without enough steps 2": {"/apis/extensions/v1beta1/", false},
		"not enough steps":             {"/api/version", false},
		"one step":                     {"/api", false},
		"zero step":                    {"/", false},
		"empty":                        {"", false},
	}

	requestInfoResolver := newTestRequestInfoResolver()

	for testName, tc := range tests {
		req, _ := http.NewRequest("GET", tc.url, nil)

		apiRequestInfo, err := requestInfoResolver.GetRequestInfo(req)
		if err != nil {
			t.Errorf("%s: Unexpected error %v", testName, err)
		}
		if e, a := tc.expected, apiRequestInfo.IsResourceRequest; e != a {
			t.Errorf("%s: expected %v, actual %v", testName, e, a)
		}
	}
}
