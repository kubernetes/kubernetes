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
	"fmt"
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
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/auth/authorizer"
	"k8s.io/kubernetes/pkg/util/sets"
)

type fakeRL bool

func (fakeRL) Stop()             {}
func (f fakeRL) TryAccept() bool { return bool(f) }
func (f fakeRL) Accept()         {}

func expectHTTP(url string, code int) error {
	r, err := http.Get(url)
	if err != nil {
		return fmt.Errorf("unexpected error: %v", err)
	}
	if r.StatusCode != code {
		return fmt.Errorf("unexpected response: %v", r.StatusCode)
	}
	return nil
}

func getPath(resource, namespace, name string) string {
	return testapi.Default.ResourcePath(resource, namespace, name)
}

func pathWithPrefix(prefix, resource, namespace, name string) string {
	return testapi.Default.ResourcePathWithPrefix(prefix, resource, namespace, name)
}

// Tests that MaxInFlightLimit works, i.e.
// - "long" requests such as proxy or watch, identified by regexp are not accounted despite
//   hanging for the long time,
// - "short" requests are correctly accounted, i.e. there can be only size of channel passed to the
//   constructor in flight at any given moment,
// - subsequent "short" requests are rejected instantly with appropriate error,
// - subsequent "long" requests are handled normally,
// - we correctly recover after some "short" requests finish, i.e. we can process new ones.
func TestMaxInFlight(t *testing.T) {
	const AllowedInflightRequestsNo = 3
	// Size of inflightRequestsChannel determines how many concurent inflight requests
	// are allowed.
	inflightRequestsChannel := make(chan bool, AllowedInflightRequestsNo)
	// notAccountedPathsRegexp specifies paths requests to which we don't account into
	// requests in flight.
	notAccountedPathsRegexp := regexp.MustCompile(".*\\/watch")
	longRunningRequestCheck := BasicLongRunningRequestCheck(notAccountedPathsRegexp, map[string]string{"watch": "true"})

	// Calls is used to wait until all server calls are received. We are sending
	// AllowedInflightRequestsNo of 'long' not-accounted requests and the same number of
	// 'short' accounted ones.
	calls := &sync.WaitGroup{}
	calls.Add(AllowedInflightRequestsNo * 2)

	// Responses is used to wait until all responses are
	// received. This prevents some async requests getting EOF
	// errors from prematurely closing the server
	responses := sync.WaitGroup{}
	responses.Add(AllowedInflightRequestsNo * 2)

	// Block is used to keep requests in flight for as long as we need to. All requests will
	// be unblocked at the same time.
	block := sync.WaitGroup{}
	block.Add(1)

	server := httptest.NewServer(
		MaxInFlightLimit(
			inflightRequestsChannel,
			longRunningRequestCheck,
			http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				// A short, accounted request that does not wait for block WaitGroup.
				if strings.Contains(r.URL.Path, "dontwait") {
					return
				}
				if calls != nil {
					calls.Done()
				}
				block.Wait()
			}),
		),
	)
	// TODO: Uncomment when fix #19254
	// defer server.Close()

	// These should hang, but not affect accounting.  use a query param match
	for i := 0; i < AllowedInflightRequestsNo; i++ {
		// These should hang waiting on block...
		go func() {
			if err := expectHTTP(server.URL+"/foo/bar?watch=true", http.StatusOK); err != nil {
				t.Error(err)
			}
			responses.Done()
		}()
	}
	// Check that sever is not saturated by not-accounted calls
	if err := expectHTTP(server.URL+"/dontwait", http.StatusOK); err != nil {
		t.Error(err)
	}

	// These should hang and be accounted, i.e. saturate the server
	for i := 0; i < AllowedInflightRequestsNo; i++ {
		// These should hang waiting on block...
		go func() {
			if err := expectHTTP(server.URL, http.StatusOK); err != nil {
				t.Error(err)
			}
			responses.Done()
		}()
	}
	// We wait for all calls to be received by the server
	calls.Wait()
	// Disable calls notifications in the server
	calls = nil

	// Do this multiple times to show that it rate limit rejected requests don't block.
	for i := 0; i < 2; i++ {
		if err := expectHTTP(server.URL, errors.StatusTooManyRequests); err != nil {
			t.Error(err)
		}
	}
	// Validate that non-accounted URLs still work.  use a path regex match
	if err := expectHTTP(server.URL+"/dontwait/watch", http.StatusOK); err != nil {
		t.Error(err)
	}

	// Let all hanging requests finish
	block.Done()

	// Show that we recover from being blocked up.
	// Too avoid flakyness we need to wait until at least one of the requests really finishes.
	responses.Wait()
	if err := expectHTTP(server.URL, http.StatusOK); err != nil {
		t.Error(err)
	}
}

func TestReadOnly(t *testing.T) {
	server := httptest.NewServer(ReadOnly(http.HandlerFunc(
		func(w http.ResponseWriter, req *http.Request) {
			if req.Method != "GET" {
				t.Errorf("Unexpected call: %v", req.Method)
			}
		},
	)))
	// TODO: Uncomment when fix #19254
	// defer server.Close()
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
	// TODO: Uncomment when fix #19254
	// defer ts.Close()

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

func TestGetAttribs(t *testing.T) {
	r := &requestAttributeGetter{api.NewRequestContextMapper(), &RequestInfoResolver{sets.NewString("api", "apis"), sets.NewString("api")}}

	testcases := map[string]struct {
		Verb               string
		Path               string
		ExpectedAttributes *authorizer.AttributesRecord
	}{
		"non-resource root": {
			Verb: "POST",
			Path: "/",
			ExpectedAttributes: &authorizer.AttributesRecord{
				Verb: "post",
				Path: "/",
			},
		},
		"non-resource api prefix": {
			Verb: "GET",
			Path: "/api/",
			ExpectedAttributes: &authorizer.AttributesRecord{
				Verb: "get",
				Path: "/api/",
			},
		},
		"non-resource group api prefix": {
			Verb: "GET",
			Path: "/apis/extensions/",
			ExpectedAttributes: &authorizer.AttributesRecord{
				Verb: "get",
				Path: "/apis/extensions/",
			},
		},

		"resource": {
			Verb: "POST",
			Path: "/api/v1/nodes/mynode",
			ExpectedAttributes: &authorizer.AttributesRecord{
				Verb:            "create",
				Path:            "/api/v1/nodes/mynode",
				ResourceRequest: true,
				Resource:        "nodes",
			},
		},
		"namespaced resource": {
			Verb: "PUT",
			Path: "/api/v1/namespaces/myns/pods/mypod",
			ExpectedAttributes: &authorizer.AttributesRecord{
				Verb:            "update",
				Path:            "/api/v1/namespaces/myns/pods/mypod",
				ResourceRequest: true,
				Namespace:       "myns",
				Resource:        "pods",
			},
		},
		"API group resource": {
			Verb: "GET",
			Path: "/apis/extensions/v1beta1/namespaces/myns/jobs",
			ExpectedAttributes: &authorizer.AttributesRecord{
				Verb:            "list",
				Path:            "/apis/extensions/v1beta1/namespaces/myns/jobs",
				ResourceRequest: true,
				APIGroup:        extensions.GroupName,
				Namespace:       "myns",
				Resource:        "jobs",
			},
		},
	}

	for k, tc := range testcases {
		req, _ := http.NewRequest(tc.Verb, tc.Path, nil)
		attribs := r.GetAttribs(req)
		if !reflect.DeepEqual(attribs, tc.ExpectedAttributes) {
			t.Errorf("%s: expected\n\t%#v\ngot\n\t%#v", k, tc.ExpectedAttributes, attribs)
		}
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
		{"HEAD", "/api/v1/namespaces/other/pods/foo", "get", "api", "", "v1", "other", "pods", "", "foo", []string{"pods", "foo"}},
		{"GET", "/api/v1/pods", "list", "api", "", "v1", api.NamespaceAll, "pods", "", "", []string{"pods"}},
		{"HEAD", "/api/v1/pods", "list", "api", "", "v1", api.NamespaceAll, "pods", "", "", []string{"pods"}},
		{"GET", "/api/v1/namespaces/other/pods/foo", "get", "api", "", "v1", "other", "pods", "", "foo", []string{"pods", "foo"}},
		{"GET", "/api/v1/namespaces/other/pods", "list", "api", "", "v1", "other", "pods", "", "", []string{"pods"}},

		// special verbs
		{"GET", "/api/v1/proxy/namespaces/other/pods/foo", "proxy", "api", "", "v1", "other", "pods", "", "foo", []string{"pods", "foo"}},
		{"GET", "/api/v1/proxy/namespaces/other/pods/foo/subpath/not/a/subresource", "proxy", "api", "", "v1", "other", "pods", "", "foo", []string{"pods", "foo", "subpath", "not", "a", "subresource"}},
		{"GET", "/api/v1/redirect/namespaces/other/pods/foo", "redirect", "api", "", "v1", "other", "pods", "", "foo", []string{"pods", "foo"}},
		{"GET", "/api/v1/redirect/namespaces/other/pods/foo/subpath/not/a/subresource", "redirect", "api", "", "v1", "other", "pods", "", "foo", []string{"pods", "foo", "subpath", "not", "a", "subresource"}},
		{"GET", "/api/v1/watch/pods", "watch", "api", "", "v1", api.NamespaceAll, "pods", "", "", []string{"pods"}},
		{"GET", "/api/v1/watch/namespaces/other/pods", "watch", "api", "", "v1", "other", "pods", "", "", []string{"pods"}},

		// subresource identification
		{"GET", "/api/v1/namespaces/other/pods/foo/status", "get", "api", "", "v1", "other", "pods", "status", "foo", []string{"pods", "foo", "status"}},
		{"GET", "/api/v1/namespaces/other/pods/foo/proxy/subpath", "get", "api", "", "v1", "other", "pods", "proxy", "foo", []string{"pods", "foo", "proxy", "subpath"}},
		{"PUT", "/api/v1/namespaces/other/finalize", "update", "api", "", "v1", "other", "namespaces", "finalize", "other", []string{"namespaces", "other", "finalize"}},
		{"PUT", "/api/v1/namespaces/other/status", "update", "api", "", "v1", "other", "namespaces", "status", "other", []string{"namespaces", "other", "status"}},

		// verb identification
		{"PATCH", "/api/v1/namespaces/other/pods/foo", "patch", "api", "", "v1", "other", "pods", "", "foo", []string{"pods", "foo"}},
		{"DELETE", "/api/v1/namespaces/other/pods/foo", "delete", "api", "", "v1", "other", "pods", "", "foo", []string{"pods", "foo"}},
		{"POST", "/api/v1/namespaces/other/pods", "create", "api", "", "v1", "other", "pods", "", "", []string{"pods"}},

		// deletecollection verb identification
		{"DELETE", "/api/v1/nodes", "deletecollection", "api", "", "v1", "", "nodes", "", "", []string{"nodes"}},
		{"DELETE", "/api/v1/namespaces", "deletecollection", "api", "", "v1", "", "namespaces", "", "", []string{"namespaces"}},
		{"DELETE", "/api/v1/namespaces/other/pods", "deletecollection", "api", "", "v1", "other", "pods", "", "", []string{"pods"}},
		{"DELETE", "/apis/extensions/v1/namespaces/other/pods", "deletecollection", "api", "extensions", "v1", "other", "pods", "", "", []string{"pods"}},

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
