/*
Copyright 2014 The Kubernetes Authors.

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

package testing

import (
	"io/ioutil"
	"net/http"
	"net/url"
	"reflect"
	"sync"
)

// TestInterface is a simple interface providing Errorf, to make injection for
// testing easier (insert 'yo dawg' meme here).
type TestInterface interface {
	Errorf(format string, args ...interface{})
	Logf(format string, args ...interface{})
}

// LogInterface is a simple interface to allow injection of Logf to report serving errors.
type LogInterface interface {
	Logf(format string, args ...interface{})
}

// FakeHandler is to assist in testing HTTP requests. Notice that FakeHandler is
// not thread safe and you must not direct traffic to except for the request
// you want to test. You can do this by hiding it in an http.ServeMux.
type FakeHandler struct {
	RequestReceived *http.Request
	RequestBody     string
	StatusCode      int
	ResponseBody    string
	// For logging - you can use a *testing.T
	// This will keep log messages associated with the test.
	T LogInterface

	// Enforce "only one use" constraint.
	lock           sync.Mutex
	requestCount   int
	hasBeenChecked bool
}

func (f *FakeHandler) ServeHTTP(response http.ResponseWriter, request *http.Request) {
	f.lock.Lock()
	defer f.lock.Unlock()
	f.requestCount++
	if f.hasBeenChecked {
		panic("got request after having been validated")
	}

	f.RequestReceived = request
	response.Header().Set("Content-Type", "application/json")
	response.WriteHeader(f.StatusCode)
	response.Write([]byte(f.ResponseBody))

	bodyReceived, err := ioutil.ReadAll(request.Body)
	if err != nil && f.T != nil {
		f.T.Logf("Received read error: %v", err)
	}
	f.RequestBody = string(bodyReceived)
}

func (f *FakeHandler) ValidateRequestCount(t TestInterface, count int) bool {
	ok := true
	f.lock.Lock()
	defer f.lock.Unlock()
	if f.requestCount != count {
		ok = false
		t.Errorf("Expected %d call, but got %d. Only the last call is recorded and checked.", count, f.requestCount)
	}
	f.hasBeenChecked = true
	return ok
}

// ValidateRequest verifies that FakeHandler received a request with expected path, method, and body.
func (f *FakeHandler) ValidateRequest(t TestInterface, expectedPath, expectedMethod string, body *string) {
	f.lock.Lock()
	defer f.lock.Unlock()
	if f.requestCount != 1 {
		t.Logf("Expected 1 call, but got %v. Only the last call is recorded and checked.", f.requestCount)
	}
	f.hasBeenChecked = true

	expectURL, err := url.Parse(expectedPath)
	if err != nil {
		t.Errorf("Couldn't parse %v as a URL.", expectedPath)
	}
	if f.RequestReceived == nil {
		t.Errorf("Unexpected nil request received for %s", expectedPath)
		return
	}
	if f.RequestReceived.URL.Path != expectURL.Path {
		t.Errorf("Unexpected request path for request %#v, received: %q, expected: %q", f.RequestReceived, f.RequestReceived.URL.Path, expectURL.Path)
	}
	if e, a := expectURL.Query(), f.RequestReceived.URL.Query(); !reflect.DeepEqual(e, a) {
		t.Errorf("Unexpected query for request %#v, received: %q, expected: %q", f.RequestReceived, a, e)
	}
	if f.RequestReceived.Method != expectedMethod {
		t.Errorf("Unexpected method: %q, expected: %q", f.RequestReceived.Method, expectedMethod)
	}
	if body != nil {
		if *body != f.RequestBody {
			t.Errorf("Received body:\n%s\n Doesn't match expected body:\n%s", f.RequestBody, *body)
		}
	}
}
