/*
Copyright 2016 The Kubernetes Authors.

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

package filters

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/util/sets"
	apirequest "k8s.io/apiserver/pkg/endpoints/request"
	apifilters "k8s.io/apiserver/pkg/endpoints/filters"
)

func createMaxInflightServer(callsWg, blockWg *sync.WaitGroup, disableCallsWg *bool, disableCallsWgMutex *sync.Mutex, nonMutating, mutating int) *httptest.Server {

	longRunningRequestCheck := BasicLongRunningRequestCheck(sets.NewString("watch"), sets.NewString("proxy"))

	requestContextMapper := apirequest.NewRequestContextMapper()
	requestInfoFactory := &apirequest.RequestInfoFactory{APIPrefixes: sets.NewString("apis", "api"), GrouplessAPIPrefixes: sets.NewString("api")}
	handler := WithMaxInFlightLimit(
		http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// A short, accounted request that does not wait for block WaitGroup.
			if strings.Contains(r.URL.Path, "dontwait") {
				return
			}
			disableCallsWgMutex.Lock()
			waitForCalls := *disableCallsWg
			disableCallsWgMutex.Unlock()
			if waitForCalls {
				callsWg.Done()
			}
			blockWg.Wait()
		}),
		nonMutating,
		mutating,
		requestContextMapper,
		longRunningRequestCheck,
	)
	handler = apifilters.WithRequestInfo(handler, requestInfoFactory, requestContextMapper)
	handler = apirequest.WithRequestContext(handler, requestContextMapper)

	return httptest.NewServer(handler)
}

// Tests that MaxInFlightLimit works, i.e.
// - "long" requests such as proxy or watch, identified by regexp are not accounted despite
//   hanging for the long time,
// - "short" requests are correctly accounted, i.e. there can be only size of channel passed to the
//   constructor in flight at any given moment,
// - subsequent "short" requests are rejected instantly with appropriate error,
// - subsequent "long" requests are handled normally,
// - we correctly recover after some "short" requests finish, i.e. we can process new ones.
func TestMaxInFlightNonMutating(t *testing.T) {
	const AllowedNonMutatingInflightRequestsNo = 3

	// Calls is used to wait until all server calls are received. We are sending
	// AllowedNonMutatingInflightRequestsNo of 'long' not-accounted requests and the same number of
	// 'short' accounted ones.
	calls := &sync.WaitGroup{}
	calls.Add(AllowedNonMutatingInflightRequestsNo * 2)

	// Responses is used to wait until all responses are
	// received. This prevents some async requests getting EOF
	// errors from prematurely closing the server
	responses := &sync.WaitGroup{}
	responses.Add(AllowedNonMutatingInflightRequestsNo * 2)

	// Block is used to keep requests in flight for as long as we need to. All requests will
	// be unblocked at the same time.
	block := &sync.WaitGroup{}
	block.Add(1)

	waitForCalls := true
	waitForCallsMutex := sync.Mutex{}

	server := createMaxInflightServer(calls, block, &waitForCalls, &waitForCallsMutex, AllowedNonMutatingInflightRequestsNo, 1)
	defer server.Close()

	// These should hang, but not affect accounting.  use a query param match
	for i := 0; i < AllowedNonMutatingInflightRequestsNo; i++ {
		// These should hang waiting on block...
		go func() {
			if err := expectHTTPGet(server.URL+"/api/v1/namespaces/default/wait?watch=true", http.StatusOK); err != nil {
				t.Error(err)
			}
			responses.Done()
		}()
	}

	// Check that sever is not saturated by not-accounted calls
	if err := expectHTTPGet(server.URL+"/dontwait", http.StatusOK); err != nil {
		t.Error(err)
	}

	// These should hang and be accounted, i.e. saturate the server
	for i := 0; i < AllowedNonMutatingInflightRequestsNo; i++ {
		// These should hang waiting on block...
		go func() {
			if err := expectHTTPGet(server.URL, http.StatusOK); err != nil {
				t.Error(err)
			}
			responses.Done()
		}()
	}
	// We wait for all calls to be received by the server
	calls.Wait()
	// Disable calls notifications in the server
	waitForCallsMutex.Lock()
	waitForCalls = false
	waitForCallsMutex.Unlock()

	// Do this multiple times to show that rate limit rejected requests don't block.
	for i := 0; i < 2; i++ {
		if err := expectHTTPGet(server.URL, errors.StatusTooManyRequests); err != nil {
			t.Error(err)
		}
	}
	// Validate that non-accounted URLs still work.  use a path regex match
	if err := expectHTTPGet(server.URL+"/api/v1/watch/namespaces/default/dontwait", http.StatusOK); err != nil {
		t.Error(err)
	}

	// We should allow a single mutating request.
	if err := expectHTTPPost(server.URL+"/dontwait", http.StatusOK); err != nil {
		t.Error(err)
	}

	// Let all hanging requests finish
	block.Done()

	// Show that we recover from being blocked up.
	// Too avoid flakyness we need to wait until at least one of the requests really finishes.
	responses.Wait()
	if err := expectHTTPGet(server.URL, http.StatusOK); err != nil {
		t.Error(err)
	}
}

func TestMaxInFlightMutating(t *testing.T) {
	const AllowedMutatingInflightRequestsNo = 3

	calls := &sync.WaitGroup{}
	calls.Add(AllowedMutatingInflightRequestsNo)

	responses := &sync.WaitGroup{}
	responses.Add(AllowedMutatingInflightRequestsNo)

	// Block is used to keep requests in flight for as long as we need to. All requests will
	// be unblocked at the same time.
	block := &sync.WaitGroup{}
	block.Add(1)

	waitForCalls := true
	waitForCallsMutex := sync.Mutex{}

	server := createMaxInflightServer(calls, block, &waitForCalls, &waitForCallsMutex, 1, AllowedMutatingInflightRequestsNo)
	defer server.Close()

	// These should hang and be accounted, i.e. saturate the server
	for i := 0; i < AllowedMutatingInflightRequestsNo; i++ {
		// These should hang waiting on block...
		go func() {
			if err := expectHTTPPost(server.URL+"/foo/bar", http.StatusOK); err != nil {
				t.Error(err)
			}
			responses.Done()
		}()
	}
	// We wait for all calls to be received by the server
	calls.Wait()
	// Disable calls notifications in the server
	// Disable calls notifications in the server
	waitForCallsMutex.Lock()
	waitForCalls = false
	waitForCallsMutex.Unlock()

	// Do this multiple times to show that rate limit rejected requests don't block.
	for i := 0; i < 2; i++ {
		if err := expectHTTPPost(server.URL+"/foo/bar/", errors.StatusTooManyRequests); err != nil {
			t.Error(err)
		}
	}
	// Validate that non-mutating URLs still work.  use a path regex match
	if err := expectHTTPGet(server.URL+"/dontwait", http.StatusOK); err != nil {
		t.Error(err)
	}

	// Let all hanging requests finish
	block.Done()

	// Show that we recover from being blocked up.
	// Too avoid flakyness we need to wait until at least one of the requests really finishes.
	responses.Wait()
	if err := expectHTTPPost(server.URL+"/foo/bar", http.StatusOK); err != nil {
		t.Error(err)
	}
}

// We use GET as a sample non-mutating request.
func expectHTTPGet(url string, code int) error {
	r, err := http.Get(url)
	if err != nil {
		return fmt.Errorf("unexpected error: %v", err)
	}
	if r.StatusCode != code {
		return fmt.Errorf("unexpected response: %v", r.StatusCode)
	}
	return nil
}

// We use POST as a sample mutating request.
func expectHTTPPost(url string, code int) error {
	r, err := http.Post(url, "text/html", strings.NewReader("foo bar"))
	if err != nil {
		return fmt.Errorf("unexpected error: %v", err)
	}
	if r.StatusCode != code {
		return fmt.Errorf("unexpected response: %v", r.StatusCode)
	}
	return nil
}
