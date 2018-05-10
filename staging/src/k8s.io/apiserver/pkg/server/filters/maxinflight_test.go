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

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/authentication/user"
	apifilters "k8s.io/apiserver/pkg/endpoints/filters"
	apirequest "k8s.io/apiserver/pkg/endpoints/request"
)

func createMaxInflightServer(callsWg, blockWg *sync.WaitGroup, disableCallsWg *bool, disableCallsWgMutex *sync.Mutex, nonMutating, mutating int) *httptest.Server {
	longRunningRequestCheck := BasicLongRunningRequestCheck(sets.NewString("watch"), sets.NewString("proxy"))

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
		longRunningRequestCheck,
	)
	handler = withFakeUser(handler)
	handler = apifilters.WithRequestInfo(handler, requestInfoFactory)

	return httptest.NewServer(handler)
}

func createMaxInflightPerUserServer(callsWg, blockWg *sync.WaitGroup, disableCallsWg *bool, disableCallsWgMutex *sync.Mutex, limit int) *httptest.Server {
	longRunningRequestCheck := BasicLongRunningRequestCheck(sets.NewString("watch"), sets.NewString("proxy"))

	requestInfoFactory := &apirequest.RequestInfoFactory{APIPrefixes: sets.NewString("apis", "api"), GrouplessAPIPrefixes: sets.NewString("api")}
	handler := WithMaxInFlightPerUserLimit(
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
		limit,
		longRunningRequestCheck,
	)
	handler = withFakeUser(handler)
	handler = apifilters.WithRequestInfo(handler, requestInfoFactory)

	return httptest.NewServer(handler)
}

func withFakeUser(handler http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		userInfo := new(user.DefaultInfo)

		if userName := r.URL.Query().Get("user"); userName != "" {
			userInfo.Name = userName
		}

		if len(r.Header["Groups"]) > 0 {
			userInfo.Groups = r.Header["Groups"]
		}

		r = r.WithContext(apirequest.WithUser(r.Context(), userInfo))

		handler.ServeHTTP(w, r)
	})
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
		if err := expectHTTPGet(server.URL, http.StatusTooManyRequests); err != nil {
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
		if err := expectHTTPPost(server.URL+"/foo/bar/", http.StatusTooManyRequests); err != nil {
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
func expectHTTPPost(url string, code int, groups ...string) error {
	req, err := http.NewRequest(http.MethodPost, url, strings.NewReader("foo bar"))
	if err != nil {
		return err
	}
	for _, group := range groups {
		req.Header.Add("Groups", group)
	}

	r, err := http.DefaultClient.Do(req)
	if err != nil {
		return fmt.Errorf("unexpected error: %v", err)
	}
	if r.StatusCode != code {
		return fmt.Errorf("unexpected response: %v", r.StatusCode)
	}
	return nil
}

func TestMaxInFlightSkipsMasters(t *testing.T) {
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
		if err := expectHTTPPost(server.URL+"/dontwait", http.StatusOK, user.SystemPrivilegedGroup); err != nil {
			t.Error(err)
		}
	}

	// Let all hanging requests finish
	block.Done()

	responses.Wait()
}

// Tests that TestMaxInFlightPerUser works, i.e.
// - "long" requests such as proxy or watch, identified by regexp are not accounted despite
//   hanging for the long time,
// - "short" requests are correctly accounted, i.e. there can be only size of channel passed to the
//   constructor in flight at any given moment,
// - subsequent "short" requests are rejected instantly with appropriate error,
// - subsequent "long" requests are handled normally,
// - we correctly recover after some "short" requests finish, i.e. we can process new ones.
// - this limit accounted for different users separately
func TestMaxInFlightPerUser(t *testing.T) {
	const AllowedInFlightPerUserRequestsNo = 3

	// Calls is used to wait until all server calls are received. We are sending
	// AllowedNonMutatingInflightRequestsNo of 'long' not-accounted requests and the same number of
	// 'short' accounted ones.
	calls := &sync.WaitGroup{}

	// Responses is used to wait until all responses are
	// received. This prevents some async requests getting EOF
	// errors from prematurely closing the server
	responses := &sync.WaitGroup{}

	// Block is used to keep requests in flight for as long as we need to. All requests will
	// be unblocked at the same time.
	block := &sync.WaitGroup{}
	block.Add(1)

	waitForCalls := true
	waitForCallsMutex := sync.Mutex{}

	server := createMaxInflightPerUserServer(calls, block, &waitForCalls, &waitForCallsMutex, AllowedInFlightPerUserRequestsNo)
	defer server.Close()

	usernames := []string{"test", "test2"}

	for _, username := range usernames {
		waitForCallsMutex.Lock()
		waitForCalls = true
		waitForCallsMutex.Unlock()

		// These should hang, but not affect accounting.  use a query param match
		for i := 0; i < AllowedInFlightPerUserRequestsNo; i++ {
			// These should hang waiting on block...
			calls.Add(1)
			responses.Add(1)
			go func() {
				if err := expectHTTPGet(server.URL+"/api/v1/watch/namespaces/default/?user="+username, http.StatusOK); err != nil {
					t.Error(err)
					calls.Done() //decrease WG to prevemnt test hang, due to when `StatusTooManyRequests`, calls.Done() didn't done by handler
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

		// Check that sever is not saturated by not-accounted calls
		if err := expectHTTPGet(server.URL+"/dontwait?user="+username, http.StatusOK); err != nil {
			t.Error(err)
		}

		waitForCallsMutex.Lock()
		waitForCalls = true
		waitForCallsMutex.Unlock()

		// These should hang and be accounted, i.e. saturate the server for user 'test'
		for i := 0; i < AllowedInFlightPerUserRequestsNo; i++ {
			// These should hang waiting on block...
			calls.Add(1)
			responses.Add(1)
			go func() {
				if err := expectHTTPGet(server.URL+"/user?&user="+username, http.StatusOK); err != nil {
					t.Error(err)
					calls.Done()
				}
				responses.Done()
			}()
		}

		// We wait for all calls to be received by the server
		calls.Wait()
	}

	// Disable calls notifications in the server
	waitForCallsMutex.Lock()
	waitForCalls = false
	waitForCallsMutex.Unlock()

	// Validate that all users are saturated
	for _, username := range usernames {
		// Do this multiple times to show that rate limit rejected requests don't block.
		for i := 0; i < 2; i++ {
			if err := expectHTTPGet(server.URL+"/dontwait?user="+username, http.StatusTooManyRequests); err != nil {
				t.Error(err)
			}
		}
		// Validate that non-accounted URLs still work.  use a path regex match
		if err := expectHTTPGet(server.URL+"/api/v1/watch/namespaces/default/dontwait?user="+username, http.StatusOK); err != nil {
			t.Error(err)
		}
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
