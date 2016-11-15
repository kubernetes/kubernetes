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
	"regexp"
	"strings"
	"sync"
	"testing"

	"k8s.io/kubernetes/pkg/api/errors"
)

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
		WithMaxInFlightLimit(
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
			AllowedInflightRequestsNo,
			longRunningRequestCheck,
		),
	)
	defer server.Close()

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
