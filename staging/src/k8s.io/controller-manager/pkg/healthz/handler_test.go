/*
Copyright 2021 The Kubernetes Authors.

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

package healthz

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"k8s.io/apiserver/pkg/server/healthz"
)

func TestMutableHealthzHandler(t *testing.T) {
	badChecker := healthz.NamedCheck("bad", func(r *http.Request) error {
		return fmt.Errorf("bad")
	})
	for _, tc := range []struct {
		name           string
		checkBatches   [][]healthz.HealthChecker
		appendBad      bool // appends bad check after batches above, and see if it fails afterwards
		path           string
		expectedBody   string
		expectedStatus int
	}{
		{
			name:           "empty",
			checkBatches:   [][]healthz.HealthChecker{},
			path:           "/healthz",
			expectedBody:   "ok",
			expectedStatus: http.StatusOK,
		},
		{
			name: "good",
			checkBatches: [][]healthz.HealthChecker{
				{NamedPingChecker("good")},
			},
			path:           "/healthz",
			expectedBody:   "ok",
			expectedStatus: http.StatusOK,
		},
		{
			name: "good verbose", // verbose only applies for successful checks
			checkBatches: [][]healthz.HealthChecker{
				{NamedPingChecker("good")}, // batch 1: good
			},
			path:           "/healthz?verbose=true",
			expectedBody:   "[+]good ok\nhealthz check passed\n",
			expectedStatus: http.StatusOK,
		},
		{
			name: "good and bad, same batch",
			checkBatches: [][]healthz.HealthChecker{
				{NamedPingChecker("good"), badChecker}, // batch 1: good, bad
			},
			path:           "/healthz",
			expectedBody:   "[+]good ok\n[-]bad failed: reason withheld\nhealthz check failed\n",
			expectedStatus: http.StatusInternalServerError,
		},
		{
			name: "good and bad, two batches",
			checkBatches: [][]healthz.HealthChecker{
				{NamedPingChecker("good")}, // batch 1: good
				{badChecker},               // batch 2: bad
			},
			path:           "/healthz",
			expectedBody:   "[+]good ok\n[-]bad failed: reason withheld\nhealthz check failed\n",
			expectedStatus: http.StatusInternalServerError,
		},
		{
			name: "two checks and append bad",
			checkBatches: [][]healthz.HealthChecker{
				{NamedPingChecker("foo"), NamedPingChecker("bar")},
			},
			path:           "/healthz",
			expectedBody:   "ok",
			expectedStatus: http.StatusOK,
			appendBad:      true,
		},
		{
			name: "subcheck",
			checkBatches: [][]healthz.HealthChecker{
				{NamedPingChecker("good")}, // batch 1: good
				{badChecker},               // batch 2: bad
			},
			path:           "/healthz/good",
			expectedBody:   "ok",
			expectedStatus: http.StatusOK,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h := NewMutableHealthzHandler()
			for _, batch := range tc.checkBatches {
				h.AddHealthChecker(batch...)
			}
			req, err := http.NewRequest("GET", fmt.Sprintf("https://example.com%v", tc.path), nil)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			w := httptest.NewRecorder()
			h.ServeHTTP(w, req)
			if w.Code != tc.expectedStatus {
				t.Errorf("unexpected status: expected %v, got %v", tc.expectedStatus, w.Result().StatusCode)
			}
			if w.Body.String() != tc.expectedBody {
				t.Errorf("unexpected body: expected %v, got %v", tc.expectedBody, w.Body.String())
			}
			if tc.appendBad {
				h.AddHealthChecker(badChecker)
				w := httptest.NewRecorder()
				h.ServeHTTP(w, req)
				// should fail
				if w.Code != http.StatusInternalServerError {
					t.Errorf("did not fail after adding bad checker")
				}
			}
		})
	}
}

// TestConcurrentChecks tests that the handler would not block on concurrent healthz requests.
func TestConcurrentChecks(t *testing.T) {
	const N = 5
	stopChan := make(chan interface{})
	defer close(stopChan) // always close no matter passing or not
	concurrentChan := make(chan interface{}, N)
	var concurrentCount int32
	pausingCheck := healthz.NamedCheck("pausing", func(r *http.Request) error {
		atomic.AddInt32(&concurrentCount, 1)
		concurrentChan <- nil
		<-stopChan
		return nil
	})

	h := NewMutableHealthzHandler(pausingCheck)
	for i := 0; i < N; i++ {
		go func() {
			req, _ := http.NewRequest(http.MethodGet, "https://example.com/healthz", nil)
			w := httptest.NewRecorder()
			h.ServeHTTP(w, req)
		}()
	}

	giveUp := time.After(1 * time.Second) // should take <1ms if passing
	for i := 0; i < N; i++ {
		select {
		case <-giveUp:
			t.Errorf("given up waiting for concurrent checks to start.")
			return
		case <-concurrentChan:
			continue
		}
	}

	if concurrentCount != N {
		t.Errorf("expected %v concurrency, got %v", N, concurrentCount)
	}
}
