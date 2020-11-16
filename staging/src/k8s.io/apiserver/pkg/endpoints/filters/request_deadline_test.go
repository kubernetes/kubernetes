/*
Copyright 2020 The Kubernetes Authors.

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
	"testing"
	"time"

	"k8s.io/apiserver/pkg/endpoints/request"
)

func TestParseTimeout(t *testing.T) {
	tests := []struct {
		name            string
		url             string
		expected        bool
		timeoutExpected time.Duration
		errExpected     error
	}{
		{
			name: "the user does not specify a timeout",
			url:  "/api/v1/namespaces",
		},
		{
			name:            "the user specifies a valid timeout",
			url:             "/api/v1/namespaces?timeout=10s",
			expected:        true,
			timeoutExpected: 10 * time.Second,
		},
		{
			name:        "the use specifies an invalid timeout",
			url:         "/api/v1/namespaces?timeout=foo",
			errExpected: errInvalidTimeoutInURL,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			request, err := http.NewRequest(http.MethodGet, test.url, nil)
			if err != nil {
				t.Fatalf("failed to create new http request - %v", err)
			}

			timeoutGot, ok, err := parseTimeout(request)

			if test.expected != ok {
				t.Errorf("expected: %t, but got: %t", test.expected, ok)
			}
			if test.errExpected != err {
				t.Errorf("expected err: %v, but got: %v", test.errExpected, err)
			}
			if test.timeoutExpected != timeoutGot {
				t.Errorf("expected timeout: %s, but got: %s", test.timeoutExpected, timeoutGot)
			}
		})
	}
}

func TestWithRequestDeadline(t *testing.T) {
	const requestTimeoutMaximum = 60 * time.Second

	tests := []struct {
		name                     string
		requestURL               string
		longRunning              bool
		hasDeadlineExpected      bool
		deadlineExpected         time.Duration
		handlerCallCountExpected int
		statusCodeExpected       int
	}{
		{
			name:                     "the user specifies a valid request timeout",
			requestURL:               "/api/v1/namespaces?timeout=15s",
			longRunning:              false,
			handlerCallCountExpected: 1,
			hasDeadlineExpected:      true,
			deadlineExpected:         14 * time.Second, // to account for the delay in verification
			statusCodeExpected:       http.StatusOK,
		},
		{
			name:                     "the user does not specify any request timeout, default deadline is expected to be set",
			requestURL:               "/api/v1/namespaces?timeout=",
			longRunning:              false,
			handlerCallCountExpected: 1,
			hasDeadlineExpected:      true,
			deadlineExpected:         requestTimeoutMaximum - time.Second, // to account for the delay in verification
			statusCodeExpected:       http.StatusOK,
		},
		{
			name:                     "the request is long running, no deadline is expected to be set",
			requestURL:               "/api/v1/namespaces?timeout=10s",
			longRunning:              true,
			hasDeadlineExpected:      false,
			handlerCallCountExpected: 1,
			statusCodeExpected:       http.StatusOK,
		},
		{
			name:               "the timeout specified is malformed, the request is aborted with HTTP 400",
			requestURL:         "/api/v1/namespaces?timeout=foo",
			longRunning:        false,
			statusCodeExpected: http.StatusBadRequest,
		},
		{
			name:               "the timeout specified exceeds the maximum deadline allowed, the request is aborted with HTTP 400",
			requestURL:         fmt.Sprintf("/api/v1/namespaces?timeout=%s", requestTimeoutMaximum+time.Second),
			longRunning:        false,
			statusCodeExpected: http.StatusBadRequest,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var (
				callCount      int
				hasDeadlineGot bool
				deadlineGot    time.Duration
			)
			handler := http.HandlerFunc(func(_ http.ResponseWriter, req *http.Request) {
				callCount++
				deadlineGot, hasDeadlineGot = deadline(req)
			})

			withDeadline := WithRequestDeadline(
				handler, func(_ *http.Request, _ *request.RequestInfo) bool { return test.longRunning }, requestTimeoutMaximum)
			withDeadline = WithRequestInfo(withDeadline, &fakeRequestResolver{})

			testRequest, err := http.NewRequest(http.MethodGet, test.requestURL, nil)
			if err != nil {
				t.Fatalf("failed to create new http request - %v", err)
			}

			// make sure a default request does not have any deadline set
			remaning, ok := deadline(testRequest)
			if ok {
				t.Fatalf("test setup failed, expected the new HTTP request context to have no deadline but got: %s", remaning)
			}

			w := httptest.NewRecorder()
			withDeadline.ServeHTTP(w, testRequest)

			if test.handlerCallCountExpected != callCount {
				t.Errorf("expected the request handler to be invoked %d times, but was actually invoked %d times", test.handlerCallCountExpected, callCount)
			}

			if test.hasDeadlineExpected != hasDeadlineGot {
				t.Errorf("expected the request context to have deadline set: %t but got: %t", test.hasDeadlineExpected, hasDeadlineGot)
			}

			deadlineGot = deadlineGot.Truncate(time.Second)
			if test.deadlineExpected != deadlineGot {
				t.Errorf("expected a request context with a deadline of %s but got: %s", test.deadlineExpected, deadlineGot)
			}

			statusCodeGot := w.Result().StatusCode
			if test.statusCodeExpected != statusCodeGot {
				t.Errorf("expected status code %d but got: %d", test.statusCodeExpected, statusCodeGot)
			}
		})
	}
}

type fakeRequestResolver struct{}

func (r fakeRequestResolver) NewRequestInfo(req *http.Request) (*request.RequestInfo, error) {
	return &request.RequestInfo{}, nil
}

func deadline(r *http.Request) (time.Duration, bool) {
	if deadline, ok := r.Context().Deadline(); ok {
		remaining := time.Until(deadline)
		return remaining, ok
	}

	return 0, false
}
