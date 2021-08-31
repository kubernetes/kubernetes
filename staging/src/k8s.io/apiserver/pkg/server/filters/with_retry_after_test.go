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

package filters

import (
	"net/http"
	"net/http/httptest"
	"testing"

	utilwaitgroup "k8s.io/apimachinery/pkg/util/waitgroup"
	apirequest "k8s.io/apiserver/pkg/endpoints/request"
)

func TestWithRetryAfter(t *testing.T) {
	tests := []struct {
		name                           string
		shutdownDelayDurationElapsedFn func() <-chan struct{}
		requestURL                     string
		userAgent                      string
		safeWaitGroupIsWaiting         bool
		handlerInvoked                 int
		closeExpected                  string
		retryAfterExpected             string
		statusCodeExpected             int
	}{
		{
			name: "retry-after disabled",
			shutdownDelayDurationElapsedFn: func() <-chan struct{} {
				return newChannel(false)
			},
			requestURL:         "/api/v1/namespaces",
			userAgent:          "foo",
			handlerInvoked:     1,
			closeExpected:      "",
			retryAfterExpected: "",
			statusCodeExpected: http.StatusOK,
		},
		{
			name: "retry-after enabled, request is not exempt",
			shutdownDelayDurationElapsedFn: func() <-chan struct{} {
				return newChannel(true)
			},
			requestURL:         "/api/v1/namespaces",
			userAgent:          "foo",
			handlerInvoked:     0,
			closeExpected:      "close",
			retryAfterExpected: "5",
			statusCodeExpected: http.StatusTooManyRequests,
		},
		{
			name: "retry-after enabled, request is exempt(/metrics)",
			shutdownDelayDurationElapsedFn: func() <-chan struct{} {
				return newChannel(true)
			},
			requestURL:         "/metrics?foo=bar",
			userAgent:          "foo",
			handlerInvoked:     1,
			closeExpected:      "",
			retryAfterExpected: "",
			statusCodeExpected: http.StatusOK,
		},
		{
			name: "retry-after enabled, request is exempt(/livez)",
			shutdownDelayDurationElapsedFn: func() <-chan struct{} {
				return newChannel(true)
			},
			requestURL:         "/livez?verbose",
			userAgent:          "foo",
			handlerInvoked:     1,
			closeExpected:      "",
			retryAfterExpected: "",
			statusCodeExpected: http.StatusOK,
		},
		{
			name: "retry-after enabled, request is exempt(/readyz)",
			shutdownDelayDurationElapsedFn: func() <-chan struct{} {
				return newChannel(true)
			},
			requestURL:         "/readyz?verbose",
			userAgent:          "foo",
			handlerInvoked:     1,
			closeExpected:      "",
			retryAfterExpected: "",
			statusCodeExpected: http.StatusOK,
		},
		{
			name: "retry-after enabled, request is exempt(/healthz)",
			shutdownDelayDurationElapsedFn: func() <-chan struct{} {
				return newChannel(true)
			},
			requestURL:         "/healthz?verbose",
			userAgent:          "foo",
			handlerInvoked:     1,
			closeExpected:      "",
			retryAfterExpected: "",
			statusCodeExpected: http.StatusOK,
		},
		{
			name: "retry-after enabled, request is exempt(local loopback)",
			shutdownDelayDurationElapsedFn: func() <-chan struct{} {
				return newChannel(true)
			},
			requestURL:         "/api/v1/namespaces",
			userAgent:          "kube-apiserver/",
			handlerInvoked:     1,
			closeExpected:      "",
			retryAfterExpected: "",
			statusCodeExpected: http.StatusOK,
		},
		{
			name: "nil channel",
			shutdownDelayDurationElapsedFn: func() <-chan struct{} {
				return nil
			},
			requestURL:         "/api/v1/namespaces",
			userAgent:          "foo",
			handlerInvoked:     1,
			closeExpected:      "",
			retryAfterExpected: "",
			statusCodeExpected: http.StatusOK,
		},
		{
			name: "retry-after enabled, request is exempt(/readyz), SafeWaitGroup is in waiting mode",
			shutdownDelayDurationElapsedFn: func() <-chan struct{} {
				return newChannel(true)
			},
			requestURL:             "/readyz?verbose",
			userAgent:              "foo",
			safeWaitGroupIsWaiting: true,
			handlerInvoked:         1,
			closeExpected:          "",
			retryAfterExpected:     "",
			statusCodeExpected:     http.StatusOK,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var handlerInvoked int
			handler := http.HandlerFunc(func(_ http.ResponseWriter, req *http.Request) {
				handlerInvoked++
			})

			safeWG := new(utilwaitgroup.SafeWaitGroup)
			if test.safeWaitGroupIsWaiting {
				// mark the safe wait group as waiting, it's a blocking call
				// but since the WaitGroup counter is zero it should not block
				safeWG.Wait()
			}

			wrapped := WithWaitGroup(handler, func(*http.Request, *apirequest.RequestInfo) bool {
				return false
			}, safeWG)
			wrapped = WithRetryAfter(wrapped, test.shutdownDelayDurationElapsedFn())

			req, err := http.NewRequest(http.MethodGet, test.requestURL, nil)
			if err != nil {
				t.Fatalf("failed to create new http request - %v", err)
			}

			req.Header.Set("User-Agent", test.userAgent)
			req = req.WithContext(apirequest.WithRequestInfo(req.Context(), &apirequest.RequestInfo{}))

			w := httptest.NewRecorder()
			wrapped.ServeHTTP(w, req)

			if test.handlerInvoked != handlerInvoked {
				t.Errorf("expected the handler to be invoked: %d timed, but got: %d", test.handlerInvoked, handlerInvoked)
			}
			if test.statusCodeExpected != w.Result().StatusCode {
				t.Errorf("expected status code: %d, but got: %d", test.statusCodeExpected, w.Result().StatusCode)
			}

			closeGot := w.Header().Get("Connection")
			if test.closeExpected != closeGot {
				t.Errorf("expected Connection close: %s, but got: %s", test.closeExpected, closeGot)
			}

			retryAfterGot := w.Header().Get("Retry-After")
			if test.retryAfterExpected != retryAfterGot {
				t.Errorf("expected Retry-After: %s, but got: %s", test.retryAfterExpected, retryAfterGot)
			}
		})
	}
}

func newChannel(closed bool) <-chan struct{} {
	ch := make(chan struct{})
	if closed {
		close(ch)
	}
	return ch
}
