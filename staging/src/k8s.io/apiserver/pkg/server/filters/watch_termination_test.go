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
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"k8s.io/apimachinery/pkg/runtime"
	apirequest "k8s.io/apiserver/pkg/endpoints/request"
)

func TestWithWatchTerminationDuringShutdown(t *testing.T) {
	tests := []struct {
		name                    string
		requestInfo             *apirequest.RequestInfo
		signal                  *fakeServerShutdownSignal
		wg                      *fakeRequestWaitGroup
		handlerInvoked          int
		statusCodeExpected      int
		retryAfterExpected      bool
		wgInvokedExpected       int
		signalAttachedToContext bool
	}{
		{
			name:               "no RequestInfo attached to request context",
			handlerInvoked:     0,
			statusCodeExpected: http.StatusInternalServerError,
		},
		{
			name:               "request is not a WATCH, not added into wait group",
			requestInfo:        &apirequest.RequestInfo{Verb: "get"},
			handlerInvoked:     1,
			statusCodeExpected: http.StatusOK,
		},
		{
			name:                    "request is a WATCH, wait group is in waiting mode",
			requestInfo:             &apirequest.RequestInfo{Verb: "watch"},
			wg:                      &fakeRequestWaitGroup{waiting: true},
			handlerInvoked:          0,
			signalAttachedToContext: false,
			wgInvokedExpected:       1,
			retryAfterExpected:      true,
			statusCodeExpected:      http.StatusServiceUnavailable,
		},
		{
			name:                    "request is a WATCH, wait group is accepting",
			requestInfo:             &apirequest.RequestInfo{Verb: "watch"},
			wg:                      &fakeRequestWaitGroup{},
			signal:                  &fakeServerShutdownSignal{},
			wgInvokedExpected:       1,
			signalAttachedToContext: true,
			handlerInvoked:          1,
			statusCodeExpected:      http.StatusOK,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var (
				handlerInvokedGot int
				signalGot         *fakeServerShutdownSignal
			)
			delegate := http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
				handlerInvokedGot++
				if signal := apirequest.ServerShutdownSignalFrom(req.Context()); signal != nil {
					signalGot, _ = signal.(*fakeServerShutdownSignal)
				}
				w.WriteHeader(http.StatusOK)
			})

			handler := WithWatchTerminationDuringShutdown(delegate, test.signal, test.wg)

			req, err := http.NewRequest(http.MethodGet, "/apis/groups.k8s.io/v1/namespaces", nil)
			if err != nil {
				t.Fatalf("failed to create new http request - %v", err)
			}
			if test.requestInfo != nil {
				req = req.WithContext(apirequest.WithRequestInfo(req.Context(), test.requestInfo))
			}

			w := httptest.NewRecorder()
			w.Code = 0
			handler.ServeHTTP(w, req)
			responseGot := w.Result()

			if test.handlerInvoked != handlerInvokedGot {
				t.Errorf("expected the handler to be invoked: %d timed, but got: %d", test.handlerInvoked, handlerInvokedGot)
			}
			if test.statusCodeExpected != responseGot.StatusCode {
				t.Errorf("expected status code: %d, but got: %d", test.statusCodeExpected, w.Result().StatusCode)
			}
			retryAfterGot := retryAfterSent(responseGot)
			if test.retryAfterExpected != retryAfterGot {
				t.Errorf("expected retry-after: %t, but got: %t, response: %v#", test.retryAfterExpected, retryAfterGot, responseGot)
			}

			switch {
			case test.signalAttachedToContext:
				if test.signal == nil || test.signal != signalGot {
					t.Errorf("expected request context to have server shutdown signal: %p, but got: %p", test.signal, signalGot)
				}
			default:
				if signalGot != nil {
					t.Errorf("expected request context to not have server shutdown signal: %p, but got: %p", test.signal, signalGot)
				}
			}
			if test.wg == nil {
				return
			}
			if test.wg.inflight != 0 {
				t.Errorf("expected wait group inflight to be zero, but got: %d", test.wg.inflight)
			}
			if test.wgInvokedExpected != test.wg.invoked {
				t.Errorf("expected wait group Add to be invoked: %d times, but got: %d", test.wgInvokedExpected, test.wg.invoked)
			}
		})
	}
}

type fakeServerShutdownSignal struct{}

func (fakeServerShutdownSignal) ShuttingDown() <-chan struct{} { return nil }

type fakeRequestWaitGroup struct {
	waiting           bool
	invoked, inflight int
}

func (f *fakeRequestWaitGroup) Add(delta int) error {
	f.invoked++
	if f.waiting {
		return fmt.Errorf("waitgroup is in waiting mode")
	}
	f.inflight += delta
	return nil
}
func (f *fakeRequestWaitGroup) Done() { f.inflight-- }

func retryAfterSent(resp *http.Response) bool {
	switch {
	case resp.StatusCode == http.StatusServiceUnavailable &&
		resp.Header.Get("Retry-After") == "1" &&
		resp.Header.Get("Content-Type") == runtime.ContentTypeJSON &&
		resp.Header.Get("X-Content-Type-Options") == "nosniff":
		return true
	default:
		return false
	}
}
