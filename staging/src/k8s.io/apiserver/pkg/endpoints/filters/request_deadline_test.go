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
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	auditinternal "k8s.io/apiserver/pkg/apis/audit"
	"k8s.io/apiserver/pkg/audit"
	"k8s.io/apiserver/pkg/audit/policy"
	"k8s.io/apiserver/pkg/endpoints/request"
	testingclock "k8s.io/utils/clock/testing"
)

func TestParseTimeout(t *testing.T) {
	tests := []struct {
		name            string
		url             string
		expected        bool
		timeoutExpected time.Duration
		message         string
	}{
		{
			name: "the user does not specify a timeout",
			url:  "/api/v1/namespaces?timeout=",
		},
		{
			name:            "the user specifies a valid timeout",
			url:             "/api/v1/namespaces?timeout=10s",
			expected:        true,
			timeoutExpected: 10 * time.Second,
		},
		{
			name:     "the user specifies a timeout of 0s",
			url:      "/api/v1/namespaces?timeout=0s",
			expected: true,
		},
		{
			name:    "the user specifies an invalid timeout",
			url:     "/api/v1/namespaces?timeout=foo",
			message: invalidTimeoutInURL,
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
			if test.timeoutExpected != timeoutGot {
				t.Errorf("expected timeout: %s, but got: %s", test.timeoutExpected, timeoutGot)
			}

			errMessageGot := message(err)
			if !strings.Contains(errMessageGot, test.message) {
				t.Errorf("expected error message to contain: %s, but got: %s", test.message, errMessageGot)
			}
		})
	}
}

func TestWithRequestDeadline(t *testing.T) {
	// we have three category of requests based on how long they can run:
	//
	// a) watch: it's a long running request, user can provide a timeout for
	// a watch request by specifying the 'timeoutSeconds' parameter in the
	// request URI. If not specified in the request URI, the apiserver
	// enforces a default (derived from the server
	// run option '--min-request-timeout'
	//
	// b) long-running: all other long running requests except for WATCH,
	// no timeout is enforced by the apiserver.
	//
	// c) short: we call it 'short' here for lack of better terms, these
	// are the non long-running requests, user can provide a timeout for
	// these request(s) by specifying the 'timeout' parameter in the
	// request URI. If not specified in the request URI, the apiserver
	// enforces a default (derived from the server run option
	// '--request-timeout'). A short request can not run longer than
	// 'request-timeout' duration, the apiserver enforces this constraint.
	shortReqDefaultTimeout := 33 * time.Second
	watchReqDefaultTimeout := 420 // 420s = 7m
	tests := []struct {
		name                     string
		timeoutSpecified         string
		reqInfo                  *request.RequestInfo
		timeoutExpected          time.Duration
		jitterExpected           bool // applies to WATCH request only
		handlerCallCountExpected int
		statusCodeExpected       int
	}{
		{
			name:                     "short, user specifies a valid timeout",
			timeoutSpecified:         "timeout=15s",
			reqInfo:                  &request.RequestInfo{Verb: "get"},
			handlerCallCountExpected: 1,
			statusCodeExpected:       http.StatusOK,
			timeoutExpected:          15 * time.Second,
		},
		{
			name:                     "short, specified timeout is 0s, default deadline is expected to be set",
			timeoutSpecified:         "timeout=0s",
			reqInfo:                  &request.RequestInfo{Verb: "get"},
			handlerCallCountExpected: 1,
			timeoutExpected:          shortReqDefaultTimeout,
			statusCodeExpected:       http.StatusOK,
		},
		{
			name:                     "short, user does not specify any timeout, default deadline is expected to be set",
			timeoutSpecified:         "timeout=",
			reqInfo:                  &request.RequestInfo{Verb: "get"},
			handlerCallCountExpected: 1,
			timeoutExpected:          shortReqDefaultTimeout,
			statusCodeExpected:       http.StatusOK,
		},
		{
			name:                     "long running (not watch), no deadline is expected to be set",
			timeoutSpecified:         "timeout=700s&timeoutSeconds=700",
			reqInfo:                  &request.RequestInfo{Verb: ""}, // long-running (non WATCH)
			handlerCallCountExpected: 1,
			timeoutExpected:          0,
			statusCodeExpected:       http.StatusOK,
		},
		{
			name:                     "short, specified timeout is malformed, the request is aborted with HTTP 400",
			timeoutSpecified:         "timeout=foo",
			reqInfo:                  &request.RequestInfo{Verb: "get"},
			handlerCallCountExpected: 0,
			timeoutExpected:          0,
			statusCodeExpected:       http.StatusBadRequest,
		},
		{
			name:                     "short, specified timeout exceeds the maximum allowed, the default deadline is used",
			timeoutSpecified:         fmt.Sprintf("timeout=%s", shortReqDefaultTimeout+time.Second),
			reqInfo:                  &request.RequestInfo{Verb: "get"},
			statusCodeExpected:       http.StatusOK,
			handlerCallCountExpected: 1,
			timeoutExpected:          shortReqDefaultTimeout,
		},
		{
			name:                     "watch, user specifies a valid timeout (shorter than the default)",
			timeoutSpecified:         fmt.Sprintf("timeoutSeconds=%d", watchReqDefaultTimeout-60),
			reqInfo:                  &request.RequestInfo{Verb: "watch"},
			handlerCallCountExpected: 1,
			timeoutExpected:          time.Duration(watchReqDefaultTimeout-60) * time.Second,
			statusCodeExpected:       http.StatusOK,
		},
		{
			name:                     "watch, user specifies a valid timeout (larger than the default)",
			timeoutSpecified:         fmt.Sprintf("timeoutSeconds=%d", watchReqDefaultTimeout+60),
			reqInfo:                  &request.RequestInfo{Verb: "watch"},
			handlerCallCountExpected: 1,
			timeoutExpected:          time.Duration(watchReqDefaultTimeout+60) * time.Second,
			statusCodeExpected:       http.StatusOK,
		},
		{
			name:                     "watch, specified timeout is invalid",
			timeoutSpecified:         "timeoutSeconds=foo",
			reqInfo:                  &request.RequestInfo{Verb: "watch"},
			handlerCallCountExpected: 1,
			timeoutExpected:          time.Duration(watchReqDefaultTimeout) * time.Second,
			jitterExpected:           true,
			statusCodeExpected:       http.StatusOK,
		},
		{
			name:                     "watch, specified timeout is empty, default is applied",
			timeoutSpecified:         "timeoutSeconds=",
			reqInfo:                  &request.RequestInfo{Verb: "watch"},
			handlerCallCountExpected: 1,
			timeoutExpected:          time.Duration(watchReqDefaultTimeout) * time.Second,
			jitterExpected:           true,
			statusCodeExpected:       http.StatusOK,
		},
		{
			name:                     "watch, specified timeout is empty, default is applied",
			timeoutSpecified:         "",
			reqInfo:                  &request.RequestInfo{Verb: "watch"},
			handlerCallCountExpected: 1,
			timeoutExpected:          time.Duration(watchReqDefaultTimeout) * time.Second,
			jitterExpected:           true,
			statusCodeExpected:       http.StatusOK,
		},
		{
			name:                     "watch, specified timeout is 0s",
			timeoutSpecified:         "timeoutSeconds=0",
			reqInfo:                  &request.RequestInfo{Verb: "watch"},
			handlerCallCountExpected: 1,
			timeoutExpected:          time.Duration(watchReqDefaultTimeout) * time.Second,
			jitterExpected:           true,
			statusCodeExpected:       http.StatusOK,
		},
		{
			// TODO: this is the current behavior, should we apply the default instead?
			name:                     "watch, specified timeout is a negative integer",
			timeoutSpecified:         "timeoutSeconds=-5",
			reqInfo:                  &request.RequestInfo{Verb: "watch"},
			handlerCallCountExpected: 1,
			timeoutExpected:          -5 * time.Second,
			statusCodeExpected:       http.StatusOK,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			fakeSink := &fakeAuditSink{}
			fakeRuleEvaluator := policy.NewFakePolicyRuleEvaluator(auditinternal.LevelRequestResponse, nil)
			longRunningFn := func(_ *http.Request, _ *request.RequestInfo) bool {
				switch test.reqInfo.Verb {
				case "get":
					return false // indicates a short request
				case "watch":
					return true // indicates a WATCH request
				}
				return true // long-running requests except for WATCH
			}
			fakeReqResolver := &fakeRequestResolver{reqInfo: test.reqInfo}

			var handlerInvokedGot int
			h := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				handlerInvokedGot++
				ctx := r.Context()

				var receivedAt, deadlineGot time.Time
				var ok bool
				if receivedAt, ok = request.ReceivedTimestampFrom(ctx); !ok {
					t.Errorf("expected the request context to have a received at timestamp, but got: %s", receivedAt)
				}

				deadlineGot, ok = ctx.Deadline()
				switch {
				case test.timeoutExpected != 0:
					if !ok {
						t.Errorf("expected the request context to have a deadline")
					}
					timeoutGot := deadlineGot.Sub(receivedAt)
					if !test.jitterExpected && test.timeoutExpected != timeoutGot {
						t.Errorf("expected the request context to have a deadline of: %s, but got: %s", test.timeoutExpected, timeoutGot)
					}
					if low, high := test.timeoutExpected, 2*test.timeoutExpected; test.jitterExpected && !(timeoutGot >= low && timeoutGot < high) {
						t.Errorf("expected the request context to have a deadline in this range[%s - %s), but got: %s", low, high, timeoutGot)
					}
				default:
					if ok {
						t.Errorf("did not expect the request context to have a deadline")
					}
				}
			})
			handler := WithRequestDeadline(h, fakeSink, fakeRuleEvaluator, longRunningFn, newSerializer(), shortReqDefaultTimeout, watchReqDefaultTimeout)
			handler = WithRequestInfo(handler, fakeReqResolver)
			handler = WithRequestReceivedTimestamp(handler)
			handler = WithAuditInit(handler)

			req, err := http.NewRequest(http.MethodGet, fmt.Sprintf("/ping?%s", test.timeoutSpecified), nil)
			if err != nil {
				t.Errorf("failed to create a new http request - %v", err)
				return
			}
			// make sure the request does not have any deadline set
			if _, ok := req.Context().Deadline(); ok {
				t.Errorf("test setup failed, expected the new HTTP request context to have no deadline")
			}

			w := httptest.NewRecorder()
			handler.ServeHTTP(w, req)

			statusCodeGot := w.Result().StatusCode
			if test.statusCodeExpected != statusCodeGot {
				t.Errorf("expected status code %d but got: %d", test.statusCodeExpected, statusCodeGot)
			}
			if test.handlerCallCountExpected != handlerInvokedGot {
				t.Errorf("expected the request handler to be invoked %d times, but was actually invoked %d times", test.handlerCallCountExpected, handlerInvokedGot)
			}
		})
	}
}

func TestWithRequestDeadlineWithClock(t *testing.T) {
	var (
		hasDeadlineGot bool
		deadlineGot    time.Duration
	)
	handler := http.HandlerFunc(func(_ http.ResponseWriter, req *http.Request) {
		deadlineGot, hasDeadlineGot = deadline(req)
	})

	// if the deadline filter uses the clock instead of using the request started timestamp from the context
	// then we will see a request deadline of about a minute.
	receivedTimestampExpected := time.Now().Add(time.Minute)
	fakeClock := testingclock.NewFakeClock(receivedTimestampExpected)

	fakeSink := &fakeAuditSink{}
	fakeRuleEvaluator := policy.NewFakePolicyRuleEvaluator(auditinternal.LevelRequestResponse, nil)
	longRunningFunc := func(_ *http.Request, _ *request.RequestInfo) bool { return false }
	parser := &timeoutParser{watchReqDefaultTimeout: 7 * time.Minute, shortReqDefaultTimeout: time.Minute}
	withDeadline := withRequestDeadline(handler, fakeSink, fakeRuleEvaluator, longRunningFunc, newSerializer(), fakeClock, parser)
	withDeadline = WithRequestInfo(withDeadline, &fakeRequestResolver{})

	testRequest := newRequest(t, "/api/v1/namespaces?timeout=1s")
	// the request has arrived just now.
	testRequest = testRequest.WithContext(request.WithReceivedTimestamp(testRequest.Context(), time.Now()))

	w := httptest.NewRecorder()
	withDeadline.ServeHTTP(w, testRequest)

	if !hasDeadlineGot {
		t.Error("expected the request context to have deadline set")
	}

	// we expect a deadline <= 1s since the filter should use the request started timestamp from the context.
	if deadlineGot > time.Second {
		t.Errorf("expected a request context with a deadline <= %s, but got: %s", time.Second, deadlineGot)
	}
}

func TestWithRequestDeadlineWithInvalidTimeoutIsAudited(t *testing.T) {
	var handlerInvoked bool
	handler := http.HandlerFunc(func(_ http.ResponseWriter, req *http.Request) {
		handlerInvoked = true
	})

	fakeSink := &fakeAuditSink{}
	fakeRuleEvaluator := policy.NewFakePolicyRuleEvaluator(auditinternal.LevelRequestResponse, nil)
	withDeadline := WithRequestDeadline(handler, fakeSink, fakeRuleEvaluator,
		func(_ *http.Request, _ *request.RequestInfo) bool { return false }, newSerializer(), time.Minute, 700)
	withDeadline = WithRequestInfo(withDeadline, &fakeRequestResolver{})

	testRequest := newRequest(t, "/api/v1/namespaces?timeout=foo")
	w := httptest.NewRecorder()
	withDeadline.ServeHTTP(w, testRequest)

	if handlerInvoked {
		t.Error("expected the request to fail and the handler to be skipped")
	}

	statusCodeGot := w.Result().StatusCode
	if statusCodeGot != http.StatusBadRequest {
		t.Errorf("expected status code %d, but got: %d", http.StatusBadRequest, statusCodeGot)
	}
	// verify that the audit event from the request context is written to the audit sink.
	if len(fakeSink.events) != 1 {
		t.Fatalf("expected audit sink to have 1 event, but got: %d", len(fakeSink.events))
	}
}

func TestWithRequestDeadlineWithPanic(t *testing.T) {
	var (
		panicErrGot interface{}
		ctxGot      context.Context
	)

	panicErrExpected := errors.New("apiserver panic'd")
	handler := http.HandlerFunc(func(_ http.ResponseWriter, req *http.Request) {
		ctxGot = req.Context()
		panic(panicErrExpected)
	})

	fakeSink := &fakeAuditSink{}
	fakeRuleEvaluator := policy.NewFakePolicyRuleEvaluator(auditinternal.LevelRequestResponse, nil)
	withDeadline := WithRequestDeadline(handler, fakeSink, fakeRuleEvaluator,
		func(_ *http.Request, _ *request.RequestInfo) bool { return false }, newSerializer(), 1*time.Minute, 700)
	withDeadline = WithRequestInfo(withDeadline, &fakeRequestResolver{})
	withPanicRecovery := http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		defer func() {
			panicErrGot = recover()
		}()
		withDeadline.ServeHTTP(w, req)
	})

	testRequest := newRequest(t, "/api/v1/namespaces?timeout=1s")
	w := httptest.NewRecorder()
	withPanicRecovery.ServeHTTP(w, testRequest)

	if panicErrExpected != panicErrGot {
		t.Errorf("expected panic error: %#v, but got: %#v", panicErrExpected, panicErrGot)
	}
	if ctxGot.Err() != context.Canceled {
		t.Error("expected the request context to be canceled on handler panic")
	}
}

func TestWithRequestDeadlineWithRequestTimesOut(t *testing.T) {
	timeout := 100 * time.Millisecond
	var errGot error
	handler := http.HandlerFunc(func(_ http.ResponseWriter, req *http.Request) {
		ctx := req.Context()
		select {
		case <-time.After(timeout + time.Second):
			errGot = fmt.Errorf("expected the request context to have timed out in %s", timeout)
		case <-ctx.Done():
			errGot = ctx.Err()
		}
	})

	fakeSink := &fakeAuditSink{}
	fakeRuleEvaluator := policy.NewFakePolicyRuleEvaluator(auditinternal.LevelRequestResponse, nil)
	withDeadline := WithRequestDeadline(handler, fakeSink, fakeRuleEvaluator,
		func(_ *http.Request, _ *request.RequestInfo) bool { return false }, newSerializer(), 1*time.Minute, 700)
	withDeadline = WithRequestInfo(withDeadline, &fakeRequestResolver{})

	testRequest := newRequest(t, fmt.Sprintf("/api/v1/namespaces?timeout=%s", timeout))
	w := httptest.NewRecorder()
	withDeadline.ServeHTTP(w, testRequest)

	if errGot != context.DeadlineExceeded {
		t.Errorf("expected error: %#v, but got: %#v", context.DeadlineExceeded, errGot)
	}
}

func TestWithFailedRequestAudit(t *testing.T) {
	tests := []struct {
		name                          string
		statusErr                     *apierrors.StatusError
		errorHandlerCallCountExpected int
		statusCodeExpected            int
		auditExpected                 bool
	}{
		{
			name:                          "bad request, the error handler is invoked and the request is audited",
			statusErr:                     apierrors.NewBadRequest("error serving request"),
			errorHandlerCallCountExpected: 1,
			statusCodeExpected:            http.StatusBadRequest,
			auditExpected:                 true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var (
				errorHandlerCallCountGot int
				rwGot                    http.ResponseWriter
				requestGot               *http.Request
			)

			errorHandler := http.HandlerFunc(func(rw http.ResponseWriter, req *http.Request) {
				http.Error(rw, "error serving request", http.StatusBadRequest)

				errorHandlerCallCountGot++
				requestGot = req
				rwGot = rw
			})

			fakeSink := &fakeAuditSink{}
			fakeRuleEvaluator := policy.NewFakePolicyRuleEvaluator(auditinternal.LevelRequestResponse, nil)

			withAudit := withFailedRequestAudit(errorHandler, test.statusErr, fakeSink, fakeRuleEvaluator)

			w := httptest.NewRecorder()
			testRequest := newRequest(t, "/apis/v1/namespaces/default/pods")
			info := request.RequestInfo{}
			testRequest = testRequest.WithContext(request.WithRequestInfo(testRequest.Context(), &info))

			withAudit.ServeHTTP(w, testRequest)

			if test.errorHandlerCallCountExpected != errorHandlerCallCountGot {
				t.Errorf("expected the testRequest handler to be invoked %d times, but was actually invoked %d times", test.errorHandlerCallCountExpected, errorHandlerCallCountGot)
			}

			statusCodeGot := w.Result().StatusCode
			if test.statusCodeExpected != statusCodeGot {
				t.Errorf("expected status code %d, but got: %d", test.statusCodeExpected, statusCodeGot)
			}

			if test.auditExpected {
				// verify that the right http.ResponseWriter is passed to the error handler
				_, ok := rwGot.(*auditResponseWriter)
				if !ok {
					t.Errorf("expected an http.ResponseWriter of type: %T but got: %T", &auditResponseWriter{}, rwGot)
				}

				auditEventGot := audit.AuditEventFrom(requestGot.Context())
				if auditEventGot == nil {
					t.Fatal("expected an audit event object but got nil")
				}
				if auditEventGot.Stage != auditinternal.StageResponseStarted {
					t.Errorf("expected audit event Stage: %s, but got: %s", auditinternal.StageResponseStarted, auditEventGot.Stage)
				}
				if auditEventGot.ResponseStatus == nil {
					t.Fatal("expected a ResponseStatus field of the audit event object, but got nil")
				}
				if test.statusCodeExpected != int(auditEventGot.ResponseStatus.Code) {
					t.Errorf("expected audit event ResponseStatus.Code: %d, but got: %d", test.statusCodeExpected, auditEventGot.ResponseStatus.Code)
				}
				if test.statusErr.Error() != auditEventGot.ResponseStatus.Message {
					t.Errorf("expected audit event ResponseStatus.Message: %s, but got: %s", test.statusErr, auditEventGot.ResponseStatus.Message)
				}

				// verify that the audit event from the request context is written to the audit sink.
				if len(fakeSink.events) != 1 {
					t.Fatalf("expected audit sink to have 1 event, but got: %d", len(fakeSink.events))
				}
				auditEventFromSink := fakeSink.events[0]
				if !reflect.DeepEqual(auditEventGot, auditEventFromSink) {
					t.Errorf("expected the audit event from the request context to be written to the audit sink, but got diffs: %s", cmp.Diff(auditEventGot, auditEventFromSink))
				}
			}
		})
	}
}

func newRequest(t *testing.T, requestURL string) *http.Request {
	req, err := http.NewRequest(http.MethodGet, requestURL, nil)
	if err != nil {
		t.Fatalf("failed to create new http request - %v", err)
	}
	ctx := audit.WithAuditContext(req.Context())
	return req.WithContext(ctx)
}

func message(err error) string {
	if err != nil {
		return err.Error()
	}

	return ""
}

func newSerializer() runtime.NegotiatedSerializer {
	scheme := runtime.NewScheme()
	return serializer.NewCodecFactory(scheme).WithoutConversion()
}

type fakeRequestResolver struct {
	reqInfo *request.RequestInfo
}

func (r fakeRequestResolver) NewRequestInfo(req *http.Request) (*request.RequestInfo, error) {
	if r.reqInfo != nil {
		return r.reqInfo, nil
	}
	return &request.RequestInfo{}, nil
}

func deadline(r *http.Request) (time.Duration, bool) {
	if deadline, ok := r.Context().Deadline(); ok {
		remaining := time.Until(deadline)
		return remaining, ok
	}

	return 0, false
}
