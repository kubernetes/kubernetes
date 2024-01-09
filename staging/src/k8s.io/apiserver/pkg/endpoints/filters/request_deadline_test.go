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
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/util/wait"
	auditinternal "k8s.io/apiserver/pkg/apis/audit"
	"k8s.io/apiserver/pkg/audit"
	"k8s.io/apiserver/pkg/audit/policy"
	"k8s.io/apiserver/pkg/endpoints/request"
	responsewritertesting "k8s.io/apiserver/pkg/endpoints/responsewriter/testing"
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
			name:                     "the user specifies a valid request timeout",
			requestURL:               "/api/v1/namespaces?timeout=15s",
			longRunning:              false,
			handlerCallCountExpected: 1,
			hasDeadlineExpected:      true,
			deadlineExpected:         14 * time.Second, // to account for the delay in verification
			statusCodeExpected:       http.StatusOK,
		},
		{
			name:                     "the specified timeout is 0s, default deadline is expected to be set",
			requestURL:               "/api/v1/namespaces?timeout=0s",
			longRunning:              false,
			handlerCallCountExpected: 1,
			hasDeadlineExpected:      true,
			deadlineExpected:         requestTimeoutMaximum - time.Second, // to account for the delay in verification
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
			name:                     "the timeout specified exceeds the maximum deadline allowed, the default deadline is used",
			requestURL:               fmt.Sprintf("/api/v1/namespaces?timeout=%s", requestTimeoutMaximum+time.Second),
			longRunning:              false,
			statusCodeExpected:       http.StatusOK,
			handlerCallCountExpected: 1,
			hasDeadlineExpected:      true,
			deadlineExpected:         requestTimeoutMaximum - time.Second, // to account for the delay in verification
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

			fakeSink := &fakeAuditSink{}
			fakeRuleEvaluator := policy.NewFakePolicyRuleEvaluator(auditinternal.LevelRequestResponse, nil)
			withDeadline := WithRequestDeadline(handler, fakeSink, fakeRuleEvaluator,
				func(_ *http.Request, _ *request.RequestInfo) bool { return test.longRunning },
				newSerializer(), requestTimeoutMaximum)
			withDeadline = WithRequestInfo(withDeadline, &fakeRequestResolver{})

			testRequest := newRequest(t, test.requestURL)

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
	withDeadline := withRequestDeadline(handler, fakeSink, fakeRuleEvaluator,
		func(_ *http.Request, _ *request.RequestInfo) bool { return false }, newSerializer(), time.Minute, fakeClock)
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
		func(_ *http.Request, _ *request.RequestInfo) bool { return false }, newSerializer(), time.Minute)
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
		func(_ *http.Request, _ *request.RequestInfo) bool { return false }, newSerializer(), 1*time.Minute)
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
		func(_ *http.Request, _ *request.RequestInfo) bool { return false }, newSerializer(), 1*time.Minute)
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

func TestPerHandlerWriteDeadlineHTTP2WithWriteShouldReturnErrorAfterTimeout(t *testing.T) {
	// This test documents the behavior of the per handler write
	// deadline with a standard net http2 server.
	//
	// scenario: http2, write deadline is set, after timeout occurs we expect
	// the Write method of the ResponseWriter object to return an error.
	t.Parallel()
	clientDoneCh, handlerDoneCh := make(chan struct{}), make(chan error, 1)
	server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		defer close(handlerDoneCh)
		if req.ProtoMajor != 2 {
			t.Errorf("expected an HTTP/2.0 request, but got: %s", req.Proto)
			return
		}

		ctrl := http.NewResponseController(w)
		if err := ctrl.SetWriteDeadline(time.Now().Add(100 * time.Millisecond)); err != nil {
			t.Errorf("expected no error from SetReadDeadline, but got: %v", err)
			return
		}

		// a) the handler blocks here indefinitely
		// b) this forces the handler write timeout
		// to occur on the server side
		// c) the http2 client should receive a stream error
		// immediately after the write timeout occurs.
		<-clientDoneCh

		// NOTE: Write to the ResponseWriter object may succeed
		// immediately after write timeout occurs, but we expect Write
		// to return an error eventually; as soon as the underlying
		// buffer is full, Write is expected to return an error.
		func() {
			now := time.Now()
			count := 0
			defer func() {
				t.Logf("After timeout, Write was invoked %d times, duration: %s", count, time.Since(now))
			}()

			for {
				count++
				if _, err := w.Write(bytes.Repeat([]byte("a"), 1024)); err != nil {
					handlerDoneCh <- err
					break
				}
			}
		}()
	}))

	defer server.Close()
	server.EnableHTTP2 = true
	server.StartTLS()

	client := server.Client()
	func() {
		defer close(clientDoneCh)
		// we intentionally don't set any transport or client timeout to
		// simulate a default client-go client.
		_, err := client.Get(server.URL + "/foo")
		if !responsewritertesting.IsStreamReadOrWriteTimeout(err) {
			t.Errorf("expected a stream reset error, but got: %v", err)
		}
	}()

	select {
	case err := <-handlerDoneCh:
		if err == nil {
			t.Errorf("expected an error from Write after timeout")
			return
		}
		t.Logf("Write (invoked after timeout) returned an error: %v", err)
	case <-time.After(wait.ForeverTestTimeout):
		t.Errorf("expected the request handler to have terminated")
	}
}

func TestPerHandlerWriteDeadlineHTTP1WithWriteShouldReturnErrorAfterTimeout(t *testing.T) {
	// This test documents the behavior of the per handler write
	// deadline with a standard net http/1x server.
	//
	// scenario: http/1x, write deadline is set, after timeout occurs we expect
	// the Write method of the ResponseWriter object to return an error.
	t.Parallel()
	handlerDoneCh := make(chan error, 1)
	server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		defer close(handlerDoneCh)
		if req.ProtoMajor != 1 {
			t.Errorf("expected an HTTP/1x request, but got: %s", req.Proto)
			return
		}

		ctrl := http.NewResponseController(w)
		if err := ctrl.SetWriteDeadline(time.Now().Add(5 * time.Millisecond)); err != nil {
			t.Errorf("expected no error from SetReadDeadline, but got: %v", err)
			return
		}

		// wait for timeout to occur, 10s to eliminate flakes in ci
		<-time.After(10 * time.Second)

		// NOTE: Write to the ResponseWriter object may succeed
		// immediately after write timeout occurs, but we expect Write
		// to return an error eventually; as soon as the underlying
		// buffer is full, Write is expected to return an error.
		func() {
			now := time.Now()
			count := 0
			defer func() {
				t.Logf("After timeout, Write was invoked %d times, duration: %s", count, time.Since(now))
			}()

			for {
				count++
				if _, err := w.Write(bytes.Repeat([]byte("a"), 1024)); err != nil {
					handlerDoneCh <- err
					break
				}
			}
		}()
	}))

	defer server.Close()
	server.StartTLS()

	client := server.Client()
	_, err := client.Get(server.URL + "/foo")
	if err == nil {
		t.Errorf("expected a stream reset error, but got: %v", err)
	}

	select {
	case err := <-handlerDoneCh:
		if err == nil {
			t.Errorf("expected an error from Write after timeout")
			return
		}
		t.Logf("Write (invoked after timeout) returned an error: %v", err)
	case <-time.After(wait.ForeverTestTimeout):
		t.Errorf("expected the request handler to have terminated")
	}
}

func TestPerHandlerWriteDeadlineHTTP2WithTimeoutBeforeHandlerWrites(t *testing.T) {
	// This test documents the behavior of the per handler write
	// deadline with a standard net http2 server.
	//
	// scenario: http2, write deadline is set, timeout occurs before
	// the handler writes to the ResponseWriter object.
	t.Parallel()
	clientDoneCh, handlerDoneCh := make(chan struct{}), make(chan struct{})
	server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		defer close(handlerDoneCh)
		if req.ProtoMajor != 2 {
			t.Errorf("expected an HTTP/2.0 request, but got: %s", req.Proto)
			return
		}
		flusher, ok := w.(interface{ FlushError() error })
		if !ok {
			t.Errorf("expected ResponseWriter object to implement FlushError")
			return
		}

		ctrl := http.NewResponseController(w)
		if err := ctrl.SetWriteDeadline(time.Now().Add(100 * time.Millisecond)); err != nil {
			t.Errorf("expected no error from SetReadDeadline, but got: %v", err)
			return
		}

		// a) the handler blocks here indefinitely
		// b) this forces the handler write timeout
		// to occur on the server side
		// c) the http2 client should receive a stream error
		// immediately after the write timeout occurs.
		<-clientDoneCh

		// Write to the ResponseWriter object may succeed immediately
		// after the timeout occurs if the underlying buffer has room.
		if _, err := w.Write([]byte("hello")); err != nil {
			t.Logf("ResponseWriter.Write returned an error after timeout: %v", err)
		}
		// FlushError, on the other hand, is expected to
		// return a timeout error immediately.
		if err := flusher.FlushError(); !errors.Is(err, os.ErrDeadlineExceeded) {
			t.Errorf("expected an os.ErrDeadlineExceeded, but got: %v", err)
		}

		// NOTE: the handler goroutine is expected to terminate as soon
		// as it receives an error from either Write or FlushError. This
		// is how we prevent a request handler from running indefinitely.
	}))

	defer server.Close()
	server.EnableHTTP2 = true
	server.StartTLS()

	client := server.Client()
	func() {
		defer close(clientDoneCh)
		_, err := client.Get(server.URL + "/foo")
		if !responsewritertesting.IsStreamReadOrWriteTimeout(err) {
			t.Errorf("expected a stream reset error, but got: %v", err)
		}
	}()

	select {
	case <-handlerDoneCh:
	case <-time.After(wait.ForeverTestTimeout):
		t.Errorf("expected the request handler to have terminated")
	}
}

func TestPerHandlerWriteDeadlineHTTP1WithTimeoutBeforeHandlerWrites(t *testing.T) {
	// This test documents the behavior of the per handler write
	// deadline with a standard net http/1x server.
	//
	// scenario: http/1x, write deadline is set, timeout occurs before
	// the handler writes to the ResponseWriter object.
	t.Parallel()
	handlerDoneCh := make(chan struct{})
	server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		defer close(handlerDoneCh)
		if req.ProtoMajor != 1 {
			t.Errorf("expected an HTTP/1x request, but got: %s", req.Proto)
			return
		}
		flusher, ok := w.(interface{ FlushError() error })
		if !ok {
			t.Errorf("expected ResponseWriter object to implement FlushError")
			return
		}

		ctrl := http.NewResponseController(w)
		if err := ctrl.SetWriteDeadline(time.Now().Add(1 * time.Millisecond)); err != nil {
			t.Errorf("expected no error from SetReadDeadline, but got: %v", err)
			return
		}

		// NOTE: if we block the handler here, the client (not setting a
		// transport or client timeout) will hang indefinitely, even
		// though we have set the write deadline on the server side.
		//
		// this could be a problem in the real-world, for example,
		// client-go does not set any transport or client timeout by
		// default, and today the legacy timeout filter in the
		// apiserver sends a timeout response to the client after 60s.
		// In order for us to keep this behavior, we have to ensure
		// that our request handler terminates by respecting the
		// read/write handler deadline semantics.

		// we wait long enough for the write deadline to occur
		// TODO: wait for 10s to avoid flakiness due to overloaded
		// ci env, is there a way to make it flake free without
		// using a sleep?
		<-time.After(10 * time.Second)

		if _, err := w.Write([]byte("hello")); err != nil {
			t.Logf("ResponseWriter.Write returned an error: %v", err)
		}
		// FlushError, on the other hand, is expected to return a
		// timeout error, the handler goroutine is expected to check for
		// an error from FlushError and terminate accordingly.
		// This is how we avoid a request handler running indefinitely.
		if err := flusher.FlushError(); !errors.Is(err, os.ErrDeadlineExceeded) {
			t.Errorf("expected an os.ErrDeadlineExceeded, but got: %v", err)
		}
	}))

	defer server.Close()
	server.StartTLS()

	client := server.Client()
	// we intentionally don't set any transport or client timeout to
	// simulate a default client-go client.
	resp, err := client.Get(server.URL + "/foo")
	t.Logf("the server has returned response: %#v, err: %v", resp, err)
	if err == nil {
		t.Errorf("expected an error, but got none")
	}

	select {
	case <-handlerDoneCh:
	case <-time.After(wait.ForeverTestTimeout):
		t.Errorf("expected the handler to have terminated")
	}
}

func TestPerHandlerWriteDeadlineHTTP2WithTimeoutAfterHandlerWrites(t *testing.T) {
	// This test documents the behavior of the per handler write
	// deadline with a standard net http2 server.
	//
	// scenario: http2, write deadline is set, timeout occurs after
	// the handler writes to ResponseWriter object.
	t.Parallel()
	clientReceivedCh, handlerDoneCh := make(chan struct{}), make(chan error, 1)
	server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		defer close(handlerDoneCh)
		if req.ProtoMajor != 2 {
			t.Errorf("expected an HTTP/2.0 request, but got: %s", req.Proto)
			return
		}
		flusher, ok := w.(interface{ FlushError() error })
		if !ok {
			t.Errorf("expected ResponseWriter object to implement FlushError")
			return
		}

		// write and flush
		if _, err := w.Write([]byte("hello")); err != nil {
			t.Errorf("expected no error from Write, but got: %v", err)
			return
		}
		if err := flusher.FlushError(); err != nil {
			t.Errorf("expected no error from FlushError, but got: %v", err)
		}

		ctrl := http.NewResponseController(w)
		if err := ctrl.SetWriteDeadline(time.Now().Add(100 * time.Millisecond)); err != nil {
			t.Errorf("expected no error from SetReadDeadline, but got: %v", err)
			return
		}

		// a) the handler blocks here indefinitely
		// b) this forces the write timeout to occur on the server side
		// c) the client should receive the response written to
		// d) wait for the client to finish reading the response body
		// after receiving it, due to the handler write timing out
		<-clientReceivedCh

		// at this point the client has already received a response
		if _, err := w.Write([]byte("world")); err != nil {
			t.Logf("ResponseWriter.Write returned an error: %v", err)
		}
		if err := flusher.FlushError(); err != nil {
			handlerDoneCh <- err
		}
	}))

	defer server.Close()
	server.EnableHTTP2 = true
	server.StartTLS()

	client := server.Client()
	func() {
		defer close(clientReceivedCh)

		resp, err := client.Get(server.URL + "/foo")
		if err != nil {
			t.Errorf("unexpected connection error: %v", err)
			return
		}
		// we expect OK due to the initial write and flush
		if resp.StatusCode != http.StatusOK {
			t.Errorf("expected staus code: %d, but got: %d", http.StatusOK, resp.StatusCode)
		}
		// we expect to a get a stream reset error while
		// reading the body of the response.
		defer closeResponseBody(t, resp)
		if _, err := io.Copy(io.Discard, resp.Body); !responsewritertesting.IsStreamReadOrWriteTimeout(err) {
			t.Errorf("expected http2 stream error, but got: %v", err)
		}
	}()

	// now wait for the handler to terminate with the right error
	select {
	case err := <-handlerDoneCh:
		if !errors.Is(err, os.ErrDeadlineExceeded) {
			t.Errorf("expected an os.ErrDeadlineExceeded, but got: %v", err)
		}
	case <-time.After(wait.ForeverTestTimeout):
		t.Errorf("expected the handler to have terminated")
	}
}

func TestPerHandlerWriteDeadlineHTTP1WithTimeoutAfterHandlerWrites(t *testing.T) {
	// This test documents the behavior of the per handler write
	// deadline with a standard net http/1x server.
	//
	// scenario: http/1x, write deadline is set, the handler writes partial
	// content to the ResponseWriter object, it sets the write deadline,
	// timeout occurs, any attempt flush the ResponseWriter object
	// should result an error.
	//
	// with http/1x, a blocked handler also blocks the client indefinitely
	// unless client-side timeout is enforced.
	t.Parallel()
	handlerDoneCh := make(chan struct{})
	server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		defer close(handlerDoneCh)
		now := time.Now()
		if req.ProtoMajor != 1 {
			t.Errorf("expected an HTTP/1x request, but got: %s", req.Proto)
			return
		}
		flusher, ok := w.(interface{ FlushError() error })
		if !ok {
			t.Errorf("expected ResponseWriter object to implement FlushError")
			return
		}
		if _, err := w.Write([]byte("hello")); err != nil {
			t.Errorf("expected no error from Write, but got: %v", err)
		}
		if err := flusher.FlushError(); err != nil {
			t.Errorf("expected no error from FlushError, but got: %v", err)
		}

		ctrl := http.NewResponseController(w)
		if err := ctrl.SetWriteDeadline(time.Now().Add(100 * time.Millisecond)); err != nil {
			t.Errorf("expected no error from SetWriteDeadline, but got: %v", err)
			return
		}

		// NOTE: if we block the handler here, the client (not setting a
		// transport or client timeout) will hang indefinitely, even
		// though we have set the write deadline on the server side.
		//
		// this could be a problem in the real-world, for example,
		// client-go does not set any transport or client timeout by
		// default, and today the legacy timeout filter in the
		// apiserver sends a timeout response to the client after 60s.
		// In order for us to keep this behavior, we have to ensure
		// that our request handler terminates by respecting the
		// read/write handler deadline semantics.

		// keep writing until write timeout occurs
		// TODO: is there a better way to detect a timeout here?
		if err := wait.PollUntilContextTimeout(context.Background(), 5*time.Millisecond, wait.ForeverTestTimeout, true, func(ctx context.Context) (bool, error) {
			if _, err := w.Write(bytes.Repeat([]byte("foo"), 1024)); err != nil {
				return true, nil
			}
			return false, nil
		}); err != nil {
			t.Errorf("ResponseWriter.Write never returned an error: %v", err)
		}

		if _, err := w.Write([]byte("foo")); err != nil {
			t.Logf("ResponseWriter.Write returned an error: %v", err)
		}
		if err := flusher.FlushError(); !errors.Is(err, os.ErrDeadlineExceeded) {
			t.Errorf("expected an os.ErrDeadlineExceeded, but got: %v", err)
		}

		t.Logf("the request handler ran for %s", time.Since(now))
	}))

	defer server.Close()
	server.StartTLS()

	client := server.Client()
	resp, err := client.Get(server.URL + "/foo")
	if err != nil {
		t.Errorf("expected no error from client.Get, but got: %v", err)
		return
	}
	// we expect OK due to the initial write and flush
	if resp.StatusCode != http.StatusOK {
		t.Errorf("expected staus code: %d, but got: %d", http.StatusOK, resp.StatusCode)
	}
	defer closeResponseBody(t, resp)
	if _, err = io.ReadAll(resp.Body); err == nil {
		t.Errorf("expected an error while reading the Response Body")
	}

	select {
	case <-handlerDoneCh:
	case <-time.After(wait.ForeverTestTimeout):
		t.Errorf("expected the request handler to have terminated")
	}
}

func TestPerHandlerWriteTimeoutHTTP2WithHandlerWritingIndefinitely(t *testing.T) {
	// This test documents the behavior of the per handler write
	// deadline with a standard net http2 server.
	//
	// scenario: http2, write deadline set, handler keeps
	// writing to the ResponseWriter object.
	t.Parallel()
	handlerDoneCh := make(chan error, 1)
	server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		defer close(handlerDoneCh)
		if req.ProtoMajor != 2 {
			t.Errorf("expected an HTTP/2.0 request, but got: %s", req.Proto)
			return
		}
		flusher, ok := w.(interface{ FlushError() error })
		if !ok {
			t.Errorf("expected ResponseWriter object to implement FlushError")
			return
		}

		if _, err := w.Write([]byte("hello")); err != nil {
			t.Errorf("expected no error from Write, but got: %v", err)
			return
		}
		if err := flusher.FlushError(); err != nil {
			t.Errorf("expected no error from FlushError, but got: %v", err)
		}

		ctrl := http.NewResponseController(w)
		if err := ctrl.SetWriteDeadline(time.Now().Add(300 * time.Millisecond)); err != nil {
			t.Errorf("expected no error from SetReadDeadline, but got: %v", err)
			return
		}

		if _, err := io.Copy(w, neverEnding('a')); err != nil {
			handlerDoneCh <- err
		}
	}))

	defer server.Close()
	server.EnableHTTP2 = true
	server.StartTLS()

	client := server.Client()
	resp, err := client.Get(server.URL + "/foo")
	if err != nil {
		t.Errorf("unexpected connection error: %v", err)
		return
	}
	// we expect OK due to the initial write and flush
	if resp.StatusCode != http.StatusOK {
		t.Errorf("expected staus code: %d, but got: %d", http.StatusOK, resp.StatusCode)
	}
	defer closeResponseBody(t, resp)
	if _, err := io.Copy(io.Discard, resp.Body); !responsewritertesting.IsStreamReadOrWriteTimeout(err) {
		t.Errorf("expected http2 stream error, but got: %v", err)
	}

	// wait for the handler to terminate with the right error
	select {
	case err := <-handlerDoneCh:
		if err == nil || !errors.Is(err, os.ErrDeadlineExceeded) {
			t.Errorf("expected an os.ErrDeadlineExceeded, but got: %v", err)
		}
	case <-time.After(wait.ForeverTestTimeout):
		t.Errorf("expected the handler to have terminated")
	}
}

func TestPerHandlerReadDeadlineHTTP2WithClientNotWritingToRequestBody(t *testing.T) {
	// This test documents the behavior of the per handler read
	// deadline with a standard net http/2.0 server.
	//
	// scenario: http2, read deadline is set, client does not
	// send content, read timeout is expected.
	t.Parallel()
	clientDoneCh, handlerDoneCh := make(chan struct{}), make(chan struct{})
	server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		defer close(handlerDoneCh)
		if req.ProtoMajor != 2 {
			t.Errorf("expected an HTTP/2.0 request, but got: %s", req.Proto)
			return
		}
		flusher, ok := w.(interface{ FlushError() error })
		if !ok {
			t.Errorf("expected ResponseWriter object to implement FlushError")
			return
		}

		ctrl := http.NewResponseController(w)
		if err := ctrl.SetReadDeadline(time.Now().Add(100 * time.Millisecond)); err != nil {
			t.Errorf("expected no error from SetReadDeadline, but got: %v", err)
			return
		}

		// without a write deadline set, if the handler blocks here with,
		//  <-clientDoneCh
		// it will cause the client to block indefinitely as well.

		// we expect a read timeout to occur here
		if _, err := io.Copy(io.Discard, req.Body); !errors.Is(err, os.ErrDeadlineExceeded) {
			t.Errorf("expected an os.ErrDeadlineExceeded, but got: %v", err)
		}

		// even though read timeout has occurred, it seems we can
		// successfully write to the ResponseWriter
		if _, err := w.Write([]byte("hello")); err != nil {
			t.Errorf("expected no error from ResponseWriter.Write, but got: %v", err)
		}
		if err := flusher.FlushError(); err != nil {
			t.Errorf("expected no error from FlushError, but got: %v", err)
		}
	}))

	defer server.Close()
	server.EnableHTTP2 = true
	server.StartTLS()

	reader, writer := io.Pipe()
	defer func() {
		if err := writer.Close(); err != nil {
			t.Errorf("expected no error from Close, but got: %v", err)
		}
	}()

	client := server.Client()
	func() {
		defer close(clientDoneCh)

		resp, err := client.Post(server.URL, "text/foo", reader)
		if err != nil {
			t.Errorf("expected no error from Post, but got: %v", err)
			return
		}
		defer closeResponseBody(t, resp)
		if resp.StatusCode != http.StatusOK {
			t.Errorf("expected staus code: %d, but got: %d", http.StatusOK, resp.StatusCode)
		}
		if got, err := io.ReadAll(resp.Body); err != nil || string(got) != "hello" {
			t.Errorf("expected the client to read the response: want: %q, got: %q, error: %v", "hello", string(got), err)
		}
	}()

	select {
	case <-handlerDoneCh:
	case <-time.After(wait.ForeverTestTimeout):
		t.Errorf("expected the handler to have terminated")
	}
}

func TestPerHandlerReadDeadlineHTTP1WithClientNotWritingToRequestBody(t *testing.T) {
	// This test documents the behavior of the per handler read
	// deadline with a standard net http/1x server.
	//
	// scenario: http/1x, read deadline is set, client does not
	// send content, read timeout is expected.
	t.Parallel()
	doneCh := make(chan struct{})
	server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		defer close(doneCh)
		if req.ProtoMajor != 1 {
			t.Errorf("expected an HTTP/1x request, but got: %s", req.Proto)
			return
		}
		flusher, ok := w.(interface{ FlushError() error })
		if !ok {
			t.Errorf("expected ResponseWriter object to implement FlushError")
			return
		}

		ctrl := http.NewResponseController(w)
		if err := ctrl.SetReadDeadline(time.Now().Add(100 * time.Millisecond)); err != nil {
			t.Errorf("expected no error from SetReadDeadline, but got: %v", err)
			return
		}

		// we expect a read timeout here
		if _, err := io.Copy(io.Discard, req.Body); !errors.Is(err, os.ErrDeadlineExceeded) {
			t.Errorf("expected an os.ErrDeadlineExceeded, but got: %v", err)
		}

		if _, err := w.Write([]byte("hello")); err != nil {
			t.Errorf("expected no error from ResponseWriter.Write, but got: %v", err)
		}
		if err := flusher.FlushError(); err != nil {
			t.Errorf("expected no error from FlushError, but got: %v", err)
		}
	}))

	defer server.Close()
	server.StartTLS()

	reader, writer := io.Pipe()
	defer func() {
		if err := writer.Close(); err != nil {
			t.Errorf("expected no error from Close, but got: %v", err)
		}
	}()

	client := server.Client()
	resp, err := client.Post(server.URL, "text/foo", reader)
	if err != nil {
		t.Errorf("expected no error from Post, but got: %v", err)
		return
	}
	if resp.StatusCode != http.StatusOK {
		t.Errorf("expected staus code: %d, but got: %d", http.StatusOK, resp.StatusCode)
	}
	defer closeResponseBody(t, resp)
	if got, err := io.ReadAll(resp.Body); err != nil || string(got) != "hello" {
		t.Errorf("expected the client to read the response: want: %q, got: %q, error: %v", "hello", string(got), err)
	}

	select {
	case <-doneCh:
	case <-time.After(wait.ForeverTestTimeout):
		t.Errorf("expected the handler to have terminated")
	}
}

func TestPerHandlerReadDeadlineHTTPWithTimeoutWhileClientIsSendingContent(t *testing.T) {
	// This test documents the behavior of the per handler read
	// deadline with a standard net http server.
	//
	// scenario: http2 or http/1x, read deadline is set, client sends some
	// content and keeps the request Body stream open, read timeout expected
	t.Parallel()
	tests := []struct {
		protoMajor int
	}{
		{protoMajor: 1},
		{protoMajor: 2},
	}
	for _, test := range tests {
		t.Run(fmt.Sprintf("HTTP/%d", test.protoMajor), func(t *testing.T) {
			msg := "hello"
			handlerDoneCh := make(chan struct{})
			server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
				defer close(handlerDoneCh)
				if req.ProtoMajor != test.protoMajor {
					t.Errorf("expected an ProtoMajor: %d, but got: %s", test.protoMajor, req.Proto)
					return
				}
				flusher, ok := w.(interface{ FlushError() error })
				if !ok {
					t.Errorf("expected ResponseWriter object to implement FlushError")
					return
				}

				b := make([]byte, len(msg))
				n, err := io.ReadFull(req.Body, b)
				if err != nil || string(b[:n]) != msg {
					t.Errorf("expected content in request body, want: %s, got: %s, error: %v", msg, string(b[:n]), err)
					return
				}

				ctrl := http.NewResponseController(w)
				if err := ctrl.SetReadDeadline(time.Now().Add(10 * time.Millisecond)); err != nil {
					t.Errorf("expected no error from SetReadDeadline, but got: %v", err)
					return
				}

				// at this point, there are some content in the
				// Body of the request, and the EOF will not be
				// reached until after this handler terminates.
				if _, err := io.Copy(io.Discard, req.Body); !errors.Is(err, os.ErrDeadlineExceeded) {
					t.Errorf("expected an os.ErrDeadlineExceeded, but got: %v", err)
				}

				if _, err := w.Write([]byte(msg)); err != nil {
					t.Errorf("expected no error from ResponseWriter.Write, but got: %v", err)
				}
				if err := flusher.FlushError(); err != nil {
					t.Errorf("expected no error from FlushError, but got: %v", err)
				}
			}))

			defer server.Close()
			if test.protoMajor == 2 {
				server.EnableHTTP2 = true
			}
			server.StartTLS()

			reader, writer := io.Pipe()
			writerDoneCh := make(chan struct{})
			go func() {
				defer close(writerDoneCh)
				if _, err := writer.Write([]byte(msg + "more")); err != nil {
					t.Errorf("expected no error from Write to Body of the request, but got: %v", err)
				}

				// wait until the request handler terminates
				// before closing the writer
				<-handlerDoneCh
				if err := writer.Close(); err != nil {
					t.Errorf("expected no error from Close, but got: %v", err)
				}
			}()

			client := server.Client()
			resp, err := client.Post(server.URL, "test/foo", reader)
			if err != nil {
				t.Errorf("expected no error, but got: %v", err)
				return
			}
			if resp.StatusCode != http.StatusOK {
				t.Errorf("expected staus code: %d, but got: %d", http.StatusOK, resp.StatusCode)
			}
			defer closeResponseBody(t, resp)
			if got, err := io.ReadAll(resp.Body); err != nil || msg != string(got) {
				t.Errorf("expected response body: %s, but got: %s, error: %v", msg, string(got), err)
			}

			select {
			case <-writerDoneCh:
			case <-time.After(wait.ForeverTestTimeout):
				t.Errorf("expected the request handler to have terminated")
			}
		})
	}
}

func TestPerHandlerReadDeadlineHTTP2WithNoRequestBody(t *testing.T) {
	// This test documents the behavior of the per handler read
	// deadline with a standard net http2 server.
	//
	// scenario: http2, read deadline is set, client request
	// body is empty, read timeout has no effect.

	// TODO: enable the test to run when
	//  https://github.com/golang/go/issues/58237 is fixed
	t.Skip("the http server panics, see https://github.com/golang/go/issues/58237")

	t.Parallel()
	clientDoneCh, handlerDoneCh := make(chan struct{}), make(chan struct{})
	server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		defer close(handlerDoneCh)
		if req.ProtoMajor != 2 {
			t.Errorf("expected an HTTP/2.0 request, but got: %s", req.Proto)
			return
		}
		flusher, ok := w.(interface{ FlushError() error })
		if !ok {
			t.Errorf("expected ResponseWriter object to implement FlushError")
			return
		}

		ctrl := http.NewResponseController(w)
		if err := ctrl.SetReadDeadline(time.Now().Add(1 * time.Millisecond)); err != nil {
			t.Errorf("expected no error from SetReadDeadline, but got: %v", err)
			return
		}

		// without a write deadline set, if the handler blocks here with,
		//  <-clientDoneCh
		// it will cause the client to block indefinitely as well.
		<-time.After(10 * time.Second)

		// we expect no timeout reading from an empty body since
		// for server requests, the Request Body is always non-nil
		// but will return EOF immediately when no body is present.
		t.Logf("request Body type: %T", req.Body)
		if _, err := io.Copy(io.Discard, req.Body); err != nil {
			t.Errorf("expected no error from reading the Body (%T) of the request, but got: %v", req.Body, err)
		}

		if _, err := w.Write([]byte("hello")); err != nil {
			t.Errorf("expected no error from ResponseWriter.Write, but got: %v", err)
		}
		if err := flusher.FlushError(); err != nil {
			t.Errorf("expected no error from FlushError, but got: %v", err)
		}
	}))

	defer server.Close()
	server.EnableHTTP2 = true
	server.StartTLS()
	client := server.Client()
	func() {
		defer close(clientDoneCh)
		resp, err := client.Get(server.URL + "/foo")
		if err != nil {
			t.Errorf("expected no error from Post, but got: %v", err)
			return
		}
		if resp.StatusCode != http.StatusOK {
			t.Errorf("expected staus code: %d, but got: %d", http.StatusOK, resp.StatusCode)
		}
		defer closeResponseBody(t, resp)
		if got, err := io.ReadAll(resp.Body); err != nil || string(got) != "hello" {
			t.Errorf("expected the client to read the response: want: %q, got: %q, error: %v", "hello", string(got), err)
		}
	}()

	select {
	case <-handlerDoneCh:
	case <-time.After(wait.ForeverTestTimeout):
		t.Errorf("expected the handler to have terminated")
	}
}

func TestPerHandlerReadDeadlineHTTP1WithNoRequestBody(t *testing.T) {
	// This test documents the behavior of the per handler read
	// deadline with a standard net http server.
	//
	// scenario: http/1x, read deadline is set, client request
	// body is empty, read timeout has no effect.
	t.Parallel()
	doneCh := make(chan struct{})
	server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		defer close(doneCh)
		if req.ProtoMajor != 1 {
			t.Errorf("expected an HTTP/1x request, but got: %s", req.Proto)
			return
		}
		flusher, ok := w.(interface{ FlushError() error })
		if !ok {
			t.Errorf("expected ResponseWriter object to implement FlushError")
			return
		}

		ctrl := http.NewResponseController(w)
		if err := ctrl.SetReadDeadline(time.Now().Add(5 * time.Millisecond)); err != nil {
			t.Errorf("expected no error from SetReadDeadline, but got: %v", err)
			return
		}

		<-time.After(10 * time.Second)

		// we expect no timeout reading from an empty body since
		// for server requests, the Request Body is always non-nil
		// but will return EOF immediately when no body is present.
		t.Logf("request Body type: %T", req.Body)
		if _, err := io.Copy(io.Discard, req.Body); err != nil {
			t.Errorf("expected no error from reading the empty Body (%T) of the request, but got: %v", req.Body, err)
		}

		if _, err := w.Write([]byte("hello")); err != nil {
			t.Errorf("expected no error from ResponseWriter.Write, but got: %v", err)
		}
		if err := flusher.FlushError(); err != nil {
			t.Errorf("expected no error from FlushError, but got: %v", err)
		}
	}))

	defer server.Close()
	server.StartTLS()

	client := server.Client()
	resp, err := client.Get(server.URL + "/foo")
	if err != nil {
		t.Errorf("expected no error from Post, but got: %v", err)
		return
	}
	if resp.StatusCode != http.StatusOK {
		t.Errorf("expected staus code: %d, but got: %d", http.StatusOK, resp.StatusCode)
	}
	defer closeResponseBody(t, resp)
	if got, err := io.ReadAll(resp.Body); err != nil || string(got) != "hello" {
		t.Errorf("expected the client to read the response: want: %q, got: %q, error: %v", "hello", string(got), err)
	}

	select {
	case <-doneCh:
	case <-time.After(wait.ForeverTestTimeout):
		t.Errorf("expected the handler to have terminated")
	}

}

type neverEnding byte

func (b neverEnding) Read(p []byte) (n int, err error) {
	for i := range p {
		p[i] = byte(b)
	}
	return len(p), nil
}

func closeResponseBody(t *testing.T, resp *http.Response) {
	t.Helper()
	if err := resp.Body.Close(); err != nil {
		t.Errorf("unexpected error while closing the Body of the Response object: %v", err)
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
