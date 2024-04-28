/*
Copyright 2024 The Kubernetes Authors.

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

package server

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/endpoints/responsewriter"
	responsewritertesting "k8s.io/apiserver/pkg/endpoints/responsewriter/testing"
	"k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
)

func TestPerHandlerReadWriteDeadlineWithNonLongRunningRequest(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PerHandlerReadWriteTimeout, true)()

	fakeAudit := &fakeAudit{}
	config, _ := setUp(t)
	config.AuditPolicyRuleEvaluator = fakeAudit
	config.AuditBackend = fakeAudit
	s, err := config.Complete(nil).New("test", NewEmptyDelegate())
	if err != nil {
		t.Fatalf("Error in setting up a GenericAPIServer object: %v", err)
	}

	longRunningFn := config.LongRunningFunc
	clientDoneCh, handlerDoneCh := make(chan struct{}), make(chan error, 1)
	timeoutWant := 1 * time.Second
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer close(handlerDoneCh)
		ctx := r.Context()

		reqInfo, ok := request.RequestInfoFrom(ctx)
		if !ok {
			t.Errorf("expected the request context to have a RequestInfo associated")
			return
		}
		if longRunningFn(r, reqInfo) {
			t.Errorf("wrong test setup, wanted a non long-running request, but got: %#v", reqInfo)
			return
		}
		receivedAt, ok := request.ReceivedTimestampFrom(ctx)
		if !ok {
			t.Errorf("expected the request context to have a received at timestamp, but got: %s", receivedAt)
			return
		}
		deadline, ok := ctx.Deadline()
		if !ok {
			t.Errorf("expected the request context to have a deadline")
			return
		}
		if timeoutGot := deadline.Sub(receivedAt); timeoutWant != timeoutGot {
			t.Errorf("expected the request context to have a deadline of: %s, but got: %s", timeoutWant, timeoutGot)
			return
		}

		<-clientDoneCh

		// a threshold of 10s to account for round trip and CI flakes
		sinceDeadline, threshold := time.Since(deadline), 10*time.Second
		t.Logf("client has received a response %s after deadline", sinceDeadline)
		if sinceDeadline > threshold {
			t.Errorf("expected the client to receive a response earlier, took %s", sinceDeadline)
		}

		func() {
			now := time.Now()
			count := 0
			defer func() {
				duration := time.Since(now)
				t.Logf("After timeout, Write (1KB of data) was invoked %d times, total duration before error: %s", count, duration)
				// 10s should be long enough to account for CI flakes
				if duration > threshold {
					t.Errorf("Write took too long to return a timeout error: %s", duration)
				}
			}()
			for {
				count++
				if _, err := w.Write(bytes.Repeat([]byte("a"), 1024)); err != nil {
					handlerDoneCh <- err
					break
				}
			}
		}()
	})
	s.Handler.NonGoRestfulMux.Handle("/ping", handler)

	server := httptest.NewUnstartedServer(s.Handler)
	defer server.Close()
	server.EnableHTTP2 = true
	server.StartTLS()

	client := server.Client()
	func() {
		defer close(clientDoneCh)
		_, err := client.Get(server.URL + fmt.Sprintf("/ping?timeout=%s", timeoutWant))
		if !responsewritertesting.IsStreamReadOrWriteTimeout(err) {
			t.Errorf("expected a stream reset error, but got: %v", err)
		}
	}()

	select {
	case err := <-handlerDoneCh:
		if err == nil || !strings.Contains(err.Error(), "i/o timeout") {
			t.Errorf("expected Write (invoked after deadline passes) to return a timeout error, but got: %v", err)
		}
	case <-time.After(wait.ForeverTestTimeout):
		t.Errorf("expected the request handler to have terminated")
	}
}

func TestPerHandlerReadWriteDeadlineWithWatchRequest(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PerHandlerReadWriteTimeout, true)()

	fakeAudit := &fakeAudit{}
	config, _ := setUp(t)
	config.AuditPolicyRuleEvaluator = fakeAudit
	config.AuditBackend = fakeAudit
	s, err := config.Complete(nil).New("test", NewEmptyDelegate())
	if err != nil {
		t.Fatalf("Error in setting up a GenericAPIServer object: %v", err)
	}

	longRunningFn := config.LongRunningFunc
	handlerDoneCh := make(chan struct{})
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer close(handlerDoneCh)
		ctx := r.Context()

		reqInfo, ok := request.RequestInfoFrom(ctx)
		if !ok {
			t.Errorf("expected the request context to have a RequestInfo associated")
			return
		}
		if !longRunningFn(r, reqInfo) || reqInfo.Verb != "watch" {
			t.Errorf("wrong test setup, wanted a watch request, but got: %#v", reqInfo)
			return
		}
		if receivedAt, ok := request.ReceivedTimestampFrom(ctx); !ok {
			t.Errorf("expected the request context to have a received at timestamp, but got: %s", receivedAt)
			return
		}
		if _, ok := ctx.Deadline(); ok {
			t.Errorf("did not expect the request context to have a deadline set")
			return
		}
	})

	path := "/api/v1/namespaces/ns1/resources"
	s.Handler.NonGoRestfulMux.Handle(path, handler)

	server := httptest.NewUnstartedServer(s.Handler)
	defer server.Close()
	server.EnableHTTP2 = true
	server.StartTLS()

	client := server.Client()
	if _, err := client.Get(server.URL + fmt.Sprintf("%s?timeout=1m&timeoutSeconds=60&watch=1", path)); err != nil {
		t.Errorf("expected no error from client.Get, but got: %v", err)
	}

	select {
	case <-handlerDoneCh:
	case <-time.After(wait.ForeverTestTimeout):
		t.Errorf("expected the request handler to have terminated")
	}
}

func TestPerHandlerWriteDeadlineWith(t *testing.T) {
	tests := []struct {
		protoMajor                  int
		perHandlerTimeoutEnabled    bool
		clientTimeout               time.Duration
		serverTimeout               time.Duration
		handlerWritesBeforeTimeout  bool
		handlerFlushesBeforeTimeout bool

		// expectations
		serverWriteErrContains       string
		clientErrContains            string
		clientStatusCodeExpected     int
		clientResBodyReadErrContains string
	}{
		// scenario:
		//  a) per-handler timeout: disabled
		//  b) proto: http/2.0
		//  c) client specifies timeout: no
		//  d) request handler writes to the ResponseWriter object before timeout happens: no
		// expected:
		//  client:
		//   a) client should receive a 'http.StatusGatewayTimeout' status code
		//   b) reading the Body of the Response object should not yield any error
		//  server:
		//   c) Write to the ResponseWriter will yield an 'http: Handler timeout' immediately (timeout filter)
		// {
		// 	perHandlerTimeoutEnabled: false, protoMajor: 2, clientTimeout: 0, serverTimeout: time.Second, handlerWritesBeforeTimeout: false, handlerFlushesBeforeTimeout: false,
		// 	serverWriteErrContains: "http: Handler timeout", clientStatusCodeExpected: http.StatusGatewayTimeout, clientErrContains: "", clientResBodyReadErrContains: "",
		// },
		//
		// scenario:
		// same as above, except a) per-handler timeout: enabled
		// expected:
		//  client:
		//   a) client should receive a http2 stream reset error status code
		//   b) reading the Body of the Response object should not yield any error
		//  server:
		//   c) Write to the ResponseWriter will yield an 'i/o timeout' error once the underlying buffer is full
		// {
		// 	perHandlerTimeoutEnabled: true, protoMajor: 2, clientTimeout: 0, serverTimeout: time.Second, handlerWritesBeforeTimeout: false, handlerFlushesBeforeTimeout: false,
		// 	serverWriteErrContains: "i/o timeout", clientStatusCodeExpected: 0, clientErrContains: "stream error: stream ID", clientResBodyReadErrContains: "",
		// },

		// ---------
		// scenario:
		//  a) per-handler timeout: disabled
		//  b) proto: http/1.x
		//  c) client specifies timeout: no
		//  d) request handler writes to the ResponseWriter object before timeout happens: no
		// expected:
		//  client:
		//   a) client should receive a 'http.StatusGatewayTimeout' status code
		//   b) reading the Body of the Response object should not yield any error
		//  server:
		//   c) Write to the ResponseWriter will yield an 'http: Handler timeout' immediately (timeout filter)
		// {
		// 	perHandlerTimeoutEnabled: false, protoMajor: 1, clientTimeout: 0, serverTimeout: time.Second, handlerWritesBeforeTimeout: false, handlerFlushesBeforeTimeout: false,
		// 	serverWriteErrContains: "http: Handler timeout", clientStatusCodeExpected: http.StatusGatewayTimeout, clientErrContains: "", clientResBodyReadErrContains: "",
		// },
		//
		// scenario:
		// same as above, except a) per-handler timeout: enabled
		// expected:
		//  client:
		//   a) client should receive an error
		//   b) reading the Body of the Response object should not yield any error
		//  server:
		//   c) Write to the ResponseWriter will yield an 'i/o timeout' error once the underlying buffer is full
		//
		// actual:
		//   client hangs indefinitely, since the request handler blocks and hence never completes
		// {
		// 	perHandlerTimeoutEnabled: true, protoMajor: 1, clientTimeout: 0, serverTimeout: time.Second, handlerWritesBeforeTimeout: false, handlerFlushesBeforeTimeout: false,
		// 	serverWriteErrContains: "i/o timeout", clientStatusCodeExpected: 0, clientErrContains: "local error: tls: bad record MAC", clientResBodyReadErrContains: "",
		// },

		//
		// ---------
		// scenario:
		//  a) per-handler timeout: disabled
		//  b) proto: http/2.0
		//  c) client specifies timeout: no
		//  d) request handler writes to the ResponseWriter object before timeout happens: yes
		// expected:
		//  client:
		//   a) client should receive an http2 stream reset error (since the client-facing goroutine in the timeout filter panics if the ResponseWriter object has already been written to)
		//   b) Response is nil
		//  server:
		//   c) Write to the ResponseWriter will yield an 'http: Handler timeout' immediately (timeout filter)
		// {
		// 	perHandlerTimeoutEnabled: true, protoMajor: 1, clientTimeout: 0, serverTimeout: time.Second, handlerWritesBeforeTimeout: true, handlerFlushesBeforeTimeout: false,
		// 	serverWriteErrContains: "i/o timeout", clientStatusCodeExpected: 0, clientErrContains: "local error: tls: bad record MAC", clientResBodyReadErrContains: "",
		// },
		//
		// scenario:
		// same as above, except a) per-handler timeout: enabled
		// expected:
		//  client:
		//   a) client should receive an http2 stream reset error (net/http will reset the stream directly once the write deadline passes, it will not wait for the request handler to return
		//   b) Response is nil
		//  server:
		//   c) Write to the ResponseWriter will yield an 'i/o timeout' error once the underlying buffer is full
		// {
		// 	perHandlerTimeoutEnabled: true, protoMajor: 2, clientTimeout: 0, serverTimeout: time.Second, handlerWritesBeforeTimeout: true, handlerFlushesBeforeTimeout: false,
		// 	serverWriteErrContains: "i/o timeout", clientStatusCodeExpected: 0, clientErrContains: "stream error: stream ID", clientResBodyReadErrContains: "",
		// },

		// ---------
		// scenario:
		//  a) per-handler timeout: disabled
		//  b) proto: http/2.0
		//  c) client specifies timeout: no
		//  d) request handler writes to and flushes the ResponseWriter object before timeout happens: yes
		// expected:
		//  client:
		//   a) client receives a non nil http.Response object, status code is 'OK 200'
		//   b) while reading the Body of the Response, a stream reset error is expected
		//  server:
		//   c) Write to the ResponseWriter will yield an 'http: Handler timeout' immediately (timeout filter)
		// {
		// 	perHandlerTimeoutEnabled: false, protoMajor: 2, clientTimeout: 0, serverTimeout: time.Second, handlerWritesBeforeTimeout: true, handlerFlushesBeforeTimeout: true,
		// 	serverWriteErrContains: "http: Handler timeout", clientStatusCodeExpected: http.StatusOK, clientErrContains: "", clientResBodyReadErrContains: "stream error: stream ID",
		// },
		//
		// scenario:
		// same as above, except a) per-handler timeout: enabled
		// expected:
		//  client:
		//   a) client receives a non nil http.Response object, status code is 'OK 200'
		//   b) while reading the Body of the Response, a stream reset error is expected
		//  server:
		//   c) Write to the ResponseWriter will yield an 'i/o timeout' error once the underlying buffer is full
		// {
		// 	perHandlerTimeoutEnabled: true, protoMajor: 2, clientTimeout: 0, serverTimeout: time.Second, handlerWritesBeforeTimeout: true, handlerFlushesBeforeTimeout: true,
		// 	serverWriteErrContains: "i/o timeout", clientStatusCodeExpected: http.StatusOK, clientErrContains: "", clientResBodyReadErrContains: "stream error: stream ID",
		// },
		// analysis: in both scenarios, the client sees the same result, on the server we can note the following change:
		// with timeout filter enabled Write yields an error immediately, in contrast,  with per-request timeout enabled Write yields an error once the underlying buffer is full:

		// ---------
		// scenario:
		//  a) per-handler timeout: disabled
		//  b) proto: http/1.x
		//  c) client specifies timeout: no
		//  d) request handler writes to and flushes the ResponseWriter object before timeout happens: yes
		// expected:
		//  client:
		//   a) client receives a non nil http.Response object, status code is 'OK 200'
		//   b) while reading the Body of the Response, an 'unexpected EOF' error is expected
		//  server:
		//   c) Write to the ResponseWriter will yield an 'http: Handler timeout' immediately (timeout filter)
		// {
		// 	perHandlerTimeoutEnabled: false, protoMajor: 1, clientTimeout: 0, serverTimeout: time.Second, handlerWritesBeforeTimeout: true, handlerFlushesBeforeTimeout: true,
		// 	serverWriteErrContains: "http: Handler timeout", clientStatusCodeExpected: http.StatusOK, clientErrContains: "", clientResBodyReadErrContains: "unexpected EOF",
		// },
		//
		// scenario:
		// same as above, except a) per-handler timeout: enabled
		// expected:
		//  client:
		//   a) client hangs indefinitely, since the request handler blocks and hence never completes
		// {
		// 	perHandlerTimeoutEnabled: true, protoMajor: 1, clientTimeout: 0, serverTimeout: time.Second, handlerWritesBeforeTimeout: true, handlerFlushesBeforeTimeout: true,
		// 	serverWriteErrContains: "i/o timeout", clientStatusCodeExpected: http.StatusOK, clientErrContains: "", clientResBodyReadErrContains: "tls: bad record MAC",
		// },
		// analysis: the timeout filter prevents from having an frozen client by returning a response

		// ---------
		// scenario:
		//  a) per-handler timeout: disabled
		//  b) proto: http/2.0
		//  c) client specifies timeout: yes
		//  d) request handler writes to and flushes the ResponseWriter object before timeout happens: yes
		// expected:
		//  client:
		//   a) client receives a non nil http.Response object, status code is 'OK 200'
		//   b) while reading the Body of the Response, an 'unexpected EOF' error is expected
		//  server:
		//   c) Write to the ResponseWriter will yield an 'http: Handler timeout' immediately (timeout filter)
		{
			perHandlerTimeoutEnabled: false, protoMajor: 2, clientTimeout: time.Second, serverTimeout: time.Minute, handlerWritesBeforeTimeout: true, handlerFlushesBeforeTimeout: true,
			serverWriteErrContains: "stream closed", clientStatusCodeExpected: http.StatusOK, clientErrContains: "", clientResBodyReadErrContains: "context deadline exceeded",
		},

		{
			perHandlerTimeoutEnabled: true, protoMajor: 2, clientTimeout: time.Second, serverTimeout: time.Minute, handlerWritesBeforeTimeout: true, handlerFlushesBeforeTimeout: true,
			serverWriteErrContains: "stream closed", clientStatusCodeExpected: http.StatusOK, clientErrContains: "", clientResBodyReadErrContains: "context deadline exceeded",
		},
	}
	for _, test := range tests {
		t.Run(fmt.Sprintf("%s=%t HTTP/%d", string(features.PerHandlerReadWriteTimeout), test.perHandlerTimeoutEnabled, test.protoMajor), func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PerHandlerReadWriteTimeout, test.perHandlerTimeoutEnabled)()
			// defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.APIServerTracing, false)()

			fakeAudit := &fakeAudit{}
			config, _ := setUp(t)
			config.RequestTimeout = test.serverTimeout
			config.AuditPolicyRuleEvaluator = fakeAudit
			config.AuditBackend = fakeAudit
			s, err := config.Complete(nil).New("test", NewEmptyDelegate())
			if err != nil {
				t.Fatalf("Error in setting up a GenericAPIServer object: %v", err)
			}

			longRunningFn := config.LongRunningFunc
			clientDoneCh, handlerDoneCh := make(chan struct{}), make(chan error, 1)
			timeoutWant := config.RequestTimeout
			if test.clientTimeout > 0 {
				timeoutWant = test.clientTimeout
			}

			handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				defer close(handlerDoneCh)
				ctx := r.Context()

				if r.ProtoMajor != test.protoMajor {
					t.Errorf("expected an ProtoMajor: %d, but got: %s", test.protoMajor, r.Proto)
					return
				}

				// NOTE: we don't support `FlushError` delegation yet, so get to the original
				// ResponseWriter object and use that to invoke FlushError
				o := responsewriter.GetOriginal(w)
				flusher, ok := o.(interface{ FlushError() error })
				if !ok {
					t.Errorf("expected ResponseWriter object to implement FlushError")
					return
				}
				reqInfo, ok := request.RequestInfoFrom(ctx)
				if !ok {
					t.Errorf("expected the request context to have a RequestInfo associated")
					return
				}
				if longRunningFn(r, reqInfo) {
					t.Errorf("wrong test setup, wanted a non long-running request, but got: %#v", reqInfo)
					return
				}
				receivedAt, ok := request.ReceivedTimestampFrom(ctx)
				if !ok {
					t.Errorf("expected the request context to have a received at timestamp, but got: %s", receivedAt)
					return
				}
				deadline, ok := ctx.Deadline()
				if !ok {
					t.Errorf("expected the request context to have a deadline")
					return
				}
				if timeoutGot := deadline.Sub(receivedAt); timeoutWant != timeoutGot {
					t.Errorf("expected the request context to have a deadline of: %s, but got: %s", timeoutWant, timeoutGot)
					return
				}

				if test.handlerWritesBeforeTimeout {
					if _, err := w.Write([]byte("hello")); err != nil {
						t.Errorf("expected no error from Write, but got: %v", err)
						return
					}
				}
				if test.handlerFlushesBeforeTimeout {
					if err := flusher.FlushError(); err != nil {
						t.Errorf("expected no error from FlushError, but got: %v", err)
					}
				}

				<-clientDoneCh

				// <-time.After(timeoutWant * 2)

				// w.Write([]byte("a"))
				// if err := flusher.FlushError(); err == nil || !strings.Contains(err.Error(), test.serverFlushErrContains) {
				// 	t.Errorf("expected error: %q from FlushError, but got: %v", test.serverFlushErrContains, err)
				// }

				for {
					if _, err := w.Write(bytes.Repeat([]byte("a"), 1024)); err != nil {
						handlerDoneCh <- err
						break
					}
				}

			})
			s.Handler.NonGoRestfulMux.Handle("/ping", handler)

			server := httptest.NewUnstartedServer(s.Handler)
			defer server.Close()
			if test.protoMajor == 2 {
				server.EnableHTTP2 = true
			}
			server.StartTLS()

			client := server.Client()
			func() {
				defer close(clientDoneCh)

				url := fmt.Sprintf("%s/ping", server.URL)
				if test.clientTimeout > 0 {
					url = fmt.Sprintf("%s?timeout=%s", url, test.clientTimeout)
				}
				req, err := http.NewRequest(http.MethodGet, url, nil)
				if err != nil {
					t.Errorf("failed to create a new http request - %v", err)
					return
				}
				if test.clientTimeout > 0 {
					ctx, cancel := context.WithTimeout(context.Background(), test.clientTimeout)
					defer cancel()

					req = req.WithContext(ctx)
				}

				// t.Logf("client url: %s", url)
				resp, err := client.Do(req)

				switch {
				case test.clientStatusCodeExpected > 0:
					if resp == nil || resp.StatusCode != test.clientStatusCodeExpected {
						t.Errorf("expected a status code: %d, but got: %#v", test.clientStatusCodeExpected, resp)
					}
				default:
					if resp != nil {
						t.Errorf("did not expect a Response from the server, but got: %#v", resp)
					}
				}
				switch {
				case len(test.clientErrContains) > 0:
					if err == nil || !strings.Contains(err.Error(), test.clientErrContains) {
						t.Errorf("expected error: %q from the server, but got: %v", test.clientErrContains, err)
					}
				default:
					if err != nil {
						t.Errorf("did not expect any error from the server, but got: %v", err)
					}
				}

				if resp != nil {
					_, err = io.ReadAll(resp.Body)
					switch {
					case len(test.clientResBodyReadErrContains) > 0:
						if err == nil || !strings.Contains(err.Error(), test.clientResBodyReadErrContains) {
							t.Errorf("expected error: %q while reading the response body, but got: %v", test.clientResBodyReadErrContains, err)
						}
					default:
						if err != nil {
							t.Errorf("did not expect any error while reading the response body, but got: %v", err)
						}
					}
					if err := resp.Body.Close(); err != nil {
						t.Errorf("unexpected error while closing the Body of the Response object: %v", err)
					}
				}
			}()

			select {
			case err := <-handlerDoneCh:
				if err == nil || !strings.Contains(err.Error(), test.serverWriteErrContains) {
					t.Errorf("expected Write (invoked after deadline passes) to return error: %q, but got: %v", test.serverWriteErrContains, err)
				}
			case <-time.After(wait.ForeverTestTimeout):
				t.Errorf("expected the request handler to have terminated")
			}
		})
	}
}
