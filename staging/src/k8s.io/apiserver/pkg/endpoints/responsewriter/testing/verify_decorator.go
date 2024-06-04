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

package testing

import (
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/endpoints/responsewriter"
)

// The user defined handler wraps the ResponseWriter object, this shared test
// verifies that the wrapped ResponseWriter object produced by the handler
// maintains interface compatibility with the original ResponseWriter
// object passed to it.
// NOTE: this test uses fake ResponseWriter implementation
func VerifyResponseWriterDecoratorWithFake(t *testing.T, chain func(http.Handler) http.Handler) {
	tests := []struct {
		name      string
		w         http.ResponseWriter
		decorator func(w http.ResponseWriter) http.ResponseWriter
	}{
		{
			name: "http/2.0",
			w: &FakeResponseWriterFlusherCloseNotifier{
				FakeResponseWriter: &FakeResponseWriter{},
			},
		},
		{
			name: "http/1x",
			w: &FakeResponseWriterFlusherCloseNotifierHijacker{
				FakeResponseWriterFlusherCloseNotifier: &FakeResponseWriterFlusherCloseNotifier{
					FakeResponseWriter: &FakeResponseWriter{},
				},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			req, err := http.NewRequest("GET", "http://example.com", nil)
			if err != nil {
				t.Fatalf("Unexpected error: %v", err)
			}

			ri := &request.RequestInfo{}
			req = req.WithContext(request.WithRequestInfo(req.Context(), ri))

			// the verifier handler verifies the ResponseWriter
			// wrapped by the user defined handler
			var invoked bool
			verifier := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				invoked = true

				// test.w is the original ResponseWriter passed
				// to the user defined handler
				// w is the wrapped ResponseWriter object
				inner, outer := test.w, w
				t.Logf("verifier handler being invoked, ResponseWriter - original: %T, wrapped: %T", inner, outer)

				if inner == w {
					t.Errorf("the user provided handler did not wrap the ResponseWriter object")
					return
				}

				decorator, ok := outer.(responsewriter.UserProvidedDecorator)
				if !ok {
					t.Errorf("expected the outer ResponseWriter object to implement responsewriter.UserProvidedDecorator")
					return
				}
				if want, got := inner, decorator.Unwrap(); want != got {
					t.Errorf("expected the decorator to return the correct inner ResponseWriter object")
				}

				AssertResponseWriterInterfaceCompatibility(t, inner, outer)
			})

			// we build a handler chain that looks like this:
			//      func(w http.ResponseWriter, r *http.Request) {
			//         // this is the user defined handler that wraps
			//         // the ResponseWriter object passed to it.
			//         w = wrap(w)
			//
			//         // invoke the verifier handler defined above,
			//         // the wrapped ResponseWriter gets passed to
			//         // the verifier handler.
			//         verifier.ServeHTTP(w, r)
			//      }
			handler := chain(verifier)
			handler.ServeHTTP(test.w, req)

			if !invoked {
				t.Errorf("the user provided handler did not invoke the verifier handler")
			}
		})
	}
}

// VerifyResponseWriterDecoratorWithHTTPServer verifies that the ResponseWriter
// decorator built with this package adheres to interface compatibility as expected.
// NOTE: each ResponseWriter decorator used by the apiserver is expected to run
// the following test, and thus it is a shared test.
func VerifyResponseWriterDecoratorWithHTTPServer(t *testing.T, chain func(http.Handler) http.Handler) {
	for _, proto := range []string{"HTTP/1.1", "HTTP/2.0"} {
		t.Run(proto, func(t *testing.T) {

			// this handler should run first, so it should be
			// appended to the chain:
			// - it saves the original ResponseWriter object passed
			// to the request handler by net/http
			// - assert that the original ResponseWriter implements
			// the expected interfaces
			var inner http.ResponseWriter
			first := func(h http.Handler) http.Handler {
				return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
					inner = w
					t.Logf("got the ResponseWriter object passed by net/http: original=%T", w)

					if req.Proto != proto {
						t.Errorf("expected protocol: %q, but got: %q", proto, req.Proto)
					}
					AssertResponseWriterImplementsExtendedInterfaces(t, inner, req)

					ri := &request.RequestInfo{}
					req = req.WithContext(request.WithRequestInfo(req.Context(), ri))
					h.ServeHTTP(w, req)
				})
			}

			// this handler should run last in the chain - it
			// receives the wrapped ResponseWriter object
			doneCh := make(chan struct{})
			last := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				defer close(doneCh)
				// a wrapped ResponseWriter object is nested, as shown below:
				//   outer
				//     |
				//     |-- middle (user defined decorator)
				//           |
				//           |-- inner
				//
				// a) inner: this is the ResponseWriter that is being wrapped
				// b) middle: it takes the inner ResponseWriter object and decorates
				// it, for example, it can override the `Flush` method, the decorator
				// does not need to maintain interface compatibility with the inner
				// ResponseWriter.
				// c) outer: it maintains interface compatibility with the inner, and
				// delegates calls to the decorator as appropriate
				outer := w
				t.Logf("the outer ResponseWriter object is of type: %T", outer)
				if inner == nil || outer == inner {
					t.Errorf("the user provided handler did not wrap the ResponseWriter object")
					return
				}

				AssertResponseWriterImplementsExtendedInterfaces(t, outer, r)
				AssertResponseWriterInterfaceCompatibility(t, inner, outer)

				accessor, ok := outer.(interface {
					GetUserProvidedDecorator() responsewriter.UserProvidedDecorator
				})
				if !ok {
					t.Errorf("the wrapped ResponseWriter is not compatible: %T", outer)
					return
				}
				middle := accessor.GetUserProvidedDecorator()
				t.Logf("got the user defined decorator - decorator=%T", middle)

				// we expect the wrapped ResponseWriter to be a UserProvidedDecorator
				decorator, ok := outer.(responsewriter.UserProvidedDecorator)
				if !ok {
					t.Errorf("expected a UserProvidedDecorator: %T", outer)
					return
				}

				if want, got := inner, decorator.Unwrap(); want != got {
					t.Errorf("expected Unwrap to return the original ResponseWriter")
				}
				if want, got := responsewriter.GetOriginal(outer), decorator.Unwrap(); want != got {
					t.Errorf("expected Unwrap to return the original ResponseWriter")
				}
			})

			// the handler chain is constructed as follows:
			//   first -> user provided handler -> last
			handler := chain(last)
			handler = first(handler)

			server := httptest.NewUnstartedServer(handler)
			defer server.Close()
			if proto == "HTTP/2.0" {
				server.EnableHTTP2 = true
			}
			server.StartTLS()

			client := server.Client()
			client.Timeout = wait.ForeverTestTimeout
			resp, err := client.Get(server.URL)
			if err != nil {
				t.Errorf("unexpected error from client.Get: %v", err)
				return
			}
			if resp.StatusCode != http.StatusOK {
				t.Errorf("Expected an OK response from the server, but got: %v", resp)
			}

			select {
			case <-doneCh:
			case <-time.After(wait.ForeverTestTimeout):
				t.Errorf("expected the request handler to have terminated")
			}
		})
	}
}
