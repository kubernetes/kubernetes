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
	"flag"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/endpoints/metrics"
	"k8s.io/apiserver/pkg/endpoints/responsewriter"
	responsewritertesting "k8s.io/apiserver/pkg/endpoints/responsewriter/testing"
	"k8s.io/klog/v2"
)

func TestWrappedResponseWriterWithFullHandlerChain(t *testing.T) {
	for _, proto := range []string{"HTTP/1.1", "HTTP/2.0"} {
		t.Run(proto, func(t *testing.T) {
			fakeAudit := &fakeAudit{}
			config, _ := setUp(t)
			config.AuditPolicyRuleEvaluator = fakeAudit
			config.AuditBackend = fakeAudit
			config.RequestTimeout = time.Minute

			// build the full handler chain of the generic apiserver
			s, err := config.Complete(nil).New("test", NewEmptyDelegate())
			if err != nil {
				t.Fatalf("Error in setting up a GenericAPIServer object: %v", err)
			}

			// this handler should run first, so it should be
			// appended to the chain
			var original http.ResponseWriter
			first := func(h http.Handler) http.Handler {
				return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					// save the original ResponseWriter
					// object passed by net/http
					original = w
					t.Logf("got the ResponseWriter object passed by net/http: original=%T", original)

					h.ServeHTTP(w, r)
				})
			}

			// this handler should run last in the chain
			handlerDoneCh := make(chan struct{})
			handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				defer close(handlerDoneCh)

				if r.Proto != proto {
					t.Errorf("expected protocol: %q, but got: %q", proto, r.Proto)
					return
				}
				if original == nil {
					t.Errorf("wrong test setup - original ResponseWriter object has not been set")
					return
				}

				// the server handler chain has wrapped the ResponseWriter object,
				// let's ensure that the wrapped ResponseWriter is interface
				// compatible with the original ResponseWriter passed by net/http
				responsewritertesting.AssertResponseWriterImplementsExtendedInterfaces(t, w, r)
				responsewritertesting.AssertResponseWriterInterfaceCompatibility(t, original, w)

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
				//
				// when a ResponseWriter is wrapped multiple times, the outer object
				// is passed as an inner to the next handler

				// let's unwrap the wrapped ResponseWriter, we start with the final
				// ResponseWriter object passed to this handler, and work our way
				// back to the original ResponseWriter handed to us by net/http.
				t.Logf("unwrapping our way back to the original ResponseWriter")
				for outer := w; ; {
					if outer == original {
						t.Logf("we have reached the original ResponseWriter object - outer=%T", outer)
						break
					}
					t.Logf("unwrapping ResponseWriter: outer=%T", outer)

					// decorators from the kube-apiserver filters
					// is expected to implement the following interface
					if accessor, ok := outer.(interface {
						GetUserProvidedDecorator() responsewriter.UserProvidedDecorator
					}); ok {
						middle := accessor.GetUserProvidedDecorator()
						t.Logf("found a decorator (known) - decorator=%T", middle)
					} else {
						t.Logf("found a decorator (unknown, possibly wrapped by a third party library) decorator=%T", outer)
					}

					// all decorators must have 'Unwrap() http.ResponseWriter' method
					unwrapper, ok := outer.(interface{ Unwrap() http.ResponseWriter })
					if !ok {
						t.Errorf("the wrapped ResponseWriter is not compatible: outer=%T", outer)
						break
					}

					inner := unwrapper.Unwrap()
					if inner == nil {
						t.Errorf("expected the decorator to hold an inner ResponseWriter object")
						break
					}
					responsewritertesting.AssertResponseWriterInterfaceCompatibility(t, inner, outer)

					// let's unwrap this inner ResponseWriter
					outer = inner
				}
			})

			// httplog is enabled only when we run in -v=3 or greater
			// TODO: does it have impact on other tests running in parallel?
			state := klog.CaptureState()
			t.Cleanup(state.Restore)
			var fs flag.FlagSet
			klog.InitFlags(&fs)
			if err := fs.Set("v", "3"); err != nil {
				t.Errorf("unexpected error while setting klog flags: %v", err)
				return
			}

			// add the metrics handler
			// TODO: register a fake type with the generic apiserver
			var chain http.Handler = metrics.InstrumentHandlerFunc("", "", "", "", "", "", "", false, "", http.HandlerFunc(s.Handler.FullHandlerChain.ServeHTTP))

			chain = first(chain)
			s.Handler.FullHandlerChain = chain

			s.Handler.NonGoRestfulMux.Handle("/ping", handler)
			server := httptest.NewUnstartedServer(s.Handler)
			defer server.Close()
			if proto == "HTTP/2.0" {
				server.EnableHTTP2 = true
			}
			server.StartTLS()

			client := server.Client()
			client.Timeout = wait.ForeverTestTimeout
			resp, err := client.Get(server.URL + "/ping")
			if err != nil {
				t.Errorf("unexpected error from client.Get: %v", err)
				return
			}
			if resp.StatusCode != http.StatusOK {
				t.Errorf("Expected an OK response from the server, but got: %v", resp)
			}

			select {
			case <-handlerDoneCh:
			case <-time.After(wait.ForeverTestTimeout):
				t.Errorf("expected the request handler to have terminated")
			}
		})
	}
}
