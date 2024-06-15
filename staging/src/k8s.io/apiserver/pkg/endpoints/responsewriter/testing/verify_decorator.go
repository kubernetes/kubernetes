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
	"testing"

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
