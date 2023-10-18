/*
Copyright 2023 The Kubernetes Authors.

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
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/endpoints/responsewriter"
	responsewritertesting "k8s.io/apiserver/pkg/endpoints/responsewriter/testing"
)

func TestLatencyTrackerResponseWriterDecoratorShouldNotUseFlush(t *testing.T) {
	var decorator interface{} = &writeLatencyTracker{}
	if _, ok := decorator.(http.Flusher); ok {
		t.Errorf("decorator should not use Flush method, use FlushError instead")
	}
}

func TestLatencyTrackerResponseWriterDecoratorConstruction(t *testing.T) {
	inner1 := &responsewritertesting.FakeResponseWriter{}
	middle := &writeLatencyTracker{ResponseWriter: inner1, ctx: context.Background()} // middle is the decorator
	var i interface{} = middle
	if innerGot := i.(responsewriter.UserProvidedDecorator).Unwrap(); inner1 != innerGot {
		t.Errorf("Expected the decorator to return the inner http.ResponseWriter object")
	}

	// simulate http2
	inner2 := &responsewritertesting.FakeResponseWriterFlusherCloseNotifier{FakeResponseWriter: inner1}
	middle = &writeLatencyTracker{ResponseWriter: inner2, ctx: context.Background()}
	outer := responsewriter.WrapForHTTP1Or2(middle)
	if innerGot := outer.(responsewriter.UserProvidedDecorator).Unwrap(); inner2 != innerGot {
		t.Errorf("Expected the decorator to return the inner http.ResponseWriter object")
	}
	responsewritertesting.AssertResponseWriterInterfaceCompatibility(t, inner2, outer)

	// simulate http/1x
	inner3 := &responsewritertesting.FakeResponseWriterFlusherCloseNotifierHijacker{FakeResponseWriterFlusherCloseNotifier: inner2}
	middle = &writeLatencyTracker{ResponseWriter: inner3, ctx: context.Background()}
	outer = responsewriter.WrapForHTTP1Or2(middle)
	if innerGot := outer.(responsewriter.UserProvidedDecorator).Unwrap(); inner3 != innerGot {
		t.Errorf("Expected the decorator to return the inner http.ResponseWriter object")
	}
	responsewritertesting.AssertResponseWriterInterfaceCompatibility(t, inner3, outer)
}

func TestLatencyTrackerResponseWriterDecoratorWithHTTPServer(t *testing.T) {
	tests := []struct {
		name  string
		http2 bool
	}{
		{
			name: "http/1.x",
		},
		{
			name:  "http2",
			http2: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var inner http.ResponseWriter
			chain := func(h http.Handler) http.Handler {
				return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					inner = w

					middle := &writeLatencyTracker{ResponseWriter: inner, ctx: r.Context()}
					outer := responsewriter.WrapForHTTP1Or2(middle)

					h.ServeHTTP(outer, r)
				})
			}

			invokedCh := make(chan struct{}, 1)
			handler := chain(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				defer close(invokedCh)
				responsewritertesting.AssertResponseWriterInterfaceCompatibility(t, inner, w)
			}))

			server := httptest.NewUnstartedServer(handler)
			defer server.Close()
			if test.http2 {
				server.EnableHTTP2 = true
			}
			server.StartTLS()
			if _, err := url.Parse(server.URL); err != nil {
				t.Errorf("Expected the server to have a valid URL, but got: %s", server.URL)
				return
			}
			req, err := http.NewRequest("GET", server.URL, nil)
			if err != nil {
				t.Errorf("error creating request: %v", err)
				return
			}

			client := server.Client()
			client.Timeout = wait.ForeverTestTimeout
			_, err = client.Do(req)
			if err != nil {
				t.Errorf("Unexpected error from the server: %v", err)
			}

			select {
			case <-invokedCh:
			case <-time.After(wait.ForeverTestTimeout):
				t.Errorf("Expected the handler to be invoked")
			}
		})
	}
}
