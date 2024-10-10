/*
Copyright 2014 The Kubernetes Authors.

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

package httplog

import (
	"net/http"
	"net/http/httptest"
	"net/url"
	"reflect"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/endpoints/responsewriter"
	responsewritertesting "k8s.io/apiserver/pkg/endpoints/responsewriter/testing"
)

func TestDefaultStacktracePred(t *testing.T) {
	for _, x := range []int{101, 200, 204, 302, 400, 404} {
		if DefaultStacktracePred(x) {
			t.Fatalf("should not log on %v by default", x)
		}
	}

	for _, x := range []int{500, 100} {
		if !DefaultStacktracePred(x) {
			t.Fatalf("should log on %v by default", x)
		}
	}
}

func TestStatusIsNot(t *testing.T) {
	statusTestTable := []struct {
		status   int
		statuses []int
		want     bool
	}{
		{http.StatusOK, []int{}, true},
		{http.StatusOK, []int{http.StatusOK}, false},
		{http.StatusCreated, []int{http.StatusOK, http.StatusAccepted}, true},
	}
	for _, tt := range statusTestTable {
		sp := StatusIsNot(tt.statuses...)
		got := sp(tt.status)
		if got != tt.want {
			t.Errorf("Expected %v, got %v", tt.want, got)
		}
	}
}

func TestWithLogging(t *testing.T) {
	req, err := http.NewRequest("GET", "http://example.com", nil)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	shouldLogRequest := func() bool { return true }
	var handler http.Handler
	handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {})
	handler = withLogging(withLogging(handler, DefaultStacktracePred, shouldLogRequest), DefaultStacktracePred, shouldLogRequest)

	func() {
		defer func() {
			if r := recover(); r == nil {
				t.Errorf("Expected newLogged to panic")
			}
		}()
		w := httptest.NewRecorder()
		handler.ServeHTTP(w, req)
	}()
}

func TestLogOf(t *testing.T) {
	tests := []struct {
		name             string
		shouldLogRequest bool
		want             string
	}{
		{
			name:             "request is being logged",
			shouldLogRequest: true,
			want:             "*httplog.respLogger",
		},
		{
			name:             "request is not being logged",
			shouldLogRequest: false,
			want:             "*httplog.passthroughLogger",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			req, err := http.NewRequest("GET", "http://example.com", nil)
			if err != nil {
				t.Errorf("Unexpected error: %v", err)
			}
			var handler http.Handler
			handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				got := reflect.TypeOf(LogOf(r, w)).String()
				if test.want != got {
					t.Errorf("Expected %v, got %v", test.want, got)
				}
			})
			handler = withLogging(handler, DefaultStacktracePred, func() bool { return test.shouldLogRequest })
			w := httptest.NewRecorder()
			handler.ServeHTTP(w, req)
		})
	}
}

func TestUnlogged(t *testing.T) {
	unloggedTests := []bool{true, false}
	for _, makeLogger := range unloggedTests {
		req, err := http.NewRequest("GET", "http://example.com", nil)
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}

		origWriter := httptest.NewRecorder()
		var handler http.Handler
		handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			got := Unlogged(r, w)
			if origWriter != got {
				t.Errorf("Expected origin writer, got %#v", got)
			}
		})
		if makeLogger {
			handler = WithLogging(handler, DefaultStacktracePred)
		}

		handler.ServeHTTP(origWriter, req)
	}
}

func TestLoggedStatus(t *testing.T) {
	req, err := http.NewRequest("GET", "http://example.com", nil)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	var tw http.ResponseWriter = new(responsewritertesting.FakeResponseWriter)
	logger := newLogged(req, tw)
	logger.Write(nil)

	if logger.status != http.StatusOK {
		t.Errorf("expected status after write to be %v, got %v", http.StatusOK, logger.status)
	}

	tw = new(responsewritertesting.FakeResponseWriter)
	logger = newLogged(req, tw)
	logger.WriteHeader(http.StatusForbidden)
	logger.Write(nil)

	if logger.status != http.StatusForbidden {
		t.Errorf("expected status after write to remain %v, got %v", http.StatusForbidden, logger.status)
	}
}

func TestRespLoggerWithDecoratedResponseWriter(t *testing.T) {
	tests := []struct {
		name       string
		r          func() http.ResponseWriter
		hijackable bool
	}{
		{
			name: "http2",
			r: func() http.ResponseWriter {
				return &responsewritertesting.FakeResponseWriterFlusherCloseNotifier{}
			},
			hijackable: false,
		},
		{
			name: "http/1.x",
			r: func() http.ResponseWriter {
				return &responsewritertesting.FakeResponseWriterFlusherCloseNotifierHijacker{}
			},
			hijackable: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			req, err := http.NewRequest("GET", "http://example.com", nil)
			if err != nil {
				t.Errorf("Unexpected error: %v", err)
			}

			var handler http.Handler
			handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				switch v := w.(type) {
				case *respLogger:
					t.Errorf("Did not expect %v", reflect.TypeOf(v))
					return
				default:
				}

				//lint:file-ignore SA1019 Keep supporting deprecated http.CloseNotifier
				if _, ok := w.(http.CloseNotifier); !ok {
					t.Errorf("Expected the ResponseWriter object to implement http.CloseNotifier")
				}
				if _, ok := w.(http.Flusher); !ok {
					t.Errorf("Expected the ResponseWriter object to implement http.Flusher")
				}
				if _, ok := w.(http.Hijacker); test.hijackable != ok {
					t.Errorf("http.Hijacker does not match, want: %t, got: %t", test.hijackable, ok)
				}
			})

			handler = withLogging(handler, DefaultStacktracePred, func() bool { return true })
			handler.ServeHTTP(test.r(), req)
		})
	}
}

func TestResponseWriterDecoratorShouldNotUseFlush(t *testing.T) {
	var decorator interface{} = &respLogger{}
	if _, ok := decorator.(http.Flusher); ok {
		t.Errorf("decorator should not use Flush method, use FlushError instead")
	}
}

func TestResponseWriterDecoratorConstruction(t *testing.T) {
	req, err := http.NewRequest("GET", "http://example.com", nil)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
		return
	}

	inner1 := &responsewritertesting.FakeResponseWriter{}
	middle := &respLogger{w: inner1} // middle is the decorator
	outer := responsewriter.WrapForHTTP1Or2(middle)
	// FakeResponseWriter does not implement http.Flusher, FlusherError,
	// http.CloseNotifier, or http.Hijacker; so WrapForHTTP1Or2 is not
	// expected to return an outer object.
	if outer != middle {
		t.Errorf("Did not expect a new outer object, but got %v", outer)
	}
	if innerGot := outer.(responsewriter.UserProvidedDecorator).Unwrap(); inner1 != innerGot {
		t.Errorf("Expected the decorator to return the inner http.ResponseWriter object")
	}

	// simulate http2
	inner2 := &responsewritertesting.FakeResponseWriterFlusherCloseNotifier{FakeResponseWriter: inner1}
	middle = newLogged(req, inner2)
	outer = responsewriter.WrapForHTTP1Or2(middle)
	if innerGot := outer.(responsewriter.UserProvidedDecorator).Unwrap(); inner2 != innerGot {
		t.Errorf("Expected the decorator to return the inner http.ResponseWriter object")
	}
	responsewritertesting.AssertResponseWriterInterfaceCompatibility(t, inner2, outer)

	// simulate http/1x
	inner3 := &responsewritertesting.FakeResponseWriterFlusherCloseNotifierHijacker{FakeResponseWriterFlusherCloseNotifier: inner2}
	middle = newLogged(req, inner3)
	outer = responsewriter.WrapForHTTP1Or2(middle)
	if innerGot := outer.(responsewriter.UserProvidedDecorator).Unwrap(); inner3 != innerGot {
		t.Errorf("Expected the decorator to return the inner http.ResponseWriter object")
	}
	responsewritertesting.AssertResponseWriterInterfaceCompatibility(t, inner3, outer)
}

func TestResponseWriterDecoratorWithHTTPServer(t *testing.T) {
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

					middle := newLogged(r, inner)
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
