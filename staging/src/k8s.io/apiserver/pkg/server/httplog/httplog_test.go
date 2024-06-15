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
	"reflect"
	"testing"

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

func TestHTTPLogResponseWriterDecoratorWithFake(t *testing.T) {
	responsewritertesting.VerifyResponseWriterDecoratorWithFake(t, func(h http.Handler) http.Handler {
		return withLogging(h, DefaultStacktracePred, func() bool { return true })
	})
}

func TestHTTPLogResponseWriterDecoratorConstruction(t *testing.T) {
	inner := &responsewritertesting.FakeResponseWriter{}
	middle := &respLogger{w: inner} // middle is the decorator
	outer := responsewriter.WrapForHTTP1Or2(middle)

	// FakeResponseWriter does not implement http.Flusher, FlusherError,
	// http.CloseNotifier, or http.Hijacker; so WrapForHTTP1Or2 is not
	// expected to return an outer object.
	if outer != middle {
		t.Errorf("did not expect a new outer object, but got %v", outer)
	}

	decorator, ok := outer.(responsewriter.UserProvidedDecorator)
	if !ok {
		t.Fatal("expected the middle to implement UserProvidedDecorator")
	}
	if want, got := inner, decorator.Unwrap(); want != got {
		t.Errorf("expected the decorator to return the inner http.ResponseWriter object")
	}
}
