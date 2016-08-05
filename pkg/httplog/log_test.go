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
	"bytes"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"
)

func TestHandler(t *testing.T) {
	want := &httptest.ResponseRecorder{
		HeaderMap: make(http.Header),
		Body:      new(bytes.Buffer),
	}
	want.WriteHeader(http.StatusOK)
	mux := http.NewServeMux()
	handler := Handler(mux, DefaultStacktracePred)
	mux.HandleFunc("/kube", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	req, err := http.NewRequest("GET", "http://example.com/kube", nil)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)
	if !reflect.DeepEqual(want, w) {
		t.Errorf("Expected %v, got %v", want, w)
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

func TestNewLogged(t *testing.T) {
	req, err := http.NewRequest("GET", "http://example.com", nil)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	handler := func(w http.ResponseWriter, r *http.Request) {
		NewLogged(req, &w)
		defer func() {
			if r := recover(); r == nil {
				t.Errorf("Expected NewLogged to panic")
			}
		}()
		NewLogged(req, &w)
	}
	w := httptest.NewRecorder()
	handler(w, req)
}

func TestLogOf(t *testing.T) {
	logOfTests := []bool{true, false}
	for _, makeLogger := range logOfTests {
		req, err := http.NewRequest("GET", "http://example.com", nil)
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
		handler := func(w http.ResponseWriter, r *http.Request) {
			var want string
			if makeLogger {
				NewLogged(req, &w)
				want = "*httplog.respLogger"
			} else {
				want = "*httplog.passthroughLogger"
			}
			got := reflect.TypeOf(LogOf(r, w)).String()
			if want != got {
				t.Errorf("Expected %v, got %v", want, got)
			}
		}
		w := httptest.NewRecorder()
		handler(w, req)
	}
}

func TestUnlogged(t *testing.T) {
	unloggedTests := []bool{true, false}
	for _, makeLogger := range unloggedTests {
		req, err := http.NewRequest("GET", "http://example.com", nil)
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
		handler := func(w http.ResponseWriter, r *http.Request) {
			want := w
			if makeLogger {
				NewLogged(req, &w)
			}
			got := Unlogged(w)
			if want != got {
				t.Errorf("Expected %v, got %v", want, got)
			}
		}
		w := httptest.NewRecorder()
		handler(w, req)
	}
}

type testResponseWriter struct{}

func (*testResponseWriter) Header() http.Header       { return nil }
func (*testResponseWriter) Write([]byte) (int, error) { return 0, nil }
func (*testResponseWriter) WriteHeader(int)           {}

func TestLoggedStatus(t *testing.T) {
	req, err := http.NewRequest("GET", "http://example.com", nil)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	var tw http.ResponseWriter = new(testResponseWriter)
	logger := NewLogged(req, &tw)
	logger.Write(nil)

	if logger.status != http.StatusOK {
		t.Errorf("expected status after write to be %v, got %v", http.StatusOK, logger.status)
	}

	tw = new(testResponseWriter)
	logger = NewLogged(req, &tw)
	logger.WriteHeader(http.StatusForbidden)
	logger.Write(nil)

	if logger.status != http.StatusForbidden {
		t.Errorf("expected status after write to remain %v, got %v", http.StatusForbidden, logger.status)
	}
}
