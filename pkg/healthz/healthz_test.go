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

package healthz

import (
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestInstallHandler(t *testing.T) {
	mux := http.NewServeMux()
	InstallHandler(mux)
	req, err := http.NewRequest("GET", "http://example.com/healthz", nil)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Errorf("Expected %v, got %v", http.StatusOK, w.Code)
	}
	if w.Body.String() != "ok" {
		t.Errorf("Expected %v, got %v", "ok", w.Body.String())
	}
}

func TestMulitipleChecks(t *testing.T) {
	tests := []struct {
		path             string
		expectedResponse string
		expectedStatus   int
		addBadCheck      bool
	}{
		{"/healthz?verbose", "[+]ping ok\nhealthz check passed\n", http.StatusOK, false},
		{"/healthz/ping", "ok", http.StatusOK, false},
		{"/healthz", "ok", http.StatusOK, false},
		{"/healthz?verbose", "[+]ping ok\n[-]bad failed: this will fail\nhealthz check failed\n", http.StatusInternalServerError, true},
		{"/healthz/ping", "ok", http.StatusOK, true},
		{"/healthz/bad", "Internal server error: this will fail\n", http.StatusInternalServerError, true},
		{"/healthz", "[+]ping ok\n[-]bad failed: this will fail\nhealthz check failed\n", http.StatusInternalServerError, true},
	}

	for i, test := range tests {
		mux := http.NewServeMux()
		checks := []HealthzChecker{PingHealthz}
		if test.addBadCheck {
			checks = append(checks, NamedCheck("bad", func(_ *http.Request) error {
				return errors.New("this will fail")
			}))
		}
		InstallHandler(mux, checks...)
		req, err := http.NewRequest("GET", fmt.Sprintf("http://example.com%v", test.path), nil)
		if err != nil {
			t.Fatalf("case[%d] Unexpected error: %v", i, err)
		}
		w := httptest.NewRecorder()
		mux.ServeHTTP(w, req)
		if w.Code != test.expectedStatus {
			t.Errorf("case[%d] Expected: %v, got: %v", i, test.expectedStatus, w.Code)
		}
		if w.Body.String() != test.expectedResponse {
			t.Errorf("case[%d] Expected:\n%v\ngot:\n%v\n", i, test.expectedResponse, w.Body.String())
		}
	}
}
