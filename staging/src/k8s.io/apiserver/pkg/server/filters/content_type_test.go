/*
Copyright 2019 The Kubernetes Authors.

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
	"net/http"
	"net/http/httptest"
	"testing"
)

func noopHandler() http.HandlerFunc {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// noop
	})
}

func TestWithContentType(t *testing.T) {
	mux := http.NewServeMux()
	mux.Handle("/text", WithContentType(noopHandler(), "text/plain"))
	mux.Handle("/json", WithContentType(noopHandler(), "application/json"))
	tests := []struct {
		description      string
		path             string
		expectedMimeType string
	}{
		{"/text should return a plain text response", "/text", "text/plain"},
		{"/json should return a json response", "/json", "application/json"},
	}
	for _, test := range tests {
		path := "http://example.com" + test.path
		t.Run(path, func(t *testing.T) {
			req, err := http.NewRequest("GET", path, nil)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			w := httptest.NewRecorder()
			mux.ServeHTTP(w, req)
			if nosniffHeader := w.Header().Get("X-Content-Type-Options"); nosniffHeader != "nosniff" {
				t.Errorf("expected nosniff header to be set, got %v", nosniffHeader)
			}
			if mimeTypeHeader := w.Header().Get("Content-Type"); mimeTypeHeader != test.expectedMimeType {
				t.Errorf("expected %v, got %v", test.expectedMimeType, mimeTypeHeader)
			}
		})
	}
}
