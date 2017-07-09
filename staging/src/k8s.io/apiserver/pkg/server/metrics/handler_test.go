/*
Copyright 2017 The Kubernetes Authors.

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

package metrics

import (
	"bytes"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestPrometheusHandler(t *testing.T) {
	req := httptest.NewRequest("GET", "http://example.com/metrics", nil)
	w := httptest.NewRecorder()
	h := NewPrometheusHandler()
	h.ServeHTTP(w, req)
	w = httptest.NewRecorder()
	h.ServeHTTP(w, req)

	resp := w.Result()
	body, _ := ioutil.ReadAll(resp.Body)

	expectedStatusCode := http.StatusOK
	if resp.StatusCode != expectedStatusCode {
		t.Errorf("Incorrect status code: expected %d, got %d", expectedStatusCode, resp.StatusCode)
	}
	if len(body) == 0 {
		t.Errorf("expected a body")
	}

	expectedCT := "text/plain"
	actualCT := resp.Header.Get("Content-Type")
	if !strings.HasPrefix(actualCT, expectedCT) {
		t.Errorf("incorrect content-type: expected %q, got %q", expectedCT, actualCT)
	}

	expectedMetric := []byte("http_request_duration_seconds")
	if bytes.Index(body, expectedMetric) == -1 {
		t.Errorf("missing metrics: expected %q, got %q", expectedMetric, string(body))
	}
}

func TestDuplicateHandler(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("Should not panic: %v\n", r)
		}
	}()
	NewPrometheusHandler()
	NewPrometheusHandler()
}
