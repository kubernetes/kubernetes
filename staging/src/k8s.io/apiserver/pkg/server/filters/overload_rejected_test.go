/*
Copyright The Kubernetes Authors.

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

func TestTooManyRequestsMarksOverloadRejection(t *testing.T) {
	w := httptest.NewRecorder()
	tooManyRequests(httptest.NewRequest(http.MethodGet, "/api/v1/pods", nil), w, "7")

	resp := w.Result()
	if resp.StatusCode != http.StatusTooManyRequests {
		t.Errorf("expected status %d, got %d", http.StatusTooManyRequests, resp.StatusCode)
	}
	if got := resp.Header.Get("Retry-After"); got != "7" {
		t.Errorf("expected Retry-After: 7, got %q", got)
	}
	// The client-go REST client matches this exact name; changing it breaks
	// rest_client_requests_rejected_total.
	if got := resp.Header.Get("X-Kubernetes-Overload-Rejected"); got != "true" {
		t.Errorf("expected X-Kubernetes-Overload-Rejected: true, got %q", got)
	}
}
