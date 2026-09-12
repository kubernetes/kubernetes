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

	apirequest "k8s.io/apiserver/pkg/endpoints/request"
)

const maxResourceNameLength = 10

func TestWithResourceNameLengthLimit(t *testing.T) {
	tests := []struct {
		name           string
		requestInfo    *apirequest.RequestInfo
		wantStatus     int
		wantNextCalled bool
	}{
		{
			name:           "allows request when request info is missing",
			requestInfo:    nil,
			wantStatus:     http.StatusOK,
			wantNextCalled: true,
		},
		{
			name: "allows named resource request within limit",
			requestInfo: &apirequest.RequestInfo{
				IsResourceRequest: true,
				Resource:          "pods",
				Name:              "nginx",
			},
			wantStatus:     http.StatusOK,
			wantNextCalled: true,
		},
		{
			name: "rejects request when name exceeds max length",
			requestInfo: &apirequest.RequestInfo{
				IsResourceRequest: true,
				Resource:          "pods",
				Name:              "fooooooooooooooo",
			},
			wantStatus:     http.StatusBadRequest,
			wantNextCalled: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			nextCalled := false
			next := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				nextCalled = true
				w.WriteHeader(http.StatusOK)
			})

			handler := WithResourceNameLengthLimit(next, maxResourceNameLength)

			req := httptest.NewRequest(http.MethodGet, "/test", nil)
			if tt.requestInfo != nil {
				req = req.WithContext(apirequest.WithRequestInfo(req.Context(), tt.requestInfo))
			}

			rr := httptest.NewRecorder()
			handler.ServeHTTP(rr, req)

			if rr.Code != tt.wantStatus {
				t.Fatalf("expected status %d, got %d", tt.wantStatus, rr.Code)
			}

			if nextCalled != tt.wantNextCalled {
				t.Fatalf("expected nextCalled=%v, got %v", tt.wantNextCalled, nextCalled)
			}
		})
	}
}
