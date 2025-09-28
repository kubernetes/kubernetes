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

package httputil

import (
	"net/http"
	"testing"
)

func TestNegotiateMediaType(t *testing.T) {
	tests := []struct {
		name           string
		reqHeader      string
		supportedTypes []string
		want           string
		wantErr        bool
	}{
		{
			name:           "valid application/json header",
			reqHeader:      "application/json",
			supportedTypes: []string{"application/json", "text/plain"},
			want:           "application/json",
		},
		{
			name:           "valid text/plain header",
			reqHeader:      "text/plain",
			supportedTypes: []string{"application/json", "text/plain"},
			want:           "text/plain",
		},
		{
			name:           "no header",
			reqHeader:      "",
			supportedTypes: []string{"application/json", "text/plain"},
			want:           "",
		},
		{
			name:           "wildcard header",
			reqHeader:      "*/*",
			supportedTypes: []string{"application/json", "text/plain"},
			want:           "application/json",
		},
		{
			name:           "invalid header",
			reqHeader:      "application/xml",
			supportedTypes: []string{"application/json", "text/plain"},
			wantErr:        true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req, err := http.NewRequest(http.MethodGet, "http://example.com/statusz", nil)
			if err != nil {
				t.Fatalf("Unexpected error while creating request: %v", err)
			}

			req.Header.Set("Accept", tt.reqHeader)
			got, err := NegotiateMediaType(req, tt.supportedTypes)
			if (err != nil) != tt.wantErr {
				t.Errorf("NegotiateMediaType() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if got != tt.want {
				t.Errorf("Unexpected response from NegotiateMediaType(), want %v, got = %v", tt.want, got)
			}
		})
	}
}
