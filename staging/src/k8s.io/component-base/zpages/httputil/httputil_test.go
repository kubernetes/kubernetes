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

func TestAcceptableMediaTypes(t *testing.T) {
	tests := []struct {
		name      string
		reqHeader string
		want      bool
	}{
		{
			name:      "valid text/plain header",
			reqHeader: "text/plain",
			want:      true,
		},
		{
			name:      "valid text/* header",
			reqHeader: "text/*",
			want:      true,
		},
		{
			name:      "valid */plain header",
			reqHeader: "*/plain",
			want:      true,
		},
		{
			name:      "valid accept args",
			reqHeader: "text/plain; charset=utf-8",
			want:      true,
		},
		{
			name:      "invalid text/foo header",
			reqHeader: "text/foo",
			want:      false,
		},
		{
			name:      "invalid text/plain params",
			reqHeader: "text/plain; foo=bar",
			want:      false,
		},
	}
	for _, tt := range tests {
		req, err := http.NewRequest(http.MethodGet, "http://example.com/statusz", nil)
		if err != nil {
			t.Fatalf("Unexpected error while creating request: %v", err)
		}

		req.Header.Set("Accept", tt.reqHeader)
		got := AcceptableMediaType(req)

		if got != tt.want {
			t.Errorf("Unexpected response from AcceptableMediaType(), want %v, got = %v", tt.want, got)
		}
	}
}

func TestNegotiateMediaTypeWithVersion(t *testing.T) {
	tests := []struct {
		name                string
		acceptHeader        string
		supportedMediaTypes []string
		version             string
		group               string
		kind                string
		wantMediaType       string
		wantErr             error
	}{
		{
			name:                "no accept header",
			supportedMediaTypes: []string{"application/json", "text/plain"},
			wantMediaType:       "text/plain",
		},
		{
			name:                "valid text/plain header",
			acceptHeader:        "text/plain",
			supportedMediaTypes: []string{"application/json", "text/plain"},
			wantMediaType:       "text/plain",
		},
		{
			name:                "application/json without params",
			acceptHeader:        "application/json",
			supportedMediaTypes: []string{"application/json", "text/plain"},
			wantErr:             ErrNotAcceptable,
		},
		{
			name:                "application/json with correct params",
			acceptHeader:        "application/json;v=v1alpha1;g=config.k8s.io;as=Statusz",
			supportedMediaTypes: []string{"application/json", "text/plain"},
			version:             "v1alpha1",
			group:               "config.k8s.io",
			kind:                "Statusz",
			wantMediaType:       "application/json",
		},
		{
			name:                "application/json with missing as",
			acceptHeader:        "application/json;v=v1alpha1;g=config.k8s.io",
			supportedMediaTypes: []string{"application/json", "text/plain"},
			version:             "v1alpha1",
			group:               "config.k8s.io",
			kind:                "Statusz",
			wantErr:             ErrNotAcceptable,
		},
		{
			name:                "invalid text/foo header",
			acceptHeader:        "text/foo",
			supportedMediaTypes: []string{"application/json", "text/plain"},
			wantErr:             ErrNotAcceptable,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req, _ := http.NewRequest("GET", "/", nil)
			if tt.acceptHeader != "" {
				req.Header.Set("Accept", tt.acceptHeader)
			}

			gotMediaType, gotErr := NegotiateMediaTypeWithVersion(req, tt.supportedMediaTypes, tt.version, tt.group, tt.kind)

			if gotErr != tt.wantErr {
				t.Errorf("NegotiateMediaTypeWithVersion() error = %v, wantErr %v", gotErr, tt.wantErr)
				return
			}

			if gotMediaType != tt.wantMediaType {
				t.Errorf("NegotiateMediaTypeWithVersion() = %v, want %v", gotMediaType, tt.wantMediaType)
			}
		})
	}
}
