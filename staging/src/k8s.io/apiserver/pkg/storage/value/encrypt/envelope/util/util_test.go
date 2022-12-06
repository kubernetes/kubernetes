/*
Copyright 2022 The Kubernetes Authors.

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

package util

import (
	"strings"
	"testing"
)

func TestParseEndpoint(t *testing.T) {
	testCases := []struct {
		desc     string
		endpoint string
		want     string
	}{
		{
			desc:     "path with prefix",
			endpoint: "unix:///@path",
			want:     "@path",
		},
		{
			desc:     "path without prefix",
			endpoint: "unix:///path",
			want:     "/path",
		},
	}

	for _, tt := range testCases {
		t.Run(tt.desc, func(t *testing.T) {
			got, err := ParseEndpoint(tt.endpoint)
			if err != nil {
				t.Errorf("ParseEndpoint(%q) error: %v", tt.endpoint, err)
			}
			if got != tt.want {
				t.Errorf("ParseEndpoint(%q) = %q, want %q", tt.endpoint, got, tt.want)
			}
		})
	}
}

func TestParseEndpointError(t *testing.T) {
	testCases := []struct {
		desc     string
		endpoint string
		wantErr  string
	}{
		{
			desc:     "empty endpoint",
			endpoint: "",
			wantErr:  "remote KMS provider can't use empty string as endpoint",
		},
		{
			desc:     "invalid scheme",
			endpoint: "http:///path",
			wantErr:  "unsupported scheme \"http\" for remote KMS provider",
		},
	}

	for _, tt := range testCases {
		t.Run(tt.desc, func(t *testing.T) {
			_, err := ParseEndpoint(tt.endpoint)
			if err == nil {
				t.Errorf("ParseEndpoint(%q) error: %v", tt.endpoint, err)
			}
			if !strings.Contains(err.Error(), tt.wantErr) {
				t.Errorf("ParseEndpoint(%q) = %q, want %q", tt.endpoint, err, tt.wantErr)
			}
		})
	}
}
