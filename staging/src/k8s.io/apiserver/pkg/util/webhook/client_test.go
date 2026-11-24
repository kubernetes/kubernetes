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

package webhook

import (
	"testing"

	"golang.org/x/net/http2"

	"k8s.io/apimachinery/pkg/runtime/schema"
)

func TestWebhookClientConfig(t *testing.T) {
	cm, _ := NewClientManager([]schema.GroupVersion{})
	authInfoResolver, err := NewDefaultAuthenticationInfoResolver("")
	if err != nil {
		t.Fatal(err)
	}
	cm.SetAuthenticationInfoResolver(authInfoResolver)
	cm.SetServiceResolver(NewDefaultServiceResolver())

	tests := []struct {
		name             string
		url              string
		expectAllowHTTP2 bool
		expectHost       string
		expectPath       string
	}{
		{
			name:             "force http1",
			url:              "https://webhook.example.com",
			expectAllowHTTP2: false,
			expectHost:       "https://webhook.example.com",
			expectPath:       "",
		},
		{
			name:             "allow http2 for localhost",
			url:              "https://localhost",
			expectAllowHTTP2: true,
			expectHost:       "https://localhost",
			expectPath:       "",
		},
		{
			name:             "allow http2 for 127.0.0.1",
			url:              "https://127.0.0.1",
			expectAllowHTTP2: true,
			expectHost:       "https://127.0.0.1",
			expectPath:       "",
		},
		{
			name:             "allow http2 for [::1]:0",
			url:              "https://[::1]",
			expectAllowHTTP2: true,
			expectHost:       "https://[::1]",
			expectPath:       "",
		},
		{
			name:       "trailing slash omitted from host",
			url:        "https://example.com/foo",
			expectHost: "https://example.com",
			expectPath: "/foo",
		},
		{
			name:       "encoded host re-encoded",
			url:        "https://[::ffff:1.1.1.1%25.zone]",
			expectHost: "https://[::ffff:1.1.1.1%25.zone]",
			expectPath: "",
		},
		{
			name:       "encoded host re-encoded with port",
			url:        "https://[::ffff:1.1.1.1%25.zone]:443",
			expectHost: "https://[::ffff:1.1.1.1%25.zone]:443",
			expectPath: "",
		},
		{
			name:       "encoded host re-encoded with port and path",
			url:        "https://[::ffff:1.1.1.1%25.zone]:443/foo",
			expectHost: "https://[::ffff:1.1.1.1%25.zone]:443",
			expectPath: "/foo",
		},
		{
			name:       "unescaped path re-escaped",
			url:        "https://example.com/path with spaces",
			expectHost: "https://example.com",
			expectPath: "/path%20with%20spaces",
		},
		{
			name:       "escaped path re-escaped",
			url:        "https://example.com/path%20with%20spaces",
			expectHost: "https://example.com",
			expectPath: "/path%20with%20spaces",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {

			cc := ClientConfig{
				URL: tc.url,
			}
			cfg, err := cm.hookClientConfig(cc)
			if err != nil {
				t.Fatal(err)
			}
			if tc.expectAllowHTTP2 && !allowHTTP2(cfg.NextProtos) {
				t.Errorf("expected allow http/2, got: %v", cfg.NextProtos)
			}
			if tc.expectHost != cfg.Host {
				t.Errorf("expected host %s, got: %s", tc.expectHost, cfg.Host)
			}
			if tc.expectPath != cfg.APIPath {
				t.Errorf("expected apiPath %s, got: %s", tc.expectPath, cfg.APIPath)
			}
		})
	}
}

func allowHTTP2(nextProtos []string) bool {
	if len(nextProtos) == 0 {
		// the transport expressed no NextProto preference, allow
		return true
	}
	for _, p := range nextProtos {
		if p == http2.NextProtoTLS {
			// the transport explicitly allowed http/2
			return true
		}
	}
	// the transport explicitly set NextProtos and excluded http/2
	return false
}
