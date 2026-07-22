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

package verify

import (
	"fmt"
	"strings"
	"testing"
)

func TestAudienceForService(t *testing.T) {
	tests := []struct {
		name      string
		svcName   string
		namespace string
		port      int32
		path      string
		want      string
	}{
		{
			name:      "explicit port and path",
			svcName:   "my-webhook",
			namespace: "kube-system",
			port:      8443,
			path:      "/validate",
			want:      "https://my-webhook.kube-system.svc:8443/validate",
		},
		{
			name:      "default port when omitted",
			svcName:   "my-webhook",
			namespace: "kube-system",
			port:      0,
			path:      "/validate",
			want:      "https://my-webhook.kube-system.svc:443/validate",
		},
		{
			name:      "empty path normalized to single slash",
			svcName:   "my-webhook",
			namespace: "default",
			port:      443,
			path:      "",
			want:      "https://my-webhook.default.svc:443/",
		},
		{
			name:      "path without leading slash is normalized",
			svcName:   "my-webhook",
			namespace: "default",
			port:      443,
			path:      "validate",
			want:      "https://my-webhook.default.svc:443/validate",
		},
		{
			name:      "explicit single-slash path is unchanged",
			svcName:   "my-webhook",
			namespace: "default",
			port:      443,
			path:      "/",
			want:      "https://my-webhook.default.svc:443/",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := AudienceForService(tc.svcName, tc.namespace, tc.port, tc.path); got != tc.want {
				t.Errorf("AudienceForService() = %q, want %q", got, tc.want)
			}
		})
	}
}

// TestAudienceForService_ServerParity asserts our derived Service audience is
// byte-identical to kube-apiserver's validateWebhookAudience
// (pkg/registry/core/serviceaccount/storage/token.go) for the same inputs. A
// divergence would let a legitimately-configured webhook deny-all on an audience
// mismatch, so this is a tripwire kept in lockstep with the server's derivation.
func TestAudienceForService_ServerParity(t *testing.T) {
	// serverAudience reproduces the server's path/port derivation verbatim (same
	// format string) so a drift on either side is caught. pathSet models the
	// server's non-nil check: the server maps a nil Service.Path to "/", which on
	// our side is an omitted (empty-string) path.
	serverAudience := func(name, namespace string, port int32, path string, pathSet bool) string {
		if port == 0 {
			port = 443
		}
		p := "/"
		if pathSet {
			p = path
			if !strings.HasPrefix(p, "/") {
				p = "/" + p
			}
		}
		return fmt.Sprintf("https://%s.%s.svc:%d%s", name, namespace, port, p)
	}

	cases := []struct {
		name    string
		port    int32
		path    string
		pathSet bool
	}{
		{name: "nil/empty path -> single slash", port: 8443, path: "", pathSet: false},
		{name: "explicit empty-string path -> single slash", port: 443, path: "", pathSet: true},
		{name: "explicit single-slash path", port: 443, path: "/", pathSet: true},
		{name: "leading-slash path", port: 8443, path: "/validate", pathSet: true},
		{name: "no-leading-slash path gets prepended", port: 443, path: "validate", pathSet: true},
		{name: "default port when zero", port: 0, path: "/validate", pathSet: true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := AudienceForService("webhook-svc", "webhook-ns", tc.port, tc.path)
			want := serverAudience("webhook-svc", "webhook-ns", tc.port, tc.path, tc.pathSet)
			if got != want {
				t.Errorf("audience diverged from server derivation:\n  ours:   %q\n  server: %q", got, want)
			}
		})
	}
}
