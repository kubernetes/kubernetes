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

package admissionhttp

import (
	"net/http/httptest"
	"testing"
)

// TestInClusterAudienceResolver covers the zero-config in-cluster derivation:
// name/namespace from request.Host, port from the kubelet-injected service env
// var (which is the trust anchor), and path from request.URL.Path.
func TestInClusterAudienceResolver(t *testing.T) {
	resolve := inClusterAudienceResolver()

	tests := []struct {
		name    string
		target  string // absolute URL, sets request.Host and request.URL.Path
		envName string
		envVal  string
		want    string
		wantErr bool
	}{
		{
			name:    "derives audience; env port is authoritative over the host port",
			target:  "https://my-webhook.webhooks.svc:8443/validate",
			envName: "MY_WEBHOOK_SERVICE_PORT",
			envVal:  "443",
			want:    "https://my-webhook.webhooks.svc:443/validate",
		},
		{
			name:    "dashed service name maps to underscored env var",
			target:  "https://audit-hook.kube-system.svc:443/v1",
			envName: "AUDIT_HOOK_SERVICE_PORT",
			envVal:  "8443",
			want:    "https://audit-hook.kube-system.svc:8443/v1",
		},
		{
			name:    "no service env var (scheduling race or spoofed host) is rejected",
			target:  "https://my-webhook.webhooks.svc:443/validate",
			envName: "UNRELATED_SERVICE_PORT",
			envVal:  "443",
			wantErr: true,
		},
		{
			name:    "non-service host is rejected",
			target:  "https://localhost:8443/validate",
			envName: "LOCALHOST_SERVICE_PORT",
			envVal:  "8443",
			wantErr: true,
		},
		{
			name:    "non-numeric port env var is rejected",
			target:  "https://my-webhook.webhooks.svc:443/validate",
			envName: "MY_WEBHOOK_SERVICE_PORT",
			envVal:  "https",
			wantErr: true,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv(tc.envName, tc.envVal)
			r := httptest.NewRequest("POST", tc.target, nil)

			got, err := resolve(r)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("expected error, got audience %q", got)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tc.want {
				t.Errorf("audience = %q, want %q", got, tc.want)
			}
		})
	}
}
