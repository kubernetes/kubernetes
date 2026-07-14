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

package oidc

import "testing"

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
			name:      "no path",
			svcName:   "my-webhook",
			namespace: "default",
			port:      443,
			path:      "",
			want:      "https://my-webhook.default.svc:443",
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
