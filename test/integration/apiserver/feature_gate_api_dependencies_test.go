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

package apiserver

import (
	"strings"
	"testing"

	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
)

// TestFeatureGateAPIDependencies exercises the startup check that refuses to boot
// kube-apiserver when an enabled feature gate requires an API resource that the
// effective runtime-config does not serve.
func TestFeatureGateAPIDependencies(t *testing.T) {
	tests := []struct {
		name            string
		flags           []string
		wantErr         bool
		wantErrContains []string
	}{
		{
			// Guards the invariant that stock defaults never trip the validation:
			// every mapped gate is off by default, so nothing is required.
			name:    "default flags start cleanly",
			flags:   nil,
			wantErr: false,
		},
		{
			// The exact failure mode from the issue: gate on, required API silently
			// not served. Startup must fail fast instead of surfacing runtime 404s.
			name:    "enabled gate without served API is rejected",
			flags:   []string{"--feature-gates=EvictionRequestAPI=true"},
			wantErr: true,
			wantErrContains: []string{
				"EvictionRequestAPI is enabled",
				"lifecycle.k8s.io/evictions",
			},
		},
		{
			// Enabling the required API via runtime-config satisfies the dependency.
			name: "enabled gate with served API starts",
			flags: []string{
				"--feature-gates=EvictionRequestAPI=true",
				"--runtime-config=lifecycle.k8s.io/v1alpha1=true",
			},
			wantErr: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			flags := append(framework.DefaultTestServerFlags(), tc.flags...)
			server, err := kubeapiservertesting.StartTestServer(t, nil, flags, framework.SharedEtcd())
			if server.TearDownFn != nil {
				defer server.TearDownFn()
			}

			if tc.wantErr {
				if err == nil {
					t.Fatalf("expected kube-apiserver startup to fail, but it started successfully")
				}
				for _, want := range tc.wantErrContains {
					if !strings.Contains(err.Error(), want) {
						t.Errorf("startup error %q does not contain %q", err.Error(), want)
					}
				}
				return
			}
			if err != nil {
				t.Fatalf("expected kube-apiserver to start, but got error: %v", err)
			}
		})
	}
}
