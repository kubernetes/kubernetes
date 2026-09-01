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

func TestFeatureGateAPIRequirements(t *testing.T) {
	tests := []struct {
		name            string
		flags           []string
		wantErr         bool
		wantErrContains []string
	}{
		{
			name:  "default flags start cleanly",
			flags: nil,
		},
		{
			name:    "enabled gate without served API is rejected",
			flags:   []string{"--feature-gates=EvictionRequestAPI=true"},
			wantErr: true,
			wantErrContains: []string{
				"EvictionRequestAPI is enabled",
				"evictions.lifecycle.k8s.io",
			},
		},
		{
			name: "enabled gate with served API starts",
			flags: []string{
				"--feature-gates=EvictionRequestAPI=true",
				"--runtime-config=lifecycle.k8s.io/v1alpha1=true",
			},
		},
		{
			name:  "default flags at an emulated version start cleanly",
			flags: []string{"--emulated-version=1.36"},
		},
		{
			// clustertrustbundles is only introduced in certificates/v1 at 1.37, so at emulation
			// 1.36 the resource config says certificates/v1 is enabled while the lifecycle filter
			// drops the resource. Validating the resource config alone would miss this.
			name: "enabled gate whose API predates the emulated version is rejected",
			flags: []string{
				"--emulated-version=1.36",
				"--feature-gates=ClusterTrustBundle=true",
			},
			wantErr: true,
			wantErrContains: []string{
				"ClusterTrustBundle is enabled",
				"clustertrustbundles.certificates.k8s.io",
			},
		},
		{
			// The same configuration is satisfiable through the version that does exist at 1.36.
			name: "enabled gate served by an older version at the emulated version starts",
			flags: []string{
				"--emulated-version=1.36",
				"--feature-gates=ClusterTrustBundle=true",
				"--runtime-config=certificates.k8s.io/v1beta1/clustertrustbundles=true",
			},
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
