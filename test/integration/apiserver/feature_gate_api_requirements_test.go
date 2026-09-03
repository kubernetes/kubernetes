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
			name:  "gate enabled by default with its API explicitly disabled starts",
			flags: []string{"--runtime-config=certificates.k8s.io/v1/clustertrustbundles=false"},
		},
		{
			name: "explicitly enabled gate with its API explicitly disabled is rejected",
			flags: []string{
				"--feature-gates=ClusterTrustBundle=true",
				"--runtime-config=certificates.k8s.io/v1/clustertrustbundles=false",
			},
			wantErr: true,
			wantErrContains: []string{
				"ClusterTrustBundle is enabled",
				"clustertrustbundles.certificates.k8s.io",
			},
		},
		{
			name:    "enabled beta gate without its beta API is rejected",
			flags:   []string{"--feature-gates=GenericWorkload=true"},
			wantErr: true,
			wantErrContains: []string{
				"GenericWorkload is enabled",
				"workloads.scheduling.k8s.io",
				"podgroups.scheduling.k8s.io",
			},
		},
		{
			name: "enabled beta gate with its beta API starts",
			flags: []string{
				"--feature-gates=GenericWorkload=true",
				"--runtime-config=scheduling.k8s.io/v1beta1=true",
			},
		},
		{
			// group/version=true asks for the version, not for the gated resources in it.
			name:  "explicitly enabled API version without its gate starts",
			flags: []string{"--runtime-config=scheduling.k8s.io/v1beta1=true"},
		},
		{
			name: "explicitly enabled API version with its gate explicitly disabled starts",
			flags: []string{
				"--feature-gates=EvictionRequestAPI=false",
				"--runtime-config=lifecycle.k8s.io/v1alpha1=true",
			},
		},
		{
			name:    "explicitly enabled API resource without its gate is rejected",
			flags:   []string{"--runtime-config=lifecycle.k8s.io/v1alpha1/evictionrequests=true"},
			wantErr: true,
			wantErrContains: []string{
				"evictionrequests.lifecycle.k8s.io was explicitly enabled with --runtime-config (lifecycle.k8s.io/v1alpha1/evictionrequests)",
				"EvictionRequestAPI",
			},
		},
		{
			name: "explicitly enabled API resource with its gate explicitly disabled is rejected",
			flags: []string{
				"--feature-gates=EvictionRequestAPI=false",
				"--runtime-config=lifecycle.k8s.io/v1alpha1/evictionrequests=true",
			},
			wantErr: true,
			wantErrContains: []string{
				"evictionrequests.lifecycle.k8s.io was explicitly enabled with --runtime-config",
				"EvictionRequestAPI",
			},
		},
		{
			name:  "default flags at an emulated version start cleanly",
			flags: []string{"--emulated-version=1.36"},
		},
		{
			name: "explicitly enabled API version at an emulated version starts",
			flags: []string{
				"--emulated-version=1.36",
				"--runtime-config=certificates.k8s.io/v1=true",
			},
		},
		{
			name: "explicitly enabled API resource not yet introduced at the emulated version is rejected for its gate",
			flags: []string{
				"--emulated-version=1.36",
				"--runtime-config=certificates.k8s.io/v1/clustertrustbundles=true",
			},
			wantErr: true,
			wantErrContains: []string{
				"clustertrustbundles.certificates.k8s.io was explicitly enabled with --runtime-config",
				"ClusterTrustBundle",
			},
		},
		{
			name: "explicitly enabled API resource not yet introduced at the emulated version is rejected by the lifecycle filter",
			flags: []string{
				"--emulated-version=1.36",
				"--feature-gates=ClusterTrustBundle=true",
				"--runtime-config=certificates.k8s.io/v1/clustertrustbundles=true",
			},
			wantErr: true,
			wantErrContains: []string{
				"cannot enable resource certificates.k8s.io/v1, Resource=clustertrustbundles in runtime-config",
				"introduced at 1.37",
			},
		},
		{
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
