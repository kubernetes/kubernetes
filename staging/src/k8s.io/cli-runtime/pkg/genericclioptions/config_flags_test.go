/*
Copyright 2018 The Kubernetes Authors.

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

package genericclioptions

import (
	"testing"

	"github.com/spf13/pflag"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	"k8s.io/utils/ptr"
)

func TestConfigFlags_ProxyURL_FlagRegistration(t *testing.T) {
	tests := []struct {
		name           string
		proxyURL       *string
		shouldRegister bool
	}{
		{
			name:           "nil proxy URL should not register flag",
			proxyURL:       nil,
			shouldRegister: false,
		},
		{
			name:           "non-nil proxy URL should register flag",
			proxyURL:       ptr.To("http://proxy.example.com:8080"),
			shouldRegister: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			configFlags := &ConfigFlags{
				ProxyURL: tt.proxyURL,
			}

			flagSet := pflag.NewFlagSet("test", pflag.ContinueOnError)
			configFlags.AddFlags(flagSet)

			flag := flagSet.Lookup("proxy-url")

			if tt.shouldRegister {
				if flag == nil {
					t.Error("Expected proxy-url flag to be registered")
				}
			} else {
				if flag != nil {
					t.Error("Expected proxy-url flag to not be registered when ProxyURL is nil")
				}
			}
		})
	}
}

func TestConfigFlags_ProxyURL_Override(t *testing.T) {
	// Test direct override mechanism using ConfigOverrides
	tests := []struct {
		name            string
		kubeconfigProxy string
		overrideProxy   string
		expectedProxy   string
	}{
		{
			name:            "Override sets proxy when kubeconfig has none",
			kubeconfigProxy: "",
			overrideProxy:   "http://localhost:8080",
			expectedProxy:   "http://localhost:8080",
		},
		{
			name:            "Override replaces kubeconfig proxy",
			kubeconfigProxy: "https://old-proxy.com",
			overrideProxy:   "http://new-proxy.com",
			expectedProxy:   "http://new-proxy.com",
		},
		{
			name:            "Empty override sets empty proxy",
			kubeconfigProxy: "",
			overrideProxy:   "",
			expectedProxy:   "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create a basic kubeconfig with proxy setting
			config := clientcmdapi.NewConfig()
			config.Clusters["test-cluster"] = &clientcmdapi.Cluster{
				Server:   "https://test-server:6443",
				ProxyURL: tt.kubeconfigProxy,
			}
			config.AuthInfos["test-user"] = &clientcmdapi.AuthInfo{}
			config.Contexts["test-context"] = &clientcmdapi.Context{
				Cluster:  "test-cluster",
				AuthInfo: "test-user",
			}
			config.CurrentContext = "test-context"

			// Create override that simulates what ConfigFlags.toRawKubeConfigLoader() does
			overrides := &clientcmd.ConfigOverrides{}
			overrides.ClusterInfo.ProxyURL = tt.overrideProxy
			clientConfig := clientcmd.NewDefaultClientConfig(*config, overrides)

			// Get merged raw config and verify proxy URL override
			mergedConfig, err := clientConfig.MergedRawConfig()
			if err != nil {
				t.Fatalf("Unexpected error getting merged raw config: %v", err)
			}

			cluster := mergedConfig.Clusters["test-cluster"]
			if cluster == nil {
				t.Fatal("Expected test-cluster to exist")
			}

			if cluster.ProxyURL != tt.expectedProxy {
				t.Errorf("Expected cluster ProxyURL to be %q, got %q", tt.expectedProxy, cluster.ProxyURL)
			}
		})
	}
}
