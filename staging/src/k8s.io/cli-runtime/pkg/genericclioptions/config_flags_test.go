/*
Copyright 2026 The Kubernetes Authors.

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
	"os"
	"path/filepath"
	"testing"

	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
)

// TestDefaultClientConfigAllowsInClusterFallback verifies that the default
// configuration used to detect a missing or empty kubeconfig only takes flags
// into account that select a different server. Flags that do not select a
// different server must not prevent the in-cluster configuration from being
// used.
func TestDefaultClientConfigAllowsInClusterFallback(t *testing.T) {
	// certificate-authority, client-certificate and client-key files are only
	// validated by opening them, so empty files are enough.
	tempDir := t.TempDir()
	caFile := filepath.Join(tempDir, "ca.crt")
	clientCertFile := filepath.Join(tempDir, "client.crt")
	clientKeyFile := filepath.Join(tempDir, "client.key")
	for _, file := range []string{caFile, clientCertFile, clientKeyFile} {
		if err := os.WriteFile(file, []byte("data"), 0600); err != nil {
			t.Fatal(err)
		}
	}

	tests := []struct {
		name      string
		override  func(*clientcmd.ConfigOverrides)
		isDefault bool
	}{
		{
			name:      "no flags",
			override:  func(*clientcmd.ConfigOverrides) {},
			isDefault: true,
		},
		{
			name: "request timeout",
			override: func(overrides *clientcmd.ConfigOverrides) {
				overrides.Timeout = "5s"
			},
			isDefault: true,
		},
		{
			name: "impersonation",
			override: func(overrides *clientcmd.ConfigOverrides) {
				overrides.AuthInfo.Impersonate = "alice"
				overrides.AuthInfo.ImpersonateUID = "uid-1234"
				overrides.AuthInfo.ImpersonateGroups = []string{"admins"}
			},
			isDefault: true,
		},
		{
			name: "disable compression",
			override: func(overrides *clientcmd.ConfigOverrides) {
				overrides.ClusterInfo.DisableCompression = true
			},
			isDefault: true,
		},
		{
			name: "insecure skip tls verify",
			override: func(overrides *clientcmd.ConfigOverrides) {
				overrides.ClusterInfo.InsecureSkipTLSVerify = true
			},
			isDefault: true,
		},
		{
			name: "certificate authority",
			override: func(overrides *clientcmd.ConfigOverrides) {
				overrides.ClusterInfo.CertificateAuthority = caFile
			},
			isDefault: true,
		},
		{
			name: "tls server name",
			override: func(overrides *clientcmd.ConfigOverrides) {
				overrides.ClusterInfo.TLSServerName = "api.example.com"
			},
			isDefault: true,
		},
		{
			name: "token",
			override: func(overrides *clientcmd.ConfigOverrides) {
				overrides.AuthInfo.Token = "explicit-token"
			},
			isDefault: true,
		},
		{
			name: "client certificate",
			override: func(overrides *clientcmd.ConfigOverrides) {
				overrides.AuthInfo.ClientCertificate = clientCertFile
				overrides.AuthInfo.ClientKey = clientKeyFile
			},
			isDefault: true,
		},
		{
			name: "server takes precedence over the in-cluster configuration",
			override: func(overrides *clientcmd.ConfigOverrides) {
				overrides.ClusterInfo.Server = "https://other.example.com:6443"
			},
			isDefault: false,
		},
		{
			name: "proxy url takes precedence over the in-cluster configuration",
			override: func(overrides *clientcmd.ConfigOverrides) {
				overrides.ClusterInfo.ProxyURL = "http://proxy.example.com:8080"
			},
			isDefault: false,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			overrides := &clientcmd.ConfigOverrides{
				ClusterDefaults: clientcmd.ClusterDefaults,
				Timeout:         "0",
			}
			test.override(overrides)

			loadingRules := clientcmd.NewDefaultClientConfigLoadingRules()
			loadingRules.DefaultClientConfig = defaultClientConfig(overrides)

			// These are the configurations the deferred client config builder
			// compares to decide whether to use the in-cluster configuration:
			// the config merged from an empty kubeconfig and the overrides,
			// against the default config for the same overrides.
			merged, err := clientcmd.NewNonInteractiveClientConfig(*clientcmdapi.NewConfig(), "", overrides, nil).ClientConfig()
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got := loadingRules.IsDefaultConfig(merged); got != test.isDefault {
				t.Errorf("IsDefaultConfig() = %v, want %v", got, test.isDefault)
			}
		})
	}
}
