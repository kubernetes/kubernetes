/*
Copyright 2025 The Kubernetes Authors.

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

func TestConfigFlagsWithCertKeyOverride(t *testing.T) {
	// Create a temporary directory for test files
	tmpDir, err := os.MkdirTemp("", "kubeconfig-test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	kubeconfigPath := filepath.Join(tmpDir, "kubeconfig")

	// Create a kubeconfig with inline certificate data
	baseConfig := &clientcmdapi.Config{
		Clusters: map[string]*clientcmdapi.Cluster{
			"test-cluster": {
				Server:                   "https://example.com:6443",
				CertificateAuthorityData: []byte("fake-ca-data"),
			},
		},
		AuthInfos: map[string]*clientcmdapi.AuthInfo{
			"test-user": {
				ClientCertificateData: []byte("base-config-cert-data"),
				ClientKeyData:         []byte("base-config-key-data"),
			},
		},
		Contexts: map[string]*clientcmdapi.Context{
			"test-context": {
				Cluster:  "test-cluster",
				AuthInfo: "test-user",
			},
		},
		CurrentContext: "test-context",
	}

	err = clientcmd.WriteToFile(*baseConfig, kubeconfigPath)
	if err != nil {
		t.Fatalf("Failed to write kubeconfig: %v", err)
	}

	// Create certificate and key files
	certFile := filepath.Join(tmpDir, "client.crt")
	keyFile := filepath.Join(tmpDir, "client.key")

	err = os.WriteFile(certFile, []byte("override-cert-content"), 0600)
	if err != nil {
		t.Fatalf("Failed to create cert file: %v", err)
	}

	err = os.WriteFile(keyFile, []byte("override-key-content"), 0600)
	if err != nil {
		t.Fatalf("Failed to create key file: %v", err)
	}

	// Set KUBECONFIG environment variable
	originalKubeconfig := os.Getenv("KUBECONFIG")
	err = os.Setenv("KUBECONFIG", kubeconfigPath)
	if err != nil {
		t.Fatalf("Failed to set KUBECONFIG env var: %v", err)
	}
	defer func() {
		if originalKubeconfig != "" {
			os.Setenv("KUBECONFIG", originalKubeconfig)
		} else {
			os.Unsetenv("KUBECONFIG")
		}
	}()

	// Test case 1: ConfigFlags with CertFile and KeyFile should work without validation errors
	configFlags := &ConfigFlags{
		CertFile: &certFile,
		KeyFile:  &keyFile,
	}

	_, err = configFlags.ToRESTConfig()
	if err != nil {
		t.Errorf("ToRESTConfig() failed with cert/key file overrides: %v", err)
	}

	// Test case 2: Verify that the overrides properly clear inline data
	rawConfig := configFlags.ToRawKubeConfigLoader()

	// Get the merged config to verify the overrides work correctly
	// We need to get the client config to trigger the merge process
	clientConfig, err := rawConfig.ClientConfig()
	if err != nil {
		t.Fatalf("Failed to get client config: %v", err)
	}

	// The client config should be valid (no validation errors)
	if clientConfig == nil {
		t.Fatal("Expected client config to be non-nil")
	}

	// Verify that the client config has the correct certificate and key files
	if clientConfig.CertFile != certFile {
		t.Errorf("Expected CertFile to be %q, got %q", certFile, clientConfig.CertFile)
	}
	if clientConfig.KeyFile != keyFile {
		t.Errorf("Expected KeyFile to be %q, got %q", keyFile, clientConfig.KeyFile)
	}
}

func TestConfigFlagsWithoutCertKeyOverride(t *testing.T) {
	// Create a temporary directory for test files
	tmpDir, err := os.MkdirTemp("", "kubeconfig-test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	kubeconfigPath := filepath.Join(tmpDir, "kubeconfig")

	// Create a kubeconfig with inline certificate data
	baseConfig := &clientcmdapi.Config{
		Clusters: map[string]*clientcmdapi.Cluster{
			"test-cluster": {
				Server:                   "https://example.com:6443",
				CertificateAuthorityData: []byte("fake-ca-data"),
			},
		},
		AuthInfos: map[string]*clientcmdapi.AuthInfo{
			"test-user": {
				ClientCertificateData: []byte("base-config-cert-data"),
				ClientKeyData:         []byte("base-config-key-data"),
			},
		},
		Contexts: map[string]*clientcmdapi.Context{
			"test-context": {
				Cluster:  "test-cluster",
				AuthInfo: "test-user",
			},
		},
		CurrentContext: "test-context",
	}

	err = clientcmd.WriteToFile(*baseConfig, kubeconfigPath)
	if err != nil {
		t.Fatalf("Failed to write kubeconfig: %v", err)
	}

	// Set KUBECONFIG environment variable
	originalKubeconfig := os.Getenv("KUBECONFIG")
	err = os.Setenv("KUBECONFIG", kubeconfigPath)
	if err != nil {
		t.Fatalf("Failed to set KUBECONFIG env var: %v", err)
	}
	defer func() {
		if originalKubeconfig != "" {
			os.Setenv("KUBECONFIG", originalKubeconfig)
		} else {
			os.Unsetenv("KUBECONFIG")
		}
	}()

	// Test case: ConfigFlags without CertFile and KeyFile should preserve inline data
	configFlags := &ConfigFlags{}

	_, err = configFlags.ToRESTConfig()
	if err != nil {
		t.Errorf("ToRESTConfig() failed without overrides: %v", err)
	}

	// Verify that inline data is preserved when no file paths are specified
	rawConfig := configFlags.ToRawKubeConfigLoader()
	config, err := rawConfig.RawConfig()
	if err != nil {
		t.Fatalf("Failed to get raw config: %v", err)
	}

	authInfo := config.AuthInfos["test-user"]
	if authInfo == nil {
		t.Fatal("Expected auth info 'test-user' not found")
	}

	// Verify that inline data is preserved
	if len(authInfo.ClientCertificateData) == 0 {
		t.Error("Expected ClientCertificateData to be preserved when no CertFile is specified")
	}
	if len(authInfo.ClientKeyData) == 0 {
		t.Error("Expected ClientKeyData to be preserved when no KeyFile is specified")
	}
}
