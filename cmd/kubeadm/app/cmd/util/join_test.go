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

package util

import (
	"encoding/base64"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"

	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
)

type kubeConfigSpec struct {
	CertificateAuthorityData string
	CertificateAuthorityFile string
	Server                   string
	ClusterName              string
	ContextName              string
	User                     string
}

func generateKubeConfig(spec kubeConfigSpec) *clientcmdapi.Config {
	cluster := clientcmdapi.NewCluster()

	if spec.CertificateAuthorityData != "" {
		decodedData, _ := base64.StdEncoding.DecodeString(spec.CertificateAuthorityData)
		cluster.CertificateAuthorityData = decodedData
	} else if spec.CertificateAuthorityFile != "" {
		cluster.CertificateAuthority = spec.CertificateAuthorityFile
	}

	cluster.Server = spec.Server

	context := clientcmdapi.NewContext()
	context.Cluster = spec.ClusterName
	context.AuthInfo = spec.User

	return &clientcmdapi.Config{
		Clusters: map[string]*clientcmdapi.Cluster{
			spec.ClusterName: cluster,
		},
		Contexts: map[string]*clientcmdapi.Context{
			spec.ContextName: context,
		},
		CurrentContext: spec.ContextName,
	}
}

func writeKubeConfigToFile(config *clientcmdapi.Config, filePath string) error {
	configBytes, err := clientcmd.Write(*config)
	if err != nil {
		return err
	}
	return os.WriteFile(filePath, configBytes, 0644)
}

func generateValidKubeConfig() *clientcmdapi.Config {
	return generateKubeConfig(kubeConfigSpec{
		CertificateAuthorityData: "LS0tLS1CRUdJTiBDRVJUSUZJQ0FURS0tLS0tCk1JSUN5RENDQWJDZ0F3SUJBZ0lCQURBTkJna3Foa2lHOXcwQkFRc0ZBREFWTVJNd0VRWURWUVFERXdwcmRXSmwKY201bGRHVnpNQjRYRFRFNU1URXlNREF3TkRrME1sb1hEVEk1TVRFeE56QXdORGswTWxvd0ZURVRNQkVHQTFVRQpBeE1LYTNWaVpYSnVaWFJsY3pDQ0FTSXdEUVlKS29aSWh2Y05BUUVCQlFBRGdnRVBBRENDQVFvQ2dnRUJBTXFRCmN0RUN6QTh5RlN1Vll1cE9VWWdyVG1mUWVLZS85QmFEV2FnYXE3b3c5K0kySXZzZldGdmxyRDhRUXI4c2VhNnEKeGpxN1RWNjdWYjRSeEJhb1lEQSt5STV2SWN1aldVeFVMdW42NGx1M1E2aUMxc2oyVW5tVXBJZGdhelJYWEVrWgp2eEE2RWJBbm94QTArbEJPbjFDWldsMjNJUTRzNzBvMmhaN3dJcC92ZXZCODhSUlJqcXR2Z2M1ZWxzanNibURGCkxTN0wxWnV5ZThjNmdTOTNiUitWalZtU0lmcjFJRXEwNzQ4dElJeVhqQVZDV1BWQ3Z1UDQxTWxmUGMvSlZwWkQKdUQyK3BPNlpZUkVjZEFuT2YyZUQ0L2VMT01La280TDFkU0Z5OUpLTTVQTG5PQzBaazBBWU9kMXZTOERUQWZ4agpYUEVJWThPQllGaGxzeGY0VEU4Q0F3RUFBYU1qTUNFd0RnWURWUjBQQVFIL0JBUURBZ0trTUE4R0ExVWRFd0VCCi93UUZNQU1CQWY4d0RRWUpLb1pJaHZjTkFRRUxCUUFEZ2dFQkFIL09ZcTh6eWwxK3pTVG11b3czeUkvMTVQTDEKZGw4aEI3SUtuWk5XbUMvTFRkbS8rbm9oM1NiMUlkUnY2SGtLZy9HVW4wVU11UlVuZ0xoanUzRU80b3pKUFFjWApxdWF4emdtVEtOV0o2RXJEdlJ2V2hHWDBaY2JkQmZaditkb3d5UnF6ZDVubEo0OWhDK05ydEZGUXE2UDA1QlluCjdTZW1ndXFlWG1Yd0lqMlNhKzFEZVI2bFJtOW84c2hBWWpueVRoVUZxYU1uMThrSTNTQU5KNXZrLzNERnJQRU8KQ0tDOUV6Rmt1Mmt1eGcyZE0xMlBiUkdaUTJvMEs2SEVaZ3JySUtUUE95M29jYjhyOU0wYVNGaGpPVi9OcUdBNApTYXVwWFNXNlhmdklpL1VIb0liVTNwTmNzblVKR25RZlF2aXA5NVhLay9ncWNVcittNTB2eGd1bXh0QT0KLS0tLS1FTkQgQ0VSVElGSUNBVEUtLS0tLQ==",
		Server:                   "test-server:6443",
		ClusterName:              "somecluster",
		ContextName:              "test@somecluster",
		User:                     "test",
	})
}

func generateMissingCAKubeConfig() *clientcmdapi.Config {
	return generateKubeConfig(kubeConfigSpec{
		Server:      "test-server:6443",
		ClusterName: "somecluster",
		ContextName: "test@somecluster",
		User:        "test",
	})
}

func generateInvalidFilePathKubeConfig() *clientcmdapi.Config {
	return generateKubeConfig(kubeConfigSpec{
		CertificateAuthorityFile: "invalid-file",
		Server:                   "test-server:6443",
		ClusterName:              "somecluster",
		ContextName:              "test@somecluster",
		User:                     "test",
	})
}

func generateInvalidCAKubeConfig() *clientcmdapi.Config {
	return generateKubeConfig(kubeConfigSpec{
		CertificateAuthorityData: "invalid-data",
		Server:                   "test-server:6443",
		ClusterName:              "somecluster",
		ContextName:              "test@somecluster",
		User:                     "test",
	})
}

func TestGetJoinCommand(t *testing.T) {
	tests := []struct {
		name         string
		kubeConfig   *clientcmdapi.Config
		expectError  bool
		errorMessage string
		token        string
		expectRes    string
	}{
		{
			name:        "Success with valid kubeconfig and token",
			kubeConfig:  generateValidKubeConfig(),
			token:       "test-token",
			expectError: false,
			expectRes:   "kubeadm join test-server:6443 --token <value withheld> \\\n\t--discovery-token-ca-cert-hash",
		},
		{

			name:         "Error to load kubeconfig",
			kubeConfig:   nil,
			token:        "test-token",
			expectError:  true,
			errorMessage: "failed to load kubeconfig",
		},

		{
			name:         "Error to get default cluster config",
			kubeConfig:   &clientcmdapi.Config{},
			token:        "test-token",
			expectError:  true,
			errorMessage: "failed to get default cluster config",
		},
		{
			name:         "Error when CA certificate is invalid",
			kubeConfig:   generateInvalidCAKubeConfig(),
			expectError:  true,
			errorMessage: "failed to parse CA certificate from kubeconfig",
		},
		{
			name:         "Error when CA certificate file path is invalid",
			kubeConfig:   generateInvalidFilePathKubeConfig(),
			token:        "test-token",
			expectError:  true,
			errorMessage: "failed to load CA certificate referenced by kubeconfig",
		},
		{
			name:         "Error when CA certificate is missing",
			kubeConfig:   generateMissingCAKubeConfig(),
			token:        "test-token",
			expectError:  true,
			errorMessage: "no CA certificates found in kubeconfig",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tmpDir, err := os.MkdirTemp("", "kubeadm-join-test")
			if err != nil {
				t.Fatalf("Unable to create temporary directory: %v", err)
			}
			defer func() {
				err := os.RemoveAll(tmpDir)
				if err != nil {
					t.Fatalf("Unable to remove temporary directory: %v", err)
				}
			}()

			configFilePath := filepath.Join(tmpDir, "test-config-file")
			if tt.kubeConfig != nil {
				err = writeKubeConfigToFile(tt.kubeConfig, configFilePath)
				require.NoError(t, err)
			}

			res, err := getJoinCommand(configFilePath, tt.token, "", true, true, true)

			if tt.expectError {
				require.Error(t, err)
				require.Contains(t, err.Error(), tt.errorMessage)
			} else {
				require.NoError(t, err)
			}
			require.Contains(t, res, tt.expectRes)
		})
	}
}
