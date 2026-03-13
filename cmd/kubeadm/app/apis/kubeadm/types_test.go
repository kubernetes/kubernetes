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

package kubeadm

import (
	"testing"

	"k8s.io/kubernetes/cmd/kubeadm/app/features"
)

func TestClusterConfigurationEncryptionAlgorithmType(t *testing.T) {
	tests := []struct {
		name           string
		cfg            *ClusterConfiguration
		expectedResult EncryptionAlgorithmType
	}{
		{
			name: "feature gate is set to true, return ECDSA-P256",
			cfg: &ClusterConfiguration{
				FeatureGates: map[string]bool{
					features.PublicKeysECDSA: true,
				},
				EncryptionAlgorithm: EncryptionAlgorithmRSA4096,
			},
			expectedResult: EncryptionAlgorithmECDSAP256,
		},
		{
			name: "feature gate is set to false, return the default RSA-2048",
			cfg: &ClusterConfiguration{
				FeatureGates: map[string]bool{
					features.PublicKeysECDSA: false,
				},
			},
			expectedResult: EncryptionAlgorithmRSA2048,
		},
		{
			name: "feature gate is not set, return the field value",
			cfg: &ClusterConfiguration{
				EncryptionAlgorithm: EncryptionAlgorithmRSA4096,
			},
			expectedResult: EncryptionAlgorithmRSA4096,
		},
		{
			name:           "feature gate and field are not set, return empty string",
			cfg:            &ClusterConfiguration{},
			expectedResult: "",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if result := tc.cfg.EncryptionAlgorithmType(); result != tc.expectedResult {
				t.Errorf("expected result: %s, got: %s", tc.expectedResult, result)
			}
		})
	}
}
