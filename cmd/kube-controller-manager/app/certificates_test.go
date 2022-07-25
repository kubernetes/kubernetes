/*
Copyright 2020 The Kubernetes Authors.

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

package app

import (
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	csrsigningconfig "k8s.io/kubernetes/pkg/controller/certificates/signer/config"
)

func TestCertSpecified(t *testing.T) {
	allConfig := csrsigningconfig.CSRSigningControllerConfiguration{
		ClusterSigningCertFile: "/cluster-signing-cert",
		ClusterSigningKeyFile:  "/cluster-signing-key",
		ClusterSigningDuration: metav1.Duration{Duration: 10 * time.Hour},
		KubeletServingSignerConfiguration: csrsigningconfig.CSRSigningConfiguration{
			CertFile: "/cluster-signing-kubelet-serving/cert-file",
			KeyFile:  "/cluster-signing-kubelet-serving/key-file",
		},
		KubeletClientSignerConfiguration: csrsigningconfig.CSRSigningConfiguration{
			CertFile: "/cluster-signing-kubelet-client/cert-file",
			KeyFile:  "/cluster-signing-kubelet-client/key-file",
		},
		KubeAPIServerClientSignerConfiguration: csrsigningconfig.CSRSigningConfiguration{
			CertFile: "/cluster-signing-kube-apiserver-client/cert-file",
			KeyFile:  "/cluster-signing-kube-apiserver-client/key-file",
		},
		LegacyUnknownSignerConfiguration: csrsigningconfig.CSRSigningConfiguration{
			CertFile: "/cluster-signing-legacy-unknown/cert-file",
			KeyFile:  "/cluster-signing-legacy-unknown/key-file",
		},
	}
	defaultOnly := csrsigningconfig.CSRSigningControllerConfiguration{
		ClusterSigningCertFile: "/cluster-signing-cert",
		ClusterSigningKeyFile:  "/cluster-signing-key",
		ClusterSigningDuration: metav1.Duration{Duration: 10 * time.Hour},
	}
	specifiedOnly := csrsigningconfig.CSRSigningControllerConfiguration{
		KubeletServingSignerConfiguration: csrsigningconfig.CSRSigningConfiguration{
			CertFile: "/cluster-signing-kubelet-serving/cert-file",
			KeyFile:  "/cluster-signing-kubelet-serving/key-file",
		},
		KubeletClientSignerConfiguration: csrsigningconfig.CSRSigningConfiguration{
			CertFile: "/cluster-signing-kubelet-client/cert-file",
			KeyFile:  "/cluster-signing-kubelet-client/key-file",
		},
		KubeAPIServerClientSignerConfiguration: csrsigningconfig.CSRSigningConfiguration{
			CertFile: "/cluster-signing-kube-apiserver-client/cert-file",
			KeyFile:  "/cluster-signing-kube-apiserver-client/key-file",
		},
		LegacyUnknownSignerConfiguration: csrsigningconfig.CSRSigningConfiguration{
			CertFile: "/cluster-signing-legacy-unknown/cert-file",
			KeyFile:  "/cluster-signing-legacy-unknown/key-file",
		},
	}
	halfASpecified := csrsigningconfig.CSRSigningControllerConfiguration{
		ClusterSigningCertFile: "/cluster-signing-cert",
		ClusterSigningKeyFile:  "/cluster-signing-key",
		ClusterSigningDuration: metav1.Duration{Duration: 10 * time.Hour},
		KubeletServingSignerConfiguration: csrsigningconfig.CSRSigningConfiguration{
			CertFile: "/cluster-signing-kubelet-serving/cert-file",
			KeyFile:  "/cluster-signing-kubelet-serving/key-file",
		},
		KubeletClientSignerConfiguration: csrsigningconfig.CSRSigningConfiguration{
			CertFile: "/cluster-signing-kubelet-client/cert-file",
			KeyFile:  "/cluster-signing-kubelet-client/key-file",
		},
	}
	halfBSpecified := csrsigningconfig.CSRSigningControllerConfiguration{
		ClusterSigningCertFile: "/cluster-signing-cert",
		ClusterSigningKeyFile:  "/cluster-signing-key",
		ClusterSigningDuration: metav1.Duration{Duration: 10 * time.Hour},
		KubeAPIServerClientSignerConfiguration: csrsigningconfig.CSRSigningConfiguration{
			CertFile: "/cluster-signing-kube-apiserver-client/cert-file",
			KeyFile:  "/cluster-signing-kube-apiserver-client/key-file",
		},
		LegacyUnknownSignerConfiguration: csrsigningconfig.CSRSigningConfiguration{
			CertFile: "/cluster-signing-legacy-unknown/cert-file",
			KeyFile:  "/cluster-signing-legacy-unknown/key-file",
		},
	}

	tests := []struct {
		name              string
		config            csrsigningconfig.CSRSigningControllerConfiguration
		specifiedFn       func(config csrsigningconfig.CSRSigningControllerConfiguration) bool
		expectedSpecified bool
		filesFn           func(config csrsigningconfig.CSRSigningControllerConfiguration) (string, string)
		expectedCert      string
		expectedKey       string
	}{
		{
			name:              "allConfig-KubeletServingSignerFilesSpecified",
			config:            allConfig,
			specifiedFn:       areKubeletServingSignerFilesSpecified,
			expectedSpecified: true,
			filesFn:           getKubeletServingSignerFiles,
			expectedCert:      "/cluster-signing-kubelet-serving/cert-file",
			expectedKey:       "/cluster-signing-kubelet-serving/key-file",
		},
		{
			name:              "defaultOnly-KubeletServingSignerFilesSpecified",
			config:            defaultOnly,
			specifiedFn:       areKubeletServingSignerFilesSpecified,
			expectedSpecified: false,
			filesFn:           getKubeletServingSignerFiles,
			expectedCert:      "/cluster-signing-cert",
			expectedKey:       "/cluster-signing-key",
		},
		{
			name:              "specifiedOnly-KubeletServingSignerFilesSpecified",
			config:            specifiedOnly,
			specifiedFn:       areKubeletServingSignerFilesSpecified,
			expectedSpecified: true,
			filesFn:           getKubeletServingSignerFiles,
			expectedCert:      "/cluster-signing-kubelet-serving/cert-file",
			expectedKey:       "/cluster-signing-kubelet-serving/key-file",
		},
		{
			name:              "halfASpecified-KubeletServingSignerFilesSpecified",
			config:            halfASpecified,
			specifiedFn:       areKubeletServingSignerFilesSpecified,
			expectedSpecified: true,
			filesFn:           getKubeletServingSignerFiles,
			expectedCert:      "/cluster-signing-kubelet-serving/cert-file",
			expectedKey:       "/cluster-signing-kubelet-serving/key-file",
		},
		{
			name:              "halfBSpecified-KubeletServingSignerFilesSpecified",
			config:            halfBSpecified,
			specifiedFn:       areKubeletServingSignerFilesSpecified,
			expectedSpecified: false,
			filesFn:           getKubeletServingSignerFiles,
			expectedCert:      "",
			expectedKey:       "",
		},

		{
			name:              "allConfig-KubeletClientSignerFiles",
			config:            allConfig,
			specifiedFn:       areKubeletClientSignerFilesSpecified,
			expectedSpecified: true,
			filesFn:           getKubeletClientSignerFiles,
			expectedCert:      "/cluster-signing-kubelet-client/cert-file",
			expectedKey:       "/cluster-signing-kubelet-client/key-file",
		},
		{
			name:              "defaultOnly-KubeletClientSignerFiles",
			config:            defaultOnly,
			specifiedFn:       areKubeletClientSignerFilesSpecified,
			expectedSpecified: false,
			filesFn:           getKubeletClientSignerFiles,
			expectedCert:      "/cluster-signing-cert",
			expectedKey:       "/cluster-signing-key",
		},
		{
			name:              "specifiedOnly-KubeletClientSignerFiles",
			config:            specifiedOnly,
			specifiedFn:       areKubeletClientSignerFilesSpecified,
			expectedSpecified: true,
			filesFn:           getKubeletClientSignerFiles,
			expectedCert:      "/cluster-signing-kubelet-client/cert-file",
			expectedKey:       "/cluster-signing-kubelet-client/key-file",
		},
		{
			name:              "halfASpecified-KubeletClientSignerFiles",
			config:            halfASpecified,
			specifiedFn:       areKubeletClientSignerFilesSpecified,
			expectedSpecified: true,
			filesFn:           getKubeletClientSignerFiles,
			expectedCert:      "/cluster-signing-kubelet-client/cert-file",
			expectedKey:       "/cluster-signing-kubelet-client/key-file",
		},
		{
			name:              "halfBSpecified-KubeletClientSignerFiles",
			config:            halfBSpecified,
			specifiedFn:       areKubeletClientSignerFilesSpecified,
			expectedSpecified: false,
			filesFn:           getKubeletClientSignerFiles,
			expectedCert:      "",
			expectedKey:       "",
		},

		{
			name:              "allConfig-KubeletClientSignerFiles",
			config:            allConfig,
			specifiedFn:       areKubeAPIServerClientSignerFilesSpecified,
			expectedSpecified: true,
			filesFn:           getKubeAPIServerClientSignerFiles,
			expectedCert:      "/cluster-signing-kube-apiserver-client/cert-file",
			expectedKey:       "/cluster-signing-kube-apiserver-client/key-file",
		},
		{
			name:              "defaultOnly-KubeletClientSignerFiles",
			config:            defaultOnly,
			specifiedFn:       areKubeAPIServerClientSignerFilesSpecified,
			expectedSpecified: false,
			filesFn:           getKubeAPIServerClientSignerFiles,
			expectedCert:      "/cluster-signing-cert",
			expectedKey:       "/cluster-signing-key",
		},
		{
			name:              "specifiedOnly-KubeletClientSignerFiles",
			config:            specifiedOnly,
			specifiedFn:       areKubeAPIServerClientSignerFilesSpecified,
			expectedSpecified: true,
			filesFn:           getKubeAPIServerClientSignerFiles,
			expectedCert:      "/cluster-signing-kube-apiserver-client/cert-file",
			expectedKey:       "/cluster-signing-kube-apiserver-client/key-file",
		},
		{
			name:              "halfASpecified-KubeletClientSignerFiles",
			config:            halfASpecified,
			specifiedFn:       areKubeAPIServerClientSignerFilesSpecified,
			expectedSpecified: false,
			filesFn:           getKubeAPIServerClientSignerFiles,
			expectedCert:      "",
			expectedKey:       "",
		},
		{
			name:              "halfBSpecified-KubeletClientSignerFiles",
			config:            halfBSpecified,
			specifiedFn:       areKubeAPIServerClientSignerFilesSpecified,
			expectedSpecified: true,
			filesFn:           getKubeAPIServerClientSignerFiles,
			expectedCert:      "/cluster-signing-kube-apiserver-client/cert-file",
			expectedKey:       "/cluster-signing-kube-apiserver-client/key-file",
		},

		{
			name:              "allConfig-LegacyUnknownSignerFiles",
			config:            allConfig,
			specifiedFn:       areLegacyUnknownSignerFilesSpecified,
			expectedSpecified: true,
			filesFn:           getLegacyUnknownSignerFiles,
			expectedCert:      "/cluster-signing-legacy-unknown/cert-file",
			expectedKey:       "/cluster-signing-legacy-unknown/key-file",
		},
		{
			name:              "defaultOnly-LegacyUnknownSignerFiles",
			config:            defaultOnly,
			specifiedFn:       areLegacyUnknownSignerFilesSpecified,
			expectedSpecified: false,
			filesFn:           getLegacyUnknownSignerFiles,
			expectedCert:      "/cluster-signing-cert",
			expectedKey:       "/cluster-signing-key",
		},
		{
			name:              "specifiedOnly-LegacyUnknownSignerFiles",
			config:            specifiedOnly,
			specifiedFn:       areLegacyUnknownSignerFilesSpecified,
			expectedSpecified: true,
			filesFn:           getLegacyUnknownSignerFiles,
			expectedCert:      "/cluster-signing-legacy-unknown/cert-file",
			expectedKey:       "/cluster-signing-legacy-unknown/key-file",
		},
		{
			name:              "halfASpecified-LegacyUnknownSignerFiles",
			config:            halfASpecified,
			specifiedFn:       areLegacyUnknownSignerFilesSpecified,
			expectedSpecified: false,
			filesFn:           getLegacyUnknownSignerFiles,
			expectedCert:      "",
			expectedKey:       "",
		},
		{
			name:              "halfBSpecified-LegacyUnknownSignerFiles",
			config:            halfBSpecified,
			specifiedFn:       areLegacyUnknownSignerFilesSpecified,
			expectedSpecified: true,
			filesFn:           getLegacyUnknownSignerFiles,
			expectedCert:      "/cluster-signing-legacy-unknown/cert-file",
			expectedKey:       "/cluster-signing-legacy-unknown/key-file",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			actualSpecified := test.specifiedFn(test.config)
			if actualSpecified != test.expectedSpecified {
				t.Error(actualSpecified)
			}

			actualCert, actualKey := test.filesFn(test.config)
			if actualCert != test.expectedCert {
				t.Error(actualCert)
			}
			if actualKey != test.expectedKey {
				t.Error(actualKey)
			}
		})
	}
}
