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

package util

import (
	"io/ioutil"
	"testing"

	"github.com/stretchr/testify/assert"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
)

func TestPaths(t *testing.T) {
	certDir, err := ioutil.TempDir("", "fake-cert-dir")
	if err != nil {
		t.Fatalf("failed to create temp cert dir: %v", err)
	}
	var testCases = []struct {
		name          string
		dryRun        bool
		certDir       string
		expectedError bool
	}{
		{
			name:          "dry run paths",
			dryRun:        true,
			certDir:       certDir,
			expectedError: false,
		},
		{
			name:          "default paths",
			dryRun:        false,
			certDir:       certDir,
			expectedError: false,
		},
		{
			name:          "invalid cert dir paths",
			dryRun:        false,
			certDir:       "/invalid/cert/dir",
			expectedError: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			paths, actualError := InitPaths(tc.dryRun, tc.certDir)
			if (actualError != nil) && !tc.expectedError {
				t.Errorf("%s unexpected failure: %v", tc.name, actualError)
				return
			} else if (actualError == nil) && tc.expectedError {
				t.Errorf("%s passed when expected to fail", tc.name)
				return
			}
			if actualError != nil {
				return
			}

			if tc.dryRun {
				assert.NotEqual(t, certDir, paths.CertificateDir())
			} else {
				assert.Equal(t, certDir, paths.CertificateDir())
				assert.Equal(t, kubeadmconstants.KubernetesDir, paths.KubernetesDir())
				assert.Equal(t, kubeadmconstants.GetStaticPodDirectory(), paths.ManifestDir())
				assert.Equal(t, kubeadmconstants.KubeletRunDirectory, paths.KubeletDir())
			}
		})
	}
}
