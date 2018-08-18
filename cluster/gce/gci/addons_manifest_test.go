/*
Copyright 2017 The Kubernetes Authors.

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

package gci

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/api/core/v1"
)

const (
	/*
	   Template for defining the environment state of configure-helper.sh
	   The environment of configure-helper.sh is initially configured via kube-env file. However, as deploy-helper
	   executes new variables are created. ManifestTestCase does not care where a variable came from. However, future
	   test scenarios, may require such a distinction.
	   The list of variables is, by no means, complete - this is what is required to run currently defined tests.
	*/
	kubeAddonsEnvTmpl = `
readonly KUBE_HOME={{.KubeHome}}
readonly SSL_CERT_FILE={{.SSLCertFile}}
`
)

type kubeAddonsEnv struct {
	KubeHome    string
	SSLCertFile string
}

func TestSetupSystemCACertsManifest(t *testing.T) {
	const fakeCertData = "-----FAKE CERT DATA-----"
	fakeCertFile, err := ioutil.TempFile("", "fake-cert")
	require.NoError(t, err)
	defer os.Remove(fakeCertFile.Name())
	_, err = fakeCertFile.WriteString(fakeCertData)
	require.NoError(t, err)

	tests := map[string]struct {
		sslCertFileEnv string
		expectations   func(t *testing.T, certsData string)
	}{
		"use host certs": {"", func(t *testing.T, certsData string) {
			assert.Contains(t, certsData, "-----BEGIN CERTIFICATE-----", "Should container certificate data")
			assert.Contains(t, certsData, "-----END CERTIFICATE-----", "Should container certificate data")
		}},
		"use fake cert": {fakeCertFile.Name(), func(t *testing.T, certsData string) {
			assert.Equal(t, fakeCertData, certsData, "Should have fake cert data")
		}},
		"use non-existent cert file": {"/bogus-certs-file.crt", func(t *testing.T, certsData string) {
			assert.Empty(t, certsData, "Should not have cert data")
		}},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			baseDir, err := ioutil.TempDir("", "configure-helper-test") // cleaned up by c.tearDown()
			require.NoError(t, err, "Failed to create temp directory")

			c := ManifestTestCase{
				t:                t,
				kubeHome:         baseDir,
				manifestFuncName: fmt.Sprintf("setup-system-cacerts-manifest %s", baseDir),
			}
			defer c.tearDown()

			c.mustInvokeFunc(kubeAddonsEnvTmpl, kubeAddonsEnv{
				KubeHome:    c.kubeHome,
				SSLCertFile: test.sslCertFileEnv,
			})

			manifestPath := filepath.Join(baseDir, "system-cacerts-configmap.yaml")
			cm := &v1.ConfigMap{}
			c.mustLoadManifest(manifestPath, cm)

			certsData := cm.Data["ca-certificates.crt"]
			certsData = strings.SplitAfter(certsData, "-----END CERTIFICATE-----")[0] // Limit size of error message
			test.expectations(t, certsData)
		})
	}
}
