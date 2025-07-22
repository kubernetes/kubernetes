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

package cmd

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/tools/clientcmd"
	"sigs.k8s.io/yaml"

	kubeadmapiv1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta4"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/pkiutil"
	testutil "k8s.io/kubernetes/cmd/kubeadm/test"
	kubeconfigtestutil "k8s.io/kubernetes/cmd/kubeadm/test/kubeconfig"
)

func generateTestKubeadmConfig(dir, id, certDir, clusterName string) (string, error) {
	cfgPath := filepath.Join(dir, id)
	initCfg := kubeadmapiv1.InitConfiguration{
		TypeMeta: metav1.TypeMeta{
			APIVersion: kubeadmapiv1.SchemeGroupVersion.String(),
			Kind:       "InitConfiguration",
		},
		LocalAPIEndpoint: kubeadmapiv1.APIEndpoint{
			AdvertiseAddress: "1.2.3.4",
			BindPort:         1234,
		},
	}
	clusterCfg := kubeadmapiv1.ClusterConfiguration{
		TypeMeta: metav1.TypeMeta{
			APIVersion: kubeadmapiv1.SchemeGroupVersion.String(),
			Kind:       "ClusterConfiguration",
		},
		CertificatesDir:   certDir,
		ClusterName:       clusterName,
		KubernetesVersion: kubeadmconstants.MinimumControlPlaneVersion.String(),
	}

	var buf bytes.Buffer
	data, err := yaml.Marshal(&initCfg)
	if err != nil {
		return "", err
	}
	buf.Write(data)
	buf.WriteString("---\n")
	data, err = yaml.Marshal(&clusterCfg)
	if err != nil {
		return "", err
	}
	buf.Write(data)

	err = os.WriteFile(cfgPath, buf.Bytes(), 0644)
	return cfgPath, err
}

func TestKubeConfigSubCommandsThatWritesToOut(t *testing.T) {

	// Temporary folders for the test case
	tmpdir := t.TempDir()

	// Adds a pki folder with a ca cert to the temp folder
	pkidir := testutil.SetupPkiDirWithCertificateAuthority(t, tmpdir)

	// Retrieves ca cert for assertions
	caCert, _, err := pkiutil.TryLoadCertAndKeyFromDisk(pkidir, kubeadmconstants.CACertAndKeyBaseName)
	if err != nil {
		t.Fatalf("couldn't retrieve ca cert: %v", err)
	}

	var tests = []struct {
		name                   string
		command                string
		clusterName            string
		withClientCert         bool
		withToken              bool
		additionalFlags        []string
		expectedValidityPeriod time.Duration
	}{
		{
			name:           "user subCommand withClientCert",
			command:        "user",
			withClientCert: true,
		},
		{
			name:           "user subCommand withClientCert",
			command:        "user",
			withClientCert: true,
			clusterName:    "my-cluster",
		},
		{
			name:            "user subCommand withToken",
			withToken:       true,
			command:         "user",
			additionalFlags: []string{"--token=123456"},
		},
		{
			name:            "user subCommand withToken",
			withToken:       true,
			command:         "user",
			clusterName:     "my-cluster-with-token",
			additionalFlags: []string{"--token=123456"},
		},
		{
			name:                   "user subCommand with validity period",
			withClientCert:         true,
			command:                "user",
			additionalFlags:        []string{"--validity-period=12h"},
			expectedValidityPeriod: 12 * time.Hour,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			buf := new(bytes.Buffer)

			// Get subcommands working in the temporary directory
			cmd := newCmdUserKubeConfig(buf)

			cfgPath, err := generateTestKubeadmConfig(tmpdir, test.name, pkidir, test.clusterName)
			if err != nil {
				t.Fatalf("Failed to generate kubeadm config: %v", err)
			}

			commonFlags := []string{
				"--client-name=myUser",
				fmt.Sprintf("--config=%s", cfgPath),
			}

			// Execute the subcommand
			allFlags := append(commonFlags, test.additionalFlags...)
			cmd.SetArgs(allFlags)
			if err := cmd.Execute(); err != nil {
				t.Fatalf("Could not execute subcommand: %v", err)
			}

			// reads kubeconfig written to stdout
			config, err := clientcmd.Load(buf.Bytes())
			if err != nil {
				t.Fatalf("couldn't read kubeconfig file from buffer: %v", err)
			}

			// checks that CLI flags are properly propagated
			kubeconfigtestutil.AssertKubeConfigCurrentCluster(t, config, "https://1.2.3.4:1234", caCert)

			if test.withClientCert {
				if test.expectedValidityPeriod == 0 {
					test.expectedValidityPeriod = kubeadmconstants.CertificateValidityPeriod
				}
				// checks that kubeconfig files have expected client cert
				startTime := kubeadmutil.StartTimeUTC()
				notAfter := startTime.Add(test.expectedValidityPeriod)
				kubeconfigtestutil.AssertKubeConfigCurrentAuthInfoWithClientCert(t, config, caCert, notAfter, "myUser")
			}

			if test.withToken {
				// checks that kubeconfig files have expected token
				kubeconfigtestutil.AssertKubeConfigCurrentAuthInfoWithToken(t, config, "myUser", "123456")
			}

			if len(test.clusterName) > 0 {
				// checks that kubeconfig files have expected cluster name
				kubeconfigtestutil.AssertKubeConfigCurrentContextWithClusterName(t, config, test.clusterName)
			}
		})
	}
}
