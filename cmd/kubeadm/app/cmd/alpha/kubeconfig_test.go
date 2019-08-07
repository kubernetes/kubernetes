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

package alpha

import (
	"bytes"
	"fmt"
	"os"
	"testing"

	"k8s.io/client-go/tools/clientcmd"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/pkiutil"
	testutil "k8s.io/kubernetes/cmd/kubeadm/test"
	kubeconfigtestutil "k8s.io/kubernetes/cmd/kubeadm/test/kubeconfig"
)

func TestKubeConfigSubCommandsThatWritesToOut(t *testing.T) {

	// Temporary folders for the test case
	tmpdir := testutil.SetupTempDir(t)
	defer os.RemoveAll(tmpdir)

	// Adds a pki folder with a ca cert to the temp folder
	pkidir := testutil.SetupPkiDirWithCertificateAuthority(t, tmpdir)

	// Retrieves ca cert for assertions
	caCert, _, err := pkiutil.TryLoadCertAndKeyFromDisk(pkidir, kubeadmconstants.CACertAndKeyBaseName)
	if err != nil {
		t.Fatalf("couldn't retrieve ca cert: %v", err)
	}

	commonFlags := []string{
		"--apiserver-advertise-address=1.2.3.4",
		"--apiserver-bind-port=1234",
		"--client-name=myUser",
		fmt.Sprintf("--cert-dir=%s", pkidir),
	}

	var tests = []struct {
		name            string
		command         string
		withClientCert  bool
		withToken       bool
		additionalFlags []string
	}{
		{
			name:           "user subCommand withClientCert",
			command:        "user",
			withClientCert: true,
		},
		{
			name:            "user subCommand withToken",
			withToken:       true,
			command:         "user",
			additionalFlags: []string{"--token=123456"},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			buf := new(bytes.Buffer)

			// Get subcommands working in the temporary directory
			cmd := newCmdUserKubeConfig(buf)

			// Execute the subcommand
			allFlags := append(commonFlags, test.additionalFlags...)
			cmd.SetArgs(allFlags)
			if err := cmd.Execute(); err != nil {
				t.Fatal("Could not execute subcommand")
			}

			// reads kubeconfig written to stdout
			config, err := clientcmd.Load(buf.Bytes())
			if err != nil {
				t.Errorf("couldn't read kubeconfig file from buffer: %v", err)
				return
			}

			// checks that CLI flags are properly propagated
			kubeconfigtestutil.AssertKubeConfigCurrentCluster(t, config, "https://1.2.3.4:1234", caCert)

			if test.withClientCert {
				// checks that kubeconfig files have expected client cert
				kubeconfigtestutil.AssertKubeConfigCurrentAuthInfoWithClientCert(t, config, caCert, "myUser")
			}

			if test.withToken {
				// checks that kubeconfig files have expected token
				kubeconfigtestutil.AssertKubeConfigCurrentAuthInfoWithToken(t, config, "myUser", "123456")
			}
		})
	}
}
