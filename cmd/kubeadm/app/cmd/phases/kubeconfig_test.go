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

package phases

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	// required for triggering api machinery startup when running unit tests
	_ "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/install"

	"k8s.io/client-go/tools/clientcmd"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/certs/pkiutil"

	testutil "k8s.io/kubernetes/cmd/kubeadm/test"
	cmdtestutil "k8s.io/kubernetes/cmd/kubeadm/test/cmd"
	kubeconfigtestutil "k8s.io/kubernetes/cmd/kubeadm/test/kubeconfig"
)

func TestKubeConfigCSubCommandsHasFlags(t *testing.T) {

	subCmds := getKubeConfigSubCommands(nil, "", phaseTestK8sVersion)

	commonFlags := []string{
		"cert-dir",
		"apiserver-advertise-address",
		"apiserver-bind-port",
	}

	var tests = []struct {
		command         string
		additionalFlags []string
	}{
		{
			command: "all",
			additionalFlags: []string{
				"config",
				"node-name",
			},
		},
		{
			command: "admin",
			additionalFlags: []string{
				"config",
			},
		},
		{
			command: "kubelet",
			additionalFlags: []string{
				"config",
				"node-name",
			},
		},
		{
			command: "controller-manager",
			additionalFlags: []string{
				"config",
			},
		},
		{
			command: "scheduler",
			additionalFlags: []string{
				"config",
			},
		},
		{
			command: "user",
			additionalFlags: []string{
				"token",
				"client-name",
			},
		},
	}

	for _, test := range tests {
		expectedFlags := append(commonFlags, test.additionalFlags...)
		cmdtestutil.AssertSubCommandHasFlags(t, subCmds, test.command, expectedFlags...)
	}
}

func TestKubeConfigSubCommandsThatCreateFilesWithFlags(t *testing.T) {

	commonFlags := []string{
		"--apiserver-advertise-address=1.2.3.4",
		"--apiserver-bind-port=1234",
	}

	var tests = []struct {
		command         string
		additionalFlags []string
		expectedFiles   []string
	}{
		{
			command:         "all",
			additionalFlags: []string{"--node-name=valid-nome-name"},
			expectedFiles: []string{
				kubeadmconstants.AdminKubeConfigFileName,
				kubeadmconstants.KubeletKubeConfigFileName,
				kubeadmconstants.ControllerManagerKubeConfigFileName,
				kubeadmconstants.SchedulerKubeConfigFileName,
			},
		},
		{
			command:       "admin",
			expectedFiles: []string{kubeadmconstants.AdminKubeConfigFileName},
		},
		{
			command:         "kubelet",
			additionalFlags: []string{"--node-name=valid-nome-name"},
			expectedFiles:   []string{kubeadmconstants.KubeletKubeConfigFileName},
		},
		{
			command:       "controller-manager",
			expectedFiles: []string{kubeadmconstants.ControllerManagerKubeConfigFileName},
		},
		{
			command:       "scheduler",
			expectedFiles: []string{kubeadmconstants.SchedulerKubeConfigFileName},
		},
	}

	var kubeConfigAssertions = map[string]struct {
		clientName    string
		organizations []string
	}{
		kubeadmconstants.AdminKubeConfigFileName: {
			clientName:    "kubernetes-admin",
			organizations: []string{kubeadmconstants.MastersGroup},
		},
		kubeadmconstants.KubeletKubeConfigFileName: {
			clientName:    "system:node:valid-nome-name",
			organizations: []string{kubeadmconstants.NodesGroup},
		},
		kubeadmconstants.ControllerManagerKubeConfigFileName: {
			clientName: kubeadmconstants.ControllerManagerUser,
		},
		kubeadmconstants.SchedulerKubeConfigFileName: {
			clientName: kubeadmconstants.SchedulerUser,
		},
	}

	for _, test := range tests {

		// Create temp folder for the test case
		tmpdir := testutil.SetupTempDir(t)
		defer os.RemoveAll(tmpdir)

		// Adds a pki folder with a ca certs to the temp folder
		pkidir := testutil.SetupPkiDirWithCertificateAuthorithy(t, tmpdir)

		// Retrives ca cert for assertions
		caCert, _, err := pkiutil.TryLoadCertAndKeyFromDisk(pkidir, kubeadmconstants.CACertAndKeyBaseName)
		if err != nil {
			t.Fatalf("couldn't retrive ca cert: %v", err)
		}

		// Get subcommands working in the temporary directory
		subCmds := getKubeConfigSubCommands(nil, tmpdir, phaseTestK8sVersion)

		// Execute the subcommand
		certDirFlag := fmt.Sprintf("--cert-dir=%s", pkidir)
		allFlags := append(commonFlags, certDirFlag)
		allFlags = append(allFlags, test.additionalFlags...)
		cmdtestutil.RunSubCommand(t, subCmds, test.command, allFlags...)

		// Checks that requested files are there
		testutil.AssertFileExists(t, tmpdir, test.expectedFiles...)

		// Checks contents of generated files
		for _, file := range test.expectedFiles {

			// reads generated files
			config, err := clientcmd.LoadFromFile(filepath.Join(tmpdir, file))
			if err != nil {
				t.Errorf("Couldn't load generated kubeconfig file: %v", err)
			}

			// checks that CLI flags are properly propagated and kubeconfig properties are correct
			kubeconfigtestutil.AssertKubeConfigCurrentCluster(t, config, "https://1.2.3.4:1234", caCert)

			expectedClientName := kubeConfigAssertions[file].clientName
			expectedOrganizations := kubeConfigAssertions[file].organizations
			kubeconfigtestutil.AssertKubeConfigCurrentAuthInfoWithClientCert(t, config, caCert, expectedClientName, expectedOrganizations...)

		}
	}
}

func TestKubeConfigSubCommandsThatCreateFilesWithConfigFile(t *testing.T) {

	var tests = []struct {
		command       string
		expectedFiles []string
	}{
		{
			command: "all",
			expectedFiles: []string{
				kubeadmconstants.AdminKubeConfigFileName,
				kubeadmconstants.KubeletKubeConfigFileName,
				kubeadmconstants.ControllerManagerKubeConfigFileName,
				kubeadmconstants.SchedulerKubeConfigFileName,
			},
		},
		{
			command:       "admin",
			expectedFiles: []string{kubeadmconstants.AdminKubeConfigFileName},
		},
		{
			command:       "kubelet",
			expectedFiles: []string{kubeadmconstants.KubeletKubeConfigFileName},
		},
		{
			command:       "controller-manager",
			expectedFiles: []string{kubeadmconstants.ControllerManagerKubeConfigFileName},
		},
		{
			command:       "scheduler",
			expectedFiles: []string{kubeadmconstants.SchedulerKubeConfigFileName},
		},
	}

	var kubeConfigAssertions = map[string]struct {
		clientName    string
		organizations []string
	}{
		kubeadmconstants.AdminKubeConfigFileName: {
			clientName:    "kubernetes-admin",
			organizations: []string{kubeadmconstants.MastersGroup},
		},
		kubeadmconstants.KubeletKubeConfigFileName: {
			clientName:    "system:node:valid-node-name",
			organizations: []string{kubeadmconstants.NodesGroup},
		},
		kubeadmconstants.ControllerManagerKubeConfigFileName: {
			clientName: kubeadmconstants.ControllerManagerUser,
		},
		kubeadmconstants.SchedulerKubeConfigFileName: {
			clientName: kubeadmconstants.SchedulerUser,
		},
	}

	for _, test := range tests {

		// Create temp folder for the test case
		tmpdir := testutil.SetupTempDir(t)
		defer os.RemoveAll(tmpdir)

		// Adds a pki folder with a ca certs to the temp folder
		pkidir := testutil.SetupPkiDirWithCertificateAuthorithy(t, tmpdir)

		// Retrives ca cert for assertions
		caCert, _, err := pkiutil.TryLoadCertAndKeyFromDisk(pkidir, kubeadmconstants.CACertAndKeyBaseName)
		if err != nil {
			t.Fatalf("couldn't retrive ca cert: %v", err)
		}

		// Adds a master configuration file
		cfg := &kubeadmapi.MasterConfiguration{
			API:             kubeadmapi.API{AdvertiseAddress: "1.2.3.4", BindPort: 1234},
			CertificatesDir: pkidir,
			NodeName:        "valid-node-name",
		}
		cfgPath := testutil.SetupMasterConfigurationFile(t, tmpdir, cfg)

		// Get subcommands working in the temporary directory
		subCmds := getKubeConfigSubCommands(nil, tmpdir, phaseTestK8sVersion)

		// Execute the subcommand
		configFlag := fmt.Sprintf("--config=%s", cfgPath)
		cmdtestutil.RunSubCommand(t, subCmds, test.command, configFlag)

		// Checks that requested files are there
		testutil.AssertFileExists(t, tmpdir, test.expectedFiles...)

		// Checks contents of generated files
		for _, file := range test.expectedFiles {

			// reads generated files
			config, err := clientcmd.LoadFromFile(filepath.Join(tmpdir, file))
			if err != nil {
				t.Errorf("Couldn't load generated kubeconfig file: %v", err)
			}

			// checks that config file properties are properly propagated and kubeconfig properties are correct
			kubeconfigtestutil.AssertKubeConfigCurrentCluster(t, config, "https://1.2.3.4:1234", caCert)

			expectedClientName := kubeConfigAssertions[file].clientName
			expectedOrganizations := kubeConfigAssertions[file].organizations
			kubeconfigtestutil.AssertKubeConfigCurrentAuthInfoWithClientCert(t, config, caCert, expectedClientName, expectedOrganizations...)

		}
	}
}

func TestKubeConfigSubCommandsThatWritesToOut(t *testing.T) {

	// Temporary folders for the test case
	tmpdir := testutil.SetupTempDir(t)
	defer os.RemoveAll(tmpdir)

	// Adds a pki folder with a ca cert to the temp folder
	pkidir := testutil.SetupPkiDirWithCertificateAuthorithy(t, tmpdir)

	// Retrives ca cert for assertions
	caCert, _, err := pkiutil.TryLoadCertAndKeyFromDisk(pkidir, kubeadmconstants.CACertAndKeyBaseName)
	if err != nil {
		t.Fatalf("couldn't retrive ca cert: %v", err)
	}

	commonFlags := []string{
		"--apiserver-advertise-address=1.2.3.4",
		"--apiserver-bind-port=1234",
		"--client-name=myUser",
		fmt.Sprintf("--cert-dir=%s", pkidir),
	}

	var tests = []struct {
		command         string
		withClientCert  bool
		withToken       bool
		additionalFlags []string
	}{
		{ // Test user subCommand withClientCert
			command:        "user",
			withClientCert: true,
		},
		{ // Test user subCommand withToken
			withToken:       true,
			command:         "user",
			additionalFlags: []string{"--token=123456"},
		},
	}

	for _, test := range tests {
		buf := new(bytes.Buffer)

		// Get subcommands working in the temporary directory
		subCmds := getKubeConfigSubCommands(buf, tmpdir, phaseTestK8sVersion)

		// Execute the subcommand
		allFlags := append(commonFlags, test.additionalFlags...)
		cmdtestutil.RunSubCommand(t, subCmds, test.command, allFlags...)

		// reads kubeconfig written to stdout
		config, err := clientcmd.Load(buf.Bytes())
		if err != nil {
			t.Errorf("Couldn't read kubeconfig file from buffer: %v", err)
			continue
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
	}
}
