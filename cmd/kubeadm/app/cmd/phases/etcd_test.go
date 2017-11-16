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
	"fmt"
	"os"
	"testing"

	// required for triggering api machinery startup when running unit tests
	_ "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/install"

	testutil "k8s.io/kubernetes/cmd/kubeadm/test"
	cmdtestutil "k8s.io/kubernetes/cmd/kubeadm/test/cmd"
)

func TestEtcdSubCommandsHasFlags(t *testing.T) {

	subCmds := getEtcdSubCommands("", phaseTestK8sVersion)

	commonFlags := []string{
		"cert-dir",
		"config",
	}

	var tests = []struct {
		command         string
		additionalFlags []string
	}{
		{
			command: "local",
		},
	}

	for _, test := range tests {
		expectedFlags := append(commonFlags, test.additionalFlags...)
		cmdtestutil.AssertSubCommandHasFlags(t, subCmds, test.command, expectedFlags...)
	}
}

func TestEtcdCreateFilesWithFlags(t *testing.T) {

	var tests = []struct {
		command         string
		additionalFlags []string
		expectedFiles   []string
	}{
		{
			command:         "local",
			expectedFiles:   []string{"etcd.yaml"},
			additionalFlags: []string{},
		},
	}

	for _, test := range tests {

		// Create temp folder for the test case
		tmpdir := testutil.SetupTempDir(t)
		defer os.RemoveAll(tmpdir)

		// Get subcommands working in the temporary directory
		subCmds := getEtcdSubCommands(tmpdir, phaseTestK8sVersion)

		// Execute the subcommand
		certDirFlag := fmt.Sprintf("--cert-dir=%s", tmpdir)
		allFlags := append(test.additionalFlags, certDirFlag)
		cmdtestutil.RunSubCommand(t, subCmds, test.command, allFlags...)

		// Checks that requested files are there
		testutil.AssertFileExists(t, tmpdir, test.expectedFiles...)
	}
}
