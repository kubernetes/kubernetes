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

package phases

import (
	"testing"

	"github.com/spf13/cobra"

	cmdtestutil "k8s.io/kubernetes/cmd/kubeadm/test/cmd"
)

func TestKubeletSubCommandsHasFlags(t *testing.T) {
	subCmds := []*cobra.Command{
		NewCmdKubeletWriteEnvFile(),
		NewCmdKubeletConfigUpload(),
		NewCmdKubeletConfigDownload(),
		NewCmdKubeletConfigWriteToDisk(),
		NewCmdKubeletConfigEnableDynamic(),
	}

	commonFlags := []string{}

	var tests = []struct {
		command         string
		additionalFlags []string
	}{
		{
			command: "write-env-file",
			additionalFlags: []string{
				"config",
			},
		},
		{
			command: "upload",
			additionalFlags: []string{
				"kubeconfig",
				"config",
			},
		},
		{
			command: "download",
			additionalFlags: []string{
				"kubeconfig",
				"kubelet-version",
			},
		},
		{
			command: "write-to-disk",
			additionalFlags: []string{
				"config",
			},
		},
		{
			command: "enable-dynamic",
			additionalFlags: []string{
				"kubeconfig",
				"node-name",
				"kubelet-version",
			},
		},
	}

	for _, test := range tests {
		expectedFlags := append(commonFlags, test.additionalFlags...)
		cmdtestutil.AssertSubCommandHasFlags(t, subCmds, test.command, expectedFlags...)
	}
}
