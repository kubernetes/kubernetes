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
	kubeConfigFile := "foo"
	subCmds := []*cobra.Command{
		NewCmdKubeletUploadConfig(&kubeConfigFile),
		NewCmdKubeletWriteConfigToDisk(&kubeConfigFile),
		NewCmdKubeletEnableDynamicConfig(&kubeConfigFile),
	}

	commonFlags := []string{}

	var tests = []struct {
		command         string
		additionalFlags []string
	}{
		{
			command: "upload-config",
			additionalFlags: []string{
				"config",
			},
		},
		{
			command: "write-config-to-disk",
			additionalFlags: []string{
				"kubelet-version",
				"config",
			},
		},
		{
			command: "enable-dynamic-config",
			additionalFlags: []string{
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
