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
	"testing"

	// required for triggering api machinery startup when running unit tests
	_ "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/install"

	cmdtestutil "k8s.io/kubernetes/cmd/kubeadm/test/cmd"
)

func TestAddonsSubCommandsHasFlags(t *testing.T) {

	subCmds := getAddonsSubCommands()

	commonFlags := []string{
		"kubeconfig",
		"config",
		"kubernetes-version",
		"image-repository",
	}

	var tests = []struct {
		command         string
		additionalFlags []string
	}{
		{
			command: "all",
			additionalFlags: []string{
				"apiserver-advertise-address",
				"apiserver-bind-port",
				"pod-network-cidr",
				"service-dns-domain",
			},
		},
		{
			command: "kube-proxy",
			additionalFlags: []string{
				"apiserver-advertise-address",
				"apiserver-bind-port",
				"pod-network-cidr",
			},
		},
		{
			command: "kube-dns",
			additionalFlags: []string{
				"service-dns-domain",
			},
		},
	}

	for _, test := range tests {
		expectedFlags := append(commonFlags, test.additionalFlags...)
		cmdtestutil.AssertSubCommandHasFlags(t, subCmds, test.command, expectedFlags...)
	}
}
