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

package renew

import (
	"strings"
	"testing"

	"github.com/spf13/cobra"
)

func TestCommandsGenerated(t *testing.T) {
	expectedFlags := []string{
		"cert-dir",
		"config",
		"use-api",
	}

	expectedCommands := []string{
		"renew",

		"renew apiserver",
		"renew apiserver-kubelet-client",
		"renew apiserver-etcd-client",

		"renew front-proxy-client",

		"renew etcd-server",
		"renew etcd-peer",
		"renew etcd-healthcheck-client",
	}

	renewCmd := NewCmdCertsRenewal()

	fakeRoot := &cobra.Command{}
	fakeRoot.AddCommand(renewCmd)

	for _, cmdPath := range expectedCommands {
		t.Run(cmdPath, func(t *testing.T) {
			cmd, rem, _ := fakeRoot.Find(strings.Split(cmdPath, " "))
			if cmd == nil || len(rem) != 0 {
				t.Fatalf("couldn't locate command %q (%v)", cmdPath, rem)
			}

			for _, flag := range expectedFlags {
				if cmd.Flags().Lookup(flag) == nil {
					t.Errorf("couldn't find expected flag --%s", flag)
				}
			}
		})
	}
}
