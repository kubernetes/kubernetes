/*
Copyright 2016 The Kubernetes Authors.

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
	"io"

	"github.com/renstrom/dedent"
	"github.com/spf13/cobra"

	"k8s.io/apiserver/pkg/util/flag"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/phases"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/upgrade"
)

// NewKubeadmCommand return cobra.Command to run kubeadm command
func NewKubeadmCommand(_ io.Reader, out, err io.Writer) *cobra.Command {
	cmds := &cobra.Command{
		Use:   "kubeadm",
		Short: "kubeadm: easily bootstrap a secure Kubernetes cluster",
		Long: dedent.Dedent(`
			kubeadm: easily bootstrap a secure Kubernetes cluster.

			    ┌──────────────────────────────────────────────────────────┐
			    │ KUBEADM IS BETA, DO NOT USE IT FOR PRODUCTION CLUSTERS!  │
			    │                                                          │
			    │ But, please try it out! Give us feedback at:             │
			    │ https://github.com/kubernetes/kubeadm/issues             │
			    │ and at-mention @kubernetes/sig-cluster-lifecycle-bugs    │
			    │ or @kubernetes/sig-cluster-lifecycle-feature-requests    │
			    └──────────────────────────────────────────────────────────┘

			Example usage:

			    Create a two-machine cluster with one master (which controls the cluster),
			    and one node (where your workloads, like Pods and Deployments run).

			    ┌──────────────────────────────────────────────────────────┐
			    │ On the first machine                                     │
			    ├──────────────────────────────────────────────────────────┤
			    │ master# kubeadm init                                     │
			    └──────────────────────────────────────────────────────────┘

			    ┌──────────────────────────────────────────────────────────┐
			    │ On the second machine                                    │
			    ├──────────────────────────────────────────────────────────┤
			    │ node# kubeadm join <arguments-returned-from-init>        │
			    └──────────────────────────────────────────────────────────┘

			    You can then repeat the second step on as many other machines as you like.

		`),
	}

	cmds.ResetFlags()
	cmds.SetGlobalNormalizationFunc(flag.WarnWordSepNormalizeFunc)

	cmds.AddCommand(NewCmdCompletion(out, ""))
	cmds.AddCommand(NewCmdConfig(out))
	cmds.AddCommand(NewCmdInit(out))
	cmds.AddCommand(NewCmdJoin(out))
	cmds.AddCommand(NewCmdReset(out))
	cmds.AddCommand(NewCmdVersion(out))
	cmds.AddCommand(NewCmdToken(out, err))
	cmds.AddCommand(upgrade.NewCmdUpgrade(out))

	// Wrap not yet fully supported commands in an alpha subcommand
	experimentalCmd := &cobra.Command{
		Use:   "alpha",
		Short: "Experimental sub-commands not yet fully functional.",
	}
	experimentalCmd.AddCommand(phases.NewCmdPhase(out))
	cmds.AddCommand(experimentalCmd)

	return cmds
}
