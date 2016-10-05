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

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/api"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/util/flag"
)

func NewKubeadmCommand(f *cmdutil.Factory, in io.Reader, out, err io.Writer, envParams map[string]string) *cobra.Command {
	cmds := &cobra.Command{
		Use:   "kubeadm",
		Short: "kubeadm: easily bootstrap a secure Kubernetes cluster.",
		Long: dedent.Dedent(`
			kubeadm: easily bootstrap a secure Kubernetes cluster.

			    ┌──────────────────────────────────────────────────────────┐
			    │ KUBEADM IS ALPHA, DO NOT USE IT FOR PRODUCTION CLUSTERS! │
			    │                                                          │
			    │ But, please try it out! Give us feedback at:             │
			    │ https://github.com/kubernetes/kubernetes/issues          │
			    │ and at-mention @kubernetes/sig-cluster-lifecycle         │
			    └──────────────────────────────────────────────────────────┘

			Example usage:

			    Create a two-machine cluster with one master (which controls the cluster),
			    and one node (where workloads, like pods and replica sets run).

			    ┌──────────────────────────────────────────────────────────┐
			    │  On the first machine                                    │
			    ├──────────────────────────────────────────────────────────┤
			    │ master# kubeadm init                                     │
			    └──────────────────────────────────────────────────────────┘

			    ┌──────────────────────────────────────────────────────────┐
			    │ On the second machine                                    │
			    ├──────────────────────────────────────────────────────────┤
			    │ node# kubeadm join --token=<token> <ip-of-master>        │
			    └──────────────────────────────────────────────────────────┘

			    You can then repeat the second step on as many other machines as you like.

		`),
	}

	s := new(kubeadmapi.KubeadmConfig)
	s.EnvParams = envParams

	cmds.ResetFlags()
	cmds.SetGlobalNormalizationFunc(flag.WarnWordSepNormalizeFunc)

	cmds.AddCommand(NewCmdInit(out, s))
	cmds.AddCommand(NewCmdJoin(out, s))
	cmds.AddCommand(NewCmdVersion(out))

	return cmds
}
