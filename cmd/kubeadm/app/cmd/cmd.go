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

	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/util/flag"
)

func NewKubeadmCommand(f cmdutil.Factory, in io.Reader, out, err io.Writer) *cobra.Command {
	cmds := &cobra.Command{
		Use:   "kubeadm",
		Short: "kubeadm: easily bootstrap a secure Kubernetes cluster",
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
	// TODO(phase2+) figure out how to avoid running as root
	//
	// TODO(phase2) detect interactive vs non-interactive use and adjust output accordingly
	// i.e. make it automation friendly
	//
	// TODO(phase2) create an abstraction that defines files and the content that needs to
	// be written to disc and write it all in one go at the end as we have a lot of
	// crapy little files written from different parts of this code; this could also
	// be useful for testing
	// by having this model we can allow users to create some files before `kubeadm init` runs, e.g. PKI assets, we
	// would then be able to look at files users has given an diff or validate if those are sane, we could also warn
	// if any of the files had been deprecated

	cmds.ResetFlags()
	cmds.SetGlobalNormalizationFunc(flag.WarnWordSepNormalizeFunc)

	cmds.AddCommand(NewCmdInit(out))
	cmds.AddCommand(NewCmdJoin(out))
	cmds.AddCommand(NewCmdReset(out))
	cmds.AddCommand(NewCmdToken(out))
	cmds.AddCommand(NewCmdVersion(out))

	return cmds
}
