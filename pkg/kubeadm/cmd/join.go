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
	"fmt"
	"io"

	"github.com/renstrom/dedent"
	"github.com/spf13/cobra"

	kubeadmapi "k8s.io/kubernetes/pkg/kubeadm/api"
	kubenode "k8s.io/kubernetes/pkg/kubeadm/node"
	kubeadmutil "k8s.io/kubernetes/pkg/kubeadm/util"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
)

var (
	join_done_msgf = dedent.Dedent(`
		Node join complete:
		* Certificate signing request sent to master and response
		  received.
		* Kubelet informed of new secure connection details.

		Run 'kubectl get nodes' on the master to see this node join.
		`)
)

func NewCmdJoin(out io.Writer, params *kubeadmapi.BootstrapParams) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "join",
		Short: "Run this on other servers to join an existing cluster.",
		Run: func(cmd *cobra.Command, args []string) {
			err := RunJoin(out, cmd, args, params)
			cmdutil.CheckErr(err)
		},
	}

	// TODO this should become `kubeadm join --token=<...> <master-ip-addr>`
	cmd.PersistentFlags().StringVarP(&params.Discovery.ApiServerURLs, "api-server-urls", "", "",
		`Comma separated list of API server URLs. Typically this might be just
		https://<address-of-master>:8080/`)
	cmd.PersistentFlags().StringVarP(&params.Discovery.GivenToken, "token", "", "",
		`Shared secret used to secure bootstrap. Must match output of 'init-master'.`)

	return cmd
}

func RunJoin(out io.Writer, cmd *cobra.Command, args []string, params *kubeadmapi.BootstrapParams) error {
	ok, err := kubeadmutil.UseGivenTokenIfValid(params)
	if !ok {
		if err != nil {
			return fmt.Errorf("%s (see --help)\n", err)
		}
		return fmt.Errorf("Must specify --token (see --help)\n")
	}
	if params.Discovery.ApiServerURLs == "" {
		return fmt.Errorf("Must specify --api-server-urls (see --help)\n")
	}

	kubeconfig, err := kubenode.RetrieveTrustedClusterInfo(params)
	if err != nil {
		return err
	}

	err = kubeadmutil.WriteKubeconfigIfNotExists(params, "kubelet", kubeconfig)
	if err != nil {
		return err
	}

	fmt.Fprintf(out, join_done_msgf)
	return nil
}
