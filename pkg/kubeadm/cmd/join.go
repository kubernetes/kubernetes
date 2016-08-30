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

package kubecmd

import (
	"fmt"
	"io"

	"github.com/renstrom/dedent"
	"github.com/spf13/cobra"

	kubeadmapi "k8s.io/kubernetes/pkg/kubeadm/api"
	kubenode "k8s.io/kubernetes/pkg/kubeadm/node"
	kubeadmutil "k8s.io/kubernetes/pkg/kubeadm/util"
)

func NewCmdJoin(out io.Writer, params *kubeadmapi.BootstrapParams) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "join",
		Short: "Run this on other servers to join an existing cluster.",
		Run: func(cmd *cobra.Command, args []string) {
			ok, err := kubeadmutil.UseGivenTokenIfValid(params)
			if !ok {
				if err != nil {
					out.Write([]byte(fmt.Sprintf("%s (see --help)\n", err)))
					return
				}
				out.Write([]byte(fmt.Sprintf("Must specify --token (see --help)\n")))
				return
			}
			if params.Discovery.ApiServerURLs == "" {
				out.Write([]byte(fmt.Sprintf("Must specify --api-server-urls (see --help)\n")))
				return
			}
			kubeconfig, err := kubenode.RetrieveTrustedClusterInfo(params)
			if err != nil {
				out.Write([]byte(fmt.Sprintf("Failed to bootstrap: %s\n", err)))
				return
			}
			//fmt.Println("recieved signed certificate from the API server, will write `/etc/kubernetes/kubelet.conf`...")

			err = kubeadmutil.WriteKubeconfigIfNotExists(params, "kubelet", kubeconfig)
			if err != nil {
				out.Write([]byte(fmt.Sprintf("Unable to write config for node:\n%s\n", err)))
				return
			}
			out.Write([]byte(dedent.Dedent(`
				Node join complete:
				* Certificate signing request sent to master and response
				  received.
				* Kubelet informed of new secure connection details.

				Run 'kubectl get nodes' on the master to see this node join.

			`)))
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
