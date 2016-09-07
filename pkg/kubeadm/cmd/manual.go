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
	kubemaster "k8s.io/kubernetes/pkg/kubeadm/master"
	kubenode "k8s.io/kubernetes/pkg/kubeadm/node"
	kubeadmutil "k8s.io/kubernetes/pkg/kubeadm/util"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	netutil "k8s.io/kubernetes/pkg/util/net"
)

var (
	manual_init_done_msgf = dedent.Dedent(`
		Master initialization complete:

		* Static pods written and kubelet's kubeconfig written.
		* Kubelet should start soon.  Try 'systemctl restart kubelet'
		  or equivalent if it doesn't.

		CA cert is written to:
		    /etc/kubernetes/pki/ca.pem.

		**Please copy this file (scp, rsync or through other means) to
		all your nodes and then run on them**:

		kubeadm manual bootstrap join-node --ca-cert-file <path-to-ca-cert> \
		    --token %s --api-server-urls https://%s:443/
		`)
	manual_join_done_msgf = dedent.Dedent(`
		Node join complete:
		* Certificate signing request sent to master and response
		  received.
		* Kubelet informed of new secure connection details.

		Run 'kubectl get nodes' on the master to see this node join.
		`)
)

// TODO --token here becomes Discovery.BearerToken and not Discovery.GivenToken
// may be we should make it the same and ask user to pass dot-separated tokens
// in any of the modes; we could also enable discovery API in the manual mode just
// as well, there is no reason we shouldn't let user mix and match modes, unless
// it is too difficult to support

func NewCmdManual(out io.Writer, params *kubeadmapi.BootstrapParams) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "manual",
		Short: "Advanced, less-automated functionality, for power users.",
		// TODO put example usage in the Long description here
	}
	cmd.AddCommand(NewCmdManualBootstrap(out, params))
	return cmd
}

func NewCmdManualBootstrap(out io.Writer, params *kubeadmapi.BootstrapParams) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "bootstrap",
		Short: "Manually bootstrap a cluster 'out-of-band'",
		Long: dedent.Dedent(`
			Manually bootstrap a cluster 'out-of-band', by generating and distributing a CA
			certificate to all your servers and specifying and (list of) API server URLs.
		`),
		Run: func(cmd *cobra.Command, args []string) {
		},
	}
	cmd.AddCommand(NewCmdManualBootstrapInitMaster(out, params))
	cmd.AddCommand(NewCmdManualBootstrapJoinNode(out, params))

	return cmd
}

func NewCmdManualBootstrapInitMaster(out io.Writer, params *kubeadmapi.BootstrapParams) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "init-master",
		Short: "Manually bootstrap a master 'out-of-band'",
		Long: dedent.Dedent(`
			Manually bootstrap a master 'out-of-band'.
			Will create TLS certificates and set up static pods for Kubernetes master
			components.
		`),
		Run: func(cmd *cobra.Command, args []string) {
			err := RunManualBootstrapInitMaster(out, cmd, args, params)
			cmdutil.CheckErr(err)
		},
	}

	params.Discovery.ApiServerURLs = "http://127.0.0.1:8080/" // On the master, assume you can talk to the API server
	cmd.PersistentFlags().StringVar(&params.Discovery.ApiServerDNSName, "api-dns-name", "",
		`(optional) DNS name for the API server, will be encoded into
		subjectAltName in the resulting (generated) TLS certificates`)
	cmd.PersistentFlags().IPVar(&params.Discovery.ListenIP, "listen-ip", nil,
		`(optional) IP address to listen on, in case autodetection fails.`)
	cmd.PersistentFlags().StringVar(&params.Discovery.BearerToken, "token", "",
		`(optional) Shared secret used to secure bootstrap. Will be generated and displayed if not provided.`)

	return cmd
}

func RunManualBootstrapInitMaster(out io.Writer, cmd *cobra.Command, args []string, params *kubeadmapi.BootstrapParams) error {
	// Auto-detect the IP
	if params.Discovery.ListenIP == nil {
		ip, err := netutil.ChooseHostInterface()
		if err != nil {
			return err
		}
		params.Discovery.ListenIP = ip
	}

	if err := kubemaster.CreateTokenAuthFile(params); err != nil {
		return err
	}
	if err := kubemaster.WriteStaticPodManifests(params); err != nil {
		return err
	}
	caKey, caCert, err := kubemaster.CreatePKIAssets(params)
	if err != nil {
		return err
	}
	kubeconfigs, err := kubemaster.CreateCertsAndConfigForClients(params, []string{"kubelet", "admin"}, caKey, caCert)
	if err != nil {
		return err
	}
	for name, kubeconfig := range kubeconfigs {
		if err := kubeadmutil.WriteKubeconfigIfNotExists(params, name, kubeconfig); err != nil {
			return err
		}
	}

	// TODO use templates to reference struct fields directly as order of args is fragile
	fmt.Fprintf(out, manual_init_done_msgf,
		params.Discovery.BearerToken,
		params.Discovery.ListenIP,
	)
	return nil
}

func NewCmdManualBootstrapJoinNode(out io.Writer, params *kubeadmapi.BootstrapParams) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "join-node",
		Short: "Manually bootstrap a node 'out-of-band', joining it into a cluster with extant control plane",

		Run: func(cmd *cobra.Command, args []string) {
			err := RunManualBootstrapJoinNode(out, cmd, args, params)
			cmdutil.CheckErr(err)
		},
	}
	cmd.PersistentFlags().StringVarP(&params.Discovery.CaCertFile, "ca-cert-file", "", "",
		`Path to a CA cert file in PEM format. The same CA cert must be distributed to
		all servers.`)
	cmd.PersistentFlags().StringVarP(&params.Discovery.ApiServerURLs, "api-server-urls", "", "",
		`Comma separated list of API server URLs. Typically this might be just
		https://<address-of-master>:8080/`)
	cmd.PersistentFlags().StringVarP(&params.Discovery.BearerToken, "token", "", "",
		`Shared secret used to secure bootstrap. Must match output of 'init-master'.`)

	return cmd
}

func RunManualBootstrapJoinNode(out io.Writer, cmd *cobra.Command, args []string, params *kubeadmapi.BootstrapParams) error {
	if params.Discovery.CaCertFile == "" {
		fmt.Fprintf(out, "Must specify --ca-cert-file (see --help)\n")
		return nil
	}

	if params.Discovery.ApiServerURLs == "" {
		fmt.Fprintf(out, "Must specify --api-server-urls (see --help)\n")
		return nil
	}

	kubeconfig, err := kubenode.PerformTLSBootstrapFromParams(params)
	if err != nil {
		fmt.Fprintf(out, "Failed to perform TLS bootstrap: %s\n", err)
		return err
	}

	err = kubeadmutil.WriteKubeconfigIfNotExists(params, "kubelet", kubeconfig)
	if err != nil {
		fmt.Fprintf(out, "Unable to write config for node:\n%s\n", err)
		return err
	}

	fmt.Fprintf(out, manual_join_done_msgf)
	return nil
}
