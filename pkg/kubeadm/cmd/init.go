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
	kubeadmutil "k8s.io/kubernetes/pkg/kubeadm/util"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	netutil "k8s.io/kubernetes/pkg/util/net"
)

var (
	init_done_msgf = dedent.Dedent(`
		Kubernetes master initialised successfully!

		You can connect any number of nodes by running:

		kubeadm join --token %s %s
		`)
)

func NewCmdInit(out io.Writer, params *kubeadmapi.BootstrapParams) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "init --token <secret> [--listen-ip <addr>]",
		Short: "Run this on the first server you deploy onto.",
		Run: func(cmd *cobra.Command, args []string) {
			err := RunInit(out, cmd, args, params)
			cmdutil.CheckErr(err)
		},
	}

	cmd.PersistentFlags().StringVarP(&params.Discovery.ListenIP, "listen-ip", "", "",
		`(optional) IP address to listen on, in case autodetection fails.`)
	cmd.PersistentFlags().StringVarP(&params.Discovery.GivenToken, "token", "", "",
		`(optional) Shared secret used to secure bootstrap. Will be generated and displayed if not provided.`)

	return cmd
}

func RunInit(out io.Writer, cmd *cobra.Command, args []string, params *kubeadmapi.BootstrapParams) error {
	if params.Discovery.ListenIP == "" {
		ip, err := netutil.ChooseHostInterface()
		if err != nil {
			return err
		}
		params.Discovery.ListenIP = ip.String()
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

	client, err := kubemaster.CreateClientAndWaitForAPI(kubeconfigs["admin"])
	if err != nil {
		return err
	}

	if err := kubemaster.CreateDiscoveryDeploymentAndSecret(params, client, caCert); err != nil {
		return err
	}

	if err := kubemaster.CreateEssentialAddons(params, client); err != nil {
		return err
	}

	// TODO use templates to reference struct fields directly as order of args is fragile
	fmt.Fprintf(out, init_done_msgf,
		params.Discovery.GivenToken,
		params.Discovery.ListenIP,
	)

	return nil
}
