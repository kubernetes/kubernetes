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

package alpha

import (
	"io"

	"github.com/pkg/errors"
	"github.com/spf13/cobra"
	kubeadmscheme "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/scheme"
	kubeadmapiv1beta2 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta2"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/options"
	cmdutil "k8s.io/kubernetes/cmd/kubeadm/app/cmd/util"
	kubeconfigphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/kubeconfig"
	configutil "k8s.io/kubernetes/cmd/kubeadm/app/util/config"
)

var (
	kubeconfigLongDesc = cmdutil.LongDesc(`
	Kubeconfig file utilities.
	` + cmdutil.AlphaDisclaimer)

	userKubeconfigLongDesc = cmdutil.LongDesc(`
	Output a kubeconfig file for an additional user.
	` + cmdutil.AlphaDisclaimer)

	userKubeconfigExample = cmdutil.Examples(`
	# Output a kubeconfig file for an additional user named foo
	kubeadm alpha kubeconfig user --client-name=foo
	`)
)

// newCmdKubeConfigUtility returns main command for kubeconfig phase
func newCmdKubeConfigUtility(out io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "kubeconfig",
		Short: "Kubeconfig file utilities",
		Long:  kubeconfigLongDesc,
	}

	cmd.AddCommand(newCmdUserKubeConfig(out))
	return cmd
}

// newCmdUserKubeConfig returns sub commands for kubeconfig phase
func newCmdUserKubeConfig(out io.Writer) *cobra.Command {

	initCfg := &kubeadmapiv1beta2.InitConfiguration{}
	clusterCfg := &kubeadmapiv1beta2.ClusterConfiguration{}

	// Default values for the cobra help text
	kubeadmscheme.Scheme.Default(initCfg)
	kubeadmscheme.Scheme.Default(clusterCfg)

	var token, clientName string
	var organizations []string

	// Creates the UX Command
	cmd := &cobra.Command{
		Use:     "user",
		Short:   "Output a kubeconfig file for an additional user",
		Long:    userKubeconfigLongDesc,
		Example: userKubeconfigExample,
		RunE: func(cmd *cobra.Command, args []string) error {
			if clientName == "" {
				return errors.New("missing required argument --client-name")
			}

			// This call returns the ready-to-use configuration based on the defaults populated by flags
			internalcfg, err := configutil.DefaultedInitConfiguration(initCfg, clusterCfg)
			if err != nil {
				return err
			}

			// if the kubeconfig file for an additional user has to use a token, use it
			if token != "" {
				return kubeconfigphase.WriteKubeConfigWithToken(out, internalcfg, clientName, token)
			}

			// Otherwise, write a kubeconfig file with a generate client cert
			return kubeconfigphase.WriteKubeConfigWithClientCert(out, internalcfg, clientName, organizations)
		},
		Args: cobra.NoArgs,
	}

	// Add ClusterConfiguration backed flags to the command
	cmd.Flags().StringVar(&clusterCfg.CertificatesDir, options.CertificatesDir, clusterCfg.CertificatesDir, "The path where certificates are stored")
	cmd.Flags().StringVar(&clusterCfg.ClusterName, "cluster-name", clusterCfg.ClusterName, "Cluster name to be used in kubeconfig")

	// Add InitConfiguration backed flags to the command
	cmd.Flags().StringVar(&initCfg.LocalAPIEndpoint.AdvertiseAddress, options.APIServerAdvertiseAddress, initCfg.LocalAPIEndpoint.AdvertiseAddress, "The IP address the API server is accessible on")
	cmd.Flags().Int32Var(&initCfg.LocalAPIEndpoint.BindPort, options.APIServerBindPort, initCfg.LocalAPIEndpoint.BindPort, "The port the API server is accessible on")

	// Add command specific flags
	cmd.Flags().StringVar(&token, options.TokenStr, token, "The token that should be used as the authentication mechanism for this kubeconfig, instead of client certificates")
	cmd.Flags().StringVar(&clientName, "client-name", clientName, "The name of user. It will be used as the CN if client certificates are created")
	cmd.Flags().StringSliceVar(&organizations, "org", organizations, "The orgnizations of the client certificate. It will be used as the O if client certificates are created")

	return cmd
}
