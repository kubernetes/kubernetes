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
	kubeadmapiv1beta1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta1"
	cmdutil "k8s.io/kubernetes/cmd/kubeadm/app/cmd/util"
	kubeconfigphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/kubeconfig"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	configutil "k8s.io/kubernetes/cmd/kubeadm/app/util/config"
	"k8s.io/kubernetes/pkg/util/normalizer"
)

var (
	kubeconfigLongDesc = normalizer.LongDesc(`
	Kubeconfig file utilities.
	` + cmdutil.AlphaDisclaimer)

	userKubeconfigLongDesc = normalizer.LongDesc(`
	Outputs a kubeconfig file for an additional user.
	` + cmdutil.AlphaDisclaimer)

	userKubeconfigExample = normalizer.Examples(`
	# Outputs a kubeconfig file for an additional user named foo
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

	cfg := &kubeadmapiv1beta1.InitConfiguration{}

	// Default values for the cobra help text
	kubeadmscheme.Scheme.Default(cfg)

	var token, clientName string
	var organizations []string

	// Creates the UX Command
	cmd := &cobra.Command{
		Use:     "user",
		Short:   "Outputs a kubeconfig file for an additional user",
		Long:    userKubeconfigLongDesc,
		Example: userKubeconfigExample,
		Run: func(cmd *cobra.Command, args []string) {
			if clientName == "" {
				kubeadmutil.CheckErr(errors.New("missing required argument --client-name"))
			}

			// This call returns the ready-to-use configuration based on the configuration file that might or might not exist and the default cfg populated by flags
			internalcfg, err := configutil.ConfigFileAndDefaultsToInternalConfig("", cfg)
			kubeadmutil.CheckErr(err)

			// if the kubeconfig file for an additional user has to use a token, use it
			if token != "" {
				kubeadmutil.CheckErr(kubeconfigphase.WriteKubeConfigWithToken(out, internalcfg, clientName, token))
				return
			}

			// Otherwise, write a kubeconfig file with a generate client cert
			kubeadmutil.CheckErr(kubeconfigphase.WriteKubeConfigWithClientCert(out, internalcfg, clientName, organizations))
		},
	}

	// Add flags to the command
	cmd.Flags().StringVar(&cfg.CertificatesDir, "cert-dir", cfg.CertificatesDir, "The path where certificates are stored")
	cmd.Flags().StringVar(&cfg.LocalAPIEndpoint.AdvertiseAddress, "apiserver-advertise-address", cfg.LocalAPIEndpoint.AdvertiseAddress, "The IP address the API server is accessible on")
	cmd.Flags().Int32Var(&cfg.LocalAPIEndpoint.BindPort, "apiserver-bind-port", cfg.LocalAPIEndpoint.BindPort, "The port the API server is accessible on")
	cmd.Flags().StringVar(&token, "token", token, "The token that should be used as the authentication mechanism for this kubeconfig, instead of client certificates")
	cmd.Flags().StringVar(&clientName, "client-name", clientName, "The name of user. It will be used as the CN if client certificates are created")
	cmd.Flags().StringSliceVar(&organizations, "org", organizations, "The orgnizations of the client certificate. It will be used as the O if client certificates are created")

	return cmd
}
