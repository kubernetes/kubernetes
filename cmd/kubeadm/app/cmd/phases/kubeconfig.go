/*
Copyright 2017 The Kubernetes Authors.

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

package phases

import (
	"fmt"
	"io"

	"github.com/spf13/cobra"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiext "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha1"
	cmdutil "k8s.io/kubernetes/cmd/kubeadm/app/cmd/util"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeconfigphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/kubeconfig"
	"k8s.io/kubernetes/pkg/api"
)

// NewCmdKubeConfig return main command for kubeconfig phase
func NewCmdKubeConfig(out io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "kubeconfig",
		Short: "Generate all kubeconfig files necessary to establish the control plane and the admin kubeconfig file.",
		RunE:  cmdutil.SubCmdRunE("kubeconfig"),
	}

	cmd.AddCommand(getKubeConfigSubCommands(out, kubeadmconstants.KubernetesDir, "")...)
	return cmd
}

// getKubeConfigSubCommands returns sub commands for kubeconfig phase
func getKubeConfigSubCommands(out io.Writer, outDir, defaultKubernetesVersion string) []*cobra.Command {

	cfg := &kubeadmapiext.MasterConfiguration{}

	// This is used for unit testing only...
	// If we wouldn't set this to something, the code would dynamically look up the version from the internet
	// By setting this explicitely for tests workarounds that
	if defaultKubernetesVersion != "" {
		cfg.KubernetesVersion = defaultKubernetesVersion
	}

	// Default values for the cobra help text
	api.Scheme.Default(cfg)

	var cfgPath, token, clientName string
	var subCmds []*cobra.Command

	subCmdProperties := []struct {
		use     string
		short   string
		cmdFunc func(outDir string, cfg *kubeadmapi.MasterConfiguration) error
	}{
		{
			use:     "all",
			short:   "Generate all kubeconfig files necessary to establish the control plane and the admin kubeconfig file.",
			cmdFunc: kubeconfigphase.CreateInitKubeConfigFiles,
		},
		{
			use:     "admin",
			short:   "Generate a kubeconfig file for the admin to use and for kubeadm itself.",
			cmdFunc: kubeconfigphase.CreateAdminKubeConfigFile,
		},
		{
			use:     "kubelet",
			short:   "Generate a kubeconfig file for the Kubelet to use. Please note that this should *only* be used for bootstrapping purposes. After your control plane is up, you should request all kubelet credentials from the CSR API.",
			cmdFunc: kubeconfigphase.CreateKubeletKubeConfigFile,
		},
		{
			use:     "controller-manager",
			short:   "Generate a kubeconfig file for the Controller Manager to use.",
			cmdFunc: kubeconfigphase.CreateControllerManagerKubeConfigFile,
		},
		{
			use:     "scheduler",
			short:   "Generate a kubeconfig file for the Scheduler to use.",
			cmdFunc: kubeconfigphase.CreateSchedulerKubeConfigFile,
		},
		{
			use:   "user",
			short: "Outputs a kubeconfig file for an additional user.",
			cmdFunc: func(outDir string, cfg *kubeadmapi.MasterConfiguration) error {
				if clientName == "" {
					return fmt.Errorf("missing required argument client-name")
				}

				// if the kubeconfig file for an additional user has to use a token, use it
				if token != "" {
					return kubeconfigphase.WriteKubeConfigWithToken(out, cfg, clientName, token)
				}

				// Otherwise, write a kubeconfig file with a generate client cert
				return kubeconfigphase.WriteKubeConfigWithClientCert(out, cfg, clientName)
			},
		},
	}

	for _, properties := range subCmdProperties {
		// Creates the UX Command
		cmd := &cobra.Command{
			Use:   properties.use,
			Short: properties.short,
			Run:   runCmdPhase(properties.cmdFunc, &outDir, &cfgPath, cfg),
		}

		// Add flags to the command
		if properties.use != "user" {
			cmd.Flags().StringVar(&cfgPath, "config", cfgPath, "Path to kubeadm config file (WARNING: Usage of a configuration file is experimental)")
		}
		cmd.Flags().StringVar(&cfg.CertificatesDir, "cert-dir", cfg.CertificatesDir, "The path where certificates are stored.")
		cmd.Flags().StringVar(&cfg.API.AdvertiseAddress, "apiserver-advertise-address", cfg.API.AdvertiseAddress, "The IP address or DNS name the API Server is accessible on.")
		cmd.Flags().Int32Var(&cfg.API.BindPort, "apiserver-bind-port", cfg.API.BindPort, "The port the API Server is accessible on.")
		if properties.use == "all" || properties.use == "kubelet" {
			cmd.Flags().StringVar(&cfg.NodeName, "node-name", cfg.NodeName, `The node name that the kubelet client cert should use.`)
		}
		if properties.use == "user" {
			cmd.Flags().StringVar(&token, "token", token, "The token that should be used as the authentication mechanism for this kubeconfig.")
			cmd.Flags().StringVar(&clientName, "client-name", clientName, "The name of the KubeConfig user that will be created. Will also be used as the CN if client certs are created.")
		}

		subCmds = append(subCmds, cmd)
	}

	return subCmds
}
