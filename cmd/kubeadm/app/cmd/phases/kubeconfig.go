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
	"path/filepath"

	"github.com/spf13/cobra"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmscheme "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/scheme"
	kubeadmapiv1alpha3 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha3"
	cmdutil "k8s.io/kubernetes/cmd/kubeadm/app/cmd/util"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeconfigphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/kubeconfig"
	"k8s.io/kubernetes/pkg/util/normalizer"
)

var (
	allKubeconfigLongDesc = normalizer.LongDesc(`
		Generates all kubeconfig files necessary to establish the control plane and the admin kubeconfig file.
		` + cmdutil.AlphaDisclaimer)

	allKubeconfigExample = normalizer.Examples(`
		# Generates all kubeconfig files, functionally equivalent to what generated
		# by kubeadm init.
		kubeadm alpha phase kubeconfig all

		# Generates all kubeconfig files using options read from a configuration file.
		kubeadm alpha phase kubeconfig all --config masterconfiguration.yaml
		`)

	adminKubeconfigLongDesc = fmt.Sprintf(normalizer.LongDesc(`
		Generates the kubeconfig file for the admin and for kubeadm itself, and saves it to %s file.
		`+cmdutil.AlphaDisclaimer), kubeadmconstants.AdminKubeConfigFileName)

	kubeletKubeconfigLongDesc = fmt.Sprintf(normalizer.LongDesc(`
		Generates the kubeconfig file for the kubelet to use and saves it to %s file.

		Please note that this should *only* be used for bootstrapping purposes. After your control plane is up,
		you should request all kubelet credentials from the CSR API.
		`+cmdutil.AlphaDisclaimer), filepath.Join(kubeadmconstants.KubernetesDir, kubeadmconstants.KubeletKubeConfigFileName))

	controllerManagerKubeconfigLongDesc = fmt.Sprintf(normalizer.LongDesc(`
		Generates the kubeconfig file for the controller manager to use and saves it to %s file.
		`+cmdutil.AlphaDisclaimer), filepath.Join(kubeadmconstants.KubernetesDir, kubeadmconstants.ControllerManagerKubeConfigFileName))

	schedulerKubeconfigLongDesc = fmt.Sprintf(normalizer.LongDesc(`
		Generates the kubeconfig file for the scheduler to use and saves it to %s file.
		`+cmdutil.AlphaDisclaimer), filepath.Join(kubeadmconstants.KubernetesDir, kubeadmconstants.SchedulerKubeConfigFileName))

	userKubeconfigLongDesc = normalizer.LongDesc(`
		Outputs a kubeconfig file for an additional user.
		` + cmdutil.AlphaDisclaimer)

	userKubeconfigExample = normalizer.Examples(`
		# Outputs a kubeconfig file for an additional user named foo
		kubeadm alpha phase kubeconfig user --client-name=foo
		`)
)

// NewCmdKubeConfig returns main command for kubeconfig phase
func NewCmdKubeConfig(out io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "kubeconfig",
		Short: "Generates all kubeconfig files necessary to establish the control plane and the admin kubeconfig file",
		Long:  cmdutil.MacroCommandLongDescription,
	}

	cmd.AddCommand(getKubeConfigSubCommands(out, kubeadmconstants.KubernetesDir, "")...)
	return cmd
}

// getKubeConfigSubCommands returns sub commands for kubeconfig phase
func getKubeConfigSubCommands(out io.Writer, outDir, defaultKubernetesVersion string) []*cobra.Command {

	cfg := &kubeadmapiv1alpha3.InitConfiguration{}

	// Default values for the cobra help text
	kubeadmscheme.Scheme.Default(cfg)

	var cfgPath, token, clientName string
	var organizations []string
	var subCmds []*cobra.Command

	subCmdProperties := []struct {
		use      string
		short    string
		long     string
		examples string
		cmdFunc  func(outDir string, cfg *kubeadmapi.InitConfiguration) error
	}{
		{
			use:      "all",
			short:    "Generates all kubeconfig files necessary to establish the control plane and the admin kubeconfig file",
			long:     allKubeconfigLongDesc,
			examples: allKubeconfigExample,
			cmdFunc:  kubeconfigphase.CreateInitKubeConfigFiles,
		},
		{
			use:     "admin",
			short:   "Generates a kubeconfig file for the admin to use and for kubeadm itself",
			long:    adminKubeconfigLongDesc,
			cmdFunc: kubeconfigphase.CreateAdminKubeConfigFile,
		},
		{
			use:     "kubelet",
			short:   "Generates a kubeconfig file for the kubelet to use. Please note that this should be used *only* for bootstrapping purposes",
			long:    kubeletKubeconfigLongDesc,
			cmdFunc: kubeconfigphase.CreateKubeletKubeConfigFile,
		},
		{
			use:     "controller-manager",
			short:   "Generates a kubeconfig file for the controller manager to use",
			long:    controllerManagerKubeconfigLongDesc,
			cmdFunc: kubeconfigphase.CreateControllerManagerKubeConfigFile,
		},
		{
			use:     "scheduler",
			short:   "Generates a kubeconfig file for the scheduler to use",
			long:    schedulerKubeconfigLongDesc,
			cmdFunc: kubeconfigphase.CreateSchedulerKubeConfigFile,
		},
		{
			use:      "user",
			short:    "Outputs a kubeconfig file for an additional user",
			long:     userKubeconfigLongDesc,
			examples: userKubeconfigExample,
			cmdFunc: func(outDir string, cfg *kubeadmapi.InitConfiguration) error {
				if clientName == "" {
					return fmt.Errorf("missing required argument --client-name")
				}

				// if the kubeconfig file for an additional user has to use a token, use it
				if token != "" {
					return kubeconfigphase.WriteKubeConfigWithToken(out, cfg, clientName, token)
				}

				// Otherwise, write a kubeconfig file with a generate client cert
				return kubeconfigphase.WriteKubeConfigWithClientCert(out, cfg, clientName, organizations)
			},
		},
	}

	for _, properties := range subCmdProperties {
		// Creates the UX Command
		cmd := &cobra.Command{
			Use:     properties.use,
			Short:   properties.short,
			Long:    properties.long,
			Example: properties.examples,
			Run:     runCmdPhase(properties.cmdFunc, &outDir, &cfgPath, cfg, defaultKubernetesVersion),
		}

		// Add flags to the command
		if properties.use != "user" {
			cmd.Flags().StringVar(&cfgPath, "config", cfgPath, "Path to kubeadm config file. WARNING: Usage of a configuration file is experimental")
		}
		cmd.Flags().StringVar(&cfg.CertificatesDir, "cert-dir", cfg.CertificatesDir, "The path where certificates are stored")
		cmd.Flags().StringVar(&cfg.APIEndpoint.AdvertiseAddress, "apiserver-advertise-address", cfg.APIEndpoint.AdvertiseAddress, "The IP address the API server is accessible on")
		cmd.Flags().Int32Var(&cfg.APIEndpoint.BindPort, "apiserver-bind-port", cfg.APIEndpoint.BindPort, "The port the API server is accessible on")
		cmd.Flags().StringVar(&outDir, "kubeconfig-dir", outDir, "The path where to save the kubeconfig file")
		if properties.use == "all" || properties.use == "kubelet" {
			cmd.Flags().StringVar(&cfg.NodeRegistration.Name, "node-name", cfg.NodeRegistration.Name, `The node name that should be used for the kubelet client certificate`)
		}
		if properties.use == "user" {
			cmd.Flags().StringVar(&token, "token", token, "The token that should be used as the authentication mechanism for this kubeconfig, instead of client certificates")
			cmd.Flags().StringVar(&clientName, "client-name", clientName, "The name of user. It will be used as the CN if client certificates are created")
			cmd.Flags().StringSliceVar(&organizations, "org", organizations, "The orgnizations of the client certificate. It will be used as the O if client certificates are created")
		}

		subCmds = append(subCmds, cmd)
	}

	return subCmds
}
