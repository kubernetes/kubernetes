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
	"github.com/spf13/cobra"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiext "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha1"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/validation"
	cmdutil "k8s.io/kubernetes/cmd/kubeadm/app/cmd/util"
	certsphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/certs"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	configutil "k8s.io/kubernetes/cmd/kubeadm/app/util/config"
	"k8s.io/kubernetes/pkg/api"
)

// NewCmdCerts return main command for certs phase
func NewCmdCerts() *cobra.Command {
	cmd := &cobra.Command{
		Use:     "certs",
		Aliases: []string{"certificates"},
		Short:   "Generate certificates for a Kubernetes cluster.",
		RunE:    cmdutil.SubCmdRunE("certs"),
	}

	cmd.AddCommand(getCertsSubCommands("")...)
	return cmd
}

// getCertsSubCommands returns sub commands for certs phase
func getCertsSubCommands(defaultKubernetesVersion string) []*cobra.Command {

	cfg := &kubeadmapiext.MasterConfiguration{}

	// This is used for unit testing only...
	// If we wouldn't set this to something, the code would dynamically look up the version from the internet
	// By setting this explicitely for tests workarounds that
	if defaultKubernetesVersion != "" {
		cfg.KubernetesVersion = defaultKubernetesVersion
	}

	// Default values for the cobra help text
	api.Scheme.Default(cfg)

	var cfgPath string
	var subCmds []*cobra.Command

	subCmdProperties := []struct {
		use     string
		short   string
		cmdFunc func(cfg *kubeadmapi.MasterConfiguration) error
	}{
		{
			use:     "all",
			short:   "Generate all PKI assets necessary to establish the control plane",
			cmdFunc: certsphase.CreatePKIAssets,
		},
		{
			use:     "ca",
			short:   "Generate CA certificate and key for a Kubernetes cluster.",
			cmdFunc: certsphase.CreateCACertAndKeyfiles,
		},
		{
			use:     "apiserver",
			short:   "Generate API Server serving certificate and key.",
			cmdFunc: certsphase.CreateAPIServerCertAndKeyFiles,
		},
		{
			use:     "apiserver-kubelet-client",
			short:   "Generate a client certificate for the API Server to connect to the kubelets securely.",
			cmdFunc: certsphase.CreateAPIServerKubeletClientCertAndKeyFiles,
		},
		{
			use:     "sa",
			short:   "Generate a private key for signing service account tokens along with its public key.",
			cmdFunc: certsphase.CreateServiceAccountKeyAndPublicKeyFiles,
		},
		{
			use:     "front-proxy-ca",
			short:   "Generate front proxy CA certificate and key for a Kubernetes cluster.",
			cmdFunc: certsphase.CreateFrontProxyCACertAndKeyFiles,
		},
		{
			use:     "front-proxy-client",
			short:   "Generate front proxy CA client certificate and key for a Kubernetes cluster.",
			cmdFunc: certsphase.CreateFrontProxyClientCertAndKeyFiles,
		},
	}

	for _, properties := range subCmdProperties {
		// Creates the UX Command
		cmd := &cobra.Command{
			Use:   properties.use,
			Short: properties.short,
			Run:   runCmdFunc(properties.cmdFunc, &cfgPath, cfg),
		}

		// Add flags to the command
		cmd.Flags().StringVar(&cfgPath, "config", cfgPath, "Path to kubeadm config file (WARNING: Usage of a configuration file is experimental)")
		cmd.Flags().StringVar(&cfg.CertificatesDir, "cert-dir", cfg.CertificatesDir, "The path where to save and store the certificates")
		if properties.use == "all" || properties.use == "apiserver" {
			cmd.Flags().StringVar(&cfg.Networking.DNSDomain, "service-dns-domain", cfg.Networking.DNSDomain, "Use alternative domain for services, e.g. \"myorg.internal\"")
			cmd.Flags().StringVar(&cfg.Networking.ServiceSubnet, "service-cidr", cfg.Networking.ServiceSubnet, "Use alternative range of IP address for service VIPs")
			cmd.Flags().StringSliceVar(&cfg.APIServerCertSANs, "apiserver-cert-extra-sans", []string{}, "Optional extra altnames to use for the API Server serving cert. Can be both IP addresses and dns names.")
			cmd.Flags().StringVar(&cfg.API.AdvertiseAddress, "apiserver-advertise-address", cfg.API.AdvertiseAddress, "The IP address the API Server will advertise it's listening on. 0.0.0.0 means the default network interface's address.")
		}

		subCmds = append(subCmds, cmd)
	}

	return subCmds
}

// runCmdFunc creates a cobra.Command Run function, by composing the call to the given cmdFunc with necessary additional steps (e.g preparation of input parameters)
func runCmdFunc(cmdFunc func(cfg *kubeadmapi.MasterConfiguration) error, cfgPath *string, cfg *kubeadmapiext.MasterConfiguration) func(cmd *cobra.Command, args []string) {

	// the following statement build a clousure that wraps a call to a cmdFunc, binding
	// the function itself with the specific parameters of each sub command.
	// Please note that specific parameter should be passed as value, while other parameters - passed as reference -
	// are shared between sub commands and gets access to current value e.g. flags value.

	return func(cmd *cobra.Command, args []string) {
		if err := validation.ValidateMixedArguments(cmd.Flags()); err != nil {
			kubeadmutil.CheckErr(err)
		}

		// This call returns the ready-to-use configuration based on the configuration file that might or might not exist and the default cfg populated by flags
		internalcfg, err := configutil.ConfigFileAndDefaultsToInternalConfig(*cfgPath, cfg)
		kubeadmutil.CheckErr(err)

		// Execute the cmdFunc
		err = cmdFunc(internalcfg)
		kubeadmutil.CheckErr(err)
	}
}
