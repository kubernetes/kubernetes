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
	"strings"

	"github.com/spf13/cobra"

	clientset "k8s.io/client-go/kubernetes"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiext "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha1"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/validation"
	cmdutil "k8s.io/kubernetes/cmd/kubeadm/app/cmd/util"
	"k8s.io/kubernetes/cmd/kubeadm/app/features"
	dnsaddon "k8s.io/kubernetes/cmd/kubeadm/app/phases/addons/dns"
	proxyaddon "k8s.io/kubernetes/cmd/kubeadm/app/phases/addons/proxy"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	configutil "k8s.io/kubernetes/cmd/kubeadm/app/util/config"
	kubeconfigutil "k8s.io/kubernetes/cmd/kubeadm/app/util/kubeconfig"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/util/normalizer"
)

var (
	allAddonsLongDesc = normalizer.LongDesc(`
		Installs the kube-dns and the kube-proxys addons components via the API server.  
		Please note that although the DNS server is deployed, it will not be scheduled until CNI is installed.
		` + cmdutil.AlphaDisclaimer)

	allAddonsExample = normalizer.Examples(`
		# Installs the kube-dns and the kube-proxys addons components via the API server, 
		# functionally equivalent to what installed by kubeadm init. 

		kubeadm alpha phase selfhosting from-staticpods
		`)

	kubednsAddonsLongDesc = normalizer.LongDesc(`
		Installs the kube-dns addon components via the API server.  
		Please note that although the DNS server is deployed, it will not be scheduled until CNI is installed.
		` + cmdutil.AlphaDisclaimer)

	kubeproxyAddonsLongDesc = normalizer.LongDesc(`
		Installs the kube-proxy addon components via the API server.  
		` + cmdutil.AlphaDisclaimer)
)

// NewCmdAddon returns the addon Cobra command
func NewCmdAddon() *cobra.Command {
	cmd := &cobra.Command{
		Use:     "addon",
		Aliases: []string{"addons"},
		Short:   "Installs required addons for passing Conformance tests",
		Long:    cmdutil.MacroCommandLongDescription,
	}

	cmd.AddCommand(getAddonsSubCommands()...)
	return cmd
}

// EnsureAllAddons installs all addons to a Kubernetes cluster
func EnsureAllAddons(cfg *kubeadmapi.MasterConfiguration, client clientset.Interface) error {

	addonActions := []func(cfg *kubeadmapi.MasterConfiguration, client clientset.Interface) error{
		dnsaddon.EnsureDNSAddon,
		proxyaddon.EnsureProxyAddon,
	}

	for _, action := range addonActions {
		err := action(cfg, client)
		if err != nil {
			return err
		}
	}

	return nil
}

// getAddonsSubCommands returns sub commands for addons phase
func getAddonsSubCommands() []*cobra.Command {
	cfg := &kubeadmapiext.MasterConfiguration{}
	// Default values for the cobra help text
	legacyscheme.Scheme.Default(cfg)

	var cfgPath, kubeConfigFile, featureGatesString string
	var subCmds []*cobra.Command

	subCmdProperties := []struct {
		use      string
		short    string
		long     string
		examples string
		cmdFunc  func(cfg *kubeadmapi.MasterConfiguration, client clientset.Interface) error
	}{
		{
			use:      "all",
			short:    "Installs all addons to a Kubernetes cluster",
			long:     allAddonsLongDesc,
			examples: allAddonsExample,
			cmdFunc:  EnsureAllAddons,
		},
		{
			use:     "kube-dns",
			short:   "Installs the kube-dns addon to a Kubernetes cluster",
			long:    kubednsAddonsLongDesc,
			cmdFunc: dnsaddon.EnsureDNSAddon,
		},
		{
			use:     "kube-proxy",
			short:   "Installs the kube-proxy addon to a Kubernetes cluster",
			long:    kubeproxyAddonsLongDesc,
			cmdFunc: proxyaddon.EnsureProxyAddon,
		},
	}

	for _, properties := range subCmdProperties {
		// Creates the UX Command
		cmd := &cobra.Command{
			Use:     properties.use,
			Short:   properties.short,
			Long:    properties.long,
			Example: properties.examples,
			Run:     runAddonsCmdFunc(properties.cmdFunc, cfg, &kubeConfigFile, &cfgPath, &featureGatesString),
		}

		// Add flags to the command
		cmd.Flags().StringVar(&kubeConfigFile, "kubeconfig", "/etc/kubernetes/admin.conf", "The KubeConfig file to use when talking to the cluster")
		cmd.Flags().StringVar(&cfgPath, "config", cfgPath, "Path to a kubeadm config file. WARNING: Usage of a configuration file is experimental!")
		cmd.Flags().StringVar(&cfg.KubernetesVersion, "kubernetes-version", cfg.KubernetesVersion, `Choose a specific Kubernetes version for the control plane`)
		cmd.Flags().StringVar(&cfg.ImageRepository, "image-repository", cfg.ImageRepository, `Choose a container registry to pull control plane images from`)

		if properties.use == "all" || properties.use == "kube-proxy" {
			cmd.Flags().StringVar(&cfg.API.AdvertiseAddress, "apiserver-advertise-address", cfg.API.AdvertiseAddress, `The IP address or DNS name the API server is accessible on`)
			cmd.Flags().Int32Var(&cfg.API.BindPort, "apiserver-bind-port", cfg.API.BindPort, `The port the API server is accessible on`)
			cmd.Flags().StringVar(&cfg.Networking.PodSubnet, "pod-network-cidr", cfg.Networking.PodSubnet, `The range of IP addresses used for the Pod network`)
		}

		if properties.use == "all" || properties.use == "kube-dns" {
			cmd.Flags().StringVar(&cfg.Networking.DNSDomain, "service-dns-domain", cfg.Networking.DNSDomain, `Alternative domain for services`)
			cmd.Flags().StringVar(&cfg.Networking.ServiceSubnet, "service-cidr", cfg.Networking.ServiceSubnet, `The range of IP address used for service VIPs`)
			cmd.Flags().StringVar(&featureGatesString, "feature-gates", featureGatesString, "A set of key=value pairs that describe feature gates for various features."+
				"Options are:\n"+strings.Join(features.KnownFeatures(&features.InitFeatureGates), "\n"))
		}
		subCmds = append(subCmds, cmd)
	}

	return subCmds
}

// runAddonsCmdFunc creates a cobra.Command Run function, by composing the call to the given cmdFunc with necessary additional steps (e.g preparation of input parameters)
func runAddonsCmdFunc(cmdFunc func(cfg *kubeadmapi.MasterConfiguration, client clientset.Interface) error, cfg *kubeadmapiext.MasterConfiguration, kubeConfigFile *string, cfgPath *string, featureGatesString *string) func(cmd *cobra.Command, args []string) {

	// the following statement build a clousure that wraps a call to a cmdFunc, binding
	// the function itself with the specific parameters of each sub command.
	// Please note that specific parameter should be passed as value, while other parameters - passed as reference -
	// are shared between sub commands and gets access to current value e.g. flags value.

	return func(cmd *cobra.Command, args []string) {
		var err error
		if err := validation.ValidateMixedArguments(cmd.Flags()); err != nil {
			kubeadmutil.CheckErr(err)
		}

		if cfg.FeatureGates, err = features.NewFeatureGate(&features.InitFeatureGates, *featureGatesString); err != nil {
			kubeadmutil.CheckErr(err)
		}

		internalcfg := &kubeadmapi.MasterConfiguration{}
		legacyscheme.Scheme.Convert(cfg, internalcfg, nil)
		client, err := kubeconfigutil.ClientSetFromFile(*kubeConfigFile)
		kubeadmutil.CheckErr(err)
		internalcfg, err = configutil.ConfigFileAndDefaultsToInternalConfig(*cfgPath, cfg)
		kubeadmutil.CheckErr(err)
		if err := features.ValidateVersion(features.InitFeatureGates, internalcfg.FeatureGates, internalcfg.KubernetesVersion); err != nil {
			kubeadmutil.CheckErr(err)
		}

		// Execute the cmdFunc
		err = cmdFunc(internalcfg, client)
		kubeadmutil.CheckErr(err)
	}
}
