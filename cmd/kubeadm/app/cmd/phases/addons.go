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

	clientset "k8s.io/client-go/kubernetes"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiext "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha1"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/validation"
	cmdutil "k8s.io/kubernetes/cmd/kubeadm/app/cmd/util"
	dnsaddon "k8s.io/kubernetes/cmd/kubeadm/app/phases/addons/dns"
	proxyaddon "k8s.io/kubernetes/cmd/kubeadm/app/phases/addons/proxy"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	configutil "k8s.io/kubernetes/cmd/kubeadm/app/util/config"
	kubeconfigutil "k8s.io/kubernetes/cmd/kubeadm/app/util/kubeconfig"
	"k8s.io/kubernetes/pkg/api"
)

// NewCmdAddon returns the addon Cobra command
func NewCmdAddon() *cobra.Command {
	cmd := &cobra.Command{
		Use:     "addon <addon-name>",
		Aliases: []string{"addons"},
		Short:   "Install an addon to a Kubernetes cluster.",
		RunE:    cmdutil.SubCmdRunE("addon"),
	}

	cmd.AddCommand(getAddonsSubCommands()...)
	return cmd
}

// EnsureAllAddons install all addons to a Kubernetes cluster.
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
	api.Scheme.Default(cfg)

	var cfgPath, kubeConfigFile string
	var subCmds []*cobra.Command

	subCmdProperties := []struct {
		use     string
		short   string
		cmdFunc func(cfg *kubeadmapi.MasterConfiguration, client clientset.Interface) error
	}{
		{
			use:     "all",
			short:   "Install all addons to a Kubernetes cluster",
			cmdFunc: EnsureAllAddons,
		},
		{
			use:     "kube-dns",
			short:   "Install the kube-dns addon to a Kubernetes cluster.",
			cmdFunc: dnsaddon.EnsureDNSAddon,
		},
		{
			use:     "kube-proxy",
			short:   "Install the kube-proxy addon to a Kubernetes cluster.",
			cmdFunc: proxyaddon.EnsureProxyAddon,
		},
	}

	for _, properties := range subCmdProperties {
		// Creates the UX Command
		cmd := &cobra.Command{
			Use:   properties.use,
			Short: properties.short,
			Run:   runAddonsCmdFunc(properties.cmdFunc, cfg, &kubeConfigFile, &cfgPath),
		}

		// Add flags to the command
		cmd.Flags().StringVar(&kubeConfigFile, "kubeconfig", "/etc/kubernetes/admin.conf", "The KubeConfig file to use for talking to the cluster")
		cmd.Flags().StringVar(&cfgPath, "config", cfgPath, "Path to kubeadm config file (WARNING: Usage of a configuration file is experimental)")
		cmd.Flags().StringVar(&cfg.KubernetesVersion, "kubernetes-version", cfg.KubernetesVersion, `Choose a specific Kubernetes version for the control plane.`)
		cmd.Flags().StringVar(&cfg.ImageRepository, "image-repository", cfg.ImageRepository, `Choose a container registry to pull control plane images from.`)

		if properties.use == "all" || properties.use == "kube-proxy" {
			cmd.Flags().StringVar(&cfg.API.AdvertiseAddress, "apiserver-advertise-address", cfg.API.AdvertiseAddress, `The IP address the API Server will advertise it's listening on. 0.0.0.0 means the default network interface's address.`)
			cmd.Flags().Int32Var(&cfg.API.BindPort, "apiserver-bind-port", cfg.API.BindPort, `Port for the API Server to bind to.`)
			cmd.Flags().StringVar(&cfg.Networking.PodSubnet, "pod-network-cidr", cfg.Networking.PodSubnet, `Specify range of IP addresses for the pod network; if set, the control plane will automatically allocate CIDRs for every node.`)
		}

		if properties.use == "all" || properties.use == "kube-dns" {
			cmd.Flags().StringVar(&cfg.Networking.DNSDomain, "service-dns-domain", cfg.Networking.DNSDomain, `Use alternative domain for services, e.g. "myorg.internal.`)
		}
		subCmds = append(subCmds, cmd)
	}

	return subCmds
}

// runAddonsCmdFunc creates a cobra.Command Run function, by composing the call to the given cmdFunc with necessary additional steps (e.g preparation of input parameters)
func runAddonsCmdFunc(cmdFunc func(cfg *kubeadmapi.MasterConfiguration, client clientset.Interface) error, cfg *kubeadmapiext.MasterConfiguration, kubeConfigFile *string, cfgPath *string) func(cmd *cobra.Command, args []string) {

	// the following statement build a clousure that wraps a call to a cmdFunc, binding
	// the function itself with the specific parameters of each sub command.
	// Please note that specific parameter should be passed as value, while other parameters - passed as reference -
	// are shared between sub commands and gets access to current value e.g. flags value.

	return func(cmd *cobra.Command, args []string) {
		if err := validation.ValidateMixedArguments(cmd.Flags()); err != nil {
			kubeadmutil.CheckErr(err)
		}

		internalcfg := &kubeadmapi.MasterConfiguration{}
		api.Scheme.Convert(cfg, internalcfg, nil)
		client, err := kubeconfigutil.ClientSetFromFile(*kubeConfigFile)
		kubeadmutil.CheckErr(err)
		internalcfg, err = configutil.ConfigFileAndDefaultsToInternalConfig(*cfgPath, cfg)
		kubeadmutil.CheckErr(err)

		// Execute the cmdFunc
		err = cmdFunc(internalcfg, client)
		kubeadmutil.CheckErr(err)
	}
}
