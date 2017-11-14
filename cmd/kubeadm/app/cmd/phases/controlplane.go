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

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiext "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha1"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/validation"
	cmdutil "k8s.io/kubernetes/cmd/kubeadm/app/cmd/util"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/features"
	controlplanephase "k8s.io/kubernetes/cmd/kubeadm/app/phases/controlplane"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	configutil "k8s.io/kubernetes/cmd/kubeadm/app/util/config"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
)

// NewCmdControlplane return main command for Controlplane phase
func NewCmdControlplane() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "controlplane",
		Short: "Generate all static pod manifest files necessary to establish the control plane.",
		RunE:  cmdutil.SubCmdRunE("controlplane"),
	}

	manifestPath := kubeadmconstants.GetStaticPodDirectory()
	cmd.AddCommand(getControlPlaneSubCommands(manifestPath, "")...)
	return cmd
}

// getControlPlaneSubCommands returns sub commands for Controlplane phase
func getControlPlaneSubCommands(outDir, defaultKubernetesVersion string) []*cobra.Command {

	cfg := &kubeadmapiext.MasterConfiguration{}

	// This is used for unit testing only...
	// If we wouldn't set this to something, the code would dynamically look up the version from the internet
	// By setting this explicitely for tests workarounds that
	if defaultKubernetesVersion != "" {
		cfg.KubernetesVersion = defaultKubernetesVersion
	}

	// Default values for the cobra help text
	legacyscheme.Scheme.Default(cfg)

	var cfgPath, featureGatesString string
	var subCmds []*cobra.Command

	subCmdProperties := []struct {
		use     string
		short   string
		cmdFunc func(outDir string, cfg *kubeadmapi.MasterConfiguration) error
	}{
		{
			use:     "all",
			short:   "Generate all static pod manifest files necessary to establish the control plane.",
			cmdFunc: controlplanephase.CreateInitStaticPodManifestFiles,
		},
		{
			use:     "apiserver",
			short:   "Generate apiserver static pod manifest.",
			cmdFunc: controlplanephase.CreateAPIServerStaticPodManifestFile,
		},
		{
			use:     "controller-manager",
			short:   "Generate controller-manager static pod manifest.",
			cmdFunc: controlplanephase.CreateControllerManagerStaticPodManifestFile,
		},
		{
			use:     "scheduler",
			short:   "Generate scheduler static pod manifest.",
			cmdFunc: controlplanephase.CreateSchedulerStaticPodManifestFile,
		},
	}

	for _, properties := range subCmdProperties {
		// Creates the UX Command
		cmd := &cobra.Command{
			Use:   properties.use,
			Short: properties.short,
			Run:   runCmdControlPlane(properties.cmdFunc, &outDir, &cfgPath, &featureGatesString, cfg),
		}

		// Add flags to the command
		cmd.Flags().StringVar(&cfg.CertificatesDir, "cert-dir", cfg.CertificatesDir, `The path where certificates are stored.`)
		cmd.Flags().StringVar(&cfg.KubernetesVersion, "kubernetes-version", cfg.KubernetesVersion, `Choose a specific Kubernetes version for the control plane.`)

		if properties.use == "all" || properties.use == "apiserver" {
			cmd.Flags().StringVar(&cfg.API.AdvertiseAddress, "apiserver-advertise-address", cfg.API.AdvertiseAddress, "The IP address or DNS name the API Server is accessible on.")
			cmd.Flags().Int32Var(&cfg.API.BindPort, "apiserver-bind-port", cfg.API.BindPort, "The port the API Server is accessible on.")
			cmd.Flags().StringVar(&cfg.Networking.ServiceSubnet, "service-cidr", cfg.Networking.ServiceSubnet, "The range of IP address used for service VIPs.")
			cmd.Flags().StringVar(&featureGatesString, "feature-gates", featureGatesString, "A set of key=value pairs that describe feature gates for various features. "+
				"Options are:\n"+strings.Join(features.KnownFeatures(&features.InitFeatureGates), "\n"))
		}

		if properties.use == "all" || properties.use == "controller-manager" {
			cmd.Flags().StringVar(&cfg.Networking.PodSubnet, "pod-network-cidr", cfg.Networking.PodSubnet, "The range of IP addresses used for the pod network.")
		}

		cmd.Flags().StringVar(&cfgPath, "config", cfgPath, "Path to a kubeadm config file. WARNING: Usage of a configuration file is experimental!")

		subCmds = append(subCmds, cmd)
	}

	return subCmds
}

// runCmdControlPlane creates a cobra.Command Run function, by composing the call to the given cmdFunc with necessary additional steps (e.g preparation of input parameters)
func runCmdControlPlane(cmdFunc func(outDir string, cfg *kubeadmapi.MasterConfiguration) error, outDir, cfgPath *string, featureGatesString *string, cfg *kubeadmapiext.MasterConfiguration) func(cmd *cobra.Command, args []string) {

	// the following statement build a clousure that wraps a call to a cmdFunc, binding
	// the function itself with the specific parameters of each sub command.
	// Please note that specific parameter should be passed as value, while other parameters - passed as reference -
	// are shared between sub commands and gets access to current value e.g. flags value.
	return func(cmd *cobra.Command, args []string) {
		var err error
		if err = validation.ValidateMixedArguments(cmd.Flags()); err != nil {
			kubeadmutil.CheckErr(err)
		}

		if cfg.FeatureGates, err = features.NewFeatureGate(&features.InitFeatureGates, *featureGatesString); err != nil {
			kubeadmutil.CheckErr(err)
		}

		// This call returns the ready-to-use configuration based on the configuration file that might or might not exist and the default cfg populated by flags
		internalcfg, err := configutil.ConfigFileAndDefaultsToInternalConfig(*cfgPath, cfg)
		kubeadmutil.CheckErr(err)

		if err := features.ValidateVersion(features.InitFeatureGates, internalcfg.FeatureGates, internalcfg.KubernetesVersion); err != nil {
			kubeadmutil.CheckErr(err)
		}

		// Execute the cmdFunc
		err = cmdFunc(*outDir, internalcfg)
		kubeadmutil.CheckErr(err)
	}
}
