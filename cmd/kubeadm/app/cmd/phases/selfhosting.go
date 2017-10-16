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
	"os"
	"strings"
	"time"

	"github.com/spf13/cobra"

	kubeadmapiext "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha1"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/validation"
	cmdutil "k8s.io/kubernetes/cmd/kubeadm/app/cmd/util"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/features"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/selfhosting"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/apiclient"
	configutil "k8s.io/kubernetes/cmd/kubeadm/app/util/config"
	kubeconfigutil "k8s.io/kubernetes/cmd/kubeadm/app/util/kubeconfig"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
)

// NewCmdSelfhosting returns the self-hosting Cobra command
func NewCmdSelfhosting() *cobra.Command {
	cmd := &cobra.Command{
		Use:     "selfhosting",
		Aliases: []string{"selfhosted"},
		Short:   "Make a kubeadm cluster self-hosted.",
		RunE:    cmdutil.SubCmdRunE("selfhosting"),
	}

	cmd.AddCommand(getSelfhostingSubCommand())
	return cmd
}

// getSelfhostingSubCommand returns sub commands for Selfhosting phase
func getSelfhostingSubCommand() *cobra.Command {

	cfg := &kubeadmapiext.MasterConfiguration{}
	// Default values for the cobra help text
	legacyscheme.Scheme.Default(cfg)

	var cfgPath, kubeConfigFile, featureGatesString string

	// Creates the UX Command
	cmd := &cobra.Command{
		Use:     "convert-from-staticpods",
		Aliases: []string{"from-staticpods"},
		Short:   "Converts a Static Pod-hosted control plane into a self-hosted one.",
		Run: func(cmd *cobra.Command, args []string) {
			var err error
			if cfg.FeatureGates, err = features.NewFeatureGate(&features.InitFeatureGates, featureGatesString); err != nil {
				kubeadmutil.CheckErr(err)
			}

			if err := validation.ValidateMixedArguments(cmd.Flags()); err != nil {
				kubeadmutil.CheckErr(err)
			}

			// This call returns the ready-to-use configuration based on the configuration file that might or might not exist and the default cfg populated by flags
			internalcfg, err := configutil.ConfigFileAndDefaultsToInternalConfig(cfgPath, cfg)
			kubeadmutil.CheckErr(err)

			// Gets the kubernetes client
			client, err := kubeconfigutil.ClientSetFromFile(kubeConfigFile)
			kubeadmutil.CheckErr(err)

			// Converts the Static Pod-hosted control plane into a self-hosted one
			waiter := apiclient.NewKubeWaiter(client, 2*time.Minute, os.Stdout)
			err = selfhosting.CreateSelfHostedControlPlane(constants.GetStaticPodDirectory(), constants.KubernetesDir, internalcfg, client, waiter)
			kubeadmutil.CheckErr(err)
		},
	}

	// Add flags to the command
	// flags bound to the configuration object
	cmd.Flags().StringVar(&cfg.CertificatesDir, "cert-dir", cfg.CertificatesDir, `The path where certificates are stored`)
	cmd.Flags().StringVar(&cfgPath, "config", cfgPath, "Path to kubeadm config file (WARNING: Usage of a configuration file is experimental)")
	cmd.Flags().StringVar(&featureGatesString, "feature-gates", featureGatesString, "A set of key=value pairs that describe feature gates for various features."+
		"Options are:\n"+strings.Join(features.KnownFeatures(&features.InitFeatureGates), "\n"))

	// flags that are not bound to the configuration object
	// Note: All flags that are not bound to the cfg object should be whitelisted in cmd/kubeadm/app/apis/kubeadm/validation/validation.go
	cmd.Flags().StringVar(&kubeConfigFile, "kubeconfig", "/etc/kubernetes/admin.conf", "The KubeConfig file to use for talking to the cluster")

	return cmd
}
