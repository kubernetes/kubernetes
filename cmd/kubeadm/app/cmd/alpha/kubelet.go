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
	"github.com/pkg/errors"
	"github.com/spf13/cobra"
	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/options"
	cmdutil "k8s.io/kubernetes/cmd/kubeadm/app/cmd/util"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeletphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/kubelet"
	"k8s.io/kubernetes/cmd/kubeadm/app/preflight"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	kubeconfigutil "k8s.io/kubernetes/cmd/kubeadm/app/util/kubeconfig"
	"k8s.io/kubernetes/pkg/util/normalizer"
	utilsexec "k8s.io/utils/exec"
)

var (
	kubeletConfigDownloadLongDesc = normalizer.LongDesc(`
		Downloads the kubelet configuration from a ConfigMap of the form "kubelet-config-1.X" in the cluster,
		where X is the minor version of the kubelet. Either kubeadm autodetects the kubelet version by exec-ing
		"kubelet --version" or respects the --kubelet-version parameter.
		` + cmdutil.AlphaDisclaimer)

	kubeletConfigDownloadExample = normalizer.Examples(`
		# Downloads the kubelet configuration from the ConfigMap in the cluster. Autodetects the kubelet version.
		kubeadm alpha phase kubelet config download

		# Downloads the kubelet configuration from the ConfigMap in the cluster. Uses a specific desired kubelet version.
		kubeadm alpha phase kubelet config download --kubelet-version v1.12.0
		`)

	kubeletConfigEnableDynamicLongDesc = normalizer.LongDesc(`
		Enables or updates dynamic kubelet configuration for a Node, against the kubelet-config-1.X ConfigMap in the cluster,
		where X is the minor version of the desired kubelet version.

		WARNING: This feature is still experimental, and disabled by default. Enable only if you know what you are doing, as it
		may have surprising side-effects at this stage.

		` + cmdutil.AlphaDisclaimer)

	kubeletConfigEnableDynamicExample = normalizer.Examples(`
		# Enables dynamic kubelet configuration for a Node.
		kubeadm alpha phase kubelet enable-dynamic-config --node-name node-1 --kubelet-version v1.12.0

		WARNING: This feature is still experimental, and disabled by default. Enable only if you know what you are doing, as it
		may have surprising side-effects at this stage.
		`)
)

// newCmdKubeletUtility returns command for `kubeadm phase kubelet`
func newCmdKubeletUtility() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "kubelet",
		Short: "Commands related to handling the kubelet",
		Long:  cmdutil.MacroCommandLongDescription,
	}

	cmd.AddCommand(newCmdKubeletConfig())
	return cmd
}

// newCmdKubeletConfig returns command for `kubeadm phase kubelet config`
func newCmdKubeletConfig() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "config",
		Short: "Utilities for kubelet configuration",
		Long:  cmdutil.MacroCommandLongDescription,
	}

	cmd.AddCommand(newCmdKubeletConfigDownload())
	cmd.AddCommand(newCmdKubeletConfigEnableDynamic())
	return cmd
}

// newCmdKubeletConfigDownload calls cobra.Command for downloading the kubelet configuration from the kubelet-config-1.X ConfigMap in the cluster
func newCmdKubeletConfigDownload() *cobra.Command {
	var kubeletVersionStr string
	// TODO: Be smarter about this and be able to load multiple kubeconfig files in different orders of precedence
	kubeConfigFile := constants.GetKubeletKubeConfigPath()

	cmd := &cobra.Command{
		Use:     "download",
		Short:   "Downloads the kubelet configuration from the cluster ConfigMap kubelet-config-1.X, where X is the minor version of the kubelet.",
		Long:    kubeletConfigDownloadLongDesc,
		Example: kubeletConfigDownloadExample,
		Run: func(cmd *cobra.Command, args []string) {
			kubeletVersion, err := getKubeletVersion(kubeletVersionStr)
			kubeadmutil.CheckErr(err)

			client, err := kubeconfigutil.ClientSetFromFile(kubeConfigFile)
			kubeadmutil.CheckErr(err)

			err = kubeletphase.DownloadConfig(client, kubeletVersion, constants.KubeletRunDirectory)
			kubeadmutil.CheckErr(err)
		},
	}

	options.AddKubeConfigFlag(cmd.Flags(), &kubeConfigFile)
	cmd.Flags().StringVar(&kubeletVersionStr, "kubelet-version", kubeletVersionStr, "The desired version for the kubelet. Defaults to being autodetected from 'kubelet --version'.")
	return cmd
}

func getKubeletVersion(kubeletVersionStr string) (*version.Version, error) {
	if len(kubeletVersionStr) > 0 {
		return version.ParseSemantic(kubeletVersionStr)
	}
	return preflight.GetKubeletVersion(utilsexec.New())
}

// newCmdKubeletConfigEnableDynamic calls cobra.Command for enabling dynamic kubelet configuration on node
// This feature is still in alpha and an experimental state
func newCmdKubeletConfigEnableDynamic() *cobra.Command {
	var nodeName, kubeletVersionStr string
	kubeConfigFile := constants.GetAdminKubeConfigPath()

	cmd := &cobra.Command{
		Use:     "enable-dynamic",
		Short:   "EXPERIMENTAL: Enables or updates dynamic kubelet configuration for a Node",
		Long:    kubeletConfigEnableDynamicLongDesc,
		Example: kubeletConfigEnableDynamicExample,
		Run: func(cmd *cobra.Command, args []string) {
			if len(nodeName) == 0 {
				kubeadmutil.CheckErr(errors.New("The --node-name argument is required"))
			}
			if len(kubeletVersionStr) == 0 {
				kubeadmutil.CheckErr(errors.New("The --kubelet-version argument is required"))
			}

			kubeletVersion, err := version.ParseSemantic(kubeletVersionStr)
			kubeadmutil.CheckErr(err)

			kubeConfigFile = cmdutil.FindExistingKubeConfig(kubeConfigFile)
			client, err := kubeconfigutil.ClientSetFromFile(kubeConfigFile)
			kubeadmutil.CheckErr(err)

			err = kubeletphase.EnableDynamicConfigForNode(client, nodeName, kubeletVersion)
			kubeadmutil.CheckErr(err)
		},
	}

	options.AddKubeConfigFlag(cmd.Flags(), &kubeConfigFile)
	cmd.Flags().StringVar(&nodeName, "node-name", nodeName, "Name of the node that should enable the dynamic kubelet configuration")
	cmd.Flags().StringVar(&kubeletVersionStr, "kubelet-version", kubeletVersionStr, "The desired version for the kubelet")
	return cmd
}
