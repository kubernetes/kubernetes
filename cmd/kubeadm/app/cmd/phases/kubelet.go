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

package phases

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"

	kubeadmapiv1alpha2 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha2"
	cmdutil "k8s.io/kubernetes/cmd/kubeadm/app/cmd/util"
	kubeletphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/kubelet"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	configutil "k8s.io/kubernetes/cmd/kubeadm/app/util/config"
	kubeconfigutil "k8s.io/kubernetes/cmd/kubeadm/app/util/kubeconfig"
	"k8s.io/kubernetes/pkg/util/normalizer"
	"k8s.io/kubernetes/pkg/util/version"
)

var (
	kubeletWriteConfigToDiskLongDesc = normalizer.LongDesc(`
		Writes kubelet configuration to disk, either based on the kubelet-config-1.X ConfigMap in the cluster, or from the
		configuration passed to the command via "--config".
		` + cmdutil.AlphaDisclaimer)

	kubeletWriteConfigToDiskExample = normalizer.Examples(`
		# Writes kubelet configuration for a node to disk. The information is fetched from the cluster ConfigMap
		kubeadm alpha phase kubelet write-config-to-disk --kubelet-version v1.11.0 --kubeconfig /etc/kubernetes/kubelet.conf

		# Writes kubelet configuration down to disk, based on the configuration flag passed to --config
		kubeadm alpha phase kubelet write-config-to-disk --kubelet-version v1.11.0 --config kubeadm.yaml
		`)

	kubeletUploadDynamicConfigLongDesc = normalizer.LongDesc(`
		Uploads kubelet configuration extracted from the kubeadm MasterConfiguration object to a ConfigMap
		of the form kubelet-config-1.X in the cluster, where X is the minor version of the current Kubernetes version
		` + cmdutil.AlphaDisclaimer)

	kubeletUploadDynamicConfigExample = normalizer.Examples(`
		# Uploads the kubelet configuration from the kubeadm Config file to a ConfigMap in the cluster.
		kubeadm alpha phase kubelet upload-config --config kubeadm.yaml
		`)

	kubeletEnableDynamicConfigLongDesc = normalizer.LongDesc(`
		Enables or updates dynamic kubelet configuration for a Node, against the kubelet-config-1.X ConfigMap in the cluster,
		where X is the minor version of the desired kubelet version.

		WARNING: This feature is still experimental, and disabled by default. Enable only if you know what you are doing, as it
		may have surprising side-effects at this stage.

		` + cmdutil.AlphaDisclaimer)

	kubeletEnableDynamicConfigExample = normalizer.Examples(`
		# Enables dynamic kubelet configuration for a Node.
		kubeadm alpha phase kubelet enable-dynamic-config --node-name node-1 --kubelet-version v1.11.0

		WARNING: This feature is still experimental, and disabled by default. Enable only if you know what you are doing, as it
		may have surprising side-effects at this stage.
		`)
)

// NewCmdKubelet returns main command for Kubelet phase
func NewCmdKubelet() *cobra.Command {
	var kubeConfigFile string
	cmd := &cobra.Command{
		Use:   "kubelet",
		Short: "Handles kubelet configuration.",
		Long:  cmdutil.MacroCommandLongDescription,
	}

	cmd.PersistentFlags().StringVar(&kubeConfigFile, "kubeconfig", "/etc/kubernetes/admin.conf", "The KubeConfig file to use when talking to the cluster")

	cmd.AddCommand(NewCmdKubeletWriteConfigToDisk(&kubeConfigFile))
	cmd.AddCommand(NewCmdKubeletUploadConfig(&kubeConfigFile))
	cmd.AddCommand(NewCmdKubeletEnableDynamicConfig(&kubeConfigFile))
	return cmd
}

// NewCmdKubeletUploadConfig calls cobra.Command for uploading dynamic kubelet configuration
func NewCmdKubeletUploadConfig(kubeConfigFile *string) *cobra.Command {
	var cfgPath string

	cmd := &cobra.Command{
		Use:     "upload-config",
		Short:   "Uploads kubelet configuration to a ConfigMap",
		Long:    kubeletUploadDynamicConfigLongDesc,
		Example: kubeletUploadDynamicConfigExample,
		Run: func(cmd *cobra.Command, args []string) {
			if len(cfgPath) == 0 {
				kubeadmutil.CheckErr(fmt.Errorf("The --config argument is required"))
			}

			// This call returns the ready-to-use configuration based on the configuration file
			internalcfg, err := configutil.ConfigFileAndDefaultsToInternalConfig(cfgPath, &kubeadmapiv1alpha2.MasterConfiguration{})
			kubeadmutil.CheckErr(err)

			client, err := kubeconfigutil.ClientSetFromFile(*kubeConfigFile)
			kubeadmutil.CheckErr(err)

			err = kubeletphase.CreateConfigMap(internalcfg, client)
			kubeadmutil.CheckErr(err)
		},
	}

	cmd.Flags().StringVar(&cfgPath, "config", cfgPath, "Path to kubeadm config file (WARNING: Usage of a configuration file is experimental)")
	return cmd
}

// NewCmdKubeletWriteConfigToDisk calls cobra.Command for writing init kubelet configuration
func NewCmdKubeletWriteConfigToDisk(kubeConfigFile *string) *cobra.Command {
	var cfgPath, kubeletVersionStr string
	cmd := &cobra.Command{
		Use:     "write-config-to-disk",
		Short:   "Writes kubelet configuration to disk, either based on the --config argument or the kubeadm-config ConfigMap.",
		Long:    kubeletWriteConfigToDiskLongDesc,
		Example: kubeletWriteConfigToDiskExample,
		Run: func(cmd *cobra.Command, args []string) {
			if len(kubeletVersionStr) == 0 {
				kubeadmutil.CheckErr(fmt.Errorf("The --kubelet-version argument is required"))
			}

			client, err := kubeconfigutil.ClientSetFromFile(*kubeConfigFile)
			kubeadmutil.CheckErr(err)

			// This call returns the ready-to-use configuration based on the configuration file
			internalcfg, err := configutil.FetchConfigFromFileOrCluster(client, os.Stdout, "kubelet", cfgPath)
			kubeadmutil.CheckErr(err)

			err = kubeletphase.WriteConfigToDisk(internalcfg.KubeletConfiguration.BaseConfig)
			kubeadmutil.CheckErr(err)
		},
	}

	cmd.Flags().StringVar(&kubeletVersionStr, "kubelet-version", kubeletVersionStr, "The desired version for the kubelet")
	cmd.Flags().StringVar(&cfgPath, "config", cfgPath, "Path to kubeadm config file (WARNING: Usage of a configuration file is experimental)")
	return cmd
}

// NewCmdKubeletEnableDynamicConfig calls cobra.Command for enabling dynamic kubelet configuration on node
// This feature is still in alpha and an experimental state
func NewCmdKubeletEnableDynamicConfig(kubeConfigFile *string) *cobra.Command {
	var nodeName, kubeletVersionStr string

	cmd := &cobra.Command{
		Use:     "enable-dynamic-config",
		Short:   "EXPERIMENTAL: Enables or updates dynamic kubelet configuration for a Node",
		Long:    kubeletEnableDynamicConfigLongDesc,
		Example: kubeletEnableDynamicConfigExample,
		Run: func(cmd *cobra.Command, args []string) {
			if len(nodeName) == 0 {
				kubeadmutil.CheckErr(fmt.Errorf("The --node-name argument is required"))
			}
			if len(kubeletVersionStr) == 0 {
				kubeadmutil.CheckErr(fmt.Errorf("The --kubelet-version argument is required"))
			}

			kubeletVersion, err := version.ParseSemantic(kubeletVersionStr)
			kubeadmutil.CheckErr(err)

			client, err := kubeconfigutil.ClientSetFromFile(*kubeConfigFile)
			kubeadmutil.CheckErr(err)

			err = kubeletphase.EnableDynamicConfigForNode(client, nodeName, kubeletVersion)
			kubeadmutil.CheckErr(err)
		},
	}

	cmd.Flags().StringVar(&nodeName, "node-name", nodeName, "Name of the node that should enable the dynamic kubelet configuration")
	cmd.Flags().StringVar(&kubeletVersionStr, "kubelet-version", kubeletVersionStr, "The desired version for the kubelet")
	return cmd
}
