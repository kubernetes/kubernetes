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

	"github.com/spf13/cobra"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiv1alpha3 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha3"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/options"
	cmdutil "k8s.io/kubernetes/cmd/kubeadm/app/cmd/util"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeletphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/kubelet"
	patchnodephase "k8s.io/kubernetes/cmd/kubeadm/app/phases/patchnode"
	"k8s.io/kubernetes/cmd/kubeadm/app/preflight"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	configutil "k8s.io/kubernetes/cmd/kubeadm/app/util/config"
	kubeconfigutil "k8s.io/kubernetes/cmd/kubeadm/app/util/kubeconfig"
	"k8s.io/kubernetes/pkg/util/normalizer"
	"k8s.io/kubernetes/pkg/util/version"
	utilsexec "k8s.io/utils/exec"
)

var (
	kubeletWriteEnvFileLongDesc = normalizer.LongDesc(`
		Writes an environment file with flags that should be passed to the kubelet executing on the master or node.
		This --config flag can either consume a InitConfiguration object or a JoinConfiguration one, as this
		function is used for both "kubeadm init" and "kubeadm join".
		` + cmdutil.AlphaDisclaimer)

	kubeletWriteEnvFileExample = normalizer.Examples(`
		# Writes a dynamic environment file with kubelet flags from a InitConfiguration file.
		kubeadm alpha phase kubelet write-env-file --config masterconfig.yaml

		# Writes a dynamic environment file with kubelet flags from a JoinConfiguration file.
		kubeadm alpha phase kubelet write-env-file --config nodeconfig.yaml
		`)

	kubeletConfigUploadLongDesc = normalizer.LongDesc(`
		Uploads kubelet configuration extracted from the kubeadm InitConfiguration object to a ConfigMap
		of the form kubelet-config-1.X in the cluster, where X is the minor version of the current (API Server) Kubernetes version.
		` + cmdutil.AlphaDisclaimer)

	kubeletConfigUploadExample = normalizer.Examples(`
		# Uploads the kubelet configuration from the kubeadm Config file to a ConfigMap in the cluster.
		kubeadm alpha phase kubelet config upload --config kubeadm.yaml
		`)

	kubeletConfigAnnotateCRILongDesc = normalizer.LongDesc(`
		Adds an annotation to the current node with the CRI socket specified in the kubeadm InitConfiguration object.
		` + cmdutil.AlphaDisclaimer)

	kubeletConfigAnnotateCRIExample = normalizer.Examples(`
		kubeadm alpha phase kubelet config annotate-cri --config kubeadm.yaml
		`)

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

	kubeletConfigWriteToDiskLongDesc = normalizer.LongDesc(`
		Writes kubelet configuration to disk, based on the kubeadm configuration passed via "--config".
		` + cmdutil.AlphaDisclaimer)

	kubeletConfigWriteToDiskExample = normalizer.Examples(`
		# Extracts the kubelet configuration from a kubeadm configuration file
		kubeadm alpha phase kubelet config write-to-disk --config kubeadm.yaml
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

// NewCmdKubelet returns command for `kubeadm phase kubelet`
func NewCmdKubelet() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "kubelet",
		Short: "Commands related to handling the kubelet.",
		Long:  cmdutil.MacroCommandLongDescription,
	}

	cmd.AddCommand(NewCmdKubeletConfig())
	cmd.AddCommand(NewCmdKubeletWriteEnvFile())
	return cmd
}

// NewCmdKubeletWriteEnvFile calls cobra.Command for writing the dynamic kubelet env file based on a InitConfiguration or JoinConfiguration object
func NewCmdKubeletWriteEnvFile() *cobra.Command {
	var cfgPath string

	cmd := &cobra.Command{
		Use:     "write-env-file",
		Short:   "Writes an environment file with runtime flags for the kubelet.",
		Long:    kubeletWriteEnvFileLongDesc,
		Example: kubeletWriteEnvFileExample,
		Run: func(cmd *cobra.Command, args []string) {
			if len(cfgPath) == 0 {
				kubeadmutil.CheckErr(fmt.Errorf("The --config flag is mandatory"))
			}

			err := RunKubeletWriteEnvFile(cfgPath)
			kubeadmutil.CheckErr(err)
		},
	}

	options.AddConfigFlag(cmd.Flags(), &cfgPath)
	return cmd
}

// RunKubeletWriteEnvFile is the function that is run when "kubeadm phase kubelet write-env-file" is executed
func RunKubeletWriteEnvFile(cfgPath string) error {
	internalcfg, err := configutil.AnyConfigFileAndDefaultsToInternal(cfgPath)
	if err != nil {
		return err
	}

	var nodeRegistrationObj *kubeadmapi.NodeRegistrationOptions
	var featureGates map[string]bool
	var registerWithTaints bool

	switch cfg := internalcfg.(type) {
	case *kubeadmapi.InitConfiguration:
		nodeRegistrationObj = &cfg.NodeRegistration
		featureGates = cfg.FeatureGates
		registerWithTaints = false
	case *kubeadmapi.JoinConfiguration:
		nodeRegistrationObj = &cfg.NodeRegistration
		featureGates = cfg.FeatureGates
		registerWithTaints = true
	default:
		return fmt.Errorf("couldn't read config file, no matching kind found")
	}

	if err := kubeletphase.WriteKubeletDynamicEnvFile(nodeRegistrationObj, featureGates, registerWithTaints, constants.KubeletRunDirectory); err != nil {
		return fmt.Errorf("error writing a dynamic environment file for the kubelet: %v", err)
	}
	return nil
}

// NewCmdKubeletConfig returns command for `kubeadm phase kubelet config`
func NewCmdKubeletConfig() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "config",
		Short: "Handles kubelet configuration.",
		Long:  cmdutil.MacroCommandLongDescription,
	}

	cmd.AddCommand(NewCmdKubeletConfigUpload())
	cmd.AddCommand(NewCmdKubeletAnnotateCRI())
	cmd.AddCommand(NewCmdKubeletConfigDownload())
	cmd.AddCommand(NewCmdKubeletConfigWriteToDisk())
	cmd.AddCommand(NewCmdKubeletConfigEnableDynamic())
	return cmd
}

// NewCmdKubeletConfigUpload calls cobra.Command for uploading dynamic kubelet configuration
func NewCmdKubeletConfigUpload() *cobra.Command {
	cfg := &kubeadmapiv1alpha3.InitConfiguration{}
	var cfgPath string
	kubeConfigFile := constants.GetAdminKubeConfigPath()

	cmd := &cobra.Command{
		Use:     "upload",
		Short:   "Uploads kubelet configuration to a ConfigMap based on a kubeadm InitConfiguration file.",
		Long:    kubeletConfigUploadLongDesc,
		Example: kubeletConfigUploadExample,
		Run: func(cmd *cobra.Command, args []string) {
			if len(cfgPath) == 0 {
				kubeadmutil.CheckErr(fmt.Errorf("The --config argument is required"))
			}

			// KubernetesVersion is not used, but we set it explicitly to avoid the lookup
			// of the version from the internet when executing ConfigFileAndDefaultsToInternalConfig
			err := SetKubernetesVersion(nil, cfg)
			kubeadmutil.CheckErr(err)

			// This call returns the ready-to-use configuration based on the configuration file
			internalcfg, err := configutil.ConfigFileAndDefaultsToInternalConfig(cfgPath, cfg)
			kubeadmutil.CheckErr(err)

			kubeConfigFile = cmdutil.FindExistingKubeConfig(kubeConfigFile)
			client, err := kubeconfigutil.ClientSetFromFile(kubeConfigFile)
			kubeadmutil.CheckErr(err)

			err = kubeletphase.CreateConfigMap(internalcfg, client)
			kubeadmutil.CheckErr(err)
		},
	}

	options.AddKubeConfigFlag(cmd.Flags(), &kubeConfigFile)
	options.AddConfigFlag(cmd.Flags(), &cfgPath)
	return cmd
}

// NewCmdKubeletAnnotateCRI calls cobra.Command for annotating the node with the given crisocket
func NewCmdKubeletAnnotateCRI() *cobra.Command {
	cfg := &kubeadmapiv1alpha3.InitConfiguration{}
	var cfgPath string
	kubeConfigFile := constants.GetAdminKubeConfigPath()

	cmd := &cobra.Command{
		Use:     "annotate-cri",
		Short:   "annotates the node with the given crisocket",
		Long:    kubeletConfigAnnotateCRILongDesc,
		Example: kubeletConfigAnnotateCRIExample,
		Run: func(cmd *cobra.Command, args []string) {
			if len(cfgPath) == 0 {
				kubeadmutil.CheckErr(fmt.Errorf("The --config argument is required"))
			}

			// KubernetesVersion is not used, but we set it explicitly to avoid the lookup
			// of the version from the internet when executing ConfigFileAndDefaultsToInternalConfig
			err := SetKubernetesVersion(nil, cfg)
			kubeadmutil.CheckErr(err)

			// This call returns the ready-to-use configuration based on the configuration file
			internalcfg, err := configutil.ConfigFileAndDefaultsToInternalConfig(cfgPath, cfg)
			kubeadmutil.CheckErr(err)

			kubeConfigFile = cmdutil.FindExistingKubeConfig(kubeConfigFile)
			client, err := kubeconfigutil.ClientSetFromFile(kubeConfigFile)
			kubeadmutil.CheckErr(err)

			err = patchnodephase.AnnotateCRISocket(client, internalcfg.NodeRegistration.Name, internalcfg.NodeRegistration.CRISocket)
			kubeadmutil.CheckErr(err)
		},
	}

	options.AddKubeConfigFlag(cmd.Flags(), &kubeConfigFile)
	options.AddConfigFlag(cmd.Flags(), &cfgPath)
	return cmd
}

// NewCmdKubeletConfigDownload calls cobra.Command for downloading the kubelet configuration from the kubelet-config-1.X ConfigMap in the cluster
func NewCmdKubeletConfigDownload() *cobra.Command {
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

// NewCmdKubeletConfigWriteToDisk calls cobra.Command for writing init kubelet configuration
func NewCmdKubeletConfigWriteToDisk() *cobra.Command {
	cfg := &kubeadmapiv1alpha3.InitConfiguration{}
	var cfgPath string
	cmd := &cobra.Command{
		Use:     "write-to-disk",
		Short:   "Writes kubelet configuration to disk, either based on the --config argument.",
		Long:    kubeletConfigWriteToDiskLongDesc,
		Example: kubeletConfigWriteToDiskExample,
		Run: func(cmd *cobra.Command, args []string) {
			if len(cfgPath) == 0 {
				kubeadmutil.CheckErr(fmt.Errorf("The --config argument is required"))
			}

			// KubernetesVersion is not used, but we set it explicitly to avoid the lookup
			// of the version from the internet when executing ConfigFileAndDefaultsToInternalConfig
			err := SetKubernetesVersion(nil, cfg)
			kubeadmutil.CheckErr(err)

			// This call returns the ready-to-use configuration based on the configuration file
			internalcfg, err := configutil.ConfigFileAndDefaultsToInternalConfig(cfgPath, cfg)
			kubeadmutil.CheckErr(err)

			err = kubeletphase.WriteConfigToDisk(internalcfg.ComponentConfigs.Kubelet, constants.KubeletRunDirectory)
			kubeadmutil.CheckErr(err)
		},
	}

	options.AddConfigFlag(cmd.Flags(), &cfgPath)
	return cmd
}

// NewCmdKubeletConfigEnableDynamic calls cobra.Command for enabling dynamic kubelet configuration on node
// This feature is still in alpha and an experimental state
func NewCmdKubeletConfigEnableDynamic() *cobra.Command {
	var nodeName, kubeletVersionStr string
	kubeConfigFile := constants.GetAdminKubeConfigPath()

	cmd := &cobra.Command{
		Use:     "enable-dynamic",
		Short:   "EXPERIMENTAL: Enables or updates dynamic kubelet configuration for a Node",
		Long:    kubeletConfigEnableDynamicLongDesc,
		Example: kubeletConfigEnableDynamicExample,
		Run: func(cmd *cobra.Command, args []string) {
			if len(nodeName) == 0 {
				kubeadmutil.CheckErr(fmt.Errorf("The --node-name argument is required"))
			}
			if len(kubeletVersionStr) == 0 {
				kubeadmutil.CheckErr(fmt.Errorf("The --kubelet-version argument is required"))
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
