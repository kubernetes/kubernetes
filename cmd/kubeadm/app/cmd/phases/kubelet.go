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
	"io/ioutil"

	"github.com/spf13/cobra"

	"k8s.io/apimachinery/pkg/runtime"
	kubeadmapiext "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha1"
	cmdutil "k8s.io/kubernetes/cmd/kubeadm/app/cmd/util"
	"k8s.io/kubernetes/cmd/kubeadm/app/features"
	kubeletphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/kubelet"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	configutil "k8s.io/kubernetes/cmd/kubeadm/app/util/config"
	kubeconfigutil "k8s.io/kubernetes/cmd/kubeadm/app/util/kubeconfig"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	nodeutil "k8s.io/kubernetes/pkg/util/node"
	"k8s.io/kubernetes/pkg/util/normalizer"
)

var (
	kubeletWriteInitConfigLongDesc = normalizer.LongDesc(`
		Writes init kubelet configuration to disk for dynamic kubelet configuration feature.
		Please note that the kubelet configuration can be passed to kubeadm as a value into the master configuration file.
		` + cmdutil.AlphaDisclaimer)

	kubeletWriteInitConfigExample = normalizer.Examples(`
		# Writes init kubelet configuration to disk.
		kubeadm alpha phase kubelet init
		`)

	kubeletUploadDynamicConfigLongDesc = normalizer.LongDesc(`
		Uploads dynamic kubelet configuration as ConfigMap and links it to the current node as ConfigMapRef.
		Please note that the kubelet configuration can be passed to kubeadm as a value into the master configuration file.
		` + cmdutil.AlphaDisclaimer)

	kubeletUploadDynamicConfigExample = normalizer.Examples(`
		# Uploads dynamic kubelet configuration as ConfigMap.
		kubeadm alpha phase kubelet upload
		`)

	kubeletEnableDynamicConfigLongDesc = normalizer.LongDesc(`
		Enables or updates dynamic kubelet configuration on node. This should be run on nodes.
		Please note that the kubelet configuration can be passed to kubeadm as a value into the master configuration file.
		` + cmdutil.AlphaDisclaimer)

	kubeletEnableDynamicConfigExample = normalizer.Examples(`
		# Enables dynamic kubelet configuration on node.
		kubeadm alpha phase kubelet enable
		`)
)

// NewCmdKubelet returns main command for Kubelet phase
func NewCmdKubelet() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "kubelet",
		Short: "Adopts dynamic kubelet configuration.",
		Long:  cmdutil.MacroCommandLongDescription,
	}

	cmd.AddCommand(NewCmdKubeletWriteInitConfig())
	cmd.AddCommand(NewCmdKubeletUploadDynamicConfig())
	cmd.AddCommand(NewCmdKubeletEnableDynamicConfig())

	return cmd
}

// NewCmdKubeletWriteInitConfig calls cobra.Command for writing init kubelet configuration
func NewCmdKubeletWriteInitConfig() *cobra.Command {
	var cfgPath string
	cmd := &cobra.Command{
		Use:     "init",
		Short:   "Writes init kubelet configuration to disk",
		Long:    kubeletWriteInitConfigLongDesc,
		Example: kubeletWriteInitConfigExample,
		Run: func(cmd *cobra.Command, args []string) {
			cfg := &kubeadmapiext.MasterConfiguration{
				// KubernetesVersion is not used by kubelet init, but we set this explicitly to avoid
				// the lookup of the version from the internet when executing ConfigFileAndDefaultsToInternalConfig
				KubernetesVersion: "v1.9.0",
			}
			legacyscheme.Scheme.Default(cfg)

			// This call returns the ready-to-use configuration based on the configuration file that might or might not exist and the default cfg populated by flags
			internalcfg, err := configutil.ConfigFileAndDefaultsToInternalConfig(cfgPath, cfg)
			kubeadmutil.CheckErr(err)
			if features.Enabled(internalcfg.FeatureGates, features.DynamicKubeletConfig) {
				err = kubeletphase.WriteInitKubeletConfigToDiskOnMaster(internalcfg)
				kubeadmutil.CheckErr(err)
			} else {
				fmt.Println("[kubelet] feature gate DynamicKubeletConfig is not enabled, do nothing.")
			}
		},
	}

	cmd.Flags().StringVar(&cfgPath, "config", cfgPath, "Path to kubeadm config file (WARNING: Usage of a configuration file is experimental)")

	return cmd
}

// NewCmdKubeletUploadDynamicConfig calls cobra.Command for uploading dynamic kubelet configuration
func NewCmdKubeletUploadDynamicConfig() *cobra.Command {
	var cfgPath, kubeConfigFile string

	cmd := &cobra.Command{
		Use:     "upload",
		Short:   "Uploads dynamic kubelet configuration as ConfigMap",
		Long:    kubeletUploadDynamicConfigLongDesc,
		Example: kubeletUploadDynamicConfigExample,
		Run: func(cmd *cobra.Command, args []string) {
			cfg := &kubeadmapiext.MasterConfiguration{
				// KubernetesVersion is not used by kubelet upload, but we set this explicitly to avoid
				// the lookup of the version from the internet when executing ConfigFileAndDefaultsToInternalConfig
				KubernetesVersion: "v1.9.0",
			}
			legacyscheme.Scheme.Default(cfg)

			// This call returns the ready-to-use configuration based on the configuration file that might or might not exist and the default cfg populated by flags
			internalcfg, err := configutil.ConfigFileAndDefaultsToInternalConfig(cfgPath, cfg)
			kubeadmutil.CheckErr(err)
			if features.Enabled(internalcfg.FeatureGates, features.DynamicKubeletConfig) {
				client, err := kubeconfigutil.ClientSetFromFile(kubeConfigFile)
				kubeadmutil.CheckErr(err)
				err = kubeletphase.CreateBaseKubeletConfiguration(internalcfg, client)
				kubeadmutil.CheckErr(err)
			} else {
				fmt.Println("[kubelet] feature gate DynamicKubeletConfig is not enabled, do nothing.")
			}
		},
	}

	cmd.Flags().StringVar(&cfgPath, "config", cfgPath, "Path to kubeadm config file (WARNING: Usage of a configuration file is experimental)")
	cmd.Flags().StringVar(&kubeConfigFile, "kubeconfig", "/etc/kubernetes/admin.conf", "The KubeConfig file to use when talking to the cluster")

	return cmd
}

// NewCmdKubeletEnableDynamicConfig calls cobra.Command for enabling dynamic kubelet configuration on node
func NewCmdKubeletEnableDynamicConfig() *cobra.Command {
	cfg := &kubeadmapiext.NodeConfiguration{}
	legacyscheme.Scheme.Default(cfg)

	var cfgPath string
	cmd := &cobra.Command{
		Use:     "enable",
		Aliases: []string{"update"},
		Short:   "Enables or updates dynamic kubelet configuration on node",
		Long:    kubeletEnableDynamicConfigLongDesc,
		Example: kubeletEnableDynamicConfigExample,
		Run: func(cmd *cobra.Command, args []string) {
			nodeName, err := getNodeName(cfgPath, cfg)
			kubeadmutil.CheckErr(err)
			if features.Enabled(cfg.FeatureGates, features.DynamicKubeletConfig) {
				err = kubeletphase.ConsumeBaseKubeletConfiguration(nodeName)
				kubeadmutil.CheckErr(err)
			} else {
				fmt.Println("[kubelet] feature gate DynamicKubeletConfig is not enabled, do nothing.")
			}
		},
	}

	cmd.Flags().StringVar(&cfgPath, "config", cfgPath, "Path to kubeadm config file (WARNING: Usage of a configuration file is experimental)")
	cmd.Flags().StringVar(&cfg.NodeName, "node-name", cfg.NodeName, "Name of the node that should enable the dynamic kubelet configuration")

	return cmd
}

func getNodeName(cfgPath string, cfg *kubeadmapiext.NodeConfiguration) (string, error) {
	if cfgPath != "" {
		b, err := ioutil.ReadFile(cfgPath)
		if err != nil {
			return "", fmt.Errorf("unable to read config from %q [%v]", cfgPath, err)
		}
		if err := runtime.DecodeInto(legacyscheme.Codecs.UniversalDecoder(), b, cfg); err != nil {
			return "", fmt.Errorf("unable to decode config from %q [%v]", cfgPath, err)
		}
	}

	if cfg.NodeName == "" {
		cfg.NodeName = nodeutil.GetHostname("")
	}

	return cfg.NodeName, nil
}
