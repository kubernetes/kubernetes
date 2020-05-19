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
	"fmt"

	"github.com/pkg/errors"
	"github.com/spf13/cobra"
	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/options"
	cmdutil "k8s.io/kubernetes/cmd/kubeadm/app/cmd/util"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeletphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/kubelet"
	kubeconfigutil "k8s.io/kubernetes/cmd/kubeadm/app/util/kubeconfig"
)

var (
	kubeletConfigEnableDynamicLongDesc = cmdutil.LongDesc(`
		Enable or update dynamic kubelet configuration for a Node, against the kubelet-config-1.X ConfigMap in the cluster,
		where X is the minor version of the desired kubelet version.

		WARNING: This feature is still experimental, and disabled by default. Enable only if you know what you are doing, as it
		may have surprising side-effects at this stage.

		` + cmdutil.AlphaDisclaimer)

	kubeletConfigEnableDynamicExample = cmdutil.Examples(fmt.Sprintf(`
		# Enable dynamic kubelet configuration for a Node.
		kubeadm alpha phase kubelet enable-dynamic-config --node-name node-1 --kubelet-version %s

		WARNING: This feature is still experimental, and disabled by default. Enable only if you know what you are doing, as it
		may have surprising side-effects at this stage.
		`, constants.CurrentKubernetesVersion))
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

	cmd.AddCommand(newCmdKubeletConfigEnableDynamic())
	return cmd
}

// newCmdKubeletConfigEnableDynamic calls cobra.Command for enabling dynamic kubelet configuration on node
// This feature is still in alpha and an experimental state
func newCmdKubeletConfigEnableDynamic() *cobra.Command {
	var nodeName, kubeletVersionStr string
	var kubeConfigFile string

	cmd := &cobra.Command{
		Use:     "enable-dynamic",
		Short:   "EXPERIMENTAL: Enable or update dynamic kubelet configuration for a Node",
		Long:    kubeletConfigEnableDynamicLongDesc,
		Example: kubeletConfigEnableDynamicExample,
		RunE: func(cmd *cobra.Command, args []string) error {
			if len(nodeName) == 0 {
				return errors.New("the --node-name argument is required")
			}
			if len(kubeletVersionStr) == 0 {
				return errors.New("the --kubelet-version argument is required")
			}

			kubeletVersion, err := version.ParseSemantic(kubeletVersionStr)
			if err != nil {
				return err
			}

			kubeConfigFile = cmdutil.GetKubeConfigPath(kubeConfigFile)
			client, err := kubeconfigutil.ClientSetFromFile(kubeConfigFile)
			if err != nil {
				return err
			}

			return kubeletphase.EnableDynamicConfigForNode(client, nodeName, kubeletVersion)
		},
	}

	options.AddKubeConfigFlag(cmd.Flags(), &kubeConfigFile)
	cmd.Flags().StringVar(&nodeName, options.NodeName, nodeName, "Name of the node that should enable the dynamic kubelet configuration")
	cmd.Flags().StringVar(&kubeletVersionStr, "kubelet-version", kubeletVersionStr, "The desired version for the kubelet")
	return cmd
}
