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
	"fmt"

	"github.com/spf13/cobra"

	kubeadmapiext "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha1"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/uploadconfig"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	configutil "k8s.io/kubernetes/cmd/kubeadm/app/util/config"
	kubeconfigutil "k8s.io/kubernetes/cmd/kubeadm/app/util/kubeconfig"
)

// NewCmdUploadConfig returns the Cobra command for running the uploadconfig phase
func NewCmdUploadConfig() *cobra.Command {
	var cfgPath, kubeConfigFile string
	cmd := &cobra.Command{
		Use:     "upload-config",
		Short:   "Upload the currently used configuration for kubeadm to a ConfigMap in the cluster for future use in reconfiguration and upgrades of the cluster.",
		Aliases: []string{"uploadconfig"},
		Run: func(_ *cobra.Command, args []string) {
			if len(cfgPath) == 0 {
				kubeadmutil.CheckErr(fmt.Errorf("The --config flag is mandatory"))
			}
			client, err := kubeconfigutil.ClientSetFromFile(kubeConfigFile)
			kubeadmutil.CheckErr(err)

			defaultcfg := &kubeadmapiext.MasterConfiguration{}
			internalcfg, err := configutil.ConfigFileAndDefaultsToInternalConfig(cfgPath, defaultcfg)
			kubeadmutil.CheckErr(err)

			err = uploadconfig.UploadConfiguration(internalcfg, client)
			kubeadmutil.CheckErr(err)
		},
	}

	cmd.Flags().StringVar(&kubeConfigFile, "kubeconfig", "/etc/kubernetes/admin.conf", "The KubeConfig file to use for talking to the cluster")
	cmd.Flags().StringVar(&cfgPath, "config", "", "Path to kubeadm config file (WARNING: Usage of a configuration file is experimental)")

	return cmd
}
