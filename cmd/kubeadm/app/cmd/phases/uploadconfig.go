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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kubeadmapiext "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha1"
	cmdutil "k8s.io/kubernetes/cmd/kubeadm/app/cmd/util"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/uploadconfig"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	configutil "k8s.io/kubernetes/cmd/kubeadm/app/util/config"
	kubeconfigutil "k8s.io/kubernetes/cmd/kubeadm/app/util/kubeconfig"
	"k8s.io/kubernetes/pkg/util/normalizer"
)

var (
	uploadConfigLongDesc = fmt.Sprintf(normalizer.LongDesc(`
		Uploads the kubeadm init configuration of your cluster to a ConfigMap called %s in the %s namespace. 
		This enables correct configuration of system components and a seamless user experience when upgrading.

		Alternatively, you can use kubeadm config.
		`+cmdutil.AlphaDisclaimer), kubeadmconstants.MasterConfigurationConfigMap, metav1.NamespaceSystem)

	uploadConfigExample = normalizer.Examples(`
		# uploads the configuration of your cluster
		kubeadm alpha phase upload-config --config=myConfig.yaml
		`)
)

// NewCmdUploadConfig returns the Cobra command for running the uploadconfig phase
func NewCmdUploadConfig() *cobra.Command {
	var cfgPath, kubeConfigFile string
	cmd := &cobra.Command{
		Use:     "upload-config",
		Short:   "Uploads the currently used configuration for kubeadm to a ConfigMap",
		Long:    uploadConfigLongDesc,
		Example: uploadConfigExample,
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

	cmd.Flags().StringVar(&kubeConfigFile, "kubeconfig", "/etc/kubernetes/admin.conf", "The KubeConfig file to use when talking to the cluster")
	cmd.Flags().StringVar(&cfgPath, "config", "", "Path to a kubeadm config file. WARNING: Usage of a configuration file is experimental!")

	return cmd
}
