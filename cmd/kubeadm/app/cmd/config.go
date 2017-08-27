/*
Copyright 2016 The Kubernetes Authors.

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

package cmd

import (
	"fmt"
	"io"

	"github.com/renstrom/dedent"
	"github.com/spf13/cobra"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	kubeadmapiext "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha1"
	cmdutil "k8s.io/kubernetes/cmd/kubeadm/app/cmd/util"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/features"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/uploadconfig"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	configutil "k8s.io/kubernetes/cmd/kubeadm/app/util/config"
	kubeconfigutil "k8s.io/kubernetes/cmd/kubeadm/app/util/kubeconfig"
	"k8s.io/kubernetes/pkg/api"
)

func NewCmdConfig(out io.Writer) *cobra.Command {

	var kubeConfigFile string
	cmd := &cobra.Command{
		Use:   "config",
		Short: "Manage configuration for a kubeadm cluster persisted in a ConfigMap in the cluster.",
		Long: fmt.Sprintf(dedent.Dedent(`
			There is a ConfigMap in the %s namespace called %q that kubeadm uses to store internal configuration about the
			cluster. kubeadm CLI v1.8.0+ automatically creates this ConfigMap with used config on 'kubeadm init', but if you
			initialized your cluster using kubeadm v1.7.x or lower, you must use the 'config upload' command to create this
			ConfigMap in order for 'kubeadm upgrade' to be able to configure your upgraded cluster correctly.
		`), metav1.NamespaceSystem, constants.MasterConfigurationConfigMap),
		// Without this callback, if a user runs just the "upload"
		// command without a subcommand, or with an invalid subcommand,
		// cobra will print usage information, but still exit cleanly.
		// We want to return an error code in these cases so that the
		// user knows that their command was invalid.
		RunE: cmdutil.SubCmdRunE("config"),
	}

	cmd.PersistentFlags().StringVar(&kubeConfigFile, "kubeconfig", "/etc/kubernetes/admin.conf", "The KubeConfig file to use for talking to the cluster")

	cmd.AddCommand(NewCmdConfigUpload(out, &kubeConfigFile))
	cmd.AddCommand(NewCmdConfigView(out, &kubeConfigFile))

	return cmd
}

func NewCmdConfigUpload(out io.Writer, kubeConfigFile *string) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "upload",
		Short: "Upload configuration about the current state so 'kubeadm upgrade' later can know how to configure the upgraded cluster",
		RunE:  cmdutil.SubCmdRunE("upload"),
	}

	cmd.AddCommand(NewCmdConfigUploadFromFile(out, kubeConfigFile))
	cmd.AddCommand(NewCmdConfigUploadFromFlags(out, kubeConfigFile))
	return cmd
}

func NewCmdConfigView(out io.Writer, kubeConfigFile *string) *cobra.Command {
	return &cobra.Command{
		Use:   "view",
		Short: "View the kubeadm configuration stored inside the cluster",
		Long: fmt.Sprintf(dedent.Dedent(`
			Using this command, you can view the ConfigMap in the cluster where the configuration for kubeadm is located

			The configuration is located in the %q namespace in the %q ConfigMap
		`), metav1.NamespaceSystem, constants.MasterConfigurationConfigMap),
		Run: func(cmd *cobra.Command, args []string) {
			client, err := kubeconfigutil.ClientSetFromFile(*kubeConfigFile)
			kubeadmutil.CheckErr(err)

			err = RunConfigView(out, client)
			kubeadmutil.CheckErr(err)
		},
	}
}

func NewCmdConfigUploadFromFile(out io.Writer, kubeConfigFile *string) *cobra.Command {
	var cfgPath string
	cmd := &cobra.Command{
		Use:   "from-file",
		Short: "Upload a configuration file to the in-cluster ConfigMap for kubeadm configuration",
		Long: fmt.Sprintf(dedent.Dedent(`
			Using from-file, you can upload configuration to the ConfigMap in the cluster using the same config file you gave to kubeadm init.
			If you initialized your cluster using a v1.7.x or lower kubeadm client and used the --config option; you need to run this command with the
			same config file before upgrading to v1.8 using 'kubeadm upgrade'.

			The configuration is located in the %q namespace in the %q ConfigMap
		`), metav1.NamespaceSystem, constants.MasterConfigurationConfigMap),
		Run: func(cmd *cobra.Command, args []string) {
			if len(cfgPath) == 0 {
				kubeadmutil.CheckErr(fmt.Errorf("The --config flag is mandatory"))
			}

			client, err := kubeconfigutil.ClientSetFromFile(*kubeConfigFile)
			kubeadmutil.CheckErr(err)

			// The default configuration is empty; everything should come from the file on disk
			defaultcfg := &kubeadmapiext.MasterConfiguration{}
			// Upload the configuration using the file; don't care about the defaultcfg really
			err = uploadConfiguration(client, cfgPath, defaultcfg)
			kubeadmutil.CheckErr(err)
		},
	}
	cmd.Flags().StringVar(&cfgPath, "config", "", "Path to kubeadm config file (WARNING: Usage of a configuration file is experimental)")
	return cmd
}

func NewCmdConfigUploadFromFlags(out io.Writer, kubeConfigFile *string) *cobra.Command {
	cfg := &kubeadmapiext.MasterConfiguration{}
	api.Scheme.Default(cfg)

	var featureGatesString string

	cmd := &cobra.Command{
		Use:   "from-flags",
		Short: "Create the in-cluster configuration file for the first time from using flags",
		Long: fmt.Sprintf(dedent.Dedent(`
			Using from-flags, you can upload configuration to the ConfigMap in the cluster using the same flags you'd give to kubeadm init.
			If you initialized your cluster using a v1.7.x or lower kubeadm client and set some flag; you need to run this command with the
			same flags before upgrading to v1.8 using 'kubeadm upgrade'.

			The configuration is located in the %q namespace in the %q ConfigMap
		`), metav1.NamespaceSystem, constants.MasterConfigurationConfigMap),
		Run: func(cmd *cobra.Command, args []string) {
			var err error
			if cfg.FeatureGates, err = features.NewFeatureGate(&features.InitFeatureGates, featureGatesString); err != nil {
				kubeadmutil.CheckErr(err)
			}

			client, err := kubeconfigutil.ClientSetFromFile(*kubeConfigFile)
			kubeadmutil.CheckErr(err)

			// Default both statically and dynamically, convert to internal API type, and validate everything
			// The cfgPath argument is unset here as we shouldn't load a config file from disk, just go with cfg
			err = uploadConfiguration(client, "", cfg)
			kubeadmutil.CheckErr(err)
		},
	}
	AddInitConfigFlags(cmd.PersistentFlags(), cfg, &featureGatesString)
	return cmd
}

// RunConfigView gets the configuration persisted in the cluster
func RunConfigView(out io.Writer, client clientset.Interface) error {

	cfgConfigMap, err := client.CoreV1().ConfigMaps(metav1.NamespaceSystem).Get(constants.MasterConfigurationConfigMap, metav1.GetOptions{})
	if err != nil {
		return err
	}
	// No need to append \n as that already exists in the ConfigMap
	fmt.Fprintf(out, "%s", cfgConfigMap.Data[constants.MasterConfigurationConfigMapKey])
	return nil
}

// uploadConfiguration handles the uploading of the configuration internally
func uploadConfiguration(client clientset.Interface, cfgPath string, defaultcfg *kubeadmapiext.MasterConfiguration) error {

	// Default both statically and dynamically, convert to internal API type, and validate everything
	// First argument is unset here as we shouldn't load a config file from disk
	internalcfg, err := configutil.ConfigFileAndDefaultsToInternalConfig(cfgPath, defaultcfg)
	if err != nil {
		return err
	}

	// Then just call the uploadconfig phase to do the rest of the work
	return uploadconfig.UploadConfiguration(internalcfg, client)
}
