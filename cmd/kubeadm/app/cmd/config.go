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
	"strings"

	"github.com/golang/glog"
	"github.com/renstrom/dedent"
	"github.com/spf13/cobra"
	flag "github.com/spf13/pflag"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiv1alpha1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha1"
	cmdutil "k8s.io/kubernetes/cmd/kubeadm/app/cmd/util"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/features"
	"k8s.io/kubernetes/cmd/kubeadm/app/images"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/uploadconfig"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	configutil "k8s.io/kubernetes/cmd/kubeadm/app/util/config"
	kubeconfigutil "k8s.io/kubernetes/cmd/kubeadm/app/util/kubeconfig"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
)

// NewCmdConfig returns cobra.Command for "kubeadm config" command
func NewCmdConfig(out io.Writer) *cobra.Command {

	var kubeConfigFile string
	cmd := &cobra.Command{
		Use:   "config",
		Short: "Manage configuration for a kubeadm cluster persisted in a ConfigMap in the cluster.",
		Long: fmt.Sprintf(dedent.Dedent(`
			There is a ConfigMap in the %s namespace called %q that kubeadm uses to store internal configuration about the
			cluster. kubeadm CLI v1.8.0+ automatically creates this ConfigMap with the config used with 'kubeadm init', but if you
			initialized your cluster using kubeadm v1.7.x or lower, you must use the 'config upload' command to create this
			ConfigMap. This is required so that 'kubeadm upgrade' can configure your upgraded cluster correctly.
		`), metav1.NamespaceSystem, constants.MasterConfigurationConfigMap),
		// Without this callback, if a user runs just the "upload"
		// command without a subcommand, or with an invalid subcommand,
		// cobra will print usage information, but still exit cleanly.
		// We want to return an error code in these cases so that the
		// user knows that their command was invalid.
		RunE: cmdutil.SubCmdRunE("config"),
	}

	cmd.PersistentFlags().StringVar(&kubeConfigFile, "kubeconfig", "/etc/kubernetes/admin.conf", "The KubeConfig file to use when talking to the cluster.")

	cmd.AddCommand(NewCmdConfigUpload(out, &kubeConfigFile))
	cmd.AddCommand(NewCmdConfigView(out, &kubeConfigFile))
	cmd.AddCommand(NewCmdConfigListImages(out))
	return cmd
}

// NewCmdConfigUpload returns cobra.Command for "kubeadm config upload" command
func NewCmdConfigUpload(out io.Writer, kubeConfigFile *string) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "upload",
		Short: "Upload configuration about the current state, so that 'kubeadm upgrade' can later know how to configure the upgraded cluster.",
		RunE:  cmdutil.SubCmdRunE("upload"),
	}

	cmd.AddCommand(NewCmdConfigUploadFromFile(out, kubeConfigFile))
	cmd.AddCommand(NewCmdConfigUploadFromFlags(out, kubeConfigFile))
	return cmd
}

// NewCmdConfigView returns cobra.Command for "kubeadm config view" command
func NewCmdConfigView(out io.Writer, kubeConfigFile *string) *cobra.Command {
	return &cobra.Command{
		Use:   "view",
		Short: "View the kubeadm configuration stored inside the cluster.",
		Long: fmt.Sprintf(dedent.Dedent(`
			Using this command, you can view the ConfigMap in the cluster where the configuration for kubeadm is located.

			The configuration is located in the %q namespace in the %q ConfigMap.
		`), metav1.NamespaceSystem, constants.MasterConfigurationConfigMap),
		Run: func(cmd *cobra.Command, args []string) {
			glog.V(1).Infoln("[config] retrieving ClientSet from file")
			client, err := kubeconfigutil.ClientSetFromFile(*kubeConfigFile)
			kubeadmutil.CheckErr(err)

			err = RunConfigView(out, client)
			kubeadmutil.CheckErr(err)
		},
	}
}

// NewCmdConfigUploadFromFile verifies given kubernetes config file and returns cobra.Command for
// "kubeadm config upload from-file" command
func NewCmdConfigUploadFromFile(out io.Writer, kubeConfigFile *string) *cobra.Command {
	var cfgPath string
	cmd := &cobra.Command{
		Use:   "from-file",
		Short: "Upload a configuration file to the in-cluster ConfigMap for kubeadm configuration.",
		Long: fmt.Sprintf(dedent.Dedent(`
			Using this command, you can upload configuration to the ConfigMap in the cluster using the same config file you gave to 'kubeadm init'.
			If you initialized your cluster using a v1.7.x or lower kubeadm client and used the --config option, you need to run this command with the
			same config file before upgrading to v1.8 using 'kubeadm upgrade'.

			The configuration is located in the %q namespace in the %q ConfigMap.
		`), metav1.NamespaceSystem, constants.MasterConfigurationConfigMap),
		Run: func(cmd *cobra.Command, args []string) {
			if len(cfgPath) == 0 {
				kubeadmutil.CheckErr(fmt.Errorf("The --config flag is mandatory"))
			}

			glog.V(1).Infoln("[config] retrieving ClientSet from file")
			client, err := kubeconfigutil.ClientSetFromFile(*kubeConfigFile)
			kubeadmutil.CheckErr(err)

			// The default configuration is empty; everything should come from the file on disk
			glog.V(1).Infoln("[config] creating empty default configuration")
			defaultcfg := &kubeadmapiv1alpha1.MasterConfiguration{}
			// Upload the configuration using the file; don't care about the defaultcfg really
			glog.V(1).Infof("[config] uploading configuration")
			err = uploadConfiguration(client, cfgPath, defaultcfg)
			kubeadmutil.CheckErr(err)
		},
	}
	cmd.Flags().StringVar(&cfgPath, "config", "", "Path to a kubeadm config file. WARNING: Usage of a configuration file is experimental.")
	return cmd
}

// NewCmdConfigUploadFromFlags returns cobra.Command for "kubeadm config upload from-flags" command
func NewCmdConfigUploadFromFlags(out io.Writer, kubeConfigFile *string) *cobra.Command {
	cfg := &kubeadmapiv1alpha1.MasterConfiguration{}
	legacyscheme.Scheme.Default(cfg)

	var featureGatesString string

	cmd := &cobra.Command{
		Use:   "from-flags",
		Short: "Create the in-cluster configuration file for the first time from using flags.",
		Long: fmt.Sprintf(dedent.Dedent(`
			Using this command, you can upload configuration to the ConfigMap in the cluster using the same flags you gave to 'kubeadm init'.
			If you initialized your cluster using a v1.7.x or lower kubeadm client and set certain flags, you need to run this command with the
			same flags before upgrading to v1.8 using 'kubeadm upgrade'.

			The configuration is located in the %q namespace in the %q ConfigMap.
		`), metav1.NamespaceSystem, constants.MasterConfigurationConfigMap),
		Run: func(cmd *cobra.Command, args []string) {
			var err error
			glog.V(1).Infoln("[config] creating new FeatureGates")
			if cfg.FeatureGates, err = features.NewFeatureGate(&features.InitFeatureGates, featureGatesString); err != nil {
				kubeadmutil.CheckErr(err)
			}
			glog.V(1).Infoln("[config] retrieving ClientSet from file")
			client, err := kubeconfigutil.ClientSetFromFile(*kubeConfigFile)
			kubeadmutil.CheckErr(err)

			// Default both statically and dynamically, convert to internal API type, and validate everything
			// The cfgPath argument is unset here as we shouldn't load a config file from disk, just go with cfg
			glog.V(1).Infof("[config] uploading configuration")
			err = uploadConfiguration(client, "", cfg)
			kubeadmutil.CheckErr(err)
		},
	}
	AddInitConfigFlags(cmd.PersistentFlags(), cfg, &featureGatesString)
	return cmd
}

// RunConfigView gets the configuration persisted in the cluster
func RunConfigView(out io.Writer, client clientset.Interface) error {

	glog.V(1).Infoln("[config] getting the cluster configuration")
	cfgConfigMap, err := client.CoreV1().ConfigMaps(metav1.NamespaceSystem).Get(constants.MasterConfigurationConfigMap, metav1.GetOptions{})
	if err != nil {
		return err
	}
	// No need to append \n as that already exists in the ConfigMap
	fmt.Fprintf(out, "%s", cfgConfigMap.Data[constants.MasterConfigurationConfigMapKey])
	return nil
}

// uploadConfiguration handles the uploading of the configuration internally
func uploadConfiguration(client clientset.Interface, cfgPath string, defaultcfg *kubeadmapiv1alpha1.MasterConfiguration) error {

	// Default both statically and dynamically, convert to internal API type, and validate everything
	// First argument is unset here as we shouldn't load a config file from disk
	glog.V(1).Infoln("[config] converting to internal API type")
	internalcfg, err := configutil.ConfigFileAndDefaultsToInternalConfig(cfgPath, defaultcfg)
	if err != nil {
		return err
	}

	// Then just call the uploadconfig phase to do the rest of the work
	return uploadconfig.UploadConfiguration(internalcfg, client)
}

// NewCmdConfigListImages returns the "kubeadm images" command
func NewCmdConfigListImages(out io.Writer) *cobra.Command {
	cfg := &kubeadmapiv1alpha1.MasterConfiguration{}
	kubeadmapiv1alpha1.SetDefaults_MasterConfiguration(cfg)
	var cfgPath, featureGatesString string
	var err error

	cmd := &cobra.Command{
		Use:   "list-images",
		Short: "Print a list of images kubeadm will use. The configuration file is used in case any images or image repositories are customized.",
		Run: func(_ *cobra.Command, _ []string) {
			if cfg.FeatureGates, err = features.NewFeatureGate(&features.InitFeatureGates, featureGatesString); err != nil {
				kubeadmutil.CheckErr(err)
			}
			listImages, err := NewListImages(cfgPath, cfg)
			kubeadmutil.CheckErr(err)
			kubeadmutil.CheckErr(listImages.Run(out))
		},
	}
	AddListImagesConfigFlag(cmd.PersistentFlags(), cfg, &featureGatesString)
	AddListImagesFlags(cmd.PersistentFlags(), &cfgPath)

	return cmd
}

// NewListImages returns a "kubeadm images" command
func NewListImages(cfgPath string, cfg *kubeadmapiv1alpha1.MasterConfiguration) (*ListImages, error) {
	internalcfg, err := configutil.ConfigFileAndDefaultsToInternalConfig(cfgPath, cfg)
	if err != nil {
		return nil, fmt.Errorf("could not convert cfg to an internal cfg: %v", err)
	}

	return &ListImages{
		cfg: internalcfg,
	}, nil
}

// ListImages defines the struct used for "kubeadm images"
type ListImages struct {
	cfg *kubeadmapi.MasterConfiguration
}

// Run runs the images command and writes the result to the io.Writer passed in
func (i *ListImages) Run(out io.Writer) error {
	imgs := images.GetAllImages(i.cfg)
	for _, img := range imgs {
		fmt.Fprintln(out, img)
	}

	return nil
}

// AddListImagesConfigFlag adds the flags that configure kubeadm
func AddListImagesConfigFlag(flagSet *flag.FlagSet, cfg *kubeadmapiv1alpha1.MasterConfiguration, featureGatesString *string) {
	flagSet.StringVar(
		&cfg.KubernetesVersion, "kubernetes-version", cfg.KubernetesVersion,
		`Choose a specific Kubernetes version for the control plane.`,
	)
	flagSet.StringVar(featureGatesString, "feature-gates", *featureGatesString, "A set of key=value pairs that describe feature gates for various features. "+
		"Options are:\n"+strings.Join(features.KnownFeatures(&features.InitFeatureGates), "\n"))
}

// AddListImagesFlags adds the flag that defines the location of the config file
func AddListImagesFlags(flagSet *flag.FlagSet, cfgPath *string) {
	flagSet.StringVar(cfgPath, "config", *cfgPath, "Path to kubeadm config file.")
}
