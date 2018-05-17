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
	kubeadmscheme "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/scheme"
	kubeadmapiv1alpha2 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha2"
	cmdutil "k8s.io/kubernetes/cmd/kubeadm/app/cmd/util"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/features"
	"k8s.io/kubernetes/cmd/kubeadm/app/images"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/uploadconfig"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	configutil "k8s.io/kubernetes/cmd/kubeadm/app/util/config"
	kubeconfigutil "k8s.io/kubernetes/cmd/kubeadm/app/util/kubeconfig"
	utilsexec "k8s.io/utils/exec"
)

const (
	masterConfig = "MasterConfiguration"
	nodeConfig   = "NodeConfiguration"
	sillyToken   = "abcdef.0123456789abcdef"
)

var availableAPIObjects = []string{masterConfig, nodeConfig}

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

	cmd.AddCommand(NewCmdConfigPrintDefault(out))
	cmd.AddCommand(NewCmdConfigUpload(out, &kubeConfigFile))
	cmd.AddCommand(NewCmdConfigView(out, &kubeConfigFile))
	cmd.AddCommand(NewCmdConfigImages(out))
	return cmd
}

// NewCmdConfigPrintDefault returns cobra.Command for "kubeadm config print-default" command
func NewCmdConfigPrintDefault(out io.Writer) *cobra.Command {
	apiObjects := []string{}
	cmd := &cobra.Command{
		Use:     "print-default",
		Aliases: []string{"print-defaults"},
		Short:   "Print the default values for a kubeadm configuration object.",
		Long: fmt.Sprintf(dedent.Dedent(`
			This command prints the default MasterConfiguration object that is used for 'kubeadm init' and 'kubeadm upgrade',
			and the default NodeConfiguration object that is used for 'kubeadm join'.

			Note that sensitive values like the Bootstrap Token fields are replaced with silly values like %q in order to pass validation but
			not perform the real computation for creating a token.
		`), sillyToken),
		Run: func(cmd *cobra.Command, args []string) {
			if len(apiObjects) == 0 {
				apiObjects = availableAPIObjects
			}
			for i, apiObject := range apiObjects {
				if i > 0 {
					fmt.Fprintln(out, "---")
				}

				cfgBytes, err := getDefaultAPIObjectBytes(apiObject)
				kubeadmutil.CheckErr(err)
				// Print the API object byte array
				fmt.Fprintf(out, "%s", cfgBytes)
			}
		},
	}
	cmd.Flags().StringSliceVar(&apiObjects, "api-objects", apiObjects,
		fmt.Sprintf("A comma-separated list for API objects to print the default values for. Available values: %v. This flag unset means 'print all known objects'", availableAPIObjects))
	return cmd
}

func getDefaultAPIObjectBytes(apiObject string) ([]byte, error) {
	if apiObject == masterConfig {

		internalcfg, err := configutil.ConfigFileAndDefaultsToInternalConfig("", &kubeadmapiv1alpha2.MasterConfiguration{
			Token: sillyToken,
		})
		kubeadmutil.CheckErr(err)

		return kubeadmutil.MarshalToYamlForCodecs(internalcfg, kubeadmapiv1alpha2.SchemeGroupVersion, kubeadmscheme.Codecs)
	}
	if apiObject == nodeConfig {
		internalcfg, err := configutil.NodeConfigFileAndDefaultsToInternalConfig("", &kubeadmapiv1alpha2.NodeConfiguration{
			Token: sillyToken,
			DiscoveryTokenAPIServers:               []string{"kube-apiserver:6443"},
			DiscoveryTokenUnsafeSkipCAVerification: true,
		})
		kubeadmutil.CheckErr(err)

		return kubeadmutil.MarshalToYamlForCodecs(internalcfg, kubeadmapiv1alpha2.SchemeGroupVersion, kubeadmscheme.Codecs)
	}
	return []byte{}, fmt.Errorf("--api-object needs to be one of %v", availableAPIObjects)
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
			defaultcfg := &kubeadmapiv1alpha2.MasterConfiguration{}
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
	cfg := &kubeadmapiv1alpha2.MasterConfiguration{}
	kubeadmscheme.Scheme.Default(cfg)

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
func uploadConfiguration(client clientset.Interface, cfgPath string, defaultcfg *kubeadmapiv1alpha2.MasterConfiguration) error {

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

// NewCmdConfigImages returns the "config images" command
func NewCmdConfigImages(out io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "images",
		Short: "Interact with container images used by kubeadm.",
		RunE:  cmdutil.SubCmdRunE("images"),
	}
	cmd.AddCommand(NewCmdConfigImagesList(out))
	cmd.AddCommand(NewCmdConfigImagesPull())
	return cmd
}

// NewCmdConfigImagesPull returns the `config images pull` command
func NewCmdConfigImagesPull() *cobra.Command {
	cfg := &kubeadmapiv1alpha2.MasterConfiguration{}
	kubeadmscheme.Scheme.Default(cfg)
	var cfgPath, featureGatesString string
	var err error
	cmd := &cobra.Command{
		Use:   "pull",
		Short: "Pull images used by kubeadm.",
		Run: func(_ *cobra.Command, _ []string) {
			cfg.FeatureGates, err = features.NewFeatureGate(&features.InitFeatureGates, featureGatesString)
			kubeadmutil.CheckErr(err)
			internalcfg, err := configutil.ConfigFileAndDefaultsToInternalConfig(cfgPath, cfg)
			kubeadmutil.CheckErr(err)
			puller, err := images.NewImagePuller(utilsexec.New(), internalcfg.GetCRISocket())
			kubeadmutil.CheckErr(err)
			imagesPull := NewImagesPull(puller, images.GetAllImages(internalcfg))
			kubeadmutil.CheckErr(imagesPull.PullAll())
		},
	}
	AddImagesCommonConfigFlags(cmd.PersistentFlags(), cfg, &featureGatesString)
	return cmd
}

// ImagesPull is the struct used to hold information relating to image pulling
type ImagesPull struct {
	puller images.Puller
	images []string
}

// NewImagesPull initializes and returns the `config images pull` command
func NewImagesPull(puller images.Puller, images []string) *ImagesPull {
	return &ImagesPull{
		puller: puller,
		images: images,
	}
}

// PullAll pulls all images that the ImagesPull knows about
func (ip *ImagesPull) PullAll() error {
	for _, image := range ip.images {
		if err := ip.puller.Pull(image); err != nil {
			return fmt.Errorf("failed to pull image %q: %v", image, err)
		}
		glog.Infof("[config/images] Pulled %s\n", image)
	}
	return nil
}

// NewCmdConfigImagesList returns the "kubeadm config images list" command
func NewCmdConfigImagesList(out io.Writer) *cobra.Command {
	cfg := &kubeadmapiv1alpha2.MasterConfiguration{}
	kubeadmscheme.Scheme.Default(cfg)
	var cfgPath, featureGatesString string
	var err error

	cmd := &cobra.Command{
		Use:   "list",
		Short: "Print a list of images kubeadm will use. The configuration file is used in case any images or image repositories are customized.",
		Run: func(_ *cobra.Command, _ []string) {
			cfg.FeatureGates, err = features.NewFeatureGate(&features.InitFeatureGates, featureGatesString)
			kubeadmutil.CheckErr(err)
			imagesList, err := NewImagesList(cfgPath, cfg)
			kubeadmutil.CheckErr(err)
			kubeadmutil.CheckErr(imagesList.Run(out))
		},
	}
	AddImagesCommonConfigFlags(cmd.PersistentFlags(), cfg, &featureGatesString)
	AddImagesListFlags(cmd.PersistentFlags(), &cfgPath)

	return cmd
}

// NewImagesList returns the underlying struct for the "kubeadm config images list" command
func NewImagesList(cfgPath string, cfg *kubeadmapiv1alpha2.MasterConfiguration) (*ImagesList, error) {
	internalcfg, err := configutil.ConfigFileAndDefaultsToInternalConfig(cfgPath, cfg)
	if err != nil {
		return nil, fmt.Errorf("could not convert cfg to an internal cfg: %v", err)
	}

	return &ImagesList{
		cfg: internalcfg,
	}, nil
}

// ImagesList defines the struct used for "kubeadm config images list"
type ImagesList struct {
	cfg *kubeadmapi.MasterConfiguration
}

// Run runs the images command and writes the result to the io.Writer passed in
func (i *ImagesList) Run(out io.Writer) error {
	imgs := images.GetAllImages(i.cfg)
	for _, img := range imgs {
		fmt.Fprintln(out, img)
	}

	return nil
}

// AddImagesCommonConfigFlags adds the flags that configure kubeadm (and affect the images kubeadm will use)
func AddImagesCommonConfigFlags(flagSet *flag.FlagSet, cfg *kubeadmapiv1alpha2.MasterConfiguration, featureGatesString *string) {
	flagSet.StringVar(
		&cfg.KubernetesVersion, "kubernetes-version", cfg.KubernetesVersion,
		`Choose a specific Kubernetes version for the control plane.`,
	)
	flagSet.StringVar(featureGatesString, "feature-gates", *featureGatesString, "A set of key=value pairs that describe feature gates for various features. "+
		"Options are:\n"+strings.Join(features.KnownFeatures(&features.InitFeatureGates), "\n"))
}

// AddImagesListFlags adds the flag that defines the location of the config file
func AddImagesListFlags(flagSet *flag.FlagSet, cfgPath *string) {
	flagSet.StringVar(cfgPath, "config", *cfgPath, "Path to kubeadm config file.")
}

// AddImagesPullFlags adds flags related to the `config images pull` command
func AddImagesPullFlags(flagSet *flag.FlagSet, criSocketPath *string) {
	flagSet.StringVar(criSocketPath, "cri-socket-path", *criSocketPath, "Path to the CRI socket.")
}
