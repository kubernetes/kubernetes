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
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"strings"

	"github.com/lithammer/dedent"
	"github.com/pkg/errors"
	"github.com/spf13/cobra"
	flag "github.com/spf13/pflag"
	"k8s.io/klog"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmscheme "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/scheme"
	kubeadmapiv1alpha3 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha3"
	kubeadmapiv1beta1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta1"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/options"
	phaseutil "k8s.io/kubernetes/cmd/kubeadm/app/cmd/phases"
	cmdutil "k8s.io/kubernetes/cmd/kubeadm/app/cmd/util"
	"k8s.io/kubernetes/cmd/kubeadm/app/componentconfigs"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/features"
	"k8s.io/kubernetes/cmd/kubeadm/app/images"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/uploadconfig"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	configutil "k8s.io/kubernetes/cmd/kubeadm/app/util/config"
	kubeconfigutil "k8s.io/kubernetes/cmd/kubeadm/app/util/kubeconfig"
	utilruntime "k8s.io/kubernetes/cmd/kubeadm/app/util/runtime"
	utilsexec "k8s.io/utils/exec"
)

var (
	// placeholderToken is only set statically to make kubeadm not randomize the token on every run
	placeholderToken = kubeadmapiv1beta1.BootstrapToken{
		Token: &kubeadmapiv1beta1.BootstrapTokenString{
			ID:     "abcdef",
			Secret: "0123456789abcdef",
		},
	}
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
		`), metav1.NamespaceSystem, constants.KubeadmConfigConfigMap),
		// Without this callback, if a user runs just the "upload"
		// command without a subcommand, or with an invalid subcommand,
		// cobra will print usage information, but still exit cleanly.
		// We want to return an error code in these cases so that the
		// user knows that their command was invalid.
		RunE: cmdutil.SubCmdRunE("config"),
	}

	options.AddKubeConfigFlag(cmd.PersistentFlags(), &kubeConfigFile)

	kubeConfigFile = cmdutil.GetKubeConfigPath(kubeConfigFile)
	cmd.AddCommand(NewCmdConfigPrint(out))
	cmd.AddCommand(NewCmdConfigMigrate(out))
	cmd.AddCommand(NewCmdConfigUpload(out, &kubeConfigFile))
	cmd.AddCommand(NewCmdConfigView(out, &kubeConfigFile))
	cmd.AddCommand(NewCmdConfigImages(out))
	return cmd
}

// NewCmdConfigPrint returns cobra.Command for "kubeadm config print" command
func NewCmdConfigPrint(out io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "print",
		Short: "Print configuration",
		Long:  "This command prints configurations for subcommands provided.",
		RunE:  cmdutil.SubCmdRunE("print"),
	}
	cmd.AddCommand(NewCmdConfigPrintInitDefaults(out))
	cmd.AddCommand(NewCmdConfigPrintJoinDefaults(out))
	return cmd
}

// NewCmdConfigPrintInitDefaults returns cobra.Command for "kubeadm config print init-defaults" command
func NewCmdConfigPrintInitDefaults(out io.Writer) *cobra.Command {
	return newCmdConfigPrintActionDefaults(out, "init", getDefaultInitConfigBytes)
}

// NewCmdConfigPrintJoinDefaults returns cobra.Command for "kubeadm config print join-defaults" command
func NewCmdConfigPrintJoinDefaults(out io.Writer) *cobra.Command {
	return newCmdConfigPrintActionDefaults(out, "join", getDefaultNodeConfigBytes)
}

func newCmdConfigPrintActionDefaults(out io.Writer, action string, configBytesProc func() ([]byte, error)) *cobra.Command {
	componentConfigs := []string{}
	cmd := &cobra.Command{
		Use:   fmt.Sprintf("%s-defaults", action),
		Short: fmt.Sprintf("Print default %s configuration, that can be used for 'kubeadm %s'", action, action),
		Long: fmt.Sprintf(dedent.Dedent(`
			This command prints objects such as the default %s configuration that is used for 'kubeadm %s'.

			Note that sensitive values like the Bootstrap Token fields are replaced with placeholder values like %q in order to pass validation but
			not perform the real computation for creating a token.
		`), action, action, placeholderToken),
		Run: func(cmd *cobra.Command, args []string) {
			runConfigPrintActionDefaults(out, componentConfigs, configBytesProc)
		},
	}
	cmd.Flags().StringSliceVar(&componentConfigs, "component-configs", componentConfigs,
		fmt.Sprintf("A comma-separated list for component config API objects to print the default values for. Available values: %v. If this flag is not set, no component configs will be printed.", getSupportedComponentConfigAPIObjects()))
	return cmd
}

func runConfigPrintActionDefaults(out io.Writer, componentConfigs []string, configBytesProc func() ([]byte, error)) {
	initialConfig, err := configBytesProc()
	kubeadmutil.CheckErr(err)

	allBytes := [][]byte{initialConfig}
	for _, componentConfig := range componentConfigs {
		cfgBytes, err := getDefaultComponentConfigBytes(componentConfig)
		kubeadmutil.CheckErr(err)
		allBytes = append(allBytes, cfgBytes)
	}

	fmt.Fprint(out, string(bytes.Join(allBytes, []byte(constants.YAMLDocumentSeparator))))
}

func getDefaultComponentConfigBytes(apiObject string) ([]byte, error) {
	registration, ok := componentconfigs.Known[componentconfigs.RegistrationKind(apiObject)]
	if !ok {
		return []byte{}, errors.Errorf("--component-configs needs to contain some of %v", getSupportedComponentConfigAPIObjects())
	}

	defaultedInitConfig, err := getDefaultedInitConfig()
	if err != nil {
		return []byte{}, err
	}

	realObj, ok := registration.GetFromInternalConfig(&defaultedInitConfig.ClusterConfiguration)
	if !ok {
		return []byte{}, errors.New("GetFromInternalConfig failed")
	}

	return registration.Marshal(realObj)
}

// getSupportedComponentConfigAPIObjects returns all currently supported component config API object names
func getSupportedComponentConfigAPIObjects() []string {
	objects := []string{}
	for componentType := range componentconfigs.Known {
		objects = append(objects, string(componentType))
	}
	return objects
}

func getDefaultedInitConfig() (*kubeadmapi.InitConfiguration, error) {
	return configutil.DefaultedInitConfiguration(&kubeadmapiv1beta1.InitConfiguration{
		// TODO: Probably move to getDefaultedClusterConfig?
		LocalAPIEndpoint: kubeadmapiv1beta1.APIEndpoint{AdvertiseAddress: "1.2.3.4"},
		ClusterConfiguration: kubeadmapiv1beta1.ClusterConfiguration{
			KubernetesVersion: fmt.Sprintf("v1.%d.0", constants.MinimumControlPlaneVersion.Minor()+1),
		},
		BootstrapTokens: []kubeadmapiv1beta1.BootstrapToken{placeholderToken},
		NodeRegistration: kubeadmapiv1beta1.NodeRegistrationOptions{
			CRISocket: constants.DefaultDockerCRISocket, // avoid CRI detection
		},
	})
}

func getDefaultInitConfigBytes() ([]byte, error) {
	internalcfg, err := getDefaultedInitConfig()
	if err != nil {
		return []byte{}, err
	}

	return configutil.MarshalKubeadmConfigObject(internalcfg)
}

func getDefaultNodeConfigBytes() ([]byte, error) {
	internalcfg, err := configutil.DefaultedJoinConfiguration(&kubeadmapiv1beta1.JoinConfiguration{
		Discovery: kubeadmapiv1beta1.Discovery{
			BootstrapToken: &kubeadmapiv1beta1.BootstrapTokenDiscovery{
				Token:                    placeholderToken.Token.String(),
				APIServerEndpoint:        "kube-apiserver:6443",
				UnsafeSkipCAVerification: true, // TODO: UnsafeSkipCAVerification: true needs to be set for validation to pass, but shouldn't be recommended as the default
			},
		},
		NodeRegistration: kubeadmapiv1beta1.NodeRegistrationOptions{
			CRISocket: constants.DefaultDockerCRISocket, // avoid CRI detection
		},
	})
	if err != nil {
		return []byte{}, err
	}

	return configutil.MarshalKubeadmConfigObject(internalcfg)
}

// NewCmdConfigMigrate returns cobra.Command for "kubeadm config migrate" command
func NewCmdConfigMigrate(out io.Writer) *cobra.Command {
	var oldCfgPath, newCfgPath string
	cmd := &cobra.Command{
		Use:   "migrate",
		Short: "Read an older version of the kubeadm configuration API types from a file, and output the similar config object for the newer version.",
		Long: fmt.Sprintf(dedent.Dedent(`
			This command lets you convert configuration objects of older versions to the latest supported version,
			locally in the CLI tool without ever touching anything in the cluster.
			In this version of kubeadm, the following API versions are supported:
			- %s
			- %s

			Further, kubeadm can only write out config of version %q, but read both types.
			So regardless of what version you pass to the --old-config parameter here, the API object will be
			read, deserialized, defaulted, converted, validated, and re-serialized when written to stdout or
			--new-config if specified.

			In other words, the output of this command is what kubeadm actually would read internally if you
			submitted this file to "kubeadm init"
		`), kubeadmapiv1alpha3.SchemeGroupVersion.String(), kubeadmapiv1beta1.SchemeGroupVersion.String(), kubeadmapiv1beta1.SchemeGroupVersion.String()),
		Run: func(cmd *cobra.Command, args []string) {
			if len(oldCfgPath) == 0 {
				kubeadmutil.CheckErr(errors.New("The --old-config flag is mandatory"))
			}

			oldCfgBytes, err := ioutil.ReadFile(oldCfgPath)
			kubeadmutil.CheckErr(err)

			outputBytes, err := configutil.MigrateOldConfig(oldCfgBytes)
			kubeadmutil.CheckErr(err)

			if newCfgPath == "" {
				fmt.Fprint(out, string(outputBytes))
			} else {
				if err := ioutil.WriteFile(newCfgPath, outputBytes, 0644); err != nil {
					kubeadmutil.CheckErr(errors.Wrapf(err, "failed to write the new configuration to the file %q", newCfgPath))
				}
			}
		},
	}
	cmd.Flags().StringVar(&oldCfgPath, "old-config", "", "Path to the kubeadm config file that is using an old API version and should be converted. This flag is mandatory.")
	cmd.Flags().StringVar(&newCfgPath, "new-config", "", "Path to the resulting equivalent kubeadm config file using the new API version. Optional, if not specified output will be sent to STDOUT.")
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
		`), metav1.NamespaceSystem, constants.KubeadmConfigConfigMap),
		Run: func(cmd *cobra.Command, args []string) {
			klog.V(1).Infoln("[config] retrieving ClientSet from file")
			client, err := kubeconfigutil.ClientSetFromFile(*kubeConfigFile)
			kubeadmutil.CheckErr(err)

			err = RunConfigView(out, client)
			kubeadmutil.CheckErr(err)
		},
	}
}

// NewCmdConfigUploadFromFile verifies given Kubernetes config file and returns cobra.Command for
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
		`), metav1.NamespaceSystem, constants.KubeadmConfigConfigMap),
		Run: func(cmd *cobra.Command, args []string) {
			if len(cfgPath) == 0 {
				kubeadmutil.CheckErr(errors.New("The --config flag is mandatory"))
			}

			klog.V(1).Infoln("[config] retrieving ClientSet from file")
			client, err := kubeconfigutil.ClientSetFromFile(*kubeConfigFile)
			kubeadmutil.CheckErr(err)

			// Default both statically and dynamically, convert to internal API type, and validate everything
			internalcfg, err := configutil.LoadInitConfigurationFromFile(cfgPath)
			kubeadmutil.CheckErr(err)

			// Upload the configuration using the file
			klog.V(1).Infof("[config] uploading configuration")
			err = uploadconfig.UploadConfiguration(internalcfg, client)
			kubeadmutil.CheckErr(err)
		},
	}
	options.AddConfigFlag(cmd.Flags(), &cfgPath)
	return cmd
}

// NewCmdConfigUploadFromFlags returns cobra.Command for "kubeadm config upload from-flags" command
func NewCmdConfigUploadFromFlags(out io.Writer, kubeConfigFile *string) *cobra.Command {
	cfg := &kubeadmapiv1beta1.InitConfiguration{}
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
		`), metav1.NamespaceSystem, constants.KubeadmConfigConfigMap),
		Run: func(cmd *cobra.Command, args []string) {
			var err error
			klog.V(1).Infoln("[config] creating new FeatureGates")
			if cfg.FeatureGates, err = features.NewFeatureGate(&features.InitFeatureGates, featureGatesString); err != nil {
				kubeadmutil.CheckErr(err)
			}
			klog.V(1).Infoln("[config] retrieving ClientSet from file")
			client, err := kubeconfigutil.ClientSetFromFile(*kubeConfigFile)
			kubeadmutil.CheckErr(err)

			// KubernetesVersion is not used, but we set it explicitly to avoid the lookup
			// of the version from the internet when executing DefaultedInitConfiguration
			phaseutil.SetKubernetesVersion(&cfg.ClusterConfiguration)

			// Default both statically and dynamically, convert to internal API type, and validate everything
			klog.V(1).Infoln("[config] converting to internal API type")
			internalcfg, err := configutil.DefaultedInitConfiguration(cfg)
			kubeadmutil.CheckErr(err)

			// Finally, upload the configuration
			klog.V(1).Infof("[config] uploading configuration")
			err = uploadconfig.UploadConfiguration(internalcfg, client)
			kubeadmutil.CheckErr(err)
		},
	}
	AddInitConfigFlags(cmd.PersistentFlags(), cfg, &featureGatesString)
	return cmd
}

// RunConfigView gets the configuration persisted in the cluster
func RunConfigView(out io.Writer, client clientset.Interface) error {

	klog.V(1).Infoln("[config] getting the cluster configuration")
	cfgConfigMap, err := client.CoreV1().ConfigMaps(metav1.NamespaceSystem).Get(constants.KubeadmConfigConfigMap, metav1.GetOptions{})
	if err != nil {
		return err
	}
	// No need to append \n as that already exists in the ConfigMap
	fmt.Fprintf(out, "%s", cfgConfigMap.Data[constants.ClusterConfigurationConfigMapKey])
	return nil
}

// NewCmdConfigImages returns the "kubeadm config images" command
func NewCmdConfigImages(out io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "images",
		Short: "Interact with container images used by kubeadm.",
		RunE:  cmdutil.SubCmdRunE("images"),
	}
	cmd.AddCommand(NewCmdConfigImagesList(out, nil))
	cmd.AddCommand(NewCmdConfigImagesPull())
	return cmd
}

// NewCmdConfigImagesPull returns the `kubeadm config images pull` command
func NewCmdConfigImagesPull() *cobra.Command {
	externalcfg := &kubeadmapiv1beta1.InitConfiguration{}
	kubeadmscheme.Scheme.Default(externalcfg)
	var cfgPath, featureGatesString string
	var err error

	cmd := &cobra.Command{
		Use:   "pull",
		Short: "Pull images used by kubeadm.",
		Run: func(_ *cobra.Command, _ []string) {
			externalcfg.ClusterConfiguration.FeatureGates, err = features.NewFeatureGate(&features.InitFeatureGates, featureGatesString)
			kubeadmutil.CheckErr(err)
			internalcfg, err := configutil.LoadOrDefaultInitConfiguration(cfgPath, externalcfg)
			kubeadmutil.CheckErr(err)
			containerRuntime, err := utilruntime.NewContainerRuntime(utilsexec.New(), internalcfg.GetCRISocket())
			kubeadmutil.CheckErr(err)
			imagesPull := NewImagesPull(containerRuntime, images.GetAllImages(&internalcfg.ClusterConfiguration))
			kubeadmutil.CheckErr(imagesPull.PullAll())
		},
	}
	AddImagesCommonConfigFlags(cmd.PersistentFlags(), externalcfg, &cfgPath, &featureGatesString)
	cmdutil.AddCRISocketFlag(cmd.PersistentFlags(), &externalcfg.NodeRegistration.CRISocket)

	return cmd
}

// ImagesPull is the struct used to hold information relating to image pulling
type ImagesPull struct {
	runtime utilruntime.ContainerRuntime
	images  []string
}

// NewImagesPull initializes and returns the `kubeadm config images pull` command
func NewImagesPull(runtime utilruntime.ContainerRuntime, images []string) *ImagesPull {
	return &ImagesPull{
		runtime: runtime,
		images:  images,
	}
}

// PullAll pulls all images that the ImagesPull knows about
func (ip *ImagesPull) PullAll() error {
	for _, image := range ip.images {
		if err := ip.runtime.PullImage(image); err != nil {
			return errors.Wrapf(err, "failed to pull image %q", image)
		}
		fmt.Printf("[config/images] Pulled %s\n", image)
	}
	return nil
}

// NewCmdConfigImagesList returns the "kubeadm config images list" command
func NewCmdConfigImagesList(out io.Writer, mockK8sVersion *string) *cobra.Command {
	externalcfg := &kubeadmapiv1beta1.InitConfiguration{}
	kubeadmscheme.Scheme.Default(externalcfg)
	var cfgPath, featureGatesString string
	var err error

	// This just sets the Kubernetes version for unit testing so kubeadm won't try to
	// lookup the latest release from the internet.
	if mockK8sVersion != nil {
		externalcfg.KubernetesVersion = *mockK8sVersion
	}

	cmd := &cobra.Command{
		Use:   "list",
		Short: "Print a list of images kubeadm will use. The configuration file is used in case any images or image repositories are customized.",
		Run: func(_ *cobra.Command, _ []string) {
			externalcfg.ClusterConfiguration.FeatureGates, err = features.NewFeatureGate(&features.InitFeatureGates, featureGatesString)
			kubeadmutil.CheckErr(err)
			imagesList, err := NewImagesList(cfgPath, externalcfg)
			kubeadmutil.CheckErr(err)
			kubeadmutil.CheckErr(imagesList.Run(out))
		},
	}
	AddImagesCommonConfigFlags(cmd.PersistentFlags(), externalcfg, &cfgPath, &featureGatesString)
	return cmd
}

// NewImagesList returns the underlying struct for the "kubeadm config images list" command
func NewImagesList(cfgPath string, cfg *kubeadmapiv1beta1.InitConfiguration) (*ImagesList, error) {
	initcfg, err := configutil.LoadOrDefaultInitConfiguration(cfgPath, cfg)
	if err != nil {
		return nil, errors.Wrap(err, "could not convert cfg to an internal cfg")
	}

	return &ImagesList{
		cfg: initcfg,
	}, nil
}

// ImagesList defines the struct used for "kubeadm config images list"
type ImagesList struct {
	cfg *kubeadmapi.InitConfiguration
}

// Run runs the images command and writes the result to the io.Writer passed in
func (i *ImagesList) Run(out io.Writer) error {
	imgs := images.GetAllImages(&i.cfg.ClusterConfiguration)
	for _, img := range imgs {
		fmt.Fprintln(out, img)
	}

	return nil
}

// AddImagesCommonConfigFlags adds the flags that configure kubeadm (and affect the images kubeadm will use)
func AddImagesCommonConfigFlags(flagSet *flag.FlagSet, cfg *kubeadmapiv1beta1.InitConfiguration, cfgPath *string, featureGatesString *string) {
	flagSet.StringVar(
		&cfg.ClusterConfiguration.KubernetesVersion, "kubernetes-version", cfg.ClusterConfiguration.KubernetesVersion,
		`Choose a specific Kubernetes version for the control plane.`,
	)
	flagSet.StringVar(featureGatesString, "feature-gates", *featureGatesString, "A set of key=value pairs that describe feature gates for various features. "+
		"Options are:\n"+strings.Join(features.KnownFeatures(&features.InitFeatureGates), "\n"))
	flagSet.StringVar(cfgPath, "config", *cfgPath, "Path to kubeadm config file.")
}
