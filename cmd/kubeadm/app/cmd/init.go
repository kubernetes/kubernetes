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
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
	"text/template"

	"github.com/golang/glog"
	"github.com/pkg/errors"
	"github.com/renstrom/dedent"
	"github.com/spf13/cobra"
	flag "github.com/spf13/pflag"
	"k8s.io/apimachinery/pkg/util/sets"
	clientset "k8s.io/client-go/kubernetes"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmscheme "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/scheme"
	kubeadmapiv1beta1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta1"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/validation"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/options"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/phases"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/phases/workflow"
	cmdutil "k8s.io/kubernetes/cmd/kubeadm/app/cmd/util"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/features"
	dnsaddonphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/addons/dns"
	proxyaddonphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/addons/proxy"
	clusterinfophase "k8s.io/kubernetes/cmd/kubeadm/app/phases/bootstraptoken/clusterinfo"
	nodebootstraptokenphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/bootstraptoken/node"
	certsphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/certs"
	kubeconfigphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/kubeconfig"
	kubeletphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/kubelet"
	markmasterphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/markmaster"
	uploadconfigphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/uploadconfig"
	"k8s.io/kubernetes/cmd/kubeadm/app/preflight"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/apiclient"
	configutil "k8s.io/kubernetes/cmd/kubeadm/app/util/config"
	kubeconfigutil "k8s.io/kubernetes/cmd/kubeadm/app/util/kubeconfig"
	utilsexec "k8s.io/utils/exec"
)

var (
	initDoneTempl = template.Must(template.New("init").Parse(dedent.Dedent(`
		Your Kubernetes master has initialized successfully!

		To start using your cluster, you need to run the following as a regular user:

		  mkdir -p $HOME/.kube
		  sudo cp -i {{.KubeConfigPath}} $HOME/.kube/config
		  sudo chown $(id -u):$(id -g) $HOME/.kube/config

		You should now deploy a pod network to the cluster.
		Run "kubectl apply -f [podnetwork].yaml" with one of the options listed at:
		  https://kubernetes.io/docs/concepts/cluster-administration/addons/

		You can now join any number of machines by running the following on each node
		as root:

		  {{.joinCommand}}

		`)))

	kubeletFailTempl = template.Must(template.New("init").Parse(dedent.Dedent(`
		Unfortunately, an error has occurred:
			{{ .Error }}

		This error is likely caused by:
			- The kubelet is not running
			- The kubelet is unhealthy due to a misconfiguration of the node in some way (required cgroups disabled)

		If you are on a systemd-powered system, you can try to troubleshoot the error with the following commands:
			- 'systemctl status kubelet'
			- 'journalctl -xeu kubelet'

		Additionally, a control plane component may have crashed or exited when started by the container runtime.
		To troubleshoot, list all containers using your preferred container runtimes CLI, e.g. docker.
		Here is one example how you may list all Kubernetes containers running in docker:
			- 'docker ps -a | grep kube | grep -v pause'
			Once you have found the failing container, you can inspect its logs with:
			- 'docker logs CONTAINERID'
		`)))
)

// initOptions defines all the init options exposed via flags by kubeadm init.
// Please note that this structure includes the public kubeadm config API, but only a subset of the options
// supported by this api will be exposed as a flag.
type initOptions struct {
	cfgPath               string
	skipTokenPrint        bool
	dryRun                bool
	kubeconfigDir         string
	kubeconfigPath        string
	featureGatesString    string
	ignorePreflightErrors []string
	bto                   *options.BootstrapTokenOptions
	externalcfg           *kubeadmapiv1beta1.InitConfiguration
}

// initData defines all the runtime information used when running the kubeadm init worklow;
// this data is shared across all the phases that are included in the workflow.
type initData struct {
	cfg                   *kubeadmapi.InitConfiguration
	skipTokenPrint        bool
	dryRun                bool
	kubeconfigDir         string
	kubeconfigPath        string
	ignorePreflightErrors sets.String
	certificatesDir       string
	dryRunDir             string
	externalCA            bool
	client                clientset.Interface
	waiter                apiclient.Waiter
	outputWriter          io.Writer
}

// NewCmdInit returns "kubeadm init" command.
func NewCmdInit(out io.Writer) *cobra.Command {
	initOptions := newInitOptions()
	initRunner := workflow.NewRunner()

	cmd := &cobra.Command{
		Use:   "init",
		Short: "Run this command in order to set up the Kubernetes master.",
		Run: func(cmd *cobra.Command, args []string) {
			c, err := initRunner.InitData()
			kubeadmutil.CheckErr(err)

			data := c.(initData)
			fmt.Printf("[init] Using Kubernetes version: %s\n", data.cfg.KubernetesVersion)

			err = initRunner.Run()
			kubeadmutil.CheckErr(err)

			// TODO: the code in runInit should be progressively converted in phases; each phase will be exposed
			// via the subcommands automatically created by initRunner.BindToCommand
			err = runInit(&data, out)
			kubeadmutil.CheckErr(err)
		},
	}

	// adds flags to the init command
	// init command local flags could be eventually inherited by the sub-commands automatically generated for phases
	AddInitConfigFlags(cmd.Flags(), initOptions.externalcfg, &initOptions.featureGatesString)
	AddInitOtherFlags(cmd.Flags(), &initOptions.cfgPath, &initOptions.skipTokenPrint, &initOptions.dryRun, &initOptions.ignorePreflightErrors)
	initOptions.bto.AddTokenFlag(cmd.Flags())
	initOptions.bto.AddTTLFlag(cmd.Flags())

	// defines additional flag that are not used by the init command but that could be eventually used
	// by the sub-commands automatically generated for phases
	initRunner.SetPhaseSubcommandsAdditionalFlags(func(flags *flag.FlagSet) {
		options.AddKubeConfigFlag(flags, &initOptions.kubeconfigPath)
		options.AddKubeConfigDirFlag(flags, &initOptions.kubeconfigDir)
		options.AddControlPlanExtraArgsFlags(flags, &initOptions.externalcfg.APIServer.ExtraArgs, &initOptions.externalcfg.ControllerManager.ExtraArgs, &initOptions.externalcfg.Scheduler.ExtraArgs)
	})

	// initialize the workflow runner with the list of phases
	initRunner.AppendPhase(phases.NewPreflightMasterPhase())
	initRunner.AppendPhase(phases.NewKubeletStartPhase())
	initRunner.AppendPhase(phases.NewCertsPhase())
	initRunner.AppendPhase(phases.NewKubeConfigPhase())
	initRunner.AppendPhase(phases.NewControlPlanePhase())
	initRunner.AppendPhase(phases.NewEtcdPhase())
	initRunner.AppendPhase(phases.NewWaitControlPlanePhase())
	initRunner.AppendPhase(phases.NewUploadConfigPhase())
	// TODO: add other phases to the runner.

	// sets the data builder function, that will be used by the runner
	// both when running the entire workflow or single phases
	initRunner.SetDataInitializer(func() (workflow.RunData, error) {
		return newInitData(cmd, initOptions, out)
	})

	// binds the Runner to kubeadm init command by altering
	// command help, adding --skip-phases flag and by adding phases subcommands
	initRunner.BindToCommand(cmd)

	return cmd
}

// AddInitConfigFlags adds init flags bound to the config to the specified flagset
func AddInitConfigFlags(flagSet *flag.FlagSet, cfg *kubeadmapiv1beta1.InitConfiguration, featureGatesString *string) {
	flagSet.StringVar(
		&cfg.APIEndpoint.AdvertiseAddress, options.APIServerAdvertiseAddress, cfg.APIEndpoint.AdvertiseAddress,
		"The IP address the API Server will advertise it's listening on. Specify '0.0.0.0' to use the address of the default network interface.",
	)
	flagSet.Int32Var(
		&cfg.APIEndpoint.BindPort, options.APIServerBindPort, cfg.APIEndpoint.BindPort,
		"Port for the API Server to bind to.",
	)
	flagSet.StringVar(
		&cfg.Networking.ServiceSubnet, options.NetworkingServiceSubnet, cfg.Networking.ServiceSubnet,
		"Use alternative range of IP address for service VIPs.",
	)
	flagSet.StringVar(
		&cfg.Networking.PodSubnet, options.NetworkingPodSubnet, cfg.Networking.PodSubnet,
		"Specify range of IP addresses for the pod network. If set, the control plane will automatically allocate CIDRs for every node.",
	)
	flagSet.StringVar(
		&cfg.Networking.DNSDomain, options.NetworkingDNSDomain, cfg.Networking.DNSDomain,
		`Use alternative domain for services, e.g. "myorg.internal".`,
	)
	flagSet.StringVar(
		&cfg.KubernetesVersion, options.KubernetesVersion, cfg.KubernetesVersion,
		`Choose a specific Kubernetes version for the control plane.`,
	)
	flagSet.StringVar(
		&cfg.CertificatesDir, options.CertificatesDir, cfg.CertificatesDir,
		`The path where to save and store the certificates.`,
	)
	flagSet.StringSliceVar(
		&cfg.APIServer.CertSANs, options.APIServerCertSANs, cfg.APIServer.CertSANs,
		`Optional extra Subject Alternative Names (SANs) to use for the API Server serving certificate. Can be both IP addresses and DNS names.`,
	)
	flagSet.StringVar(
		&cfg.NodeRegistration.Name, options.NodeName, cfg.NodeRegistration.Name,
		`Specify the node name.`,
	)
	flagSet.StringVar(
		&cfg.NodeRegistration.CRISocket, options.NodeCRISocket, cfg.NodeRegistration.CRISocket,
		`Specify the CRI socket to connect to.`,
	)
	flagSet.StringVar(featureGatesString, options.FeatureGatesString, *featureGatesString, "A set of key=value pairs that describe feature gates for various features. "+
		"Options are:\n"+strings.Join(features.KnownFeatures(&features.InitFeatureGates), "\n"))
}

// AddInitOtherFlags adds init flags that are not bound to a configuration file to the given flagset
func AddInitOtherFlags(flagSet *flag.FlagSet, cfgPath *string, skipTokenPrint, dryRun *bool, ignorePreflightErrors *[]string) {
	flagSet.StringVar(
		cfgPath, options.CfgPath, *cfgPath,
		"Path to kubeadm config file. WARNING: Usage of a configuration file is experimental.",
	)
	flagSet.StringSliceVar(
		ignorePreflightErrors, options.IgnorePreflightErrors, *ignorePreflightErrors,
		"A list of checks whose errors will be shown as warnings. Example: 'IsPrivilegedUser,Swap'. Value 'all' ignores errors from all checks.",
	)
	// Note: All flags that are not bound to the cfg object should be whitelisted in cmd/kubeadm/app/apis/kubeadm/validation/validation.go
	flagSet.BoolVar(
		skipTokenPrint, options.SkipTokenPrint, *skipTokenPrint,
		"Skip printing of the default bootstrap token generated by 'kubeadm init'.",
	)
	// Note: All flags that are not bound to the cfg object should be whitelisted in cmd/kubeadm/app/apis/kubeadm/validation/validation.go
	flagSet.BoolVar(
		dryRun, options.DryRun, *dryRun,
		"Don't apply any changes; just output what would be done.",
	)
}

// newInitOptions returns a struct ready for being used for creating cmd init flags.
func newInitOptions() *initOptions {
	// initialize the public kubeadm config API by appling defaults
	externalcfg := &kubeadmapiv1beta1.InitConfiguration{}
	kubeadmscheme.Scheme.Default(externalcfg)

	// Create the options object for the bootstrap token-related flags, and override the default value for .Description
	bto := options.NewBootstrapTokenOptions()
	bto.Description = "The default bootstrap token generated by 'kubeadm init'."

	return &initOptions{
		externalcfg:    externalcfg,
		bto:            bto,
		kubeconfigDir:  kubeadmconstants.KubernetesDir,
		kubeconfigPath: kubeadmconstants.GetAdminKubeConfigPath(),
	}
}

// newInitData returns a new initData struct to be used for the execution of the kubeadm init workflow.
// This func takes care of validating initOptions passed to the command, and then it converts
// options into the internal InitConfiguration type that is used as input all the phases in the kubeadm init workflow
func newInitData(cmd *cobra.Command, options *initOptions, out io.Writer) (initData, error) {
	// Re-apply defaults to the public kubeadm API (this will set only values not exposed/not set as a flags)
	kubeadmscheme.Scheme.Default(options.externalcfg)

	// Validate standalone flags values and/or combination of flags and then assigns
	// validated values to the public kubeadm config API when applicable
	var err error
	if options.externalcfg.FeatureGates, err = features.NewFeatureGate(&features.InitFeatureGates, options.featureGatesString); err != nil {
		return initData{}, err
	}

	ignorePreflightErrorsSet, err := validation.ValidateIgnorePreflightErrors(options.ignorePreflightErrors)
	kubeadmutil.CheckErr(err)

	if err = validation.ValidateMixedArguments(cmd.Flags()); err != nil {
		return initData{}, err
	}

	if err = options.bto.ApplyTo(options.externalcfg); err != nil {
		return initData{}, err
	}

	// Either use the config file if specified, or convert public kubeadm API to the internal InitConfiguration
	// and validates InitConfiguration
	cfg, err := configutil.ConfigFileAndDefaultsToInternalConfig(options.cfgPath, options.externalcfg)
	if err != nil {
		return initData{}, err
	}
	if err := configutil.VerifyAPIServerBindAddress(cfg.APIEndpoint.AdvertiseAddress); err != nil {
		return initData{}, err
	}
	if err := features.ValidateVersion(features.InitFeatureGates, cfg.FeatureGates, cfg.KubernetesVersion); err != nil {
		return initData{}, err
	}

	// if dry running creates a temporary folder for saving kubeadm generated files
	dryRunDir := ""
	if options.dryRun {
		if dryRunDir, err = ioutil.TempDir("", "kubeadm-init-dryrun"); err != nil {
			return initData{}, errors.Wrap(err, "couldn't create a temporary directory")
		}
	}

	// Checks if an external CA is provided by the user.
	externalCA, _ := certsphase.UsingExternalCA(cfg)
	if externalCA {
		kubeconfigDir := kubeadmconstants.KubernetesDir
		if options.dryRun {
			kubeconfigDir = dryRunDir
		}
		if err := kubeconfigphase.ValidateKubeconfigsForExternalCA(kubeconfigDir, cfg); err != nil {
			return initData{}, err
		}
	}

	return initData{
		cfg:                   cfg,
		certificatesDir:       cfg.CertificatesDir,
		skipTokenPrint:        options.skipTokenPrint,
		dryRun:                options.dryRun,
		dryRunDir:             dryRunDir,
		kubeconfigDir:         options.kubeconfigDir,
		kubeconfigPath:        options.kubeconfigPath,
		ignorePreflightErrors: ignorePreflightErrorsSet,
		externalCA:            externalCA,
		outputWriter:          out,
	}, nil
}

// Cfg returns initConfiguration.
func (d initData) Cfg() *kubeadmapi.InitConfiguration {
	return d.cfg
}

// DryRun returns the DryRun flag.
func (d initData) DryRun() bool {
	return d.dryRun
}

// SkipTokenPrint returns the SkipTokenPrint flag.
func (d initData) SkipTokenPrint() bool {
	return d.skipTokenPrint
}

// IgnorePreflightErrors returns the IgnorePreflightErrors flag.
func (d initData) IgnorePreflightErrors() sets.String {
	return d.ignorePreflightErrors
}

// CertificateWriteDir returns the path to the certificate folder or the temporary folder path in case of DryRun.
func (d initData) CertificateWriteDir() string {
	if d.dryRun {
		return d.dryRunDir
	}
	return d.certificatesDir
}

// CertificateDir returns the CertificateDir as originally specified by the user.
func (d initData) CertificateDir() string {
	return d.certificatesDir
}

// KubeConfigDir returns the path of the Kubernetes configuration folder or the temporary folder path in case of DryRun.
func (d initData) KubeConfigDir() string {
	if d.dryRun {
		return d.dryRunDir
	}
	return d.kubeconfigDir
}

// KubeConfigPath returns the path to the kubeconfig file to use for connecting to Kubernetes
func (d initData) KubeConfigPath() string {
	return d.kubeconfigPath
}

// ManifestDir returns the path where manifest should be stored or the temporary folder path in case of DryRun.
func (d initData) ManifestDir() string {
	if d.dryRun {
		return d.dryRunDir
	}
	return kubeadmconstants.GetStaticPodDirectory()
}

// KubeletDir returns path of the kubelet configuration folder or the temporary folder in case of DryRun.
func (d initData) KubeletDir() string {
	if d.dryRun {
		return d.dryRunDir
	}
	return kubeadmconstants.KubeletRunDirectory
}

// ExternalCA returns true if an external CA is provided by the user.
func (d initData) ExternalCA() bool {
	return d.externalCA
}

// OutputWriter returns the io.Writer used to write output to by this command.
func (d initData) OutputWriter() io.Writer {
	return d.outputWriter
}

// Client returns a Kubernetes client to be used by kubeadm.
// This function is implemented as a singleton, thus avoiding to recreate the client when it is used by different phases.
// Important. This function must be called after the admin.conf kubeconfig file is created.
func (d initData) Client() (clientset.Interface, error) {
	if d.client == nil {
		if d.dryRun {
			// If we're dry-running; we should create a faked client that answers some GETs in order to be able to do the full init flow and just logs the rest of requests
			dryRunGetter := apiclient.NewInitDryRunGetter(d.cfg.NodeRegistration.Name, d.cfg.Networking.ServiceSubnet)
			d.client = apiclient.NewDryRunClient(dryRunGetter, os.Stdout)
		} else {
			// If we're acting for real, we should create a connection to the API server and wait for it to come up
			var err error
			d.client, err = kubeconfigutil.ClientSetFromFile(d.KubeConfigPath())
			if err != nil {
				return nil, err
			}
		}
	}
	return d.client, nil
}

// Tokens returns an array of token strings.
func (d initData) Tokens() []string {
	tokens := []string{}
	for _, bt := range d.cfg.BootstrapTokens {
		tokens = append(tokens, bt.Token.String())
	}
	return tokens
}

// runInit executes master node provisioning
func runInit(i *initData, out io.Writer) error {

	// Get directories to write files to; can be faked if we're dry-running
	glog.V(1).Infof("[init] Getting certificates directory from configuration")
	certsDirToWriteTo, kubeConfigDir, _, _, err := getDirectoriesToUse(i.dryRun, i.dryRunDir, i.cfg.CertificatesDir)
	if err != nil {
		return errors.Wrap(err, "error getting directories to use")
	}

	// certsDirToWriteTo is gonna equal cfg.CertificatesDir in the normal case, but gonna be a temp directory if dryrunning
	i.cfg.CertificatesDir = certsDirToWriteTo

	adminKubeConfigPath := filepath.Join(kubeConfigDir, kubeadmconstants.AdminKubeConfigFileName)

	// TODO: client and waiter are temporary until the rest of the phases that use them
	// are removed from this function.
	client, err := i.Client()
	if err != nil {
		return errors.Wrap(err, "failed to create client")
	}

	// Upload currently used configuration to the cluster
	// Note: This is done right in the beginning of cluster initialization; as we might want to make other phases
	// depend on centralized information from this source in the future
	glog.V(1).Infof("[init] uploading currently used configuration to the cluster")
	if err := uploadconfigphase.UploadConfiguration(i.cfg, client); err != nil {
		return errors.Wrap(err, "error uploading configuration")
	}

	glog.V(1).Infof("[init] creating kubelet configuration configmap")
	if err := kubeletphase.CreateConfigMap(i.cfg, client); err != nil {
		return errors.Wrap(err, "error creating kubelet configuration ConfigMap")
	}

	// PHASE 4: Mark the master with the right label/taint
	glog.V(1).Infof("[init] marking the master with right label")
	if err := markmasterphase.MarkMaster(client, i.cfg.NodeRegistration.Name, i.cfg.NodeRegistration.Taints); err != nil {
		return errors.Wrap(err, "error marking master")
	}

	// This feature is disabled by default
	if features.Enabled(i.cfg.FeatureGates, features.DynamicKubeletConfig) {
		kubeletVersion, err := preflight.GetKubeletVersion(utilsexec.New())
		if err != nil {
			return err
		}

		// Enable dynamic kubelet configuration for the node.
		if err := kubeletphase.EnableDynamicConfigForNode(client, i.cfg.NodeRegistration.Name, kubeletVersion); err != nil {
			return errors.Wrap(err, "error enabling dynamic kubelet configuration")
		}
	}

	// PHASE 5: Set up the node bootstrap tokens
	tokens := []string{}
	for _, bt := range i.cfg.BootstrapTokens {
		tokens = append(tokens, bt.Token.String())
	}
	if !i.skipTokenPrint {
		if len(tokens) == 1 {
			fmt.Printf("[bootstraptoken] using token: %s\n", tokens[0])
		} else if len(tokens) > 1 {
			fmt.Printf("[bootstraptoken] using tokens: %v\n", tokens)
		}
	}

	// Create the default node bootstrap token
	glog.V(1).Infof("[init] creating RBAC rules to generate default bootstrap token")
	if err := nodebootstraptokenphase.UpdateOrCreateTokens(client, false, i.cfg.BootstrapTokens); err != nil {
		return errors.Wrap(err, "error updating or creating token")
	}
	// Create RBAC rules that makes the bootstrap tokens able to post CSRs
	glog.V(1).Infof("[init] creating RBAC rules to allow bootstrap tokens to post CSR")
	if err := nodebootstraptokenphase.AllowBootstrapTokensToPostCSRs(client); err != nil {
		return errors.Wrap(err, "error allowing bootstrap tokens to post CSRs")
	}
	// Create RBAC rules that makes the bootstrap tokens able to get their CSRs approved automatically
	glog.V(1).Infof("[init] creating RBAC rules to automatic approval of CSRs automatically")
	if err := nodebootstraptokenphase.AutoApproveNodeBootstrapTokens(client); err != nil {
		return errors.Wrap(err, "error auto-approving node bootstrap tokens")
	}

	// Create/update RBAC rules that makes the nodes to rotate certificates and get their CSRs approved automatically
	glog.V(1).Infof("[init] creating/updating RBAC rules for rotating certificate")
	if err := nodebootstraptokenphase.AutoApproveNodeCertificateRotation(client); err != nil {
		return err
	}

	// Create the cluster-info ConfigMap with the associated RBAC rules
	glog.V(1).Infof("[init] creating bootstrap configmap")
	if err := clusterinfophase.CreateBootstrapConfigMapIfNotExists(client, adminKubeConfigPath); err != nil {
		return errors.Wrap(err, "error creating bootstrap configmap")
	}
	glog.V(1).Infof("[init] creating ClusterInfo RBAC rules")
	if err := clusterinfophase.CreateClusterInfoRBACRules(client); err != nil {
		return errors.Wrap(err, "error creating clusterinfo RBAC rules")
	}

	glog.V(1).Infof("[init] ensuring DNS addon")
	if err := dnsaddonphase.EnsureDNSAddon(i.cfg, client); err != nil {
		return errors.Wrap(err, "error ensuring dns addon")
	}

	glog.V(1).Infof("[init] ensuring proxy addon")
	if err := proxyaddonphase.EnsureProxyAddon(i.cfg, client); err != nil {
		return errors.Wrap(err, "error ensuring proxy addon")
	}

	// Exit earlier if we're dryrunning
	if i.dryRun {
		fmt.Println("[dryrun] finished dry-running successfully. Above are the resources that would be created")
		return nil
	}

	// Prints the join command, multiple times in case the user has multiple tokens
	for _, token := range tokens {
		if err := printJoinCommand(out, adminKubeConfigPath, token, i.skipTokenPrint); err != nil {
			return errors.Wrap(err, "failed to print join command")
		}
	}
	return nil
}

func printJoinCommand(out io.Writer, adminKubeConfigPath, token string, skipTokenPrint bool) error {
	joinCommand, err := cmdutil.GetJoinCommand(adminKubeConfigPath, token, skipTokenPrint)
	if err != nil {
		return err
	}

	ctx := map[string]string{
		"KubeConfigPath": adminKubeConfigPath,
		"joinCommand":    joinCommand,
	}

	return initDoneTempl.Execute(out, ctx)
}

// getDirectoriesToUse returns the (in order) certificates, kubeconfig and Static Pod manifest directories, followed by a possible error
// This behaves differently when dry-running vs the normal flow
func getDirectoriesToUse(dryRun bool, dryRunDir string, defaultPkiDir string) (string, string, string, string, error) {
	if dryRun {
		// Use the same temp dir for all
		return dryRunDir, dryRunDir, dryRunDir, dryRunDir, nil
	}

	return defaultPkiDir, kubeadmconstants.KubernetesDir, kubeadmconstants.GetStaticPodDirectory(), kubeadmconstants.KubeletRunDirectory, nil
}
