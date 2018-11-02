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
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
	"text/template"
	"time"

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
	kubeletphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/kubelet"
	markmasterphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/markmaster"
	patchnodephase "k8s.io/kubernetes/cmd/kubeadm/app/phases/patchnode"
	selfhostingphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/selfhosting"
	uploadconfigphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/uploadconfig"
	"k8s.io/kubernetes/cmd/kubeadm/app/preflight"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/apiclient"
	configutil "k8s.io/kubernetes/cmd/kubeadm/app/util/config"
	dryrunutil "k8s.io/kubernetes/cmd/kubeadm/app/util/dryrun"
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
	ignorePreflightErrors sets.String
	certificatesDir       string
	dryRunDir             string
	externalCA            bool
	client                clientset.Interface
}

// NewCmdInit returns "kubeadm init" command.
func NewCmdInit(out io.Writer) *cobra.Command {
	options := newInitOptions()
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

	// adds command flags
	AddInitConfigFlags(cmd.PersistentFlags(), options.externalcfg, &options.featureGatesString)
	AddInitOtherFlags(cmd.PersistentFlags(), &options.cfgPath, &options.skipTokenPrint, &options.dryRun, &options.ignorePreflightErrors)
	options.bto.AddTokenFlag(cmd.PersistentFlags())
	options.bto.AddTTLFlag(cmd.PersistentFlags())

	// initialize the workflow runner with the list of phases
	initRunner.AppendPhase(phases.NewPreflightMasterPhase())
	initRunner.AppendPhase(phases.NewKubeletStartPhase())
	initRunner.AppendPhase(phases.NewCertsPhase())
	initRunner.AppendPhase(phases.NewKubeConfigPhase())
	initRunner.AppendPhase(phases.NewControlPlanePhase())
	initRunner.AppendPhase(phases.NewEtcdPhase())
	// TODO: add other phases to the runner.

	// sets the data builder function, that will be used by the runner
	// both when running the entire workflow or single phases
	initRunner.SetDataInitializer(func() (workflow.RunData, error) {
		return newInitData(cmd, options)
	})

	// binds the Runner to kubeadm init command by altering
	// command help, adding --skip-phases flag and by adding phases subcommands
	initRunner.BindToCommand(cmd)

	return cmd
}

// AddInitConfigFlags adds init flags bound to the config to the specified flagset
func AddInitConfigFlags(flagSet *flag.FlagSet, cfg *kubeadmapiv1beta1.InitConfiguration, featureGatesString *string) {
	flagSet.StringVar(
		&cfg.APIEndpoint.AdvertiseAddress, "apiserver-advertise-address", cfg.APIEndpoint.AdvertiseAddress,
		"The IP address the API Server will advertise it's listening on. Specify '0.0.0.0' to use the address of the default network interface.",
	)
	flagSet.Int32Var(
		&cfg.APIEndpoint.BindPort, "apiserver-bind-port", cfg.APIEndpoint.BindPort,
		"Port for the API Server to bind to.",
	)
	flagSet.StringVar(
		&cfg.Networking.ServiceSubnet, "service-cidr", cfg.Networking.ServiceSubnet,
		"Use alternative range of IP address for service VIPs.",
	)
	flagSet.StringVar(
		&cfg.Networking.PodSubnet, "pod-network-cidr", cfg.Networking.PodSubnet,
		"Specify range of IP addresses for the pod network. If set, the control plane will automatically allocate CIDRs for every node.",
	)
	flagSet.StringVar(
		&cfg.Networking.DNSDomain, "service-dns-domain", cfg.Networking.DNSDomain,
		`Use alternative domain for services, e.g. "myorg.internal".`,
	)
	flagSet.StringVar(
		&cfg.KubernetesVersion, "kubernetes-version", cfg.KubernetesVersion,
		`Choose a specific Kubernetes version for the control plane.`,
	)
	flagSet.StringVar(
		&cfg.CertificatesDir, "cert-dir", cfg.CertificatesDir,
		`The path where to save and store the certificates.`,
	)
	flagSet.StringSliceVar(
		&cfg.APIServer.CertSANs, "apiserver-cert-extra-sans", cfg.APIServer.CertSANs,
		`Optional extra Subject Alternative Names (SANs) to use for the API Server serving certificate. Can be both IP addresses and DNS names.`,
	)
	flagSet.StringVar(
		&cfg.NodeRegistration.Name, "node-name", cfg.NodeRegistration.Name,
		`Specify the node name.`,
	)
	flagSet.StringVar(
		&cfg.NodeRegistration.CRISocket, "cri-socket", cfg.NodeRegistration.CRISocket,
		`Specify the CRI socket to connect to.`,
	)
	flagSet.StringVar(featureGatesString, "feature-gates", *featureGatesString, "A set of key=value pairs that describe feature gates for various features. "+
		"Options are:\n"+strings.Join(features.KnownFeatures(&features.InitFeatureGates), "\n"))
}

// AddInitOtherFlags adds init flags that are not bound to a configuration file to the given flagset
func AddInitOtherFlags(flagSet *flag.FlagSet, cfgPath *string, skipTokenPrint, dryRun *bool, ignorePreflightErrors *[]string) {
	flagSet.StringVar(
		cfgPath, "config", *cfgPath,
		"Path to kubeadm config file. WARNING: Usage of a configuration file is experimental.",
	)
	flagSet.StringSliceVar(
		ignorePreflightErrors, "ignore-preflight-errors", *ignorePreflightErrors,
		"A list of checks whose errors will be shown as warnings. Example: 'IsPrivilegedUser,Swap'. Value 'all' ignores errors from all checks.",
	)
	// Note: All flags that are not bound to the cfg object should be whitelisted in cmd/kubeadm/app/apis/kubeadm/validation/validation.go
	flagSet.BoolVar(
		skipTokenPrint, "skip-token-print", *skipTokenPrint,
		"Skip printing of the default bootstrap token generated by 'kubeadm init'.",
	)
	// Note: All flags that are not bound to the cfg object should be whitelisted in cmd/kubeadm/app/apis/kubeadm/validation/validation.go
	flagSet.BoolVar(
		dryRun, "dry-run", *dryRun,
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
		externalcfg: externalcfg,
		bto:         bto,
	}
}

// newInitData returns a new initData struct to be used for the execution of the kubeadm init workflow.
// This func takes care of validating initOptions passed to the command, and then it converts
// options into the internal InitConfiguration type that is used as input all the phases in the kubeadm init workflow
func newInitData(cmd *cobra.Command, options *initOptions) (initData, error) {
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

	return initData{
		cfg:                   cfg,
		certificatesDir:       cfg.CertificatesDir,
		skipTokenPrint:        options.skipTokenPrint,
		dryRun:                options.dryRun,
		dryRunDir:             dryRunDir,
		ignorePreflightErrors: ignorePreflightErrorsSet,
		externalCA:            externalCA,
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
	return kubeadmconstants.KubernetesDir
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
			d.client, err = kubeconfigutil.ClientSetFromFile(kubeadmconstants.GetAdminKubeConfigPath())
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
	certsDirToWriteTo, kubeConfigDir, manifestDir, _, err := getDirectoriesToUse(i.dryRun, i.dryRunDir, i.cfg.CertificatesDir)
	if err != nil {
		return errors.Wrap(err, "error getting directories to use")
	}

	// certsDirToWriteTo is gonna equal cfg.CertificatesDir in the normal case, but gonna be a temp directory if dryrunning
	i.cfg.CertificatesDir = certsDirToWriteTo

	adminKubeConfigPath := filepath.Join(kubeConfigDir, kubeadmconstants.AdminKubeConfigFileName)

	// If we're dry-running, print the generated manifests
	if err := printFilesIfDryRunning(i.dryRun, manifestDir); err != nil {
		return errors.Wrap(err, "error printing files on dryrun")
	}

	// Create a Kubernetes client and wait for the API server to be healthy (if not dryrunning)
	glog.V(1).Infof("creating Kubernetes client")
	client, err := createClient(i.cfg, i.dryRun)
	if err != nil {
		return errors.Wrap(err, "error creating client")
	}

	// waiter holds the apiclient.Waiter implementation of choice, responsible for querying the API server in various ways and waiting for conditions to be fulfilled
	glog.V(1).Infof("[init] waiting for the API server to be healthy")
	waiter := getWaiter(i, client)

	fmt.Printf("[init] waiting for the kubelet to boot up the control plane as Static Pods from directory %q \n", kubeadmconstants.GetStaticPodDirectory())

	if err := waitForKubeletAndFunc(waiter, waiter.WaitForAPI); err != nil {
		ctx := map[string]string{
			"Error": fmt.Sprintf("%v", err),
		}

		kubeletFailTempl.Execute(out, ctx)

		return errors.New("couldn't initialize a Kubernetes cluster")
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

	glog.V(1).Infof("[init] preserving the crisocket information for the master")
	if err := patchnodephase.AnnotateCRISocket(client, i.cfg.NodeRegistration.Name, i.cfg.NodeRegistration.CRISocket); err != nil {
		return errors.Wrap(err, "error uploading crisocket")
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

	// PHASE 7: Make the control plane self-hosted if feature gate is enabled
	if features.Enabled(i.cfg.FeatureGates, features.SelfHosting) {
		glog.V(1).Infof("[init] feature gate is enabled. Making control plane self-hosted")
		// Temporary control plane is up, now we create our self hosted control
		// plane components and remove the static manifests:
		fmt.Println("[self-hosted] creating self-hosted control plane")
		if err := selfhostingphase.CreateSelfHostedControlPlane(manifestDir, kubeConfigDir, i.cfg, client, waiter, i.dryRun); err != nil {
			return errors.Wrap(err, "error creating self hosted control plane")
		}
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

// createClient creates a clientset.Interface object
func createClient(cfg *kubeadmapi.InitConfiguration, dryRun bool) (clientset.Interface, error) {
	if dryRun {
		// If we're dry-running; we should create a faked client that answers some GETs in order to be able to do the full init flow and just logs the rest of requests
		dryRunGetter := apiclient.NewInitDryRunGetter(cfg.NodeRegistration.Name, cfg.Networking.ServiceSubnet)
		return apiclient.NewDryRunClient(dryRunGetter, os.Stdout), nil
	}

	// If we're acting for real, we should create a connection to the API server and wait for it to come up
	return kubeconfigutil.ClientSetFromFile(kubeadmconstants.GetAdminKubeConfigPath())
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

// printFilesIfDryRunning prints the Static Pod manifests to stdout and informs about the temporary directory to go and lookup
func printFilesIfDryRunning(dryRun bool, manifestDir string) error {
	if !dryRun {
		return nil
	}

	fmt.Printf("[dryrun] wrote certificates, kubeconfig files and control plane manifests to the %q directory\n", manifestDir)
	fmt.Println("[dryrun] the certificates or kubeconfig files would not be printed due to their sensitive nature")
	fmt.Printf("[dryrun] please examine the %q directory for details about what would be written\n", manifestDir)

	// Print the contents of the upgraded manifests and pretend like they were in /etc/kubernetes/manifests
	files := []dryrunutil.FileToPrint{}
	// Print static pod manifests
	for _, component := range kubeadmconstants.MasterComponents {
		realPath := kubeadmconstants.GetStaticPodFilepath(component, manifestDir)
		outputPath := kubeadmconstants.GetStaticPodFilepath(component, kubeadmconstants.GetStaticPodDirectory())
		files = append(files, dryrunutil.NewFileToPrint(realPath, outputPath))
	}
	// Print kubelet config manifests
	kubeletConfigFiles := []string{kubeadmconstants.KubeletConfigurationFileName, kubeadmconstants.KubeletEnvFileName}
	for _, filename := range kubeletConfigFiles {
		realPath := filepath.Join(manifestDir, filename)
		outputPath := filepath.Join(kubeadmconstants.KubeletRunDirectory, filename)
		files = append(files, dryrunutil.NewFileToPrint(realPath, outputPath))
	}

	return dryrunutil.PrintDryRunFiles(files, os.Stdout)
}

// getWaiter gets the right waiter implementation for the right occasion
func getWaiter(ctx *initData, client clientset.Interface) apiclient.Waiter {
	if ctx.dryRun {
		return dryrunutil.NewWaiter()
	}

	// We know that the images should be cached locally already as we have pulled them using
	// crictl in the preflight checks. Hence we can have a pretty short timeout for the kubelet
	// to start creating Static Pods.
	timeout := 4 * time.Minute
	return apiclient.NewKubeWaiter(client, timeout, os.Stdout)
}

// waitForKubeletAndFunc waits primarily for the function f to execute, even though it might take some time. If that takes a long time, and the kubelet
// /healthz continuously are unhealthy, kubeadm will error out after a period of exponential backoff
func waitForKubeletAndFunc(waiter apiclient.Waiter, f func() error) error {
	errorChan := make(chan error)

	go func(errC chan error, waiter apiclient.Waiter) {
		// This goroutine can only make kubeadm init fail. If this check succeeds, it won't do anything special
		if err := waiter.WaitForHealthyKubelet(40*time.Second, fmt.Sprintf("http://localhost:%d/healthz", kubeadmconstants.KubeletHealthzPort)); err != nil {
			errC <- err
		}
	}(errorChan, waiter)

	go func(errC chan error, waiter apiclient.Waiter) {
		// This main goroutine sends whatever the f function returns (error or not) to the channel
		// This in order to continue on success (nil error), or just fail if the function returns an error
		errC <- f()
	}(errorChan, waiter)

	// This call is blocking until one of the goroutines sends to errorChan
	return <-errorChan
}
