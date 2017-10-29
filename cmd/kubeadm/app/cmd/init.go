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

	"github.com/renstrom/dedent"
	"github.com/spf13/cobra"
	flag "github.com/spf13/pflag"

	"k8s.io/apimachinery/pkg/runtime"
	clientset "k8s.io/client-go/kubernetes"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiext "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha1"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/validation"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/features"
	"k8s.io/kubernetes/cmd/kubeadm/app/images"
	dnsaddonphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/addons/dns"
	proxyaddonphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/addons/proxy"
	clusterinfophase "k8s.io/kubernetes/cmd/kubeadm/app/phases/bootstraptoken/clusterinfo"
	nodebootstraptokenphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/bootstraptoken/node"
	certsphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/certs"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/certs/pkiutil"
	controlplanephase "k8s.io/kubernetes/cmd/kubeadm/app/phases/controlplane"
	etcdphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/etcd"
	kubeconfigphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/kubeconfig"
	markmasterphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/markmaster"
	selfhostingphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/selfhosting"
	uploadconfigphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/uploadconfig"
	"k8s.io/kubernetes/cmd/kubeadm/app/preflight"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/apiclient"
	configutil "k8s.io/kubernetes/cmd/kubeadm/app/util/config"
	dryrunutil "k8s.io/kubernetes/cmd/kubeadm/app/util/dryrun"
	kubeconfigutil "k8s.io/kubernetes/cmd/kubeadm/app/util/kubeconfig"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/pubkeypin"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/util/version"
)

var (
	initDoneTempl = template.Must(template.New("init").Parse(dedent.Dedent(`
		Your Kubernetes master has initialized successfully!

		To start using your cluster, you need to run (as a regular user):

		  mkdir -p $HOME/.kube
		  sudo cp -i {{.KubeConfigPath}} $HOME/.kube/config
		  sudo chown $(id -u):$(id -g) $HOME/.kube/config

		You should now deploy a pod network to the cluster.
		Run "kubectl apply -f [podnetwork].yaml" with one of the options listed at:
		  https://kubernetes.io/docs/concepts/cluster-administration/addons/

		You can now join any number of machines by running the following on each node
		as root:

		  kubeadm join --token {{.Token}} {{.MasterHostPort}} --discovery-token-ca-cert-hash {{.CAPubKeyPin}}

		`)))

	kubeletFailTempl = template.Must(template.New("init").Parse(dedent.Dedent(`
		Unfortunately, an error has occurred:
			{{ .Error }}

		This error is likely caused by that:
			- The kubelet is not running
			- The kubelet is unhealthy due to a misconfiguration of the node in some way (required cgroups disabled)
			- There is no internet connection; so the kubelet can't pull the following control plane images:
				- {{ .APIServerImage }}
				- {{ .ControllerManagerImage }}
				- {{ .SchedulerImage }}

		You can troubleshoot this for example with the following commands if you're on a systemd-powered system:
			- 'systemctl status kubelet'
			- 'journalctl -xeu kubelet'
		`)))
)

// NewCmdInit returns "kubeadm init" command.
func NewCmdInit(out io.Writer) *cobra.Command {
	cfg := &kubeadmapiext.MasterConfiguration{}
	legacyscheme.Scheme.Default(cfg)

	var cfgPath string
	var skipPreFlight bool
	var skipTokenPrint bool
	var dryRun bool
	var featureGatesString string

	cmd := &cobra.Command{
		Use:   "init",
		Short: "Run this in order to set up the Kubernetes master",
		Run: func(cmd *cobra.Command, args []string) {
			var err error
			if cfg.FeatureGates, err = features.NewFeatureGate(&features.InitFeatureGates, featureGatesString); err != nil {
				kubeadmutil.CheckErr(err)
			}

			legacyscheme.Scheme.Default(cfg)
			internalcfg := &kubeadmapi.MasterConfiguration{}
			legacyscheme.Scheme.Convert(cfg, internalcfg, nil)

			i, err := NewInit(cfgPath, internalcfg, skipPreFlight, skipTokenPrint, dryRun)
			kubeadmutil.CheckErr(err)
			kubeadmutil.CheckErr(i.Validate(cmd))
			kubeadmutil.CheckErr(i.Run(out))
		},
	}

	AddInitConfigFlags(cmd.PersistentFlags(), cfg, &featureGatesString)
	AddInitOtherFlags(cmd.PersistentFlags(), &cfgPath, &skipPreFlight, &skipTokenPrint, &dryRun)

	return cmd
}

// AddInitConfigFlags adds init flags bound to the config to the specified flagset
func AddInitConfigFlags(flagSet *flag.FlagSet, cfg *kubeadmapiext.MasterConfiguration, featureGatesString *string) {
	flagSet.StringVar(
		&cfg.API.AdvertiseAddress, "apiserver-advertise-address", cfg.API.AdvertiseAddress,
		"The IP address the API Server will advertise it's listening on. 0.0.0.0 means the default network interface's address.",
	)
	flagSet.Int32Var(
		&cfg.API.BindPort, "apiserver-bind-port", cfg.API.BindPort,
		"Port for the API Server to bind to.",
	)
	flagSet.StringVar(
		&cfg.Networking.ServiceSubnet, "service-cidr", cfg.Networking.ServiceSubnet,
		"Use alternative range of IP address for service VIPs.",
	)
	flagSet.StringVar(
		&cfg.Networking.PodSubnet, "pod-network-cidr", cfg.Networking.PodSubnet,
		"Specify range of IP addresses for the pod network; if set, the control plane will automatically allocate CIDRs for every node.",
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
		&cfg.APIServerCertSANs, "apiserver-cert-extra-sans", cfg.APIServerCertSANs,
		`Optional extra altnames to use for the API Server serving cert. Can be both IP addresses and dns names.`,
	)
	flagSet.StringVar(
		&cfg.NodeName, "node-name", cfg.NodeName,
		`Specify the node name.`,
	)
	flagSet.StringVar(
		&cfg.Token, "token", cfg.Token,
		"The token to use for establishing bidirectional trust between nodes and masters.",
	)
	flagSet.DurationVar(
		&cfg.TokenTTL.Duration, "token-ttl", cfg.TokenTTL.Duration,
		"The duration before the bootstrap token is automatically deleted. 0 means 'never expires'.",
	)
	flagSet.StringVar(featureGatesString, "feature-gates", *featureGatesString, "A set of key=value pairs that describe feature gates for various features. "+
		"Options are:\n"+strings.Join(features.KnownFeatures(&features.InitFeatureGates), "\n"))
}

// AddInitOtherFlags adds init flags that are not bound to a configuration file to the given flagset
func AddInitOtherFlags(flagSet *flag.FlagSet, cfgPath *string, skipPreFlight, skipTokenPrint, dryRun *bool) {
	flagSet.StringVar(
		cfgPath, "config", *cfgPath,
		"Path to kubeadm config file. WARNING: Usage of a configuration file is experimental.",
	)
	// Note: All flags that are not bound to the cfg object should be whitelisted in cmd/kubeadm/app/apis/kubeadm/validation/validation.go
	flagSet.BoolVar(
		skipPreFlight, "skip-preflight-checks", *skipPreFlight,
		"Skip preflight checks normally run before modifying the system.",
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

// NewInit validates given arguments and instantiates Init struct with provided information.
func NewInit(cfgPath string, cfg *kubeadmapi.MasterConfiguration, skipPreFlight, skipTokenPrint, dryRun bool) (*Init, error) {

	fmt.Println("[kubeadm] WARNING: kubeadm is in beta, please do not use it for production clusters.")

	if cfgPath != "" {
		b, err := ioutil.ReadFile(cfgPath)
		if err != nil {
			return nil, fmt.Errorf("unable to read config from %q [%v]", cfgPath, err)
		}
		if err := runtime.DecodeInto(legacyscheme.Codecs.UniversalDecoder(), b, cfg); err != nil {
			return nil, fmt.Errorf("unable to decode config from %q [%v]", cfgPath, err)
		}
	}

	// Set defaults dynamically that the API group defaulting can't (by fetching information from the internet, looking up network interfaces, etc.)
	err := configutil.SetInitDynamicDefaults(cfg)
	if err != nil {
		return nil, err
	}

	if err := features.ValidateVersion(features.InitFeatureGates, cfg.FeatureGates, cfg.KubernetesVersion); err != nil {
		return nil, err
	}

	fmt.Printf("[init] Using Kubernetes version: %s\n", cfg.KubernetesVersion)
	fmt.Printf("[init] Using Authorization modes: %v\n", cfg.AuthorizationModes)

	// Warn about the limitations with the current cloudprovider solution.
	if cfg.CloudProvider != "" {
		fmt.Println("[init] WARNING: For cloudprovider integrations to work --cloud-provider must be set for all kubelets in the cluster.")
		fmt.Println("\t(/etc/systemd/system/kubelet.service.d/10-kubeadm.conf should be edited for this purpose)")
	}

	if !skipPreFlight {
		fmt.Println("[preflight] Running pre-flight checks.")

		if err := preflight.RunInitMasterChecks(cfg); err != nil {
			return nil, err
		}

		// Try to start the kubelet service in case it's inactive
		preflight.TryStartKubelet()
	} else {
		fmt.Println("[preflight] Skipping pre-flight checks.")
	}

	return &Init{cfg: cfg, skipTokenPrint: skipTokenPrint, dryRun: dryRun}, nil
}

// Init defines struct used by "kubeadm init" command
type Init struct {
	cfg            *kubeadmapi.MasterConfiguration
	skipTokenPrint bool
	dryRun         bool
}

// Validate validates configuration passed to "kubeadm init"
func (i *Init) Validate(cmd *cobra.Command) error {
	if err := validation.ValidateMixedArguments(cmd.Flags()); err != nil {
		return err
	}
	return validation.ValidateMasterConfiguration(i.cfg).ToAggregate()
}

// Run executes master node provisioning, including certificates, needed static pod manifests, etc.
func (i *Init) Run(out io.Writer) error {

	k8sVersion, err := version.ParseSemantic(i.cfg.KubernetesVersion)
	if err != nil {
		return fmt.Errorf("could not parse Kubernetes version %q: %v", i.cfg.KubernetesVersion, err)
	}

	// Get directories to write files to; can be faked if we're dry-running
	realCertsDir := i.cfg.CertificatesDir
	certsDirToWriteTo, kubeConfigDir, manifestDir, err := getDirectoriesToUse(i.dryRun, i.cfg.CertificatesDir)
	if err != nil {
		return fmt.Errorf("error getting directories to use: %v", err)
	}
	// certsDirToWriteTo is gonna equal cfg.CertificatesDir in the normal case, but gonna be a temp directory if dryrunning
	i.cfg.CertificatesDir = certsDirToWriteTo

	adminKubeConfigPath := filepath.Join(kubeConfigDir, kubeadmconstants.AdminKubeConfigFileName)

	if res, _ := certsphase.UsingExternalCA(i.cfg); !res {

		// PHASE 1: Generate certificates
		if err := certsphase.CreatePKIAssets(i.cfg); err != nil {
			return err
		}

		// PHASE 2: Generate kubeconfig files for the admin and the kubelet
		if err := kubeconfigphase.CreateInitKubeConfigFiles(kubeConfigDir, i.cfg); err != nil {
			return err
		}

	} else {
		fmt.Println("[externalca] The file 'ca.key' was not found, yet all other certificates are present. Using external CA mode - certificates or kubeconfig will not be generated.")
	}

	// Temporarily set cfg.CertificatesDir to the "real value" when writing controlplane manifests
	// This is needed for writing the right kind of manifests
	i.cfg.CertificatesDir = realCertsDir

	// PHASE 3: Bootstrap the control plane
	if err := controlplanephase.CreateInitStaticPodManifestFiles(manifestDir, i.cfg); err != nil {
		return fmt.Errorf("error creating init static pod manifest files: %v", err)
	}
	// Add etcd static pod spec only if external etcd is not configured
	if len(i.cfg.Etcd.Endpoints) == 0 {
		if err := etcdphase.CreateLocalEtcdStaticPodManifestFile(manifestDir, i.cfg); err != nil {
			return fmt.Errorf("error creating local etcd static pod manifest file: %v", err)
		}
	}

	// Revert the earlier CertificatesDir assignment to the directory that can be written to
	i.cfg.CertificatesDir = certsDirToWriteTo

	// If we're dry-running, print the generated manifests
	if err := printFilesIfDryRunning(i.dryRun, manifestDir); err != nil {
		return fmt.Errorf("error printing files on dryrun: %v", err)
	}

	// Create a kubernetes client and wait for the API server to be healthy (if not dryrunning)
	client, err := createClient(i.cfg, i.dryRun)
	if err != nil {
		return fmt.Errorf("error creating client: %v", err)
	}

	// waiter holds the apiclient.Waiter implementation of choice, responsible for querying the API server in various ways and waiting for conditions to be fulfilled
	waiter := getWaiter(i.dryRun, client)

	if err := waitForAPIAndKubelet(waiter); err != nil {
		ctx := map[string]string{
			"Error":                  fmt.Sprintf("%v", err),
			"APIServerImage":         images.GetCoreImage(kubeadmconstants.KubeAPIServer, i.cfg.GetControlPlaneImageRepository(), i.cfg.KubernetesVersion, i.cfg.UnifiedControlPlaneImage),
			"ControllerManagerImage": images.GetCoreImage(kubeadmconstants.KubeControllerManager, i.cfg.GetControlPlaneImageRepository(), i.cfg.KubernetesVersion, i.cfg.UnifiedControlPlaneImage),
			"SchedulerImage":         images.GetCoreImage(kubeadmconstants.KubeScheduler, i.cfg.GetControlPlaneImageRepository(), i.cfg.KubernetesVersion, i.cfg.UnifiedControlPlaneImage),
		}

		kubeletFailTempl.Execute(out, ctx)

		return fmt.Errorf("couldn't initialize a Kubernetes cluster")
	}

	// Upload currently used configuration to the cluster
	// Note: This is done right in the beginning of cluster initialization; as we might want to make other phases
	// depend on centralized information from this source in the future
	if err := uploadconfigphase.UploadConfiguration(i.cfg, client); err != nil {
		return fmt.Errorf("error uploading configuration: %v", err)
	}

	// PHASE 4: Mark the master with the right label/taint
	if err := markmasterphase.MarkMaster(client, i.cfg.NodeName); err != nil {
		return fmt.Errorf("error marking master: %v", err)
	}

	// PHASE 5: Set up the node bootstrap tokens
	if !i.skipTokenPrint {
		fmt.Printf("[bootstraptoken] Using token: %s\n", i.cfg.Token)
	}

	// Create the default node bootstrap token
	tokenDescription := "The default bootstrap token generated by 'kubeadm init'."
	if err := nodebootstraptokenphase.UpdateOrCreateToken(client, i.cfg.Token, false, i.cfg.TokenTTL.Duration, kubeadmconstants.DefaultTokenUsages, []string{kubeadmconstants.NodeBootstrapTokenAuthGroup}, tokenDescription); err != nil {
		return fmt.Errorf("error updating or creating token: %v", err)
	}
	// Create RBAC rules that makes the bootstrap tokens able to post CSRs
	if err := nodebootstraptokenphase.AllowBootstrapTokensToPostCSRs(client, k8sVersion); err != nil {
		return fmt.Errorf("error allowing bootstrap tokens to post CSRs: %v", err)
	}
	// Create RBAC rules that makes the bootstrap tokens able to get their CSRs approved automatically
	if err := nodebootstraptokenphase.AutoApproveNodeBootstrapTokens(client, k8sVersion); err != nil {
		return fmt.Errorf("error auto-approving node bootstrap tokens: %v", err)
	}

	// Create/update RBAC rules that makes the nodes to rotate certificates and get their CSRs approved automatically
	if err := nodebootstraptokenphase.AutoApproveNodeCertificateRotation(client, k8sVersion); err != nil {
		return err
	}

	// Create the cluster-info ConfigMap with the associated RBAC rules
	if err := clusterinfophase.CreateBootstrapConfigMapIfNotExists(client, adminKubeConfigPath); err != nil {
		return fmt.Errorf("error creating bootstrap configmap: %v", err)
	}
	if err := clusterinfophase.CreateClusterInfoRBACRules(client); err != nil {
		return fmt.Errorf("error creating clusterinfo RBAC rules: %v", err)
	}

	if err := dnsaddonphase.EnsureDNSAddon(i.cfg, client); err != nil {
		return fmt.Errorf("error ensuring dns addon: %v", err)
	}

	if err := proxyaddonphase.EnsureProxyAddon(i.cfg, client); err != nil {
		return fmt.Errorf("error ensuring proxy addon: %v", err)
	}

	// PHASE 7: Make the control plane self-hosted if feature gate is enabled
	if features.Enabled(i.cfg.FeatureGates, features.SelfHosting) {
		// Temporary control plane is up, now we create our self hosted control
		// plane components and remove the static manifests:
		fmt.Println("[self-hosted] Creating self-hosted control plane...")
		if err := selfhostingphase.CreateSelfHostedControlPlane(manifestDir, kubeConfigDir, i.cfg, client, waiter); err != nil {
			return fmt.Errorf("error creating self hosted control plane: %v", err)
		}
	}

	// Exit earlier if we're dryrunning
	if i.dryRun {
		fmt.Println("[dryrun] Finished dry-running successfully; above are the resources that would be created.")
		return nil
	}

	// Load the CA certificate from so we can pin its public key
	caCert, err := pkiutil.TryLoadCertFromDisk(i.cfg.CertificatesDir, kubeadmconstants.CACertAndKeyBaseName)
	if err != nil {
		return fmt.Errorf("error loading ca cert from disk: %v", err)
	}

	// Generate the Master host/port pair used by initDoneTempl
	masterHostPort, err := kubeadmutil.GetMasterHostPort(i.cfg)
	if err != nil {
		return fmt.Errorf("error getting master host port: %v", err)
	}

	ctx := map[string]string{
		"KubeConfigPath": adminKubeConfigPath,
		"Token":          i.cfg.Token,
		"CAPubKeyPin":    pubkeypin.Hash(caCert),
		"MasterHostPort": masterHostPort,
	}
	if i.skipTokenPrint {
		ctx["Token"] = "<value withheld>"
	}

	return initDoneTempl.Execute(out, ctx)
}

// createClient creates a clientset.Interface object
func createClient(cfg *kubeadmapi.MasterConfiguration, dryRun bool) (clientset.Interface, error) {
	if dryRun {
		// If we're dry-running; we should create a faked client that answers some GETs in order to be able to do the full init flow and just logs the rest of requests
		dryRunGetter := apiclient.NewInitDryRunGetter(cfg.NodeName, cfg.Networking.ServiceSubnet)
		return apiclient.NewDryRunClient(dryRunGetter, os.Stdout), nil
	}

	// If we're acting for real, we should create a connection to the API server and wait for it to come up
	return kubeconfigutil.ClientSetFromFile(kubeadmconstants.GetAdminKubeConfigPath())
}

// getDirectoriesToUse returns the (in order) certificates, kubeconfig and Static Pod manifest directories, followed by a possible error
// This behaves differently when dry-running vs the normal flow
func getDirectoriesToUse(dryRun bool, defaultPkiDir string) (string, string, string, error) {
	if dryRun {
		dryRunDir, err := ioutil.TempDir("", "kubeadm-init-dryrun")
		if err != nil {
			return "", "", "", fmt.Errorf("couldn't create a temporary directory: %v", err)
		}
		// Use the same temp dir for all
		return dryRunDir, dryRunDir, dryRunDir, nil
	}

	return defaultPkiDir, kubeadmconstants.KubernetesDir, kubeadmconstants.GetStaticPodDirectory(), nil
}

// printFilesIfDryRunning prints the Static Pod manifests to stdout and informs about the temporary directory to go and lookup
func printFilesIfDryRunning(dryRun bool, manifestDir string) error {
	if !dryRun {
		return nil
	}

	fmt.Printf("[dryrun] Wrote certificates, kubeconfig files and control plane manifests to %q\n", manifestDir)
	fmt.Println("[dryrun] Won't print certificates or kubeconfig files due to the sensitive nature of them")
	fmt.Printf("[dryrun] Please go and examine the %q directory for details about what would be written\n", manifestDir)

	// Print the contents of the upgraded manifests and pretend like they were in /etc/kubernetes/manifests
	files := []dryrunutil.FileToPrint{}
	for _, component := range kubeadmconstants.MasterComponents {
		realPath := kubeadmconstants.GetStaticPodFilepath(component, manifestDir)
		outputPath := kubeadmconstants.GetStaticPodFilepath(component, kubeadmconstants.GetStaticPodDirectory())
		files = append(files, dryrunutil.NewFileToPrint(realPath, outputPath))
	}

	return dryrunutil.PrintDryRunFiles(files, os.Stdout)
}

// getWaiter gets the right waiter implementation for the right occasion
func getWaiter(dryRun bool, client clientset.Interface) apiclient.Waiter {
	if dryRun {
		return dryrunutil.NewWaiter()
	}
	return apiclient.NewKubeWaiter(client, 30*time.Minute, os.Stdout)
}

// waitForAPIAndKubelet waits primarily for the API server to come up. If that takes a long time, and the kubelet
// /healthz and /healthz/syncloop endpoints continuously are unhealthy, kubeadm will error out after a period of
// backoffing exponentially
func waitForAPIAndKubelet(waiter apiclient.Waiter) error {
	errorChan := make(chan error)

	fmt.Printf("[init] Waiting for the kubelet to boot up the control plane as Static Pods from directory %q.\n", kubeadmconstants.GetStaticPodDirectory())
	fmt.Println("[init] This often takes around a minute; or longer if the control plane images have to be pulled.")

	go func(errC chan error, waiter apiclient.Waiter) {
		// This goroutine can only make kubeadm init fail. If this check succeeds, it won't do anything special
		if err := waiter.WaitForHealthyKubelet(40*time.Second, "http://localhost:10255/healthz"); err != nil {
			errC <- err
		}
	}(errorChan, waiter)

	go func(errC chan error, waiter apiclient.Waiter) {
		// This goroutine can only make kubeadm init fail. If this check succeeds, it won't do anything special
		if err := waiter.WaitForHealthyKubelet(60*time.Second, "http://localhost:10255/healthz/syncloop"); err != nil {
			errC <- err
		}
	}(errorChan, waiter)

	go func(errC chan error, waiter apiclient.Waiter) {
		// This main goroutine sends whatever WaitForAPI returns (error or not) to the channel
		// This in order to continue on success (nil error), or just fail if
		errC <- waiter.WaitForAPI()
	}(errorChan, waiter)

	// This call is blocking until one of the goroutines sends to errorChan
	return <-errorChan
}
