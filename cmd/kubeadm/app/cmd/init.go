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
	"github.com/renstrom/dedent"
	"github.com/spf13/cobra"
	flag "github.com/spf13/pflag"

	"k8s.io/apimachinery/pkg/util/sets"
	clientset "k8s.io/client-go/kubernetes"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmscheme "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/scheme"
	kubeadmapiv1alpha2 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha2"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/validation"
	cmdutil "k8s.io/kubernetes/cmd/kubeadm/app/cmd/util"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/features"
	"k8s.io/kubernetes/cmd/kubeadm/app/images"
	dnsaddonphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/addons/dns"
	proxyaddonphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/addons/proxy"
	clusterinfophase "k8s.io/kubernetes/cmd/kubeadm/app/phases/bootstraptoken/clusterinfo"
	nodebootstraptokenphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/bootstraptoken/node"
	certsphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/certs"
	controlplanephase "k8s.io/kubernetes/cmd/kubeadm/app/phases/controlplane"
	etcdphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/etcd"
	kubeconfigphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/kubeconfig"
	kubeletphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/kubelet"
	markmasterphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/markmaster"
	selfhostingphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/selfhosting"
	uploadconfigphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/uploadconfig"
	"k8s.io/kubernetes/cmd/kubeadm/app/preflight"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/apiclient"
	auditutil "k8s.io/kubernetes/cmd/kubeadm/app/util/audit"
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
			- No internet connection is available so the kubelet cannot pull or find the following control plane images:
				- {{ .APIServerImage }}
				- {{ .ControllerManagerImage }}
				- {{ .SchedulerImage }}
{{ .EtcdImage }}
				- You can check or miligate this in beforehand with "kubeadm config images pull" to make sure the images
				  are downloaded locally and cached.

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

// NewCmdInit returns "kubeadm init" command.
func NewCmdInit(out io.Writer) *cobra.Command {
	externalcfg := &kubeadmapiv1alpha2.MasterConfiguration{}
	kubeadmscheme.Scheme.Default(externalcfg)

	var cfgPath string
	var skipPreFlight bool
	var skipTokenPrint bool
	var dryRun bool
	var featureGatesString string
	var ignorePreflightErrors []string

	cmd := &cobra.Command{
		Use:   "init",
		Short: "Run this command in order to set up the Kubernetes master.",
		Run: func(cmd *cobra.Command, args []string) {

			kubeadmscheme.Scheme.Default(externalcfg)

			var err error
			if externalcfg.FeatureGates, err = features.NewFeatureGate(&features.InitFeatureGates, featureGatesString); err != nil {
				kubeadmutil.CheckErr(err)
			}

			ignorePreflightErrorsSet, err := validation.ValidateIgnorePreflightErrors(ignorePreflightErrors, skipPreFlight)
			kubeadmutil.CheckErr(err)

			err = validation.ValidateMixedArguments(cmd.Flags())
			kubeadmutil.CheckErr(err)

			i, err := NewInit(cfgPath, externalcfg, ignorePreflightErrorsSet, skipTokenPrint, dryRun)
			kubeadmutil.CheckErr(err)
			kubeadmutil.CheckErr(i.Run(out))
		},
	}

	AddInitConfigFlags(cmd.PersistentFlags(), externalcfg, &featureGatesString)
	AddInitOtherFlags(cmd.PersistentFlags(), &cfgPath, &skipPreFlight, &skipTokenPrint, &dryRun, &ignorePreflightErrors)

	return cmd
}

// AddInitConfigFlags adds init flags bound to the config to the specified flagset
func AddInitConfigFlags(flagSet *flag.FlagSet, cfg *kubeadmapiv1alpha2.MasterConfiguration, featureGatesString *string) {
	flagSet.StringVar(
		&cfg.API.AdvertiseAddress, "apiserver-advertise-address", cfg.API.AdvertiseAddress,
		"The IP address the API Server will advertise it's listening on. Specify '0.0.0.0' to use the address of the default network interface.",
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
		&cfg.APIServerCertSANs, "apiserver-cert-extra-sans", cfg.APIServerCertSANs,
		`Optional extra Subject Alternative Names (SANs) to use for the API Server serving certificate. Can be both IP addresses and DNS names.`,
	)
	flagSet.StringVar(
		&cfg.NodeRegistration.Name, "node-name", cfg.NodeRegistration.Name,
		`Specify the node name.`,
	)
	flagSet.StringVar(
		&cfg.Token, "token", cfg.Token,
		"The token to use for establishing bidirectional trust between nodes and masters. The format is [a-z0-9]{6}\\.[a-z0-9]{16} - e.g. abcdef.0123456789abcdef",
	)
	flagSet.DurationVar(
		&cfg.TokenTTL.Duration, "token-ttl", cfg.TokenTTL.Duration,
		"The duration before the bootstrap token is automatically deleted. If set to '0', the token will never expire.",
	)
	flagSet.StringVar(
		&cfg.NodeRegistration.CRISocket, "cri-socket", cfg.NodeRegistration.CRISocket,
		`Specify the CRI socket to connect to.`,
	)
	flagSet.StringVar(featureGatesString, "feature-gates", *featureGatesString, "A set of key=value pairs that describe feature gates for various features. "+
		"Options are:\n"+strings.Join(features.KnownFeatures(&features.InitFeatureGates), "\n"))

}

// AddInitOtherFlags adds init flags that are not bound to a configuration file to the given flagset
func AddInitOtherFlags(flagSet *flag.FlagSet, cfgPath *string, skipPreFlight, skipTokenPrint, dryRun *bool, ignorePreflightErrors *[]string) {
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
		skipPreFlight, "skip-preflight-checks", *skipPreFlight,
		"Skip preflight checks which normally run before modifying the system.",
	)
	flagSet.MarkDeprecated("skip-preflight-checks", "it is now equivalent to --ignore-preflight-errors=all")
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
func NewInit(cfgPath string, externalcfg *kubeadmapiv1alpha2.MasterConfiguration, ignorePreflightErrors sets.String, skipTokenPrint, dryRun bool) (*Init, error) {

	// Either use the config file if specified, or convert the defaults in the external to an internal cfg representation
	cfg, err := configutil.ConfigFileAndDefaultsToInternalConfig(cfgPath, externalcfg)
	if err != nil {
		return nil, err
	}

	glog.V(1).Infof("[init] validating feature gates")
	if err := features.ValidateVersion(features.InitFeatureGates, cfg.FeatureGates, cfg.KubernetesVersion); err != nil {
		return nil, err
	}

	glog.Infof("[init] using Kubernetes version: %s\n", cfg.KubernetesVersion)

	glog.Infoln("[preflight] running pre-flight checks")

	if err := preflight.RunInitMasterChecks(utilsexec.New(), cfg, ignorePreflightErrors); err != nil {
		return nil, err
	}
	if err := preflight.RunPullImagesCheck(utilsexec.New(), cfg, ignorePreflightErrors); err != nil {
		return nil, err
	}

	return &Init{cfg: cfg, skipTokenPrint: skipTokenPrint, dryRun: dryRun, ignorePreflightErrors: ignorePreflightErrors}, nil
}

// Init defines struct used by "kubeadm init" command
type Init struct {
	cfg                   *kubeadmapi.MasterConfiguration
	skipTokenPrint        bool
	dryRun                bool
	ignorePreflightErrors sets.String
}

// Run executes master node provisioning, including certificates, needed static pod manifests, etc.
func (i *Init) Run(out io.Writer) error {

	// Write env file with flags for the kubelet to use. We do not need to write the --register-with-taints for the master,
	// as we handle that ourselves in the markmaster phase
	// TODO: Maybe we want to do that some time in the future, in order to remove some logic from the markmaster phase?
	if err := kubeletphase.WriteKubeletDynamicEnvFile(&i.cfg.NodeRegistration, false); err != nil {
		return err
	}

	// Try to start the kubelet service in case it's inactive
	glog.V(1).Infof("Starting kubelet")
	preflight.TryStartKubelet(i.ignorePreflightErrors)

	// Get directories to write files to; can be faked if we're dry-running
	glog.V(1).Infof("[init] Getting certificates directory from configuration")
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
		glog.V(1).Infof("[init] creating PKI Assets")
		if err := certsphase.CreatePKIAssets(i.cfg); err != nil {
			return err
		}

		// PHASE 2: Generate kubeconfig files for the admin and the kubelet
		glog.V(2).Infof("[init] generating kubeconfig files")
		if err := kubeconfigphase.CreateInitKubeConfigFiles(kubeConfigDir, i.cfg); err != nil {
			return err
		}

	} else {
		glog.Infoln("[externalca] the file 'ca.key' was not found, yet all other certificates are present. Using external CA mode - certificates or kubeconfig will not be generated")
	}

	if features.Enabled(i.cfg.FeatureGates, features.Auditing) {
		// Setup the AuditPolicy (either it was passed in and exists or it wasn't passed in and generate a default policy)
		if i.cfg.AuditPolicyConfiguration.Path != "" {
			// TODO(chuckha) ensure passed in audit policy is valid so users don't have to find the error in the api server log.
			if _, err := os.Stat(i.cfg.AuditPolicyConfiguration.Path); err != nil {
				return fmt.Errorf("error getting file info for audit policy file %q [%v]", i.cfg.AuditPolicyConfiguration.Path, err)
			}
		} else {
			i.cfg.AuditPolicyConfiguration.Path = filepath.Join(kubeConfigDir, kubeadmconstants.AuditPolicyDir, kubeadmconstants.AuditPolicyFile)
			if err := auditutil.CreateDefaultAuditLogPolicy(i.cfg.AuditPolicyConfiguration.Path); err != nil {
				return fmt.Errorf("error creating default audit policy %q [%v]", i.cfg.AuditPolicyConfiguration.Path, err)
			}
		}
	}

	// Temporarily set cfg.CertificatesDir to the "real value" when writing controlplane manifests
	// This is needed for writing the right kind of manifests
	i.cfg.CertificatesDir = realCertsDir

	// PHASE 3: Bootstrap the control plane
	glog.V(1).Infof("[init] bootstraping the control plane")
	glog.V(1).Infof("[init] creating static pod manifest")
	if err := controlplanephase.CreateInitStaticPodManifestFiles(manifestDir, i.cfg); err != nil {
		return fmt.Errorf("error creating init static pod manifest files: %v", err)
	}
	// Add etcd static pod spec only if external etcd is not configured
	if i.cfg.Etcd.External == nil {
		glog.V(1).Infof("[init] no external etcd found. Creating manifest for local etcd static pod")
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

	kubeletVersion, err := preflight.GetKubeletVersion(utilsexec.New())
	if err != nil {
		return err
	}

	// Write the kubelet configuration to disk.
	if err := kubeletphase.WriteConfigToDisk(i.cfg.KubeletConfiguration.BaseConfig); err != nil {
		return fmt.Errorf("error writing kubelet configuration to disk: %v", err)
	}

	// Create a kubernetes client and wait for the API server to be healthy (if not dryrunning)
	glog.V(1).Infof("creating Kubernetes client")
	client, err := createClient(i.cfg, i.dryRun)
	if err != nil {
		return fmt.Errorf("error creating client: %v", err)
	}

	// waiter holds the apiclient.Waiter implementation of choice, responsible for querying the API server in various ways and waiting for conditions to be fulfilled
	glog.V(1).Infof("[init] waiting for the API server to be healthy")
	waiter := getWaiter(i, client)

	if err := waitForAPIAndKubelet(waiter); err != nil {
		ctx := map[string]string{
			"Error":                  fmt.Sprintf("%v", err),
			"APIServerImage":         images.GetCoreImage(kubeadmconstants.KubeAPIServer, i.cfg.GetControlPlaneImageRepository(), i.cfg.KubernetesVersion, i.cfg.UnifiedControlPlaneImage),
			"ControllerManagerImage": images.GetCoreImage(kubeadmconstants.KubeControllerManager, i.cfg.GetControlPlaneImageRepository(), i.cfg.KubernetesVersion, i.cfg.UnifiedControlPlaneImage),
			"SchedulerImage":         images.GetCoreImage(kubeadmconstants.KubeScheduler, i.cfg.GetControlPlaneImageRepository(), i.cfg.KubernetesVersion, i.cfg.UnifiedControlPlaneImage),
		}
		// Set .EtcdImage conditionally
		if i.cfg.Etcd.Local != nil {
			ctx["EtcdImage"] = fmt.Sprintf("				- %s", images.GetCoreImage(kubeadmconstants.Etcd, i.cfg.ImageRepository, i.cfg.KubernetesVersion, i.cfg.Etcd.Local.Image))
		} else {
			ctx["EtcdImage"] = ""
		}

		kubeletFailTempl.Execute(out, ctx)

		return fmt.Errorf("couldn't initialize a Kubernetes cluster")
	}

	// Upload currently used configuration to the cluster
	// Note: This is done right in the beginning of cluster initialization; as we might want to make other phases
	// depend on centralized information from this source in the future
	glog.V(1).Infof("[init] uploading currently used configuration to the cluster")
	if err := uploadconfigphase.UploadConfiguration(i.cfg, client); err != nil {
		return fmt.Errorf("error uploading configuration: %v", err)
	}

	glog.V(1).Infof("[init] creating kubelet configuration configmap")
	if err := kubeletphase.CreateConfigMap(i.cfg, client); err != nil {
		return fmt.Errorf("error creating kubelet configuration ConfigMap: %v", err)
	}

	// PHASE 4: Mark the master with the right label/taint
	glog.V(1).Infof("[init] marking the master with right label")
	if err := markmasterphase.MarkMaster(client, i.cfg.NodeRegistration.Name, i.cfg.NodeRegistration.Taints); err != nil {
		return fmt.Errorf("error marking master: %v", err)
	}

	// NOTE: flag "--dynamic-config-dir" should be specified in /etc/systemd/system/kubelet.service.d/10-kubeadm.conf
	// This feature is disabled by default, as it is alpha still
	if features.Enabled(i.cfg.FeatureGates, features.DynamicKubeletConfig) {
		// Enable dynamic kubelet configuration for the node.
		if err := kubeletphase.EnableDynamicConfigForNode(client, i.cfg.NodeRegistration.Name, kubeletVersion); err != nil {
			return fmt.Errorf("error enabling dynamic kubelet configuration: %v", err)
		}
	}

	// PHASE 5: Set up the node bootstrap tokens
	if !i.skipTokenPrint {
		glog.Infof("[bootstraptoken] using token: %s\n", i.cfg.Token)
	}

	// Create the default node bootstrap token
	glog.V(1).Infof("[init] creating RBAC rules to generate default bootstrap token")
	tokenDescription := "The default bootstrap token generated by 'kubeadm init'."
	if err := nodebootstraptokenphase.UpdateOrCreateToken(client, i.cfg.Token, false, i.cfg.TokenTTL.Duration, i.cfg.TokenUsages, i.cfg.TokenGroups, tokenDescription); err != nil {
		return fmt.Errorf("error updating or creating token: %v", err)
	}
	// Create RBAC rules that makes the bootstrap tokens able to post CSRs
	glog.V(1).Infof("[init] creating RBAC rules to allow bootstrap tokens to post CSR")
	if err := nodebootstraptokenphase.AllowBootstrapTokensToPostCSRs(client); err != nil {
		return fmt.Errorf("error allowing bootstrap tokens to post CSRs: %v", err)
	}
	// Create RBAC rules that makes the bootstrap tokens able to get their CSRs approved automatically
	glog.V(1).Infof("[init] creating RBAC rules to automatic approval of CSRs automatically")
	if err := nodebootstraptokenphase.AutoApproveNodeBootstrapTokens(client); err != nil {
		return fmt.Errorf("error auto-approving node bootstrap tokens: %v", err)
	}

	// Create/update RBAC rules that makes the nodes to rotate certificates and get their CSRs approved automatically
	glog.V(1).Infof("[init] creating/updating RBAC rules for rotating certificate")
	if err := nodebootstraptokenphase.AutoApproveNodeCertificateRotation(client); err != nil {
		return err
	}

	// Create the cluster-info ConfigMap with the associated RBAC rules
	glog.V(1).Infof("[init] creating bootstrap configmap")
	if err := clusterinfophase.CreateBootstrapConfigMapIfNotExists(client, adminKubeConfigPath); err != nil {
		return fmt.Errorf("error creating bootstrap configmap: %v", err)
	}
	glog.V(1).Infof("[init] creating ClusterInfo RBAC rules")
	if err := clusterinfophase.CreateClusterInfoRBACRules(client); err != nil {
		return fmt.Errorf("error creating clusterinfo RBAC rules: %v", err)
	}

	glog.V(1).Infof("[init] ensuring DNS addon")
	if err := dnsaddonphase.EnsureDNSAddon(i.cfg, client); err != nil {
		return fmt.Errorf("error ensuring dns addon: %v", err)
	}

	glog.V(1).Infof("[init] ensuring proxy addon")
	if err := proxyaddonphase.EnsureProxyAddon(i.cfg, client); err != nil {
		return fmt.Errorf("error ensuring proxy addon: %v", err)
	}

	// PHASE 7: Make the control plane self-hosted if feature gate is enabled
	glog.V(1).Infof("[init] feature gate is enabled. Making control plane self-hosted")
	if features.Enabled(i.cfg.FeatureGates, features.SelfHosting) {
		// Temporary control plane is up, now we create our self hosted control
		// plane components and remove the static manifests:
		glog.Infoln("[self-hosted] creating self-hosted control plane")
		if err := selfhostingphase.CreateSelfHostedControlPlane(manifestDir, kubeConfigDir, i.cfg, client, waiter, i.dryRun); err != nil {
			return fmt.Errorf("error creating self hosted control plane: %v", err)
		}
	}

	// Exit earlier if we're dryrunning
	if i.dryRun {
		fmt.Println("[dryrun] finished dry-running successfully. Above are the resources that would be created")
		return nil
	}

	// Gets the join command
	joinCommand, err := cmdutil.GetJoinCommand(kubeadmconstants.GetAdminKubeConfigPath(), i.cfg.Token, i.skipTokenPrint)
	if err != nil {
		return fmt.Errorf("failed to get join command: %v", err)
	}

	ctx := map[string]string{
		"KubeConfigPath": adminKubeConfigPath,
		"joinCommand":    joinCommand,
	}

	return initDoneTempl.Execute(out, ctx)
}

// createClient creates a clientset.Interface object
func createClient(cfg *kubeadmapi.MasterConfiguration, dryRun bool) (clientset.Interface, error) {
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

	glog.Infof("[dryrun] wrote certificates, kubeconfig files and control plane manifests to the %q directory\n", manifestDir)
	glog.Infoln("[dryrun] the certificates or kubeconfig files would not be printed due to their sensitive nature")
	glog.Infof("[dryrun] please examine the %q directory for details about what would be written\n", manifestDir)

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
func getWaiter(i *Init, client clientset.Interface) apiclient.Waiter {
	if i.dryRun {
		return dryrunutil.NewWaiter()
	}

	// TODO: List images locally using `crictl` and pull in preflight checks if not available
	// When we do that, we can always assume the images exist at this point and have a shorter timeout.
	timeout := 30 * time.Minute
	return apiclient.NewKubeWaiter(client, timeout, os.Stdout)
}

// waitForAPIAndKubelet waits primarily for the API server to come up. If that takes a long time, and the kubelet
// /healthz and /healthz/syncloop endpoints continuously are unhealthy, kubeadm will error out after a period of
// backoffing exponentially
func waitForAPIAndKubelet(waiter apiclient.Waiter) error {
	errorChan := make(chan error)

	glog.Infof("[init] waiting for the kubelet to boot up the control plane as Static Pods from directory %q \n", kubeadmconstants.GetStaticPodDirectory())
	glog.Infoln("[init] this might take a minute or longer if the control plane images have to be pulled")

	go func(errC chan error, waiter apiclient.Waiter) {
		// This goroutine can only make kubeadm init fail. If this check succeeds, it won't do anything special
		if err := waiter.WaitForHealthyKubelet(40*time.Second, "http://localhost:10248/healthz"); err != nil {
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
