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

	"github.com/renstrom/dedent"
	"github.com/spf13/cobra"
	flag "github.com/spf13/pflag"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	certutil "k8s.io/client-go/util/cert"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiext "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha1"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/validation"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/discovery"
	"k8s.io/kubernetes/cmd/kubeadm/app/features"
	certsphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/certs"
	controlplanephase "k8s.io/kubernetes/cmd/kubeadm/app/phases/controlplane"
	kubeconfigphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/kubeconfig"
	kubeletphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/kubelet"
	markmasterphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/markmaster"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/upgrade"
	"k8s.io/kubernetes/cmd/kubeadm/app/preflight"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	kubeconfigutil "k8s.io/kubernetes/cmd/kubeadm/app/util/kubeconfig"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	nodeutil "k8s.io/kubernetes/pkg/util/node"
	utilsexec "k8s.io/utils/exec"
)

var (
	joinWorkerNodeDoneMsgf = dedent.Dedent(`
		This node has joined the cluster:

		* Certificate signing request was sent to master and approval was received.
		* The Kubelet was informed of the new secure connection details.

		Run 'kubectl get nodes' on the master to see this node join the cluster.

		`)

	joinMasterNodeDoneTemp = template.Must(template.New("join").Parse(dedent.Dedent(`
		This node has joined the cluster as a master node:
		
		* Certificate signing request was sent to master and approval was received.
		* The Kubelet was informed of the new secure connection details.
		* Master label and taint were applied to the new node.

		Run 'kubectl get nodes' on the master to see this node join the cluster.

		To start administering your cluster from this node, you need to run the following as a regular user:

			mkdir -p $HOME/.kube
			sudo cp -i {{.KubeConfigPath}} $HOME/.kube/config
			sudo chown $(id -u):$(id -g) $HOME/.kube/config

		`)))

	joinLongDescription = dedent.Dedent(`
		When joining a kubeadm initialized cluster, we need to establish
		bidirectional trust. This is split into discovery (having the Node
		trust the Kubernetes Master) and TLS bootstrap (having the Kubernetes
		Master trust the Node).

		There are 2 main schemes for discovery. The first is to use a shared
		token along with the IP address of the API server. The second is to
		provide a file - a subset of the standard kubeconfig file. This file
		can be a local file or downloaded via an HTTPS URL. The forms are
		kubeadm join --discovery-token abcdef.1234567890abcdef 1.2.3.4:6443,
		kubeadm join --discovery-file path/to/file.conf, or kubeadm join
		--discovery-file https://url/file.conf. Only one form can be used. If
		the discovery information is loaded from a URL, HTTPS must be used.
		Also, in that case the host installed CA bundle is used to verify
		the connection.

		If you use a shared token for discovery, you should also pass the
		--discovery-token-ca-cert-hash flag to validate the public key of the
		root certificate authority (CA) presented by the Kubernetes Master. The
		value of this flag is specified as "<hash-type>:<hex-encoded-value>",
		where the supported hash type is "sha256". The hash is calculated over
		the bytes of the Subject Public Key Info (SPKI) object (as in RFC7469).
		This value is available in the output of "kubeadm init" or can be
		calcuated using standard tools. The --discovery-token-ca-cert-hash flag
		may be repeated multiple times to allow more than one public key.

		If you cannot know the CA public key hash ahead of time, you can pass
		the --discovery-token-unsafe-skip-ca-verification flag to disable this
		verification. This weakens the kubeadm security model since other nodes
		can potentially impersonate the Kubernetes Master.

		The TLS bootstrap mechanism is also driven via a shared token. This is
		used to temporarily authenticate with the Kubernetes Master to submit a
		certificate signing request (CSR) for a locally created key pair. By
		default, kubeadm will set up the Kubernetes Master to automatically
		approve these signing requests. This token is passed in with the
		--tls-bootstrap-token abcdef.1234567890abcdef flag.

		Often times the same token is used for both parts. In this case, the
		--token flag can be used instead of specifying each token individually.
		`)
)

// NewCmdJoin returns "kubeadm join" command.
func NewCmdJoin(out io.Writer) *cobra.Command {
	cfg := &kubeadmapiext.NodeConfiguration{}
	legacyscheme.Scheme.Default(cfg)

	var skipPreFlight bool
	var cfgPath string
	var criSocket string
	var featureGatesString string
	var ignorePreflightErrors []string

	cmd := &cobra.Command{
		Use:   "join [flags]",
		Short: "Run this on any machine you wish to join an existing cluster",
		Long:  joinLongDescription,
		Run: func(cmd *cobra.Command, args []string) {
			cfg.DiscoveryTokenAPIServers = args

			var err error
			if cfg.FeatureGates, err = features.NewFeatureGate(&features.InitFeatureGates, featureGatesString); err != nil {
				kubeadmutil.CheckErr(err)
			}

			legacyscheme.Scheme.Default(cfg)
			internalcfg := &kubeadmapi.NodeConfiguration{}
			legacyscheme.Scheme.Convert(cfg, internalcfg, nil)

			ignorePreflightErrorsSet, err := validation.ValidateIgnorePreflightErrors(ignorePreflightErrors, skipPreFlight)
			kubeadmutil.CheckErr(err)

			j, err := NewJoin(cfgPath, args, internalcfg, ignorePreflightErrorsSet, criSocket)
			kubeadmutil.CheckErr(err)
			kubeadmutil.CheckErr(j.Validate(cmd))
			kubeadmutil.CheckErr(j.Run(out))
		},
	}

	AddJoinConfigFlags(cmd.PersistentFlags(), cfg, &featureGatesString)
	AddJoinOtherFlags(cmd.PersistentFlags(), &cfgPath, &skipPreFlight, &criSocket, &ignorePreflightErrors)

	return cmd
}

// AddJoinConfigFlags adds join flags bound to the config to the specified flagset
func AddJoinConfigFlags(flagSet *flag.FlagSet, cfg *kubeadmapiext.NodeConfiguration, featureGatesString *string) {
	flagSet.StringVar(
		&cfg.DiscoveryFile, "discovery-file", "",
		"A file or url from which to load cluster information.")
	flagSet.StringVar(
		&cfg.DiscoveryToken, "discovery-token", "",
		"A token used to validate cluster information fetched from the master.")
	flagSet.StringVar(
		&cfg.NodeName, "node-name", "",
		"Specify the node name.")
	flagSet.StringVar(
		&cfg.TLSBootstrapToken, "tls-bootstrap-token", "",
		"A token used for TLS bootstrapping.")
	flagSet.StringSliceVar(
		&cfg.DiscoveryTokenCACertHashes, "discovery-token-ca-cert-hash", []string{},
		"For token-based discovery, validate that the root CA public key matches this hash (format: \"<type>:<value>\").")
	flagSet.BoolVar(
		&cfg.DiscoveryTokenUnsafeSkipCAVerification, "discovery-token-unsafe-skip-ca-verification", false,
		"For token-based discovery, allow joining without --discovery-token-ca-cert-hash pinning.")
	flagSet.StringVar(
		&cfg.Token, "token", "",
		"Use this token for both discovery-token and tls-bootstrap-token.")
	flagSet.StringVar(
		featureGatesString, "feature-gates", *featureGatesString,
		"A set of key=value pairs that describe feature gates for various features. "+
			"Options are:\n"+strings.Join(features.KnownFeatures(&features.InitFeatureGates), "\n"))
	flagSet.BoolVar(
		&cfg.Master, "master", cfg.Master,
		"Join as a master node")
	flagSet.StringVar(
		&cfg.MasterConfigurationFile, "masterConfiguration", cfg.MasterConfigurationFile,
		"Path to kubeadm master configuration file, to be provided only if access to those information is explicitly denied to the boostrap tokens")
}

// AddJoinOtherFlags adds join flags that are not bound to a configuration file to the given flagset
func AddJoinOtherFlags(flagSet *flag.FlagSet, cfgPath *string, skipPreFlight *bool, criSocket *string, ignorePreflightErrors *[]string) {
	flagSet.StringVar(
		cfgPath, "config", *cfgPath,
		"Path to kubeadm config file.")

	flagSet.StringSliceVar(
		ignorePreflightErrors, "ignore-preflight-errors", *ignorePreflightErrors,
		"A list of checks whose errors will be shown as warnings. Example: 'IsPrivilegedUser,Swap'. Value 'all' ignores errors from all checks.",
	)
	flagSet.BoolVar(
		skipPreFlight, "skip-preflight-checks", false,
		"Skip preflight checks which normally run before modifying the system.",
	)
	flagSet.MarkDeprecated("skip-preflight-checks", "it is now equivalent to --ignore-preflight-errors=all")
	flagSet.StringVar(
		criSocket, "cri-socket", "/var/run/dockershim.sock",
		`Specify the CRI socket to connect to.`,
	)
}

// Join defines struct used by kubeadm join command
type Join struct {
	cfg                   *kubeadmapi.NodeConfiguration
	ignorePreflightErrors sets.String
	criSocket             string
}

// NewJoin instantiates Join struct with given arguments
func NewJoin(cfgPath string, args []string, cfg *kubeadmapi.NodeConfiguration, ignorePreflightErrors sets.String, criSocket string) (*Join, error) {

	if cfg.NodeName == "" {
		cfg.NodeName = nodeutil.GetHostname("")
	}

	if cfgPath != "" {
		b, err := ioutil.ReadFile(cfgPath)
		if err != nil {
			return nil, fmt.Errorf("unable to read config from %q [%v]", cfgPath, err)
		}
		if err := runtime.DecodeInto(legacyscheme.Codecs.UniversalDecoder(), b, cfg); err != nil {
			return nil, fmt.Errorf("unable to decode config from %q [%v]", cfgPath, err)
		}
	}

	// Enforce usage of --master flag with --feature-gates=HighAvailability
	if (cfg.Master == true) &&
		!features.Enabled(cfg.FeatureGates, features.HighAvailability) {
		return nil, fmt.Errorf("usage of '--master' is not allowed without '--feature-gates=HighAvailability=true'")
	}

	fmt.Println("[preflight] Running pre-flight checks.")

	// Then continue with the others...
	if err := preflight.RunJoinNodeChecks(utilsexec.New(), cfg, criSocket, ignorePreflightErrors); err != nil {
		return nil, err
	}

	// Try to start the kubelet service in case it's inactive
	preflight.TryStartKubelet(ignorePreflightErrors)

	return &Join{
		cfg: cfg,
		ignorePreflightErrors: ignorePreflightErrors,
		criSocket:             criSocket,
	}, nil
}

// Validate validates mixed arguments passed to cobra.Command
func (j *Join) Validate(cmd *cobra.Command) error {
	if err := validation.ValidateMixedArguments(cmd.PersistentFlags()); err != nil {
		return err
	}
	return validation.ValidateNodeConfiguration(j.cfg).ToAggregate()
}

// Run executes node provisioning and joins an existing cluster.
func (j *Join) Run(out io.Writer) error {

	// Discovery cluster information required for the tls bootstrap procedure
	tlsBootstrapCfg, err := discovery.For(j.cfg)
	if err != nil {
		return err
	}

	// If the node is joining as a master
	if j.cfg.Master == true {

		// Retrives the kubeadm configuration used during kubeadm init
		masterConfiguration, err := j.FetchInitClusterConfiguration(tlsBootstrapCfg)
		if err != nil {
			return err
		}

		// Checks if the cluster is in a configuration that supports
		// joining an additional master node
		err = j.CheckIfReadyForJoinAsMaster(masterConfiguration)
		if err != nil {
			return err
		}

		// Prepares the node for joining as a master
		err = j.PrepareForJoinAsMaster(masterConfiguration)
		if err != nil {
			return err
		}
	}

	// Executes the kubelet TLS bootstrap process, that completes with the node
	// joining the cluster with a dedicates set of credentials as required by
	// the node authorizer.
	// In case of a cluster that uses static pods for the control plane,
	// as soon as it starts, the kubelet will take charge of creating control plane
	// components on the node.
	err = j.BootstrapKubelet(tlsBootstrapCfg)
	if err != nil {
		return err
	}

	// if the node is joining as a master
	if j.cfg.Master == true {

		// Marks the node with master taint and label.
		// In case of a cluster that uses a self hosted cluster, this action triggers
		// the deployment of control plane components on the node.
		err := j.MarkMaster()
		if err != nil {
			return err
		}

		// outputs the join master done template and exits
		ctx := map[string]string{
			"KubeConfigPath": kubeadmconstants.GetAdminKubeConfigPath(),
		}
		joinMasterNodeDoneTemp.Execute(out, ctx)
		return nil
	}

	// otherwise, if the node joined as a worker node

	// outputs the join done message and exits
	fmt.Fprintf(out, joinWorkerNodeDoneMsgf)
	return nil
}

// FetchInitClusterConfiguration reads the master configuration from the kubeadm-admin configMap,
// or, as fallback strategy, it loads the master configuration from the given MasterConfigurationFile path.
// Nb. The fallback strategy addresses scenarios where authorizations for bootstrap tokens
// will be explicitly revoked after kubeadm init
func (j *Join) FetchInitClusterConfiguration(tlsBootstrapCfg *clientcmdapi.Config) (*kubeadmapi.MasterConfiguration, error) {

	// creates a client to access the cluster using the bootstrap token identity
	tlsClient, err := kubeconfigutil.ToClientSet(tlsBootstrapCfg)
	if err != nil {
		return nil, fmt.Errorf("error creating the kubernetes client with tls bootstrap authentity: %v", err)
	}

	// Fetches the master configuration
	// ** Note for the reviewers:
	// ** see notes in phases/upgrade about how to share FetchConfiguration between phase/update and join
	masterConfigurationExt, err := upgrade.FetchConfiguration(tlsClient, os.Stdout, j.cfg.MasterConfigurationFile)
	if err != nil {
		return nil, fmt.Errorf("error reading the master configuration for the cluster: %v", err)
	}

	// Converts public API to internal API
	masterConfiguration := &kubeadmapi.MasterConfiguration{}
	legacyscheme.Scheme.Convert(masterConfigurationExt, masterConfiguration, nil)

	return masterConfiguration, nil
}

// CheckIfReadyForJoinAsMaster ensures that the cluster is in a configuration that supports
// joining additional masters
func (j *Join) CheckIfReadyForJoinAsMaster(masterConfiguration *kubeadmapi.MasterConfiguration) error {

	// checks if the cluster was created with features.HighAvailability enabled
	if !features.Enabled(masterConfiguration.FeatureGates, features.HighAvailability) {
		return fmt.Errorf("unable to join a new master node on a cluster that wasn't created without '--feature-gates=HighAvailability=true'")
	}

	// ** Note for reviewers:
	// ** If the cluster was initialized with --feature-gates=HighAvailability=true following checks were already executed
	// ** during kubeadm init:
	// ** 1. the cluster uses an external etcd
	// ** 2. the cluster uses an external load balancer
	// ** Should we double check the above assumptions before joining?

	// if the certificates for the control plane are not stored in secrets (if the certificates are stored in a local pki folder),
	if !features.Enabled(masterConfiguration.FeatureGates, features.StoreCertsInSecrets) {

		// checks if the certificates for the control plane were already provided by the user in
		// the local pki folder and checks if those certs are compliant with the requirements
		// ** Note for the reviewers:
		// ** see notes in phases/cert about how sharing validate cert functions between the two features
		err := certsphase.CheckIfLocalPkiReadyForJoinAsMaster(masterConfiguration)
		if err != nil {
			return err
		}
	}

	return nil
}

// PrepareForJoinAsMaster makes all preparation activities require for a node joining as a master
func (j *Join) PrepareForJoinAsMaster(masterConfiguration *kubeadmapi.MasterConfiguration) error {

	// if the certificates for the control plane are stored in a local pki folder
	// (if certificates are not stored in secrets)
	if !features.Enabled(masterConfiguration.FeatureGates, features.StoreCertsInSecrets) {

		// Creates the missing certificates
		// NB. This function will use the certificate authority and service account signing key
		//     provided by the user before executing init --master.
		//     (those certificates/keys must be the same across all master instances)
		certsphase.CreatePKIAssets(masterConfiguration)
	}

	// if the cluster uses StaticPods manifests (if the cluster is not self hosted)
	if !features.Enabled(masterConfiguration.FeatureGates, features.SelfHosting) {

		// Generate kubeconfig files for controller manager, scheduler and for the admin/kubeadm itself
		// NB. The kubeconfig file for kubelet will be generated by the TLS bootstrap process in
		// following steps of the join --master workflow
		err := kubeconfigphase.CreateJoinMasterKubeConfigFiles(kubeadmconstants.KubernetesDir, masterConfiguration)
		if err != nil {
			return fmt.Errorf("error generating kubeconfig files: %v", err)
		}

		// Creates static pod manifests file for the control plane components to be deployed on this node
		// Static pods will be created and managed by the kubelet as soon as it starts
		err = controlplanephase.CreateInitStaticPodManifestFiles(kubeadmconstants.GetStaticPodDirectory(), masterConfiguration)
		if err != nil {
			return fmt.Errorf("error creating static pod manifest files for the control plane components: %v", err)
		}

		return nil
	}

	// Otherwise, if the cluster uses a SelfHosting control plane

	// Creates the admin kubeconfig file for the admin and for kubeadm itself.
	err := kubeconfigphase.CreateAdminKubeConfigFile(kubeadmconstants.KubernetesDir, masterConfiguration)
	if err != nil {
		return fmt.Errorf("error generating the admin kubeconfig file: %v", err)
	}

	// NB. self hosted clusters desn't requires preparation activities,
	// because the existing DeamonSets for the control plane components will automatically trigger
	// the deploymento of new instances of such components on this node as soon as will join the cluster.

	return nil
}

// BootstrapKubelet initializes the kubelet TLS bootstrap process.
// This process is executed by the kubelet and completes with the node joining the cluster
// with a dedicates set of credentials as required by the node authorizer
func (j *Join) BootstrapKubelet(tlsBootstrapCfg *clientcmdapi.Config) error {
	bootstrapKubeconfigFile := filepath.Join(kubeadmconstants.KubernetesDir, kubeadmconstants.KubeletBootstrapKubeConfigFileName)

	// Writes the bootstrap kubelet config; this will trigger the kubelet TLS bootstrap process
	err := kubeconfigutil.WriteToDisk(bootstrapKubeconfigFile, tlsBootstrapCfg)
	if err != nil {
		return fmt.Errorf("couldn't save bootstrap-kubelet.conf to disk: %v", err)
	}

	// Writes the ca certificate to disk so kubelet can use it for authentication
	cluster := tlsBootstrapCfg.Contexts[tlsBootstrapCfg.CurrentContext].Cluster
	err = certutil.WriteCert(j.cfg.CACertPath, tlsBootstrapCfg.Clusters[cluster].CertificateAuthorityData)
	if err != nil {
		return fmt.Errorf("couldn't save the CA certificate to disk: %v", err)
	}

	// Eventually, completes the kubelet dynamic configuration as soon as the kubelet bootstrap is finished
	// NOTE: flag "--dynamic-config-dir" should be specified in /etc/systemd/system/kubelet.service.d/10-kubeadm.conf
	if features.Enabled(j.cfg.FeatureGates, features.DynamicKubeletConfig) {
		err := kubeletphase.ConsumeBaseKubeletConfiguration(j.cfg.NodeName)
		if err != nil {
			return fmt.Errorf("error applying the dynamic kubelet configuration to the node: %v", err)
		}
	}

	return nil
}

// MarkMaster marks the new node as master
func (j *Join) MarkMaster() error {
	kubeConfigFile := filepath.Join(kubeadmconstants.KubernetesDir, kubeadmconstants.AdminKubeConfigFileName)

	client, err := kubeconfigutil.ClientSetFromFile(kubeConfigFile)
	if err != nil {
		return fmt.Errorf("couldn't create kubernetes client: %v", err)
	}

	err = markmasterphase.MarkMaster(client, j.cfg.NodeName)
	if err != nil {
		return fmt.Errorf("error applying master label and taints: %v", err)
	}

	return nil
}
