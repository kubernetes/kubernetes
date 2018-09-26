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
	"bytes"
	"fmt"
	"io"
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
	"k8s.io/apimachinery/pkg/util/wait"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	certutil "k8s.io/client-go/util/cert"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmscheme "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/scheme"
	kubeadmapiv1alpha3 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha3"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/validation"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/discovery"
	"k8s.io/kubernetes/cmd/kubeadm/app/features"
	certsphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/certs"
	controlplanephase "k8s.io/kubernetes/cmd/kubeadm/app/phases/controlplane"
	kubeconfigphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/kubeconfig"
	kubeletphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/kubelet"
	markmasterphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/markmaster"
	patchnodephase "k8s.io/kubernetes/cmd/kubeadm/app/phases/patchnode"
	uploadconfigphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/uploadconfig"
	"k8s.io/kubernetes/cmd/kubeadm/app/preflight"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/apiclient"
	configutil "k8s.io/kubernetes/cmd/kubeadm/app/util/config"
	kubeconfigutil "k8s.io/kubernetes/cmd/kubeadm/app/util/kubeconfig"
	utilsexec "k8s.io/utils/exec"
)

var (
	joinWorkerNodeDoneMsg = dedent.Dedent(`
		This node has joined the cluster:
		* Certificate signing request was sent to apiserver and a response was received.
		* The Kubelet was informed of the new secure connection details.

		Run 'kubectl get nodes' on the master to see this node join the cluster.

		`)

	notReadyToJoinControPlaneTemp = template.Must(template.New("join").Parse(dedent.Dedent(`
		One or more conditions for hosting a new control plane instance is not satisfied. 

		{{.Error}}

		Please ensure that:
		* The cluster has a stable controlPlaneEndpoint address.
		* The cluster uses an external etcd.
		* The certificates that must be shared among control plane instances are provided.

		`)))

	joinControPlaneDoneTemp = template.Must(template.New("join").Parse(dedent.Dedent(`
		This node has joined the cluster and a new control plane instance was created:
		
		* Certificate signing request was sent to apiserver and approval was received.
		* The Kubelet was informed of the new secure connection details.
		* Master label and taint were applied to the new node.
		* The kubernetes control plane instances scaled up.

		To start administering your cluster from this node, you need to run the following as a regular user:

			mkdir -p $HOME/.kube
			sudo cp -i {{.KubeConfigPath}} $HOME/.kube/config
			sudo chown $(id -u):$(id -g) $HOME/.kube/config

		Run 'kubectl get nodes' to see this node join the cluster.
		
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
		calculated using standard tools. The --discovery-token-ca-cert-hash flag
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

	kubeadmJoinFailMsg = dedent.Dedent(`
		Unfortunately, an error has occurred:
			%v

		This error is likely caused by:
			- The kubelet is not running
			- The kubelet is unhealthy due to a misconfiguration of the node in some way (required cgroups disabled)

		If you are on a systemd-powered system, you can try to troubleshoot the error with the following commands:
			- 'systemctl status kubelet'
			- 'journalctl -xeu kubelet'
		`)
)

// NewCmdJoin returns "kubeadm join" command.
func NewCmdJoin(out io.Writer) *cobra.Command {
	cfg := &kubeadmapiv1alpha3.JoinConfiguration{}
	kubeadmscheme.Scheme.Default(cfg)

	var cfgPath string
	var featureGatesString string
	var ignorePreflightErrors []string

	cmd := &cobra.Command{
		Use:   "join",
		Short: "Run this on any machine you wish to join an existing cluster",
		Long:  joinLongDescription,
		Run: func(cmd *cobra.Command, args []string) {
			j, err := NewValidJoin(cmd.PersistentFlags(), cfg, args, cfgPath, featureGatesString, ignorePreflightErrors)
			kubeadmutil.CheckErr(err)
			kubeadmutil.CheckErr(j.Run(out))
		},
	}

	AddJoinConfigFlags(cmd.PersistentFlags(), cfg, &featureGatesString)
	AddJoinOtherFlags(cmd.PersistentFlags(), &cfgPath, &ignorePreflightErrors)

	return cmd
}

// NewValidJoin validates the command line that are passed to the cobra command
func NewValidJoin(flagSet *flag.FlagSet, cfg *kubeadmapiv1alpha3.JoinConfiguration, args []string, cfgPath, featureGatesString string, ignorePreflightErrors []string) (*Join, error) {
	cfg.DiscoveryTokenAPIServers = args

	var err error
	if cfg.FeatureGates, err = features.NewFeatureGate(&features.InitFeatureGates, featureGatesString); err != nil {
		return nil, err
	}

	if err := validation.ValidateMixedArguments(flagSet); err != nil {
		return nil, err
	}

	ignorePreflightErrorsSet, err := validation.ValidateIgnorePreflightErrors(ignorePreflightErrors)
	if err != nil {
		return nil, err
	}

	return NewJoin(cfgPath, args, cfg, ignorePreflightErrorsSet)
}

// AddJoinConfigFlags adds join flags bound to the config to the specified flagset
func AddJoinConfigFlags(flagSet *flag.FlagSet, cfg *kubeadmapiv1alpha3.JoinConfiguration, featureGatesString *string) {
	flagSet.StringVar(
		&cfg.DiscoveryFile, "discovery-file", "",
		"A file or url from which to load cluster information.")
	flagSet.StringVar(
		&cfg.DiscoveryToken, "discovery-token", "",
		"A token used to validate cluster information fetched from the api server.")
	flagSet.StringVar(
		&cfg.NodeRegistration.Name, "node-name", cfg.NodeRegistration.Name,
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
	flagSet.StringVar(
		&cfg.NodeRegistration.CRISocket, "cri-socket", cfg.NodeRegistration.CRISocket,
		`Specify the CRI socket to connect to.`,
	)
	flagSet.BoolVar(
		&cfg.ControlPlane, "experimental-control-plane", cfg.ControlPlane,
		"Create a new control plane instance on this node")
	flagSet.StringVar(
		&cfg.APIEndpoint.AdvertiseAddress, "apiserver-advertise-address", cfg.APIEndpoint.AdvertiseAddress,
		"If the node should host a new control plane instance, the IP address the API Server will advertise it's listening on.",
	)
	flagSet.Int32Var(
		&cfg.APIEndpoint.BindPort, "apiserver-bind-port", cfg.APIEndpoint.BindPort,
		"If the node should host a new control plane instance, the port for the API Server to bind to.",
	)
}

// AddJoinOtherFlags adds join flags that are not bound to a configuration file to the given flagset
func AddJoinOtherFlags(flagSet *flag.FlagSet, cfgPath *string, ignorePreflightErrors *[]string) {
	flagSet.StringVar(
		cfgPath, "config", *cfgPath,
		"Path to kubeadm config file.")

	flagSet.StringSliceVar(
		ignorePreflightErrors, "ignore-preflight-errors", *ignorePreflightErrors,
		"A list of checks whose errors will be shown as warnings. Example: 'IsPrivilegedUser,Swap'. Value 'all' ignores errors from all checks.",
	)
}

// Join defines struct used by kubeadm join command
type Join struct {
	cfg                   *kubeadmapi.JoinConfiguration
	ignorePreflightErrors sets.String
}

// NewJoin instantiates Join struct with given arguments
func NewJoin(cfgPath string, args []string, defaultcfg *kubeadmapiv1alpha3.JoinConfiguration, ignorePreflightErrors sets.String) (*Join, error) {

	if defaultcfg.NodeRegistration.Name == "" {
		glog.V(1).Infoln("[join] found NodeName empty; using OS hostname as NodeName")
	}

	if defaultcfg.APIEndpoint.AdvertiseAddress == "" {
		glog.V(1).Infoln("[join] found advertiseAddress empty; using default interface's IP address as advertiseAddress")
	}

	internalcfg, err := configutil.NodeConfigFileAndDefaultsToInternalConfig(cfgPath, defaultcfg)
	if err != nil {
		return nil, err
	}
	if err := configutil.VerifyAPIServerBindAddress(internalcfg.APIEndpoint.AdvertiseAddress); err != nil {
		return nil, err
	}

	fmt.Println("[preflight] running pre-flight checks")

	// Then continue with the others...
	glog.V(1).Infoln("[preflight] running various checks on all nodes")
	if err := preflight.RunJoinNodeChecks(utilsexec.New(), internalcfg, ignorePreflightErrors); err != nil {
		return nil, err
	}

	return &Join{cfg: internalcfg, ignorePreflightErrors: ignorePreflightErrors}, nil
}

// Run executes worker node provisioning and tries to join an existing cluster.
func (j *Join) Run(out io.Writer) error {
	// Perform the Discovery, which turns a Bootstrap Token and optionally (and preferably) a CA cert hash into a KubeConfig
	// file that may be used for the TLS Bootstrapping process the kubelet performs using the Certificates API.
	glog.V(1).Infoln("[join] discovering cluster-info")
	tlsBootstrapCfg, err := discovery.For(j.cfg)
	if err != nil {
		return err
	}

	// If the node should host a new control plane instance
	var initConfiguration *kubeadmapi.InitConfiguration
	if j.cfg.ControlPlane == true {
		// Retrives the kubeadm configuration used during kubeadm init
		glog.V(1).Infoln("[join] retrieving KubeConfig objects")
		initConfiguration, err = j.FetchInitConfiguration(tlsBootstrapCfg)
		if err != nil {
			return err
		}

		// injects into the kubeadm configuration the information about the joining node
		initConfiguration.NodeRegistration = j.cfg.NodeRegistration
		initConfiguration.APIEndpoint = j.cfg.APIEndpoint

		// Checks if the cluster configuration supports
		// joining a new control plane instance and if all the necessary certificates are provided
		if err = j.CheckIfReadyForAdditionalControlPlane(initConfiguration); err != nil {
			// outputs the not ready for hosting a new control plane instance message
			ctx := map[string]string{
				"Error": err.Error(),
			}

			var msg bytes.Buffer
			notReadyToJoinControPlaneTemp.Execute(&msg, ctx)
			return errors.New(msg.String())
		}

		// run kubeadm init preflight checks for checking all the prequisites
		fmt.Printf("[join] running pre-flight checks before initializing the new control plane instance\n")
		preflight.RunInitMasterChecks(utilsexec.New(), initConfiguration, j.ignorePreflightErrors)

		// Prepares the node for hosting a new control plane instance by writing necessary
		// KubeConfig files, and static pod manifests
		if err = j.PrepareForHostingControlPlane(initConfiguration); err != nil {
			return err
		}
	}

	// Executes the kubelet TLS bootstrap process, that completes with the node
	// joining the cluster with a dedicates set of credentials as required by
	// the node authorizer.
	// if the node is hosting a new control plane instance, since it uses static pods for the control plane,
	// as soon as the kubelet starts it will take charge of creating control plane
	// components on the node.
	if err = j.BootstrapKubelet(tlsBootstrapCfg); err != nil {
		return err
	}

	// if the node is hosting a new control plane instance
	if j.cfg.ControlPlane == true {
		// Completes the control plane setup
		if err := j.PostInstallControlPlane(initConfiguration); err != nil {
			return err
		}

		// outputs the join control plane done template and exits
		ctx := map[string]string{
			"KubeConfigPath": kubeadmconstants.GetAdminKubeConfigPath(),
		}
		joinControPlaneDoneTemp.Execute(out, ctx)
		return nil
	}

	// otherwise, if the node joined as a worker node;
	// outputs the join done message and exits
	fmt.Fprintf(out, joinWorkerNodeDoneMsg)
	return nil
}

// FetchInitConfiguration reads the cluster configuration from the kubeadm-admin configMap,
func (j *Join) FetchInitConfiguration(tlsBootstrapCfg *clientcmdapi.Config) (*kubeadmapi.InitConfiguration, error) {
	// creates a client to access the cluster using the bootstrap token identity
	tlsClient, err := kubeconfigutil.ToClientSet(tlsBootstrapCfg)
	if err != nil {
		return nil, errors.Wrap(err, "Unable to access the cluster")
	}

	// Fetches the init configuration
	initConfiguration, err := configutil.FetchConfigFromFileOrCluster(tlsClient, os.Stdout, "join", "", true)
	if err != nil {
		return nil, errors.Wrap(err, "Unable to fetch the kubeadm-config ConfigMap")
	}

	return initConfiguration, nil
}

// CheckIfReadyForAdditionalControlPlane ensures that the cluster is in a state that supports
// joining an additional control plane instance and if the node is ready to join
func (j *Join) CheckIfReadyForAdditionalControlPlane(initConfiguration *kubeadmapi.InitConfiguration) error {
	// blocks if the cluster was created without a stable control plane endpoint
	if initConfiguration.ControlPlaneEndpoint == "" {
		return fmt.Errorf("unable to add a new control plane instance a cluster that doesn't have a stable controlPlaneEndpoint address")
	}

	// blocks if the cluster was created without an external etcd cluster
	if initConfiguration.Etcd.External == nil {
		return fmt.Errorf("unable to add a new control plane instance on a cluster that doesn't use an external etcd")
	}

	// blocks if control plane is self-hosted
	if features.Enabled(initConfiguration.FeatureGates, features.SelfHosting) {
		return fmt.Errorf("self-hosted clusters are deprecated and won't be supported by `kubeadm join --experimental-control-plane`")
	}

	// blocks if the certificates for the control plane are stored in secrets (instead of the local pki folder)
	if features.Enabled(initConfiguration.FeatureGates, features.StoreCertsInSecrets) {
		return fmt.Errorf("certificates stored in secrets, as well as self-hosted clusters are deprecated and won't be supported by `kubeadm join --experimental-control-plane`")
	}

	// checks if the certificates that must be equal across contolplane instances are provided
	if ret, err := certsphase.SharedCertificateExists(initConfiguration); !ret {
		return err
	}

	return nil
}

// PrepareForHostingControlPlane makes all preparation activities require for a node hosting a new control plane instance
func (j *Join) PrepareForHostingControlPlane(initConfiguration *kubeadmapi.InitConfiguration) error {

	// Creates the admin kubeconfig file for the admin and for kubeadm itself.
	if err := kubeconfigphase.CreateAdminKubeConfigFile(kubeadmconstants.KubernetesDir, initConfiguration); err != nil {
		return errors.Wrap(err, "error generating the admin kubeconfig file")
	}

	// Generate missing certificates (if any)
	if err := certsphase.CreatePKIAssets(initConfiguration); err != nil {
		return err
	}

	// Generate kubeconfig files for controller manager, scheduler and for the admin/kubeadm itself
	// NB. The kubeconfig file for kubelet will be generated by the TLS bootstrap process in
	// following steps of the join --experimental-control plane workflow
	if err := kubeconfigphase.CreateJoinControlPlaneKubeConfigFiles(kubeadmconstants.KubernetesDir, initConfiguration); err != nil {
		return errors.Wrap(err, "error generating kubeconfig files")
	}

	// Creates static pod manifests file for the control plane components to be deployed on this node
	// Static pods will be created and managed by the kubelet as soon as it starts
	if err := controlplanephase.CreateInitStaticPodManifestFiles(kubeadmconstants.GetStaticPodDirectory(), initConfiguration); err != nil {
		return errors.Wrap(err, "error creating static pod manifest files for the control plane components")
	}

	return nil
}

// BootstrapKubelet executes the kubelet TLS bootstrap process.
// This process is executed by the kubelet and completes with the node joining the cluster
// with a dedicates set of credentials as required by the node authorizer
func (j *Join) BootstrapKubelet(tlsBootstrapCfg *clientcmdapi.Config) error {
	bootstrapKubeConfigFile := kubeadmconstants.GetBootstrapKubeletKubeConfigPath()

	// Write the bootstrap kubelet config file or the TLS-Boostrapped kubelet config file down to disk
	glog.V(1).Infoln("[join] writing bootstrap kubelet config file at", bootstrapKubeConfigFile)
	if err := kubeconfigutil.WriteToDisk(bootstrapKubeConfigFile, tlsBootstrapCfg); err != nil {
		return errors.Wrap(err, "couldn't save bootstrap-kubelet.conf to disk")
	}

	// Write the ca certificate to disk so kubelet can use it for authentication
	cluster := tlsBootstrapCfg.Contexts[tlsBootstrapCfg.CurrentContext].Cluster
	if _, err := os.Stat(j.cfg.CACertPath); os.IsNotExist(err) {
		if err := certutil.WriteCert(j.cfg.CACertPath, tlsBootstrapCfg.Clusters[cluster].CertificateAuthorityData); err != nil {
			return errors.Wrap(err, "couldn't save the CA certificate to disk")
		}
	}

	kubeletVersion, err := preflight.GetKubeletVersion(utilsexec.New())
	if err != nil {
		return err
	}

	bootstrapClient, err := kubeconfigutil.ClientSetFromFile(bootstrapKubeConfigFile)
	if err != nil {
		return fmt.Errorf("couldn't create client from kubeconfig file %q", bootstrapKubeConfigFile)
	}

	// Configure the kubelet. In this short timeframe, kubeadm is trying to stop/restart the kubelet
	// Try to stop the kubelet service so no race conditions occur when configuring it
	glog.V(1).Infof("Stopping the kubelet")
	kubeletphase.TryStopKubelet()

	// Write the configuration for the kubelet (using the bootstrap token credentials) to disk so the kubelet can start
	if err := kubeletphase.DownloadConfig(bootstrapClient, kubeletVersion, kubeadmconstants.KubeletRunDirectory); err != nil {
		return err
	}

	// Write env file with flags for the kubelet to use. We do not need to write the --register-with-taints for the master,
	// as we handle that ourselves in the markmaster phase
	// TODO: Maybe we want to do that some time in the future, in order to remove some logic from the markmaster phase?
	if err := kubeletphase.WriteKubeletDynamicEnvFile(&j.cfg.NodeRegistration, j.cfg.FeatureGates, false, kubeadmconstants.KubeletRunDirectory); err != nil {
		return err
	}

	// Try to start the kubelet service in case it's inactive
	glog.V(1).Infof("Starting the kubelet")
	kubeletphase.TryStartKubelet()

	// Now the kubelet will perform the TLS Bootstrap, transforming /etc/kubernetes/bootstrap-kubelet.conf to /etc/kubernetes/kubelet.conf
	// Wait for the kubelet to create the /etc/kubernetes/kubelet.conf KubeConfig file. If this process
	// times out, display a somewhat user-friendly message.
	waiter := apiclient.NewKubeWaiter(nil, kubeadmconstants.TLSBootstrapTimeout, os.Stdout)
	if err := waitForKubeletAndFunc(waiter, waitForTLSBootstrappedClient); err != nil {
		fmt.Printf(kubeadmJoinFailMsg, err)
		return err
	}

	// When we know the /etc/kubernetes/kubelet.conf file is available, get the client
	client, err := kubeconfigutil.ClientSetFromFile(kubeadmconstants.GetKubeletKubeConfigPath())
	if err != nil {
		return err
	}

	glog.V(1).Infof("[join] preserving the crisocket information for the node")
	if err := patchnodephase.AnnotateCRISocket(client, j.cfg.NodeRegistration.Name, j.cfg.NodeRegistration.CRISocket); err != nil {
		return errors.Wrap(err, "error uploading crisocket")
	}

	// This feature is disabled by default in kubeadm
	if features.Enabled(j.cfg.FeatureGates, features.DynamicKubeletConfig) {
		if err := kubeletphase.EnableDynamicConfigForNode(client, j.cfg.NodeRegistration.Name, kubeletVersion); err != nil {
			return errors.Wrap(err, "error consuming base kubelet configuration")
		}
	}

	return nil
}

// PostInstallControlPlane marks the new node as master and update the cluster status with information about current node
func (j *Join) PostInstallControlPlane(initConfiguration *kubeadmapi.InitConfiguration) error {
	kubeConfigFile := filepath.Join(kubeadmconstants.KubernetesDir, kubeadmconstants.AdminKubeConfigFileName)

	client, err := kubeconfigutil.ClientSetFromFile(kubeConfigFile)
	if err != nil {
		return errors.Wrap(err, "couldn't create kubernetes client")
	}

	glog.V(1).Info("[join] uploading currently used configuration to the cluster")
	if err := uploadconfigphase.UploadConfiguration(initConfiguration, client); err != nil {
		return fmt.Errorf("error uploading configuration: %v", err)
	}

	glog.V(1).Info("[join] marking the master with right label")
	if err = markmasterphase.MarkMaster(client, initConfiguration.NodeRegistration.Name, initConfiguration.NodeRegistration.Taints); err != nil {
		return errors.Wrap(err, "error applying master label and taints")
	}

	return nil
}

// waitForTLSBootstrappedClient waits for the /etc/kubernetes/kubelet.conf file to be available
func waitForTLSBootstrappedClient() error {
	fmt.Println("[tlsbootstrap] Waiting for the kubelet to perform the TLS Bootstrap...")

	kubeletKubeConfig := filepath.Join(kubeadmconstants.KubernetesDir, kubeadmconstants.KubeletKubeConfigFileName)
	// Loop on every falsy return. Return with an error if raised. Exit successfully if true is returned.
	return wait.PollImmediate(kubeadmconstants.APICallRetryInterval, kubeadmconstants.TLSBootstrapTimeout, func() (bool, error) {
		_, err := os.Stat(kubeletKubeConfig)
		return (err == nil), nil
	})
}
