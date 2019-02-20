/*
Copyright 2019 The Kubernetes Authors.

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
	"os"
	"text/template"

	"github.com/lithammer/dedent"
	"github.com/pkg/errors"
	"github.com/spf13/cobra"
	flag "github.com/spf13/pflag"
	"k8s.io/apimachinery/pkg/util/sets"
	clientset "k8s.io/client-go/kubernetes"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	"k8s.io/klog"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmscheme "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/scheme"
	kubeadmapiv1beta1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta1"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/validation"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/options"
	phases "k8s.io/kubernetes/cmd/kubeadm/app/cmd/phases/join"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/phases/workflow"
	cmdutil "k8s.io/kubernetes/cmd/kubeadm/app/cmd/util"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/discovery"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	configutil "k8s.io/kubernetes/cmd/kubeadm/app/util/config"
	kubeconfigutil "k8s.io/kubernetes/cmd/kubeadm/app/util/kubeconfig"
)

var (
	joinWorkerNodeDoneMsg = dedent.Dedent(`
		This node has joined the cluster:
		* Certificate signing request was sent to apiserver and a response was received.
		* The Kubelet was informed of the new secure connection details.

		Run 'kubectl get nodes' on the control-plane to see this node join the cluster.

		`)

	joinControPlaneDoneTemp = template.Must(template.New("join").Parse(dedent.Dedent(`
		This node has joined the cluster and a new control plane instance was created:

		* Certificate signing request was sent to apiserver and approval was received.
		* The Kubelet was informed of the new secure connection details.
		* Control plane (master) label and taint were applied to the new node.
		* The Kubernetes control plane instances scaled up.
		{{.etcdMessage}}

		To start administering your cluster from this node, you need to run the following as a regular user:

			mkdir -p $HOME/.kube
			sudo cp -i {{.KubeConfigPath}} $HOME/.kube/config
			sudo chown $(id -u):$(id -g) $HOME/.kube/config

		Run 'kubectl get nodes' to see this node join the cluster.

		`)))

	joinLongDescription = dedent.Dedent(`
		When joining a kubeadm initialized cluster, we need to establish
		bidirectional trust. This is split into discovery (having the Node
		trust the Kubernetes Control Plane) and TLS bootstrap (having the 
		Kubernetes Control Plane trust the Node).

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
		root certificate authority (CA) presented by the Kubernetes Control Plane. 
		The value of this flag is specified as "<hash-type>:<hex-encoded-value>",
		where the supported hash type is "sha256". The hash is calculated over
		the bytes of the Subject Public Key Info (SPKI) object (as in RFC7469).
		This value is available in the output of "kubeadm init" or can be
		calculated using standard tools. The --discovery-token-ca-cert-hash flag
		may be repeated multiple times to allow more than one public key.

		If you cannot know the CA public key hash ahead of time, you can pass
		the --discovery-token-unsafe-skip-ca-verification flag to disable this
		verification. This weakens the kubeadm security model since other nodes
		can potentially impersonate the Kubernetes Control Plane.

		The TLS bootstrap mechanism is also driven via a shared token. This is
		used to temporarily authenticate with the Kubernetes Control Plane to submit a
		certificate signing request (CSR) for a locally created key pair. By
		default, kubeadm will set up the Kubernetes Control Plane to automatically
		approve these signing requests. This token is passed in with the
		--tls-bootstrap-token abcdef.1234567890abcdef flag.

		Often times the same token is used for both parts. In this case, the
		--token flag can be used instead of specifying each token individually.
		`)
)

// joinOptions defines all the options exposed via flags by kubeadm join.
// Please note that this structure includes the public kubeadm config API, but only a subset of the options
// supported by this api will be exposed as a flag.
type joinOptions struct {
	cfgPath               string
	token                 string
	controlPlane          bool
	ignorePreflightErrors []string
	externalcfg           *kubeadmapiv1beta1.JoinConfiguration
}

// joinData defines all the runtime information used when running the kubeadm join worklow;
// this data is shared across all the phases that are included in the workflow.
type joinData struct {
	cfg                   *kubeadmapi.JoinConfiguration
	skipTokenPrint        bool
	initCfg               *kubeadmapi.InitConfiguration
	tlsBootstrapCfg       *clientcmdapi.Config
	clientSets            map[string]*clientset.Clientset
	ignorePreflightErrors sets.String
	outputWriter          io.Writer
}

// NewCmdJoin returns "kubeadm join" command.
// NB. joinOptions is exposed as parameter for allowing unit testing of
//     the newJoinData method, that implements all the command options validation logic
func NewCmdJoin(out io.Writer, joinOptions *joinOptions) *cobra.Command {
	if joinOptions == nil {
		joinOptions = newJoinOptions()
	}
	joinRunner := workflow.NewRunner()

	cmd := &cobra.Command{
		Use:   "join",
		Short: "Run this on any machine you wish to join an existing cluster",
		Long:  joinLongDescription,
		Run: func(cmd *cobra.Command, args []string) {

			c, err := joinRunner.InitData(args)
			kubeadmutil.CheckErr(err)

			data := c.(*joinData)

			err = joinRunner.Run(args)
			kubeadmutil.CheckErr(err)

			// if the node is hosting a new control plane instance
			if data.cfg.ControlPlane != nil {
				// outputs the join control plane done message and exit
				etcdMessage := ""
				if data.initCfg.Etcd.External == nil {
					etcdMessage = "* A new etcd member was added to the local/stacked etcd cluster."
				}

				ctx := map[string]string{
					"KubeConfigPath": kubeadmconstants.GetAdminKubeConfigPath(),
					"etcdMessage":    etcdMessage,
				}
				joinControPlaneDoneTemp.Execute(data.outputWriter, ctx)

			} else {
				// otherwise, if the node joined as a worker node;
				// outputs the join done message and exit
				fmt.Fprintf(data.outputWriter, joinWorkerNodeDoneMsg)
			}
		},
		// We accept the control-plane location as an optional positional argument
		Args: cobra.MaximumNArgs(1),
	}

	addJoinConfigFlags(cmd.Flags(), joinOptions.externalcfg)
	addJoinOtherFlags(cmd.Flags(), &joinOptions.cfgPath, &joinOptions.ignorePreflightErrors, &joinOptions.controlPlane, &joinOptions.token)

	joinRunner.AppendPhase(phases.NewPreflightPhase())
	joinRunner.AppendPhase(phases.NewControlPlanePreparePhase())
	joinRunner.AppendPhase(phases.NewCheckEtcdPhase())
	joinRunner.AppendPhase(phases.NewKubeletStartPhase())
	joinRunner.AppendPhase(phases.NewControlPlaneJoinPhase())

	// sets the data builder function, that will be used by the runner
	// both when running the entire workflow or single phases
	joinRunner.SetDataInitializer(func(cmd *cobra.Command, args []string) (workflow.RunData, error) {
		return newJoinData(cmd, args, joinOptions, out)
	})

	// binds the Runner to kubeadm join command by altering
	// command help, adding --skip-phases flag and by adding phases subcommands
	joinRunner.BindToCommand(cmd)

	return cmd
}

// addJoinConfigFlags adds join flags bound to the config to the specified flagset
func addJoinConfigFlags(flagSet *flag.FlagSet, cfg *kubeadmapiv1beta1.JoinConfiguration) {
	flagSet.StringVar(
		&cfg.NodeRegistration.Name, options.NodeName, cfg.NodeRegistration.Name,
		`Specify the node name.`,
	)
	// add control plane endpoint flags to the specified flagset
	flagSet.StringVar(
		&cfg.ControlPlane.LocalAPIEndpoint.AdvertiseAddress, options.APIServerAdvertiseAddress, cfg.ControlPlane.LocalAPIEndpoint.AdvertiseAddress,
		"If the node should host a new control plane instance, the IP address the API Server will advertise it's listening on. If not set the default network interface will be used.",
	)
	flagSet.Int32Var(
		&cfg.ControlPlane.LocalAPIEndpoint.BindPort, options.APIServerBindPort, cfg.ControlPlane.LocalAPIEndpoint.BindPort,
		"If the node should host a new control plane instance, the port for the API Server to bind to.",
	)
	// adds bootstrap token specific discovery flags to the specified flagset
	flagSet.StringVar(
		&cfg.Discovery.BootstrapToken.Token, options.TokenDiscovery, "",
		"For token-based discovery, the token used to validate cluster information fetched from the API server.",
	)
	flagSet.StringSliceVar(
		&cfg.Discovery.BootstrapToken.CACertHashes, options.TokenDiscoveryCAHash, []string{},
		"For token-based discovery, validate that the root CA public key matches this hash (format: \"<type>:<value>\").",
	)
	flagSet.BoolVar(
		&cfg.Discovery.BootstrapToken.UnsafeSkipCAVerification, options.TokenDiscoverySkipCAHash, false,
		"For token-based discovery, allow joining without --discovery-token-ca-cert-hash pinning.",
	)
	//	discovery via kube config file flag
	flagSet.StringVar(
		&cfg.Discovery.File.KubeConfigPath, options.FileDiscovery, "",
		"For file-based discovery, a file or URL from which to load cluster information.",
	)
	flagSet.StringVar(
		&cfg.Discovery.TLSBootstrapToken, options.TLSBootstrapToken, cfg.Discovery.TLSBootstrapToken,
		`Specify the token used to temporarily authenticate with the Kubernetes Control Plane while joining the node.`,
	)
	cmdutil.AddCRISocketFlag(flagSet, &cfg.NodeRegistration.CRISocket)
}

// addJoinOtherFlags adds join flags that are not bound to a configuration file to the given flagset
func addJoinOtherFlags(flagSet *flag.FlagSet, cfgPath *string, ignorePreflightErrors *[]string, controlPlane *bool, token *string) {
	flagSet.StringVar(
		cfgPath, options.CfgPath, *cfgPath,
		"Path to kubeadm config file.",
	)
	flagSet.StringSliceVar(
		ignorePreflightErrors, options.IgnorePreflightErrors, *ignorePreflightErrors,
		"A list of checks whose errors will be shown as warnings. Example: 'IsPrivilegedUser,Swap'. Value 'all' ignores errors from all checks.",
	)
	flagSet.StringVar(
		token, options.TokenStr, "",
		"Use this token for both discovery-token and tls-bootstrap-token when those values are not provided.",
	)
	flagSet.BoolVar(
		controlPlane, options.ControlPlane, *controlPlane,
		"Create a new control plane instance on this node",
	)
}

// newJoinOptions returns a struct ready for being used for creating cmd join flags.
func newJoinOptions() *joinOptions {
	// initialize the public kubeadm config API by appling defaults
	externalcfg := &kubeadmapiv1beta1.JoinConfiguration{}

	// Add optional config objects to host flags.
	// un-set objects will be cleaned up afterwards (into newJoinData func)
	externalcfg.Discovery.File = &kubeadmapiv1beta1.FileDiscovery{}
	externalcfg.Discovery.BootstrapToken = &kubeadmapiv1beta1.BootstrapTokenDiscovery{}
	externalcfg.ControlPlane = &kubeadmapiv1beta1.JoinControlPlane{}

	// Apply defaults
	kubeadmscheme.Scheme.Default(externalcfg)

	return &joinOptions{
		externalcfg: externalcfg,
	}
}

// newJoinData returns a new joinData struct to be used for the execution of the kubeadm join workflow.
// This func takes care of validating joinOptions passed to the command, and then it converts
// options into the internal JoinConfiguration type that is used as input all the phases in the kubeadm join workflow
func newJoinData(cmd *cobra.Command, args []string, options *joinOptions, out io.Writer) (*joinData, error) {
	// Re-apply defaults to the public kubeadm API (this will set only values not exposed/not set as a flags)
	kubeadmscheme.Scheme.Default(options.externalcfg)

	// Validate standalone flags values and/or combination of flags and then assigns
	// validated values to the public kubeadm config API when applicable

	// if a token is provided, use this value for both discovery-token and tls-bootstrap-token when those values are not provided
	if len(options.token) > 0 {
		if len(options.externalcfg.Discovery.TLSBootstrapToken) == 0 {
			options.externalcfg.Discovery.TLSBootstrapToken = options.token
		}
		if len(options.externalcfg.Discovery.BootstrapToken.Token) == 0 {
			options.externalcfg.Discovery.BootstrapToken.Token = options.token
		}
	}

	// if a file or URL from which to load cluster information was not provided, unset the Discovery.File object
	if len(options.externalcfg.Discovery.File.KubeConfigPath) == 0 {
		options.externalcfg.Discovery.File = nil
	}

	// if an APIServerEndpoint from which to retrive cluster information was not provided, unset the Discovery.BootstrapToken object
	if len(args) == 0 {
		options.externalcfg.Discovery.BootstrapToken = nil
	} else {
		if len(options.cfgPath) == 0 && len(args) > 1 {
			klog.Warningf("[join] WARNING: More than one API server endpoint supplied on command line %v. Using the first one.", args)
		}
		options.externalcfg.Discovery.BootstrapToken.APIServerEndpoint = args[0]
	}

	// if not joining a control plane, unset the ControlPlane object
	if !options.controlPlane {
		options.externalcfg.ControlPlane = nil
	}

	ignorePreflightErrorsSet, err := validation.ValidateIgnorePreflightErrors(options.ignorePreflightErrors)
	if err != nil {
		return nil, err
	}

	if err = validation.ValidateMixedArguments(cmd.Flags()); err != nil {
		return nil, err
	}

	// Either use the config file if specified, or convert public kubeadm API to the internal JoinConfiguration
	// and validates JoinConfiguration
	if options.externalcfg.NodeRegistration.Name == "" {
		klog.V(1).Infoln("[join] found NodeName empty; using OS hostname as NodeName")
	}

	if options.externalcfg.ControlPlane != nil && options.externalcfg.ControlPlane.LocalAPIEndpoint.AdvertiseAddress == "" {
		klog.V(1).Infoln("[join] found advertiseAddress empty; using default interface's IP address as advertiseAddress")
	}

	cfg, err := configutil.LoadOrDefaultJoinConfiguration(options.cfgPath, options.externalcfg)
	if err != nil {
		return nil, err
	}

	// override node name and CRI socket from the command line options
	if options.externalcfg.NodeRegistration.Name != "" {
		cfg.NodeRegistration.Name = options.externalcfg.NodeRegistration.Name
	}
	if options.externalcfg.NodeRegistration.CRISocket != "" {
		cfg.NodeRegistration.CRISocket = options.externalcfg.NodeRegistration.CRISocket
	}

	if cfg.ControlPlane != nil {
		if err := configutil.VerifyAPIServerBindAddress(cfg.ControlPlane.LocalAPIEndpoint.AdvertiseAddress); err != nil {
			return nil, err
		}
	}

	return &joinData{
		cfg:                   cfg,
		clientSets:            map[string]*clientset.Clientset{},
		ignorePreflightErrors: ignorePreflightErrorsSet,
		outputWriter:          out,
	}, nil
}

// Cfg returns the JoinConfiguration.
func (j *joinData) Cfg() *kubeadmapi.JoinConfiguration {
	return j.cfg
}

// TLSBootstrapCfg returns the cluster-info (kubeconfig).
func (j *joinData) TLSBootstrapCfg() (*clientcmdapi.Config, error) {
	if j.tlsBootstrapCfg != nil {
		return j.tlsBootstrapCfg, nil
	}
	klog.V(1).Infoln("[join] Discovering cluster-info")
	tlsBootstrapCfg, err := discovery.For(j.cfg)
	j.tlsBootstrapCfg = tlsBootstrapCfg
	return tlsBootstrapCfg, err
}

// InitCfg returns the InitConfiguration.
func (j *joinData) InitCfg() (*kubeadmapi.InitConfiguration, error) {
	if j.initCfg != nil {
		return j.initCfg, nil
	}
	if _, err := j.TLSBootstrapCfg(); err != nil {
		return nil, err
	}
	klog.V(1).Infoln("[join] Fetching init configuration")
	initCfg, err := fetchInitConfigurationFromJoinConfiguration(j.cfg, j.tlsBootstrapCfg)
	j.initCfg = initCfg
	return initCfg, err
}

func (j *joinData) ClientSetFromFile(path string) (*clientset.Clientset, error) {
	if client, ok := j.clientSets[path]; ok {
		return client, nil
	}
	client, err := kubeconfigutil.ClientSetFromFile(path)
	if err != nil {
		return nil, errors.Wrap(err, "[join] couldn't create Kubernetes client")
	}
	j.clientSets[path] = client
	return client, nil
}

// IgnorePreflightErrors returns the list of preflight errors to ignore.
func (j *joinData) IgnorePreflightErrors() sets.String {
	return j.ignorePreflightErrors
}

// OutputWriter returns the io.Writer used to write messages such as the "join done" message.
func (j *joinData) OutputWriter() io.Writer {
	return j.outputWriter
}

// fetchInitConfigurationFromJoinConfiguration retrieves the init configuration from a join configuration, performing the discovery
func fetchInitConfigurationFromJoinConfiguration(cfg *kubeadmapi.JoinConfiguration, tlsBootstrapCfg *clientcmdapi.Config) (*kubeadmapi.InitConfiguration, error) {
	// Retrieves the kubeadm configuration
	klog.V(1).Infoln("[join] Retrieving KubeConfig objects")
	initConfiguration, err := fetchInitConfiguration(tlsBootstrapCfg)
	if err != nil {
		return nil, err
	}

	// Create the final KubeConfig file with the cluster name discovered after fetching the cluster configuration
	clusterinfo := kubeconfigutil.GetClusterFromKubeConfig(tlsBootstrapCfg)
	tlsBootstrapCfg.Clusters = map[string]*clientcmdapi.Cluster{
		initConfiguration.ClusterName: clusterinfo,
	}
	tlsBootstrapCfg.Contexts[tlsBootstrapCfg.CurrentContext].Cluster = initConfiguration.ClusterName

	// injects into the kubeadm configuration the information about the joining node
	initConfiguration.NodeRegistration = cfg.NodeRegistration
	if cfg.ControlPlane != nil {
		initConfiguration.LocalAPIEndpoint = cfg.ControlPlane.LocalAPIEndpoint
	}

	return initConfiguration, nil
}

// fetchInitConfiguration reads the cluster configuration from the kubeadm-admin configMap
func fetchInitConfiguration(tlsBootstrapCfg *clientcmdapi.Config) (*kubeadmapi.InitConfiguration, error) {
	// creates a client to access the cluster using the bootstrap token identity
	tlsClient, err := kubeconfigutil.ToClientSet(tlsBootstrapCfg)
	if err != nil {
		return nil, errors.Wrap(err, "unable to access the cluster")
	}

	// Fetches the init configuration
	initConfiguration, err := configutil.FetchInitConfigurationFromCluster(tlsClient, os.Stdout, "join", true)
	if err != nil {
		return nil, errors.Wrap(err, "unable to fetch the kubeadm-config ConfigMap")
	}

	return initConfiguration, nil
}
