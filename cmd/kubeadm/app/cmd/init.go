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
	"text/template"
	"time"

	"github.com/renstrom/dedent"
	"github.com/spf13/cobra"

	"k8s.io/apimachinery/pkg/runtime"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiext "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha1"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/validation"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/features"
	cmdphases "k8s.io/kubernetes/cmd/kubeadm/app/cmd/phases"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	dnsaddonphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/addons/dns"
	proxyaddonphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/addons/proxy"
	apiconfigphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/apiconfig"
	clusterinfophase "k8s.io/kubernetes/cmd/kubeadm/app/phases/bootstraptoken/clusterinfo"
	nodebootstraptokenphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/bootstraptoken/node"
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
	kubeconfigutil "k8s.io/kubernetes/cmd/kubeadm/app/util/kubeconfig"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/pubkeypin"
	"k8s.io/kubernetes/pkg/api"
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
		  http://kubernetes.io/docs/admin/addons/

		You can now join any number of machines by running the following on each node
		as root:

		  kubeadm join --token {{.Token}} {{.MasterHostPort}} --discovery-token-ca-cert-hash {{.CAPubKeyPin}}

		`)))
)

// NewCmdInit returns "kubeadm init" command.
func NewCmdInit(out io.Writer) *cobra.Command {
	cfg := &kubeadmapiext.MasterConfiguration{}
	api.Scheme.Default(cfg)

	var cfgPath string
	var skipPreFlight bool
	var skipTokenPrint bool
	cmd := &cobra.Command{
		Use:   "init",
		Short: "Run this in order to set up the Kubernetes master",
		Run: func(cmd *cobra.Command, args []string) {
			api.Scheme.Default(cfg)
			internalcfg := &kubeadmapi.MasterConfiguration{}
			api.Scheme.Convert(cfg, internalcfg, nil)

			i, err := NewInit(cfgPath, internalcfg, skipPreFlight, skipTokenPrint)
			kubeadmutil.CheckErr(err)
			kubeadmutil.CheckErr(i.Validate(cmd))

			// TODO: remove this warning in 1.9
			if !cmd.Flags().Lookup("token-ttl").Changed {
				fmt.Println("[kubeadm] WARNING: starting in 1.8, tokens expire after 24 hours by default (if you require a non-expiring token use --token-ttl 0)")
			}

			kubeadmutil.CheckErr(i.Run(out))
		},
	}

	cmd.PersistentFlags().StringVar(
		&cfg.API.AdvertiseAddress, "apiserver-advertise-address", cfg.API.AdvertiseAddress,
		"The IP address the API Server will advertise it's listening on. 0.0.0.0 means the default network interface's address.",
	)
	cmd.PersistentFlags().Int32Var(
		&cfg.API.BindPort, "apiserver-bind-port", cfg.API.BindPort,
		"Port for the API Server to bind to",
	)
	cmd.PersistentFlags().StringVar(
		&cfg.Networking.ServiceSubnet, "service-cidr", cfg.Networking.ServiceSubnet,
		"Use alternative range of IP address for service VIPs",
	)
	cmd.PersistentFlags().StringVar(
		&cfg.Networking.PodSubnet, "pod-network-cidr", cfg.Networking.PodSubnet,
		"Specify range of IP addresses for the pod network; if set, the control plane will automatically allocate CIDRs for every node",
	)
	cmd.PersistentFlags().StringVar(
		&cfg.Networking.DNSDomain, "service-dns-domain", cfg.Networking.DNSDomain,
		`Use alternative domain for services, e.g. "myorg.internal"`,
	)
	cmd.PersistentFlags().StringVar(
		&cfg.KubernetesVersion, "kubernetes-version", cfg.KubernetesVersion,
		`Choose a specific Kubernetes version for the control plane`,
	)
	cmd.PersistentFlags().StringVar(
		&cfg.CertificatesDir, "cert-dir", cfg.CertificatesDir,
		`The path where to save and store the certificates`,
	)
	cmd.PersistentFlags().StringSliceVar(
		&cfg.APIServerCertSANs, "apiserver-cert-extra-sans", cfg.APIServerCertSANs,
		`Optional extra altnames to use for the API Server serving cert. Can be both IP addresses and dns names.`,
	)
	cmd.PersistentFlags().StringVar(
		&cfg.NodeName, "node-name", cfg.NodeName,
		`Specify the node name`,
	)

	cmd.PersistentFlags().StringVar(&cfgPath, "config", cfgPath, "Path to kubeadm config file (WARNING: Usage of a configuration file is experimental)")

	// Note: All flags that are not bound to the cfg object should be whitelisted in cmd/kubeadm/app/apis/kubeadm/validation/validation.go
	cmd.PersistentFlags().BoolVar(
		&skipPreFlight, "skip-preflight-checks", skipPreFlight,
		"Skip preflight checks normally run before modifying the system",
	)
	// Note: All flags that are not bound to the cfg object should be whitelisted in cmd/kubeadm/app/apis/kubeadm/validation/validation.go
	cmd.PersistentFlags().BoolVar(
		&skipTokenPrint, "skip-token-print", skipTokenPrint,
		"Skip printing of the default bootstrap token generated by 'kubeadm init'",
	)

	cmd.PersistentFlags().StringVar(
		&cfg.Token, "token", cfg.Token,
		"The token to use for establishing bidirectional trust between nodes and masters.")

	cmd.PersistentFlags().DurationVar(
		&cfg.TokenTTL, "token-ttl", cfg.TokenTTL,
		"The duration before the bootstrap token is automatically deleted. 0 means 'never expires'.")

	return cmd
}

func NewInit(cfgPath string, cfg *kubeadmapi.MasterConfiguration, skipPreFlight, skipTokenPrint bool) (*Init, error) {

	fmt.Println("[kubeadm] WARNING: kubeadm is in beta, please do not use it for production clusters.")

	if cfgPath != "" {
		b, err := ioutil.ReadFile(cfgPath)
		if err != nil {
			return nil, fmt.Errorf("unable to read config from %q [%v]", cfgPath, err)
		}
		if err := runtime.DecodeInto(api.Codecs.UniversalDecoder(), b, cfg); err != nil {
			return nil, fmt.Errorf("unable to decode config from %q [%v]", cfgPath, err)
		}
	}

	// Set defaults dynamically that the API group defaulting can't (by fetching information from the internet, looking up network interfaces, etc.)
	err := configutil.SetInitDynamicDefaults(cfg)
	if err != nil {
		return nil, err
	}

	fmt.Printf("[init] Using Kubernetes version: %s\n", cfg.KubernetesVersion)
	fmt.Printf("[init] Using Authorization mode: %v\n", cfg.AuthorizationModes)

	// Warn about the limitations with the current cloudprovider solution.
	if cfg.CloudProvider != "" {
		fmt.Println("[init] WARNING: For cloudprovider integrations to work --cloud-provider must be set for all kubelets in the cluster.")
		fmt.Println("\t(/etc/systemd/system/kubelet.service.d/10-kubeadm.conf should be edited for this purpose)")
	}

	if !skipPreFlight {
		fmt.Println("[preflight] Running pre-flight checks")

		if err := preflight.RunInitMasterChecks(cfg); err != nil {
			return nil, err
		}

		// Try to start the kubelet service in case it's inactive
		preflight.TryStartKubelet()
	} else {
		fmt.Println("[preflight] Skipping pre-flight checks")
	}

	return &Init{cfg: cfg, skipTokenPrint: skipTokenPrint}, nil
}

type Init struct {
	cfg            *kubeadmapi.MasterConfiguration
	skipTokenPrint bool
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
		return fmt.Errorf("couldn't parse kubernetes version %q: %v", i.cfg.KubernetesVersion, err)
	}

	// PHASE 1: Generate certificates
	if err := cmdphases.CreatePKIAssets(i.cfg); err != nil {
		return err
	}

	// PHASE 2: Generate kubeconfig files for the admin and the kubelet
	if err := kubeconfigphase.CreateInitKubeConfigFiles(kubeadmconstants.KubernetesDir, i.cfg); err != nil {
		return err
	}

	// PHASE 3: Bootstrap the control plane
	manifestPath := kubeadmconstants.GetStaticPodDirectory()
	if err := controlplanephase.CreateInitStaticPodManifestFiles(manifestPath, i.cfg); err != nil {
		return err
	}
	// Add etcd static pod spec only if external etcd is not configured
	if len(i.cfg.Etcd.Endpoints) == 0 {
		if err := etcdphase.CreateLocalEtcdStaticPodManifestFile(manifestPath, i.cfg); err != nil {
			return err
		}
	}

	client, err := kubeconfigutil.ClientSetFromFile(kubeadmconstants.GetAdminKubeConfigPath())
	if err != nil {
		return err
	}

	fmt.Printf("[init] Waiting for the kubelet to boot up the control plane as Static Pods from directory %q\n", kubeadmconstants.GetStaticPodDirectory())
	if err := apiclient.WaitForAPI(client, 30*time.Minute); err != nil {
		return err
	}

	// PHASE 4: Mark the master with the right label/taint
	if err := markmasterphase.MarkMaster(client, i.cfg.NodeName); err != nil {
		return err
	}

	// PHASE 5: Set up the node bootstrap tokens
	if !i.skipTokenPrint {
		fmt.Printf("[token] Using token: %s\n", i.cfg.Token)
	}

	// Create the default node bootstrap token
	tokenDescription := "The default bootstrap token generated by 'kubeadm init'."
	if err := nodebootstraptokenphase.UpdateOrCreateToken(client, i.cfg.Token, false, i.cfg.TokenTTL, kubeadmconstants.DefaultTokenUsages, tokenDescription); err != nil {
		return err
	}
	// Create RBAC rules that makes the bootstrap tokens able to post CSRs
	if err := nodebootstraptokenphase.AllowBootstrapTokensToPostCSRs(client); err != nil {
		return err
	}
	// Create RBAC rules that makes the bootstrap tokens able to get their CSRs approved automatically
	if err := nodebootstraptokenphase.AutoApproveNodeBootstrapTokens(client, k8sVersion); err != nil {
		return err
	}

	// Create the cluster-info ConfigMap with the associated RBAC rules
	if err := clusterinfophase.CreateBootstrapConfigMapIfNotExists(client, kubeadmconstants.GetAdminKubeConfigPath()); err != nil {
		return err
	}
	if err := clusterinfophase.CreateClusterInfoRBACRules(client); err != nil {
		return err
	}

	// PHASE 6: Install and deploy all addons, and configure things as necessary

	// Upload currently used configuration to the cluster
	if err := uploadconfigphase.UploadConfiguration(i.cfg, client); err != nil {
		return err
	}

	if err := apiconfigphase.CreateRBACRules(client, k8sVersion); err != nil {
		return err
	}

	if err := dnsaddonphase.EnsureDNSAddon(i.cfg, client); err != nil {
		return err
	}

	if err := proxyaddonphase.EnsureProxyAddon(i.cfg, client); err != nil {
		return err
	}

	// PHASE 7: Make the control plane self-hosted if feature gate is enabled
	if features.Enabled(i.cfg.FeatureFlags, features.SelfHosting) {
		// Temporary control plane is up, now we create our self hosted control
		// plane components and remove the static manifests:
		fmt.Println("[self-hosted] Creating self-hosted control plane...")
		if err := selfhostingphase.CreateSelfHostedControlPlane(i.cfg, client); err != nil {
			return err
		}
	}

	// Load the CA certificate from so we can pin its public key
	caCert, err := pkiutil.TryLoadCertFromDisk(i.cfg.CertificatesDir, kubeadmconstants.CACertAndKeyBaseName)

	// Generate the Master host/port pair used by initDoneTempl
	masterHostPort, err := kubeadmutil.GetMasterHostPort(i.cfg)
	if err != nil {
		return err
	}

	ctx := map[string]string{
		"KubeConfigPath": kubeadmconstants.GetAdminKubeConfigPath(),
		"KubeConfigName": kubeadmconstants.AdminKubeConfigFileName,
		"Token":          i.cfg.Token,
		"CAPubKeyPin":    pubkeypin.Hash(caCert),
		"MasterHostPort": masterHostPort,
	}
	if i.skipTokenPrint {
		ctx["Token"] = "<value withheld>"
	}

	return initDoneTempl.Execute(out, ctx)
}
