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
	"path"
	"strconv"
	"text/template"

	"github.com/renstrom/dedent"
	"github.com/spf13/cobra"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/pkg/api"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiext "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha1"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/validation"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubemaster "k8s.io/kubernetes/cmd/kubeadm/app/master"
	addonsphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/addons"
	apiconfigphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/apiconfig"
	certphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/certs"
	kubeconfigphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/kubeconfig"
	tokenphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/token"
	"k8s.io/kubernetes/cmd/kubeadm/app/preflight"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
)

var (
	initDoneTempl = template.Must(template.New("init").Parse(dedent.Dedent(`
		Your Kubernetes master has initialized successfully!

		To start using your cluster, you need to run (as a regular user):

		  sudo cp {{.KubeConfigPath}} $HOME/
		  sudo chown $(id -u):$(id -g) $HOME/{{.KubeConfigName}}
		  export KUBECONFIG=$HOME/{{.KubeConfigName}}

		You should now deploy a pod network to the cluster.
		Run "kubectl apply -f [podnetwork].yaml" with one of the options listed at:
		  http://kubernetes.io/docs/admin/addons/

		You can now join any number of machines by running the following on each node
		as root:

		  kubeadm join --token {{.Token}} {{.MasterIP}}:{{.MasterPort}}

		`)))
)

// NewCmdInit returns "kubeadm init" command.
func NewCmdInit(out io.Writer) *cobra.Command {
	cfg := &kubeadmapiext.MasterConfiguration{}
	api.Scheme.Default(cfg)

	var cfgPath string
	var skipPreFlight bool
	cmd := &cobra.Command{
		Use:   "init",
		Short: "Run this in order to set up the Kubernetes master",
		Run: func(cmd *cobra.Command, args []string) {
			api.Scheme.Default(cfg)
			internalcfg := &kubeadmapi.MasterConfiguration{}
			api.Scheme.Convert(cfg, internalcfg, nil)

			i, err := NewInit(cfgPath, internalcfg, skipPreFlight)
			kubeadmutil.CheckErr(err)
			kubeadmutil.CheckErr(i.Validate())
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

	cmd.PersistentFlags().StringVar(&cfgPath, "config", cfgPath, "Path to kubeadm config file (WARNING: Usage of a configuration file is experimental)")

	cmd.PersistentFlags().BoolVar(
		&skipPreFlight, "skip-preflight-checks", skipPreFlight,
		"Skip preflight checks normally run before modifying the system",
	)

	cmd.PersistentFlags().StringVar(
		&cfg.Token, "token", cfg.Token,
		"The token to use for establishing bidirectional trust between nodes and masters.")

	cmd.PersistentFlags().DurationVar(
		&cfg.TokenTTL, "token-ttl", cfg.TokenTTL,
		"The duration before the bootstrap token is automatically deleted. 0 means 'never expires'.")

	return cmd
}

func NewInit(cfgPath string, cfg *kubeadmapi.MasterConfiguration, skipPreFlight bool) (*Init, error) {

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
	err := setInitDynamicDefaults(cfg)
	if err != nil {
		return nil, err
	}

	if !skipPreFlight {
		fmt.Println("[preflight] Running pre-flight checks")

		// First, check if we're root separately from the other preflight checks and fail fast
		if err := preflight.RunRootCheckOnly(); err != nil {
			return nil, err
		}

		// Then continue with the others...
		if err := preflight.RunInitMasterChecks(cfg); err != nil {
			return nil, err
		}
	} else {
		fmt.Println("[preflight] Skipping pre-flight checks")
	}

	// Try to start the kubelet service in case it's inactive
	preflight.TryStartKubelet()

	return &Init{cfg: cfg}, nil
}

type Init struct {
	cfg *kubeadmapi.MasterConfiguration
}

// Validate validates configuration passed to "kubeadm init"
func (i *Init) Validate() error {
	return validation.ValidateMasterConfiguration(i.cfg).ToAggregate()
}

// Run executes master node provisioning, including certificates, needed static pod manifests, etc.
func (i *Init) Run(out io.Writer) error {

	// PHASE 1: Generate certificates
	err := certphase.CreatePKIAssets(i.cfg)
	if err != nil {
		return err
	}

	// PHASE 2: Generate kubeconfig files for the admin and the kubelet

	// TODO this is not great, but there is only one address we can use here
	// so we'll pick the first one, there is much of chance to have an empty
	// slice by the time this gets called
	masterEndpoint := fmt.Sprintf("https://%s:%d", i.cfg.API.AdvertiseAddress, i.cfg.API.BindPort)
	err = kubeconfigphase.CreateInitKubeConfigFiles(masterEndpoint, i.cfg.CertificatesDir, kubeadmapi.GlobalEnvParams.KubernetesDir)
	if err != nil {
		return err
	}

	// PHASE 3: Bootstrap the control plane
	if err := kubemaster.WriteStaticPodManifests(i.cfg); err != nil {
		return err
	}

	adminKubeConfigPath := path.Join(kubeadmapi.GlobalEnvParams.KubernetesDir, kubeadmconstants.AdminKubeConfigFileName)
	client, err := kubemaster.CreateClientAndWaitForAPI(adminKubeConfigPath)
	if err != nil {
		return err
	}

	if err := apiconfigphase.UpdateMasterRoleLabelsAndTaints(client); err != nil {
		return err
	}

	// Is deployment type self-hosted?
	if i.cfg.SelfHosted {
		// Temporary control plane is up, now we create our self hosted control
		// plane components and remove the static manifests:
		fmt.Println("[self-hosted] Creating self-hosted control plane...")
		if err := kubemaster.CreateSelfHostedControlPlane(i.cfg, client); err != nil {
			return err
		}
	}

	// PHASE 4: Set up the bootstrap tokens
	fmt.Printf("[token] Using token: %s\n", i.cfg.Token)

	tokenDescription := "The default bootstrap token generated by 'kubeadm init'."
	if err := tokenphase.UpdateOrCreateToken(client, i.cfg.Token, false, i.cfg.TokenTTL, kubeadmconstants.DefaultTokenUsages, tokenDescription); err != nil {
		return err
	}

	if err := tokenphase.CreateBootstrapConfigMap(adminKubeConfigPath); err != nil {
		return err
	}

	// PHASE 5: Install and deploy all addons, and configure things as necessary

	// Create the necessary ServiceAccounts
	err = apiconfigphase.CreateServiceAccounts(client)
	if err != nil {
		return err
	}

	err = apiconfigphase.CreateRBACRules(client)
	if err != nil {
		return err
	}

	if err := addonsphase.CreateEssentialAddons(i.cfg, client); err != nil {
		return err
	}

	ctx := map[string]string{
		"KubeConfigPath": path.Join(kubeadmapi.GlobalEnvParams.KubernetesDir, kubeadmconstants.AdminKubeConfigFileName),
		"KubeConfigName": kubeadmconstants.AdminKubeConfigFileName,
		"Token":          i.cfg.Token,
		"MasterIP":       i.cfg.API.AdvertiseAddress,
		"MasterPort":     strconv.Itoa(int(i.cfg.API.BindPort)),
	}

	return initDoneTempl.Execute(out, ctx)
}
