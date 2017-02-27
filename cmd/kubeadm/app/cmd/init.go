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

	"github.com/renstrom/dedent"
	"github.com/spf13/cobra"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/pkg/api"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiext "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha1"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/validation"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/discovery"
	kubemaster "k8s.io/kubernetes/cmd/kubeadm/app/master"
	addonsphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/addons"
	apiconfigphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/apiconfig"
	certphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/certs"
	kubeconfigphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/kubeconfig"
	tokenphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/token"
	"k8s.io/kubernetes/cmd/kubeadm/app/preflight"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	tokenutil "k8s.io/kubernetes/cmd/kubeadm/app/util/token"
)

var (
	initDoneMsgf = dedent.Dedent(`
		Your Kubernetes master has initialized successfully!

		To start using your cluster, you need to run:
		export KUBECONFIG=%s

		You should now deploy a pod network to the cluster.
		Run "kubectl apply -f [podnetwork].yaml" with one of the options listed at:
		    http://kubernetes.io/docs/admin/addons/

		You can now join any number of machines by running the following on each node:

		kubeadm join --discovery %s
		`)
)

// NewCmdInit returns "kubeadm init" command.
func NewCmdInit(out io.Writer) *cobra.Command {
	versioned := &kubeadmapiext.MasterConfiguration{}
	api.Scheme.Default(versioned)
	cfg := kubeadmapi.MasterConfiguration{}
	api.Scheme.Convert(versioned, &cfg, nil)

	var cfgPath string
	var skipPreFlight bool
	cmd := &cobra.Command{
		Use:   "init",
		Short: "Run this in order to set up the Kubernetes master",
		Run: func(cmd *cobra.Command, args []string) {
			i, err := NewInit(cfgPath, &cfg, skipPreFlight)
			kubeadmutil.CheckErr(err)
			kubeadmutil.CheckErr(i.Validate())
			kubeadmutil.CheckErr(i.Run(out))
		},
	}

	cmd.PersistentFlags().StringSliceVar(
		&cfg.API.AdvertiseAddresses, "api-advertise-addresses", cfg.API.AdvertiseAddresses,
		"The IP addresses to advertise, in case autodetection fails",
	)
	cmd.PersistentFlags().Int32Var(
		&cfg.API.Port, "api-port", cfg.API.Port,
		"Port for API to bind to",
	)
	cmd.PersistentFlags().StringSliceVar(
		&cfg.API.ExternalDNSNames, "api-external-dns-names", cfg.API.ExternalDNSNames,
		"The DNS names to advertise, in case you have configured them yourself",
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

	cmd.PersistentFlags().StringVar(&cfgPath, "config", cfgPath, "Path to kubeadm config file")

	cmd.PersistentFlags().BoolVar(
		&skipPreFlight, "skip-preflight-checks", skipPreFlight,
		"Skip preflight checks normally run before modifying the system",
	)

	cmd.PersistentFlags().Var(
		discovery.NewDiscoveryValue(&cfg.Discovery), "discovery",
		"The discovery method kubeadm will use for connecting nodes to the master",
	)

	return cmd
}

func NewInit(cfgPath string, cfg *kubeadmapi.MasterConfiguration, skipPreFlight bool) (*Init, error) {

	fmt.Println("[kubeadm] WARNING: kubeadm is in alpha, please do not use it for production clusters.")

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
	err := certphase.CreatePKIAssets(i.cfg, kubeadmapi.GlobalEnvParams.HostPKIPath)
	if err != nil {
		return err
	}

	// PHASE 2: Generate kubeconfig files for the admin and the kubelet

	// TODO this is not great, but there is only one address we can use here
	// so we'll pick the first one, there is much of chance to have an empty
	// slice by the time this gets called
	masterEndpoint := fmt.Sprintf("https://%s:%d", i.cfg.API.AdvertiseAddresses[0], i.cfg.API.Port)
	err = kubeconfigphase.CreateInitKubeConfigFiles(masterEndpoint, kubeadmapi.GlobalEnvParams.HostPKIPath, kubeadmapi.GlobalEnvParams.KubernetesDir)
	if err != nil {
		return err
	}

	// TODO: It's not great to have an exception for token here, but necessary because the apiserver doesn't handle this properly in the API yet
	// but relies on files on disk for now, which is daunting.
	if i.cfg.Discovery.Token != nil {
		if err := tokenphase.CreateTokenAuthFile(tokenutil.BearerToken(i.cfg.Discovery.Token)); err != nil {
			return err
		}
	}

	// PHASE 3: Bootstrap the control plane
	if err := kubemaster.WriteStaticPodManifests(i.cfg); err != nil {
		return err
	}

	client, err := kubemaster.CreateClientAndWaitForAPI(path.Join(kubeadmapi.GlobalEnvParams.KubernetesDir, kubeadmconstants.AdminKubeConfigFileName))
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
		fmt.Println("[init] Creating self-hosted control plane...")
		if err := kubemaster.CreateSelfHostedControlPlane(i.cfg, client); err != nil {
			return err
		}
	}

	// PHASE 4: Set up the bootstrap tokens
	if i.cfg.Discovery.Token != nil {
		fmt.Printf("[token-discovery] Using token: %s\n", tokenutil.BearerToken(i.cfg.Discovery.Token))
		if err := kubemaster.CreateDiscoveryDeploymentAndSecret(i.cfg, client); err != nil {
			return err
		}
		tokenDescription := "The default bootstrap token generated by 'kubeadm init'."
		if err := tokenphase.UpdateOrCreateToken(client, i.cfg.Discovery.Token, false, kubeadmconstants.DefaultTokenDuration, kubeadmconstants.DefaultTokenUsages, tokenDescription); err != nil {
			return err
		}
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

	fmt.Fprintf(out, initDoneMsgf, path.Join(kubeadmapi.GlobalEnvParams.KubernetesDir, kubeadmconstants.AdminKubeConfigFileName), generateJoinArgs(i.cfg))
	return nil
}

// generateJoinArgs generates kubeadm join arguments
func generateJoinArgs(cfg *kubeadmapi.MasterConfiguration) string {
	return discovery.NewDiscoveryValue(&cfg.Discovery).String()
}
