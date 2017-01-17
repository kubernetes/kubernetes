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

	"github.com/renstrom/dedent"
	"github.com/spf13/cobra"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiext "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha1"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/validation"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/flags"
	"k8s.io/kubernetes/cmd/kubeadm/app/discovery"
	kubemaster "k8s.io/kubernetes/cmd/kubeadm/app/master"

	certphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/certs"
	kubeconfigphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/kubeconfig"

	"k8s.io/apimachinery/pkg/runtime"
	netutil "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/kubernetes/cmd/kubeadm/app/preflight"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	"k8s.io/kubernetes/pkg/api"
)

var (
	initDoneMsgf = dedent.Dedent(`
		Your Kubernetes master has initialized successfully!

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
	cmd.PersistentFlags().Var(
		flags.NewCloudProviderFlag(&cfg.CloudProvider), "cloud-provider",
		`Enable cloud provider features (external load-balancers, storage, etc). Note that you have to configure all kubelets manually`,
	)

	cmd.PersistentFlags().StringVar(
		&cfg.KubernetesVersion, "use-kubernetes-version", cfg.KubernetesVersion,
		`Choose a specific Kubernetes version for the control plane`,
	)

	cmd.PersistentFlags().StringVar(&cfgPath, "config", cfgPath, "Path to kubeadm config file")

	cmd.PersistentFlags().BoolVar(
		&skipPreFlight, "skip-preflight-checks", skipPreFlight,
		"skip preflight checks normally run before modifying the system",
	)

	cmd.PersistentFlags().Var(
		discovery.NewDiscoveryValue(&cfg.Discovery), "discovery",
		"The discovery method kubeadm will use for connecting nodes to the master",
	)

	return cmd
}

type Init struct {
	cfg *kubeadmapi.MasterConfiguration
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

	// Auto-detect the IP
	if len(cfg.API.AdvertiseAddresses) == 0 {
		ip, err := netutil.ChooseHostInterface()
		if err != nil {
			return nil, err
		}
		cfg.API.AdvertiseAddresses = []string{ip.String()}
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

	// validate version argument
	ver, err := kubeadmutil.KubernetesReleaseVersion(cfg.KubernetesVersion)
	if err != nil {
		if cfg.KubernetesVersion != kubeadmapiext.DefaultKubernetesVersion {
			return nil, err
		} else {
			ver = kubeadmapiext.DefaultKubernetesFallbackVersion
		}
	}
	cfg.KubernetesVersion = ver
	fmt.Println("[init] Using Kubernetes version:", ver)

	// Warn about the limitations with the current cloudprovider solution.
	if cfg.CloudProvider != "" {
		fmt.Println("WARNING: For cloudprovider integrations to work --cloud-provider must be set for all kubelets in the cluster.")
		fmt.Println("\t(/etc/systemd/system/kubelet.service.d/10-kubeadm.conf should be edited for this purpose)")
	}

	return &Init{cfg: cfg}, nil
}

func (i *Init) Validate() error {
	return validation.ValidateMasterConfiguration(i.cfg).ToAggregate()
}

// Run executes master node provisioning, including certificates, needed static pod manifests, etc.
func (i *Init) Run(out io.Writer) error {

	// PHASE 1: Generate certificates
	caCert, err := certphase.CreatePKIAssets(i.cfg, kubeadmapi.GlobalEnvParams.HostPKIPath)
	if err != nil {
		return err
	}

	// Exception:
	if i.cfg.Discovery.Token != nil {
		// Validate token
		if valid, err := kubeadmutil.ValidateToken(i.cfg.Discovery.Token); valid == false {
			return err
		}

		// Make sure there is at least one address
		if len(i.cfg.Discovery.Token.Addresses) == 0 {
			ip, err := netutil.ChooseHostInterface()
			if err != nil {
				return err
			}
			i.cfg.Discovery.Token.Addresses = []string{ip.String() + ":" + strconv.Itoa(kubeadmapiext.DefaultDiscoveryBindPort)}
		}

		if err := kubemaster.CreateTokenAuthFile(kubeadmutil.BearerToken(i.cfg.Discovery.Token)); err != nil {
			return err
		}
	}

	// PHASE 2: Generate kubeconfig files for the admin and the kubelet

	// TODO this is not great, but there is only one address we can use here
	// so we'll pick the first one, there is much of chance to have an empty
	// slice by the time this gets called
	masterEndpoint := fmt.Sprintf("https://%s:%d", i.cfg.API.AdvertiseAddresses[0], i.cfg.API.Port)
	err = kubeconfigphase.CreateAdminAndKubeletKubeConfig(masterEndpoint, kubeadmapi.GlobalEnvParams.HostPKIPath, kubeadmapi.GlobalEnvParams.KubernetesDir)
	if err != nil {
		return err
	}

	// Phase 3: Bootstrap the control plane
	if err := kubemaster.WriteStaticPodManifests(i.cfg); err != nil {
		return err
	}

	client, err := kubemaster.CreateClientAndWaitForAPI(path.Join(kubeadmapi.GlobalEnvParams.KubernetesDir, kubeconfigphase.AdminKubeConfigFileName))
	if err != nil {
		return err
	}

	if err := kubemaster.UpdateMasterRoleLabelsAndTaints(client, false); err != nil {
		return err
	}

	if i.cfg.Discovery.Token != nil {
		fmt.Printf("[token-discovery] Using token: %s\n", kubeadmutil.BearerToken(i.cfg.Discovery.Token))
		if err := kubemaster.CreateDiscoveryDeploymentAndSecret(i.cfg, client, caCert); err != nil {
			return err
		}
		if err := kubeadmutil.UpdateOrCreateToken(client, i.cfg.Discovery.Token, kubeadmutil.DefaultTokenDuration); err != nil {
			return err
		}
	}

	if err := kubemaster.CreateEssentialAddons(i.cfg, client); err != nil {
		return err
	}

	fmt.Fprintf(out, initDoneMsgf, generateJoinArgs(i.cfg))
	return nil
}

// generateJoinArgs generates kubeadm join arguments
func generateJoinArgs(cfg *kubeadmapi.MasterConfiguration) string {
	return discovery.NewDiscoveryValue(&cfg.Discovery).String()
}
