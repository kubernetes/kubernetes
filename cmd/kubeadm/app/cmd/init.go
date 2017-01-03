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

	"github.com/renstrom/dedent"
	"github.com/spf13/cobra"
	"github.com/spf13/pflag"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiext "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha1"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/validation"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/flags"
	"k8s.io/kubernetes/cmd/kubeadm/app/discovery"
	kubemaster "k8s.io/kubernetes/cmd/kubeadm/app/master"
	"k8s.io/kubernetes/cmd/kubeadm/app/preflight"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/runtime"
	netutil "k8s.io/kubernetes/pkg/util/net"
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
	// advancedHelpOnlyFlags records all flags that will only be displayed in advanced help.
	advancedHelpOnlyFlags = []string{"external-etcd-endpoints", "external-etcd-cafile", "external-etcd-certfile", "external-etcd-keyfile"}
)

// NewCmdInit returns "kubeadm init" command.
func NewCmdInit(out io.Writer) *cobra.Command {
	versioned := &kubeadmapiext.MasterConfiguration{}
	api.Scheme.Default(versioned)
	cfg := kubeadmapi.MasterConfiguration{}
	api.Scheme.Convert(versioned, &cfg, nil)

	var cfgPath string
	var skipPreFlight bool
	// advancedHelp indicates whether user call kubeadm init command with --advanced-help flag or not.
	var advancedHelp = false
	var deprecatedFlagWarningSuffix = ", Warning: this flag is deprecated and will be removed when componentconfig exists"
	cmd := &cobra.Command{
		Use:   "init",
		Short: "Run this in order to set up the Kubernetes master",
		Run: func(cmd *cobra.Command, args []string) {
			if advancedHelp {
				markAdvancedHelpOnlyFlagsDisplayed(cmd.Flags())
				cmd.Help()
				return
			}
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

	cmd.PersistentFlags().StringSliceVar(
		&cfg.Etcd.Endpoints, "external-etcd-endpoints", cfg.Etcd.Endpoints,
		"etcd endpoints to use, in case you have an external cluster"+
			deprecatedFlagWarningSuffix,
	)
	cmd.PersistentFlags().MarkHidden("external-etcd-endpoints")

	cmd.PersistentFlags().StringVar(
		&cfg.Etcd.CAFile, "external-etcd-cafile", cfg.Etcd.CAFile,
		"etcd certificate authority certificate file. Note: The path must be in /etc/ssl/certs"+
			deprecatedFlagWarningSuffix,
	)
	cmd.PersistentFlags().MarkHidden("external-etcd-cafile")

	cmd.PersistentFlags().StringVar(
		&cfg.Etcd.CertFile, "external-etcd-certfile", cfg.Etcd.CertFile,
		"etcd client certificate file. Note: The path must be in /etc/ssl/certs"+
			deprecatedFlagWarningSuffix,
	)
	cmd.PersistentFlags().MarkHidden("external-etcd-certfile")

	cmd.PersistentFlags().StringVar(
		&cfg.Etcd.KeyFile, "external-etcd-keyfile", cfg.Etcd.KeyFile,
		"etcd client key file. Note: The path must be in /etc/ssl/certs"+
			deprecatedFlagWarningSuffix,
	)
	cmd.PersistentFlags().MarkHidden("external-etcd-keyfile")

	cmd.PersistentFlags().BoolVar(
		&skipPreFlight, "skip-preflight-checks", skipPreFlight,
		"skip preflight checks normally run before modifying the system",
	)

	cmd.PersistentFlags().Var(
		discovery.NewDiscoveryValue(&cfg.Discovery), "discovery",
		"The discovery method kubeadm will use for connecting nodes to the master",
	)
	cmd.PersistentFlags().BoolVar(
		&advancedHelp, "advanced-help", advancedHelp,
		"more flags will be displayed in advanced help informations")

	return cmd
}

// markDisplayed sets a flag to be displayed in your program.this function is the opposite of pflag.FlagSet#MarkHidden().
func markDisplayed(f *pflag.FlagSet, name string) error {
	flag := f.Lookup(name)
	if flag == nil {
		return fmt.Errorf("flag %q does not exist", name)
	}
	flag.Hidden = false
	return nil
}

// markAdvancedHelpOnlyFlagsDisplayed set all advanced-help-only flags to be displayed.
func markAdvancedHelpOnlyFlagsDisplayed(fs *pflag.FlagSet) {
	for _, flag := range advancedHelpOnlyFlags {
		markDisplayed(fs, flag)
	}
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

	if i.cfg.Discovery.Token != nil {
		if err := kubemaster.PrepareTokenDiscovery(i.cfg.Discovery.Token); err != nil {
			return err
		}
		if err := kubemaster.CreateTokenAuthFile(kubeadmutil.BearerToken(i.cfg.Discovery.Token)); err != nil {
			return err
		}
	}

	if err := kubemaster.WriteStaticPodManifests(i.cfg); err != nil {
		return err
	}

	caKey, caCert, err := kubemaster.CreatePKIAssets(i.cfg)
	if err != nil {
		return err
	}

	kubeconfigs, err := kubemaster.CreateCertsAndConfigForClients(i.cfg.API, []string{"kubelet", "admin"}, caKey, caCert)
	if err != nil {
		return err
	}

	// kubeadm is responsible for writing the following kubeconfig file, which
	// kubelet should be waiting for. Help user avoid foot-shooting by refusing to
	// write a file that has already been written (the kubelet will be up and
	// running in that case - they'd need to stop the kubelet, remove the file, and
	// start it again in that case).
	// TODO(phase1+) this is no longer the right place to guard against foo-shooting,
	// we need to decide how to handle existing files (it may be handy to support
	// importing existing files, may be we could even make our command idempotant,
	// or at least allow for external PKI and stuff)
	for name, kubeconfig := range kubeconfigs {
		if err := kubeadmutil.WriteKubeconfigIfNotExists(name, kubeconfig); err != nil {
			return err
		}
	}

	client, err := kubemaster.CreateClientAndWaitForAPI(kubeconfigs["admin"])
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
