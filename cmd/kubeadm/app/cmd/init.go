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
	"html/template"
	"io"
	"io/ioutil"

	"github.com/renstrom/dedent"
	"github.com/spf13/cobra"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiext "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha1"
	kubemaster "k8s.io/kubernetes/cmd/kubeadm/app/master"
	"k8s.io/kubernetes/cmd/kubeadm/app/preflight"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/cloudprovider"
	_ "k8s.io/kubernetes/pkg/cloudprovider/providers"
	"k8s.io/kubernetes/pkg/runtime"
	netutil "k8s.io/kubernetes/pkg/util/net"
)

const (
	joinArgsTemplateLiteral = `--token={{.Cfg.Secrets.GivenToken -}}
		{{if ne .Cfg.API.BindPort .DefaultAPIBindPort -}}
		{{" --api-port="}}{{.Cfg.API.BindPort -}}
		{{end -}}
		{{if ne .Cfg.Discovery.BindPort .DefaultDiscoveryBindPort -}}
		{{" --discovery-port="}}{{.Cfg.Discovery.BindPort -}}
		{{end -}}
		{{" "}}{{index .Cfg.API.AdvertiseAddresses 0 -}}
`
)

var (
	initDoneMsgf = dedent.Dedent(`
		Kubernetes master initialised successfully!

		You can now join any number of machines by running the following on each node:

		kubeadm join %s
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
			kubeadmutil.CheckErr(i.Run(out))
		},
	}

	cmd.PersistentFlags().StringVar(
		&cfg.Secrets.GivenToken, "token", cfg.Secrets.GivenToken,
		"Shared secret used to secure cluster bootstrap; if none is provided, one will be generated for you",
	)
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
	cmd.PersistentFlags().StringVar(
		&cfg.CloudProvider, "cloud-provider", cfg.CloudProvider,
		`Enable cloud provider features (external load-balancers, storage, etc), e.g. "gce"`,
	)

	cmd.PersistentFlags().StringVar(
		&cfg.KubernetesVersion, "use-kubernetes-version", cfg.KubernetesVersion,
		`Choose a specific Kubernetes version for the control plane`,
	)

	cmd.PersistentFlags().StringVar(&cfgPath, "config", cfgPath, "Path to kubeadm config file")

	// TODO (phase1+) @errordeveloper make the flags below not show up in --help but rather on --advanced-help
	cmd.PersistentFlags().StringSliceVar(
		&cfg.Etcd.Endpoints, "external-etcd-endpoints", cfg.Etcd.Endpoints,
		"etcd endpoints to use, in case you have an external cluster",
	)
	cmd.PersistentFlags().MarkDeprecated("external-etcd-endpoints", "this flag will be removed when componentconfig exists")

	cmd.PersistentFlags().StringVar(
		&cfg.Etcd.CAFile, "external-etcd-cafile", cfg.Etcd.CAFile,
		"etcd certificate authority certificate file. Note: The path must be in /etc/ssl/certs",
	)
	cmd.PersistentFlags().MarkDeprecated("external-etcd-cafile", "this flag will be removed when componentconfig exists")

	cmd.PersistentFlags().StringVar(
		&cfg.Etcd.CertFile, "external-etcd-certfile", cfg.Etcd.CertFile,
		"etcd client certificate file. Note: The path must be in /etc/ssl/certs",
	)
	cmd.PersistentFlags().MarkDeprecated("external-etcd-certfile", "this flag will be removed when componentconfig exists")

	cmd.PersistentFlags().StringVar(
		&cfg.Etcd.KeyFile, "external-etcd-keyfile", cfg.Etcd.KeyFile,
		"etcd client key file. Note: The path must be in /etc/ssl/certs",
	)
	cmd.PersistentFlags().MarkDeprecated("external-etcd-keyfile", "this flag will be removed when componentconfig exists")

	cmd.PersistentFlags().BoolVar(
		&skipPreFlight, "skip-preflight-checks", skipPreFlight,
		"skip preflight checks normally run before modifying the system",
	)

	cmd.PersistentFlags().Int32Var(
		&cfg.API.BindPort, "api-port", cfg.API.BindPort,
		"Port for API to bind to",
	)

	cmd.PersistentFlags().Int32Var(
		&cfg.Discovery.BindPort, "discovery-port", cfg.Discovery.BindPort,
		"Port for JWS discovery service to bind to",
	)

	return cmd
}

type Init struct {
	cfg *kubeadmapi.MasterConfiguration
}

func NewInit(cfgPath string, cfg *kubeadmapi.MasterConfiguration, skipPreFlight bool) (*Init, error) {
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
		// TODO(phase1+) perhaps we could actually grab eth0 and eth1
		ip, err := netutil.ChooseHostInterface()
		if err != nil {
			return nil, err
		}
		cfg.API.AdvertiseAddresses = []string{ip.String()}
	}

	if !skipPreFlight {
		fmt.Println("Running pre-flight checks")
		err := preflight.RunInitMasterChecks(cfg)
		if err != nil {
			return nil, &preflight.PreFlightError{Msg: err.Error()}
		}
	} else {
		fmt.Println("Skipping pre-flight checks")
	}

	// TODO(phase1+) create a custom flag
	if cfg.CloudProvider != "" {
		if cloudprovider.IsCloudProvider(cfg.CloudProvider) {
			fmt.Printf("cloud provider %q initialized for the control plane. Remember to set the same cloud provider flag on the kubelet.\n", cfg.CloudProvider)
		} else {
			return nil, fmt.Errorf("cloud provider %q is not supported, you can use any of %v, or leave it unset.\n", cfg.CloudProvider, cloudprovider.CloudProviders())
		}
	}
	return &Init{cfg: cfg}, nil
}

// joinArgsData denotes a data object which is needed by function generateJoinArgs to generate kubeadm join arguments.
type joinArgsData struct {
	Cfg                      *kubeadmapi.MasterConfiguration
	DefaultAPIBindPort       int32
	DefaultDiscoveryBindPort int32
}

// Run executes master node provisioning, including certificates, needed static pod manifests, etc.
func (i *Init) Run(out io.Writer) error {
	if err := kubemaster.CreateTokenAuthFile(&i.cfg.Secrets); err != nil {
		return err
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
	// TODO(phase1+) this is no longer the right place to guard agains foo-shooting,
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

	schedulePodsOnMaster := false
	if err := kubemaster.UpdateMasterRoleLabelsAndTaints(client, schedulePodsOnMaster); err != nil {
		return err
	}

	if err := kubemaster.CreateDiscoveryDeploymentAndSecret(i.cfg, client, caCert); err != nil {
		return err
	}

	if err := kubemaster.CreateEssentialAddons(i.cfg, client); err != nil {
		return err
	}

	data := joinArgsData{i.cfg, kubeadmapiext.DefaultAPIBindPort, kubeadmapiext.DefaultDiscoveryBindPort}
	if joinArgs, err := generateJoinArgs(data); err != nil {
		return err
	} else {
		fmt.Fprintf(out, initDoneMsgf, joinArgs)
	}
	return nil
}

// generateJoinArgs generates kubeadm join arguments
func generateJoinArgs(data joinArgsData) (string, error) {
	joinArgsTemplate := template.Must(template.New("joinArgsTemplate").Parse(joinArgsTemplateLiteral))
	var b bytes.Buffer
	if err := joinArgsTemplate.Execute(&b, data); err != nil {
		return "", err
	}
	return b.String(), nil
}
