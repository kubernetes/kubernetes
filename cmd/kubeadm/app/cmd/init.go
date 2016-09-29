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

	"github.com/renstrom/dedent"
	"github.com/spf13/cobra"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/api"
	kubemaster "k8s.io/kubernetes/cmd/kubeadm/app/master"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	"k8s.io/kubernetes/pkg/cloudprovider"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	netutil "k8s.io/kubernetes/pkg/util/net"
)

var (
	initDoneMsgf = dedent.Dedent(`
		Kubernetes master initialised successfully!

		You can now join any number of machines by running the following on each node:

		kubeadm join --token %s %s
		`)
)

// NewCmdInit returns "kubeadm init" command.
func NewCmdInit(out io.Writer) *cobra.Command {
	cfg := &kubeadmapi.MasterConfiguration{}
	cmd := &cobra.Command{
		Use:   "init",
		Short: "Run this in order to set up the Kubernetes master.",
		Run: func(cmd *cobra.Command, args []string) {
			err := RunInit(out, cmd, args, cfg)
			cmdutil.CheckErr(err)
		},
	}

	cmd.PersistentFlags().StringVar(
		&cfg.Secrets.GivenToken, "token", "",
		"Shared secret used to secure cluster bootstrap; if none is provided, one will be generated for you",
	)
	cmd.PersistentFlags().StringSliceVar(
		&cfg.API.AdvertiseAddresses, "api-advertise-addresses", []string{},
		"The IP addresses to advertise, in case autodetection fails",
	)
	cmd.PersistentFlags().StringSliceVar(
		&cfg.API.ExternalDNSNames, "api-external-dns-names", []string{},
		"The DNS names to advertise, in case you have configured them yourself",
	)
	cmd.PersistentFlags().StringVar(
		&cfg.Networking.ServiceSubnet, "service-cidr", kubeadmapi.DefaultServicesSubnet,
		"Use alterantive range of IP address for service VIPs",
	)
	cmd.PersistentFlags().StringVar(
		&cfg.Networking.PodSubnet, "pod-network-cidr", "",
		"Specify range of IP addresses for the pod network; if set, the control plane will automatically allocate CIDRs for every node",
	)
	cmd.PersistentFlags().StringVar(
		&cfg.Networking.DNSDomain, "service-dns-domain", kubeadmapi.DefaultServiceDNSDomain,
		`Use alternative domain for services, e.g. "myorg.internal"`,
	)
	cmd.PersistentFlags().StringVar(
		&cfg.CloudProvider, "cloud-provider", "",
		`Enable cloud provider features (external load-balancers, storage, etc), e.g. "gce"`,
	)

	cmd.PersistentFlags().StringVar(
		&cfg.KubernetesVersion, "use-kubernetes-version", kubeadmapi.DefaultKubernetesVersion,
		`Choose a specific Kubernetes version for the control plane`,
	)

	// TODO (phase1+) @errordeveloper make the flags below not show up in --help but rather on --advanced-help
	cmd.PersistentFlags().StringSliceVar(
		&cfg.Etcd.Endpoints, "external-etcd-endpoints", []string{},
		"etcd endpoints to use, in case you have an external cluster",
	)
	cmd.PersistentFlags().StringVar(
		&cfg.Etcd.CAFile, "external-etcd-cafile", "",
		"etcd certificate authority certificate file. Note: The path must be in /etc/ssl/certs",
	)
	cmd.PersistentFlags().StringVar(
		&cfg.Etcd.CertFile, "external-etcd-certfile", "",
		"etcd client certificate file. Note: The path must be in /etc/ssl/certs",
	)
	cmd.PersistentFlags().StringVar(
		&cfg.Etcd.KeyFile, "external-etcd-keyfile", "",
		"etcd client key file. Note: The path must be in /etc/ssl/certs",
	)

	return cmd
}

// RunInit executes master node provisioning, including certificates, needed static pod manifests, etc.
func RunInit(out io.Writer, cmd *cobra.Command, args []string, cfg *kubeadmapi.MasterConfiguration) error {
	// Auto-detect the IP
	if len(cfg.API.AdvertiseAddresses) == 0 {
		// TODO(phase1+) perhaps we could actually grab eth0 and eth1
		ip, err := netutil.ChooseHostInterface()
		if err != nil {
			return err
		}
		cfg.API.AdvertiseAddresses = []string{ip.String()}
	}

	// TODO(phase1+) create a custom flag
	if cfg.CloudProvider != "" {
		if cloudprovider.IsCloudProvider(cfg.CloudProvider) {
			fmt.Printf("<cmd/init> cloud provider %q initialized for the control plane. Remember to set the same cloud provider flag on the kubelet.\n", cfg.CloudProvider)
		} else {
			return fmt.Errorf("<cmd/init> cloud provider %q is not supported, you can use any of %v, or leave it unset.\n", cfg.CloudProvider, cloudprovider.CloudProviders())
		}
	}

	if err := kubemaster.CreateTokenAuthFile(&cfg.Secrets); err != nil {
		return err
	}

	if err := kubemaster.WriteStaticPodManifests(cfg); err != nil {
		return err
	}

	caKey, caCert, err := kubemaster.CreatePKIAssets(cfg)
	if err != nil {
		return err
	}

	kubeconfigs, err := kubemaster.CreateCertsAndConfigForClients(cfg.API.AdvertiseAddresses, []string{"kubelet", "admin"}, caKey, caCert)
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

	if err := kubemaster.CreateDiscoveryDeploymentAndSecret(cfg, client, caCert); err != nil {
		return err
	}

	if err := kubemaster.CreateEssentialAddons(cfg, client); err != nil {
		return err
	}

	// TODO(phase1+) use templates to reference struct fields directly as order of args is fragile
	fmt.Fprintf(out, initDoneMsgf,
		cfg.Secrets.GivenToken,
		cfg.API.AdvertiseAddresses[0],
	)

	return nil
}
