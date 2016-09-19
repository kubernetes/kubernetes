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
	"net"

	"github.com/renstrom/dedent"
	"github.com/spf13/cobra"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/api"
	kubemaster "k8s.io/kubernetes/cmd/kubeadm/app/master"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
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
func NewCmdInit(out io.Writer, s *kubeadmapi.KubeadmConfig) *cobra.Command {
	advertiseAddrs := &[]string{}
	cmd := &cobra.Command{
		Use:   "init",
		Short: "Run this on the first machine.",
		Run: func(cmd *cobra.Command, args []string) {
			err := RunInit(out, cmd, args, s, advertiseAddrs)
			cmdutil.CheckErr(err) // TODO(phase1+) append alpha warning with bugs URL etc
		},
	}

	cmd.PersistentFlags().StringVar(
		&s.Secrets.GivenToken, "token", "",
		`(optional) Shared secret used to secure cluster bootstrap. If none is provided, one will be generated for you.`,
	)
	cmd.PersistentFlags().StringSliceVar(
		advertiseAddrs, "api-advertise-addresses", []string{},
		`(optional) The IP addresses to advertise, in case autodetection fails.`,
	)
	cmd.PersistentFlags().StringSliceVar(
		&s.InitFlags.API.ExternalDNSNames, "api-external-dns-names", []string{},
		`(optional) The DNS names to advertise, in case you have configured them yourself.`,
	)

	cmd.PersistentFlags().IPNetVar(
		&s.InitFlags.Services.CIDR, "service-cidr", *kubeadmapi.DefaultServicesCIDR,
		`(optional) use alternative range of IP address for service VIPs, e.g. "10.16.0.0/12"`,
	)
	cmd.PersistentFlags().StringVar(
		&s.InitFlags.Services.DNSDomain, "service-dns-domain", kubeadmapi.DefaultServiceDNSDomain,
		`(optional) use alternative domain for services, e.g. "myorg.internal"`,
	)
	cmd.PersistentFlags().StringVar(
		&s.InitFlags.CloudProvider, "cloud-provider", "",
		`(optional) enable a specific cloud provider features (external load-balancers, storage, etc), e.g. "gce"`,
	)
	cmd.PersistentFlags().BoolVar(
		&s.InitFlags.Schedulable, "schedule-workload", false,
		`(optional) allow to schedule workload to the node`,
	)

	return cmd
}

// RunInit executes master node provisioning, including certificates, needed static pod manifests, etc.
func RunInit(out io.Writer, cmd *cobra.Command, args []string, s *kubeadmapi.KubeadmConfig, advertiseAddrs *[]string) error {
	// Auto-detect the IP
	if len(*advertiseAddrs) == 0 {
		// TODO(phase1+) perhaps we could actually grab eth0 and eth1
		ip, err := netutil.ChooseHostInterface()
		if err != nil {
			return err
		}
		s.InitFlags.API.AdvertiseAddrs = []net.IP{ip}
	} else {
		for _, i := range *advertiseAddrs {
			addr := net.ParseIP(i)
			if addr == nil {
				return fmt.Errorf("<cmd/init> failed to parse flag (%q) as an IP address", "--api-advertise-addresses="+i)
			}
			s.InitFlags.API.AdvertiseAddrs = append(s.InitFlags.API.AdvertiseAddrs, addr)
		}
	}

	if s.InitFlags.CloudProvider != "" {
		// TODO(phase2) we should be able to auto-detect it and check whether things like IAM roles are correct
		if _, ok := kubeadmapi.SupportedCloudProviders[s.InitFlags.CloudProvider]; !ok {
			return fmt.Errorf("<cmd/init> cloud provider %q is not supported, you can use any of %v, or leave it unset", s.InitFlags.CloudProvider, kubeadmapi.ListOfCloudProviders)
		}
	}

	if err := kubemaster.CreateTokenAuthFile(s); err != nil {
		return err
	}

	if err := kubemaster.WriteStaticPodManifests(s); err != nil {
		return err
	}

	caKey, caCert, err := kubemaster.CreatePKIAssets(s)
	if err != nil {
		return err
	}

	kubeconfigs, err := kubemaster.CreateCertsAndConfigForClients(s, []string{"kubelet", "admin"}, caKey, caCert)
	if err != nil {
		return err
	}
	for name, kubeconfig := range kubeconfigs {
		if err := kubeadmutil.WriteKubeconfigIfNotExists(s, name, kubeconfig); err != nil {
			return err
		}
	}

	client, err := kubemaster.CreateClientAndWaitForAPI(kubeconfigs["admin"])
	if err != nil {
		return err
	}

	if err := kubemaster.UpdateMasterRoleLabelsAndTaints(client, s.Schedulable); err != nil {
		return err
	}

	if err := kubemaster.CreateDiscoveryDeploymentAndSecret(s, client, caCert); err != nil {
		return err
	}

	if err := kubemaster.CreateEssentialAddons(s, client); err != nil {
		return err
	}

	// TODO(phase1+) use templates to reference struct fields directly as order of args is fragile
	fmt.Fprintf(out, initDoneMsgf,
		s.Secrets.GivenToken,
		s.InitFlags.API.AdvertiseAddrs[0].String(),
	)

	return nil
}
