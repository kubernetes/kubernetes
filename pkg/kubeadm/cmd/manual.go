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

	kubeadmapi "k8s.io/kubernetes/pkg/kubeadm/api"
	kubemaster "k8s.io/kubernetes/pkg/kubeadm/master"
	kubenode "k8s.io/kubernetes/pkg/kubeadm/node"
	kubeadmutil "k8s.io/kubernetes/pkg/kubeadm/util"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	netutil "k8s.io/kubernetes/pkg/util/net"
)

var (
	manual_init_done_msgf = dedent.Dedent(`
		Master initialization complete:

		* Static pods written and kubelet's kubeconfig written.
		* Kubelet should start soon.  Try 'systemctl restart kubelet'
		  or equivalent if it doesn't.

		CA cert is written to:
		    /etc/kubernetes/pki/ca.pem.

		**Please copy this file (scp, rsync or through other means) to
		all your nodes and then run on them**:

		kubeadm manual bootstrap join-node --ca-cert-file <path-to-ca-cert> \
		    --token %s --api-server-urls https://%s:443/
		`)
	manual_join_done_msgf = dedent.Dedent(`
		Node join complete:
		* Certificate signing request sent to master and response
		  received.
		* Kubelet informed of new secure connection details.

		Run 'kubectl get nodes' on the master to see this node join.
		`)
)

// TODO --token here becomes `s.Secrets.BearerToken` and not `s.Secrets.GivenToken`
// may be we should make it the same and ask user to pass dot-separated tokens
// in any of the modes; we could also enable discovery API in the manual mode just
// as well, there is no reason we shouldn't let user mix and match modes, unless
// it is too difficult to support

func NewCmdManual(out io.Writer, s *kubeadmapi.KubeadmConfig) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "manual",
		Short: "Advanced, less-automated functionality, for power users.",
		// TODO put example usage in the Long description here
	}
	cmd.AddCommand(NewCmdManualBootstrap(out, s))
	return cmd
}

func NewCmdManualBootstrap(out io.Writer, s *kubeadmapi.KubeadmConfig) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "bootstrap",
		Short: "Manually bootstrap a cluster 'out-of-band'",
		Long: dedent.Dedent(`
			Manually bootstrap a cluster 'out-of-band', by generating and distributing a CA
			certificate to all your servers and specifying and (list of) API server URLs.
		`),
		Run: func(cmd *cobra.Command, args []string) {
		},
	}
	cmd.AddCommand(NewCmdManualBootstrapInitMaster(out, s))
	cmd.AddCommand(NewCmdManualBootstrapJoinNode(out, s))

	return cmd
}

func NewCmdManualBootstrapInitMaster(out io.Writer, s *kubeadmapi.KubeadmConfig) *cobra.Command {
	advertiseAddrs := &[]string{}
	cmd := &cobra.Command{
		Use:   "init-master",
		Short: "Manually bootstrap a master 'out-of-band'",
		Long: dedent.Dedent(`
			Manually bootstrap a master 'out-of-band'.
			Will create TLS certificates and set up static pods for Kubernetes master
			components.
		`),
		Run: func(cmd *cobra.Command, args []string) {
			err := RunManualBootstrapInitMaster(out, cmd, args, s, advertiseAddrs)
			cmdutil.CheckErr(err)
		},
	}

	cmd.PersistentFlags().StringVar(
		&s.Secrets.BearerToken, "token", "",
		`(optional) Shared secret used to secure bootstrap. Will be generated and displayed if not provided.`,
	)
	cmd.PersistentFlags().StringSliceVar(
		advertiseAddrs, "api-advertise-addr", nil,
		`(optional) IP address to advertise, in case autodetection fails.`,
	)
	cmd.PersistentFlags().StringSliceVar(
		&s.InitFlags.API.ExternalDNSName, "api-external-dns-name", []string{},
		`(optional) DNS name to advertise, in case you have configured one yourself.`,
	)
	cmd.PersistentFlags().IPNetVar(
		&s.InitFlags.Services.CIDR, "service-cidr", *kubeadmapi.DefaultServicesCIDR,
		`(optional) use alterantive range of IP address for service VIPs, e.g. "10.16.0.0/12"`,
	)
	cmd.PersistentFlags().StringVar(
		&s.InitFlags.Services.DNSDomain, "service-dns-domain", "cluster.local",
		`(optional) use alterantive domain name for services, e.g. "myorg.internal"`,
	)
	cmd.PersistentFlags().BoolVar(
		&s.InitFlags.Schedulable, "schedule-workload", false,
		`(optional) allow to schedule workload to the node`,
	)

	return cmd
}

func RunManualBootstrapInitMaster(out io.Writer, cmd *cobra.Command, args []string, s *kubeadmapi.KubeadmConfig, advertiseAddrs *[]string) error {
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
				return fmt.Errorf("<cmd/init> failed to parse flag (%q) as an IP address", "--api-advertise-addr="+i)
			}
			s.InitFlags.API.AdvertiseAddrs = append(s.InitFlags.API.AdvertiseAddrs, addr)
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

	// TODO we have most of cmd/init functionality here, except for `CreateDiscoveryDeploymentAndSecret()`
	// it may be a good idea to just merge the two commands into one, and it's something we have started talking
	// about, the only question is where disco service should be an opt-out...

	client, err := kubemaster.CreateClientAndWaitForAPI(kubeconfigs["admin"])
	if err != nil {
		return err
	}

	if err := kubemaster.UpdateMasterRoleLabelsAndTaints(client, s.Schedulable); err != nil {
		return err
	}

	if err := kubemaster.CreateEssentialAddons(s, client); err != nil {
		return err
	}

	// TODO use templates to reference struct fields directly as order of args is fragile
	fmt.Fprintf(out, manual_init_done_msgf,
		s.Secrets.BearerToken,
		s.InitFlags.API.AdvertiseAddrs[0].String(),
	)
	return nil
}

func NewCmdManualBootstrapJoinNode(out io.Writer, s *kubeadmapi.KubeadmConfig) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "join-node",
		Short: "Manually bootstrap a node 'out-of-band', joining it into a cluster with extant control plane",

		Run: func(cmd *cobra.Command, args []string) {
			err := RunManualBootstrapJoinNode(out, cmd, args, s)
			cmdutil.CheckErr(err)
		},
	}
	cmd.PersistentFlags().StringVarP(&s.ManualFlags.CaCertFile, "ca-cert-file", "", "",
		`Path to a CA cert file in PEM format. The same CA cert must be distributed to
		all servers.`)
	cmd.PersistentFlags().StringVarP(&s.ManualFlags.ApiServerURLs, "api-server-urls", "", "",
		`Comma separated list of API server URLs. Typically this might be just
		https://<address-of-master>:8080/`)
	cmd.PersistentFlags().StringVarP(&s.ManualFlags.BearerToken, "token", "", "",
		`Shared secret used to secure bootstrap. Must match output of 'init-master'.`)

	return cmd
}

func RunManualBootstrapJoinNode(out io.Writer, cmd *cobra.Command, args []string, s *kubeadmapi.KubeadmConfig) error {
	if s.ManualFlags.CaCertFile == "" {
		fmt.Fprintf(out, "Must specify --ca-cert-file (see --help)\n")
		return nil
	}

	if s.ManualFlags.ApiServerURLs == "" {
		fmt.Fprintf(out, "Must specify --api-server-urls (see --help)\n")
		return nil
	}

	kubeconfig, err := kubenode.PerformTLSBootstrapFromConfig(s)
	if err != nil {
		fmt.Fprintf(out, "Failed to perform TLS bootstrap: %s\n", err)
		return err
	}

	err = kubeadmutil.WriteKubeconfigIfNotExists(s, "kubelet", kubeconfig)
	if err != nil {
		fmt.Fprintf(out, "Unable to write config for node:\n%s\n", err)
		return err
	}

	fmt.Fprintf(out, manual_join_done_msgf)
	return nil
}
