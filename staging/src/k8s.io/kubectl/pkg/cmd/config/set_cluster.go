/*
Copyright 2014 The Kubernetes Authors.

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

package config

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/spf13/cobra"

	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	cliflag "k8s.io/component-base/cli/flag"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/templates"
)

var (
	setClusterLong = templates.LongDesc(i18n.T(`
		Set a cluster entry in kubeconfig.

		Specifying a name that already exists will merge new fields on top of existing values for those fields.`))

	setClusterExample = templates.Examples(`
		# Set only the server field on the e2e cluster entry without touching other values
		kubectl config set-cluster e2e --server=https://1.2.3.4

		# Embed certificate authority data for the e2e cluster entry
		kubectl config set-cluster e2e --embed-certs --certificate-authority=~/.kube/e2e/kubernetes.ca.crt

		# Disable cert checking for the e2e cluster entry
		kubectl config set-cluster e2e --insecure-skip-tls-verify=true

		# Set custom TLS server name to use for validation for the e2e cluster entry
		kubectl config set-cluster e2e --tls-server-name=my-cluster-name

		# Set proxy url for the e2e cluster entry
		kubectl config set-cluster e2e --proxy-url=https://1.2.3.4`)
)

type SetClusterFlags struct {
	certificateAuthority  cliflag.StringFlag
	embedCAData           bool
	insecureSkipTLSVerify bool
	proxyURL              cliflag.StringFlag
	server                cliflag.StringFlag
	tlsServerName         cliflag.StringFlag

	configAccess clientcmd.ConfigAccess
	ioStreams    genericclioptions.IOStreams
}

type SetClusterOptions struct {
	CertificateAuthority  cliflag.StringFlag
	EmbedCAData           bool
	InsecureSkipTLSVerify bool
	Name                  string
	ProxyURL              cliflag.StringFlag
	Server                cliflag.StringFlag
	TlsServerName         cliflag.StringFlag

	ConfigAccess clientcmd.ConfigAccess
	IOStreams    genericclioptions.IOStreams
}

// NewCmdConfigSetCluster returns a Command instance for 'config set-cluster' sub command
func NewCmdConfigSetCluster(streams genericclioptions.IOStreams, configAccess clientcmd.ConfigAccess) *cobra.Command {
	flags := NewSetClusterFlags(streams, configAccess)

	cmd := &cobra.Command{
		Use:                   fmt.Sprintf("set-cluster NAME [--%v=server] [--%v=path/to/certificate/authority] [--%v=true] [--%v=example.com]", clientcmd.FlagAPIServer, clientcmd.FlagCAFile, clientcmd.FlagInsecure, clientcmd.FlagTLSServerName),
		DisableFlagsInUseLine: true,
		Short:                 i18n.T("Set a cluster entry in kubeconfig"),
		Long:                  setClusterLong,
		Example:               setClusterExample,
		Run: func(cmd *cobra.Command, args []string) {
			options, err := flags.ToOptions(args)
			cmdutil.CheckErr(err)
			cmdutil.CheckErr(options.RunSetCluster())
		},
	}

	if err := flags.AddFlags(cmd); err != nil {
		cmdutil.CheckErr(err)
	}

	return cmd
}

func NewSetClusterFlags(streams genericclioptions.IOStreams, configAccess clientcmd.ConfigAccess) *SetClusterFlags {
	return &SetClusterFlags{
		configAccess:          configAccess,
		ioStreams:             streams,
		server:                cliflag.StringFlag{},
		tlsServerName:         cliflag.StringFlag{},
		insecureSkipTLSVerify: false,
		certificateAuthority:  cliflag.StringFlag{},
		embedCAData:           false,
		proxyURL:              cliflag.StringFlag{},
	}
}

func (flags *SetClusterFlags) ToOptions(args []string) (*SetClusterOptions, error) {
	if len(args) != 1 {
		return nil, fmt.Errorf("unexpected args: %v", args)
	}
	if flags.insecureSkipTLSVerify && flags.certificateAuthority.Provided() {
		return nil, fmt.Errorf("you cannot specify a certificate authority and insecure mode at the same time")
	}
	if flags.embedCAData {
		if !flags.certificateAuthority.Provided() {
			return nil, fmt.Errorf("you must specify a --%s to embed", clientcmd.FlagCAFile)
		}
		if _, err := os.Stat(flags.certificateAuthority.Value()); err != nil {
			return nil, fmt.Errorf("could not stat %s file %s: %v", clientcmd.FlagCAFile, flags.certificateAuthority.Value(), err)
		}
	}

	options := &SetClusterOptions{
		ConfigAccess:          flags.configAccess,
		IOStreams:             flags.ioStreams,
		Name:                  args[0],
		Server:                flags.server,
		TlsServerName:         flags.tlsServerName,
		InsecureSkipTLSVerify: flags.insecureSkipTLSVerify,
		CertificateAuthority:  flags.certificateAuthority,
		EmbedCAData:           flags.embedCAData,
		ProxyURL:              flags.proxyURL,
	}

	return options, nil
}

// AddFlags registers flags for a cli
func (flags *SetClusterFlags) AddFlags(cmd *cobra.Command) error {
	cmd.Flags().Var(&flags.server, clientcmd.FlagAPIServer, clientcmd.FlagAPIServer+" for the cluster entry in kubeconfig")
	cmd.Flags().Var(&flags.tlsServerName, clientcmd.FlagTLSServerName, clientcmd.FlagTLSServerName+" for the cluster entry in kubeconfig")
	cmd.Flags().BoolVar(&flags.insecureSkipTLSVerify, clientcmd.FlagInsecure, false, clientcmd.FlagInsecure+" for the cluster entry in kubeconfig")
	cmd.Flags().Var(&flags.certificateAuthority, clientcmd.FlagCAFile, "Path to "+clientcmd.FlagCAFile+" file for the cluster entry in kubeconfig")
	if err := cmd.MarkFlagFilename(clientcmd.FlagCAFile); err != nil {
		return err
	}
	cmd.Flags().BoolVar(&flags.embedCAData, clientcmd.FlagEmbedCerts, false, clientcmd.FlagEmbedCerts+" for the cluster entry in kubeconfig")
	cmd.Flags().Var(&flags.proxyURL, clientcmd.FlagProxyURL, clientcmd.FlagProxyURL+" for the cluster entry in kubeconfig")
	return nil
}

func (o *SetClusterOptions) RunSetCluster() error {
	config, _, err := loadConfig(o.ConfigAccess)
	if err != nil {
		return err
	}

	startingStanza, exists := config.Clusters[o.Name]
	if !exists {
		startingStanza = clientcmdapi.NewCluster()
	}
	cluster := o.modifyCluster(*startingStanza)
	config.Clusters[o.Name] = &cluster

	if err := clientcmd.ModifyConfig(o.ConfigAccess, *config, true); err != nil {
		return err
	}

	_, err = fmt.Fprintf(o.IOStreams.Out, "Cluster %q set.\n", o.Name)
	if err != nil {
		return err
	}

	return nil
}

func (o *SetClusterOptions) modifyCluster(existingCluster clientcmdapi.Cluster) clientcmdapi.Cluster {
	modifiedCluster := existingCluster

	if o.Server.Provided() {
		modifiedCluster.Server = o.Server.Value()
		// specifying a --server on the command line, overrides the TLSServerName that was specified in the kubeconfig file.
		// if both are specified, then the next if block will write the new TLSServerName.
		modifiedCluster.TLSServerName = ""
	}
	if o.TlsServerName.Provided() {
		modifiedCluster.TLSServerName = o.TlsServerName.Value()
	}
	if o.InsecureSkipTLSVerify {
		modifiedCluster.InsecureSkipTLSVerify = o.InsecureSkipTLSVerify
		// Specifying insecure mode clears any certificate authority
		if modifiedCluster.InsecureSkipTLSVerify {
			modifiedCluster.CertificateAuthority = ""
			modifiedCluster.CertificateAuthorityData = nil
		}
	}
	if o.CertificateAuthority.Provided() {
		caPath := o.CertificateAuthority
		if o.EmbedCAData {
			modifiedCluster.CertificateAuthorityData, _ = os.ReadFile(caPath.Value())
			modifiedCluster.InsecureSkipTLSVerify = false
			modifiedCluster.CertificateAuthority = ""
		} else {
			caPathAbs, _ := filepath.Abs(caPath.Value())
			modifiedCluster.CertificateAuthority = caPathAbs
			// Specifying a certificate authority file clears certificate authority data and insecure mode
			if caPath.Provided() {
				modifiedCluster.InsecureSkipTLSVerify = false
				modifiedCluster.CertificateAuthorityData = nil
			}
		}
	}

	if o.ProxyURL.Provided() {
		modifiedCluster.ProxyURL = o.ProxyURL.Value()
	}

	return modifiedCluster
}
