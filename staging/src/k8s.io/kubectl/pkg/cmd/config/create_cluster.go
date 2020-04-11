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
	"encoding/base64"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"path/filepath"

	"github.com/spf13/cobra"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	cliflag "k8s.io/component-base/cli/flag"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/templates"
)

type createClusterOptions struct {
	configAccess             clientcmd.ConfigAccess
	name                     string
	server                   cliflag.StringFlag
	tlsServerName            cliflag.StringFlag
	insecureSkipTLSVerify    cliflag.Tristate
	certificateAuthority     cliflag.StringFlag
	certificateAuthorityData cliflag.StringFlag
	embedCAData              cliflag.Tristate
}

var (
	createClusterLong = templates.LongDesc(`
		Sets a cluster entry in kubeconfig.

		Specifying a name that already exists will merge new fields on top of existing values for those fields.`)

	createClusterExample = templates.Examples(`
		# Set only the server field on the e2e cluster entry without touching other values.
		kubectl config set-cluster e2e --server=https://1.2.3.4

		# Set certificate authority file for the e2e cluster entry
		kubectl config set-cluster e2e --certificate-authority=~/.kube/e2e/kubernetes.ca.crt

		# Embed certificate authority data from a file for the e2e cluster entry
		kubectl config set-cluster e2e --certificate-authority=~/.kube/e2e/kubernetes.ca.crt --embed-certs

		# Embed base64 encoded certificate authority data for the e2e cluster entry
		kubectl config set-cluster e2e --certificate-authority-data="base64_encoded_certificate_authority_data_here"

		# Disable cert checking for the dev cluster entry
		kubectl config set-cluster e2e --insecure-skip-tls-verify=true

		# Set custom TLS server name to use for validation for the e2e cluster entry
		kubectl config set-cluster e2e --tls-server-name=my-cluster-name`)
)

// NewCmdConfigSetCluster returns a Command instance for 'config set-cluster' sub command
func NewCmdConfigSetCluster(out io.Writer, configAccess clientcmd.ConfigAccess) *cobra.Command {
	options := &createClusterOptions{configAccess: configAccess}

	cmd := &cobra.Command{
		Use:                   fmt.Sprintf("set-cluster NAME [--%v=server] [--%v=path/to/certificate/authority] [--%v=base64_encoded_certificate_authority_data] [--%v=true] [--%v=example.com]", clientcmd.FlagAPIServer, clientcmd.FlagCAFile, clientcmd.FlagCAData, clientcmd.FlagInsecure, clientcmd.FlagTLSServerName),
		DisableFlagsInUseLine: true,
		Short:                 i18n.T("Sets a cluster entry in kubeconfig"),
		Long:                  createClusterLong,
		Example:               createClusterExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(options.complete(cmd))
			cmdutil.CheckErr(options.run())
			fmt.Fprintf(out, "Cluster %q set.\n", options.name)
		},
	}

	options.insecureSkipTLSVerify.Default(false)

	cmd.Flags().Var(&options.server, clientcmd.FlagAPIServer, clientcmd.FlagAPIServer+" for the cluster entry in kubeconfig")
	cmd.Flags().Var(&options.tlsServerName, clientcmd.FlagTLSServerName, clientcmd.FlagTLSServerName+" for the cluster entry in kubeconfig")
	f := cmd.Flags().VarPF(&options.insecureSkipTLSVerify, clientcmd.FlagInsecure, "", clientcmd.FlagInsecure+" for the cluster entry in kubeconfig")
	f.NoOptDefVal = "true"
	cmd.Flags().Var(&options.certificateAuthority, clientcmd.FlagCAFile, "Path to "+clientcmd.FlagCAFile+" file for the cluster entry in kubeconfig")
	cmd.MarkFlagFilename(clientcmd.FlagCAFile)
	cmd.Flags().Var(&options.certificateAuthorityData, clientcmd.FlagCAData, "Base64 encoded certificate authority data")
	f = cmd.Flags().VarPF(&options.embedCAData, clientcmd.FlagEmbedCerts, "", clientcmd.FlagEmbedCerts+" for the cluster entry in kubeconfig")
	f.NoOptDefVal = "true"

	return cmd
}

func (o createClusterOptions) run() error {
	err := o.validate()
	if err != nil {
		return err
	}

	config, err := o.configAccess.GetStartingConfig()
	if err != nil {
		return err
	}

	startingStanza, exists := config.Clusters[o.name]
	if !exists {
		startingStanza = clientcmdapi.NewCluster()
	}
	cluster, err := o.modifyCluster(*startingStanza)
	if err != nil {
		return err
	}
	config.Clusters[o.name] = cluster

	if err := clientcmd.ModifyConfig(o.configAccess, *config, true); err != nil {
		return err
	}

	return nil
}

// cluster builds a Cluster object from the options
func (o *createClusterOptions) modifyCluster(existingCluster clientcmdapi.Cluster) (*clientcmdapi.Cluster, error) {
	modifiedCluster := existingCluster

	if o.server.Provided() {
		modifiedCluster.Server = o.server.Value()
		// specifying a --server on the command line, overrides the TLSServerName that was specified in the kubeconfig file.
		// if both are specified, then the next if block will write the new TLSServerName.
		modifiedCluster.TLSServerName = ""
	}
	if o.tlsServerName.Provided() {
		modifiedCluster.TLSServerName = o.tlsServerName.Value()
	}
	if o.insecureSkipTLSVerify.Provided() {
		modifiedCluster.InsecureSkipTLSVerify = o.insecureSkipTLSVerify.Value()
		// Specifying insecure mode clears any certificate authority
		if modifiedCluster.InsecureSkipTLSVerify {
			modifiedCluster.CertificateAuthority = ""
			modifiedCluster.CertificateAuthorityData = nil
		}
	}
	if o.certificateAuthority.Provided() {
		caPath := o.certificateAuthority.Value()
		if o.embedCAData.Value() {
			caData, err := ioutil.ReadFile(caPath)
			if err != nil {
				return nil, fmt.Errorf("could not read certificate authority data from %s: %v", caPath, err)
			}
			modifiedCluster.CertificateAuthorityData = caData
			modifiedCluster.InsecureSkipTLSVerify = false
			modifiedCluster.CertificateAuthority = ""
		} else {
			caPath, _ = filepath.Abs(caPath)
			modifiedCluster.CertificateAuthority = caPath
			// Specifying a certificate authority file clears certificate authority data and insecure mode
			if caPath != "" {
				modifiedCluster.InsecureSkipTLSVerify = false
				modifiedCluster.CertificateAuthorityData = nil
			}
		}
	}
	if o.certificateAuthorityData.Provided() {
		caData, err := base64.StdEncoding.DecodeString(o.certificateAuthorityData.Value())
		if err != nil {
			return nil, fmt.Errorf("could not decode certificate authority data: %v", err)
		}
		modifiedCluster.CertificateAuthorityData = caData
		modifiedCluster.InsecureSkipTLSVerify = false
		modifiedCluster.CertificateAuthority = ""
	}

	return &modifiedCluster, nil
}

func (o *createClusterOptions) complete(cmd *cobra.Command) error {
	args := cmd.Flags().Args()
	if len(args) != 1 {
		return helpErrorf(cmd, "Unexpected args: %v", args)
	}

	o.name = args[0]
	return nil
}

func (o createClusterOptions) validate() error {
	if len(o.name) == 0 {
		return errors.New("you must specify a non-empty cluster name")
	}
	if o.insecureSkipTLSVerify.Value() && (o.certificateAuthority.Value() != "" || o.certificateAuthorityData.Value() != "") {
		return errors.New("you cannot specify a certificate authority and insecure mode at the same time")
	}
	if o.certificateAuthority.Value() != "" && o.certificateAuthorityData.Value() != "" {
		return fmt.Errorf("you cannot specify a certificate authority file and certificate authority data at the same time")
	}
	if o.embedCAData.Value() && o.certificateAuthority.Value() == "" {
		return fmt.Errorf("you must specify a --%s to embed", clientcmd.FlagCAFile)
	}
	return nil
}
