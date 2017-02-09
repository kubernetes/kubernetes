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
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"path/filepath"

	"github.com/spf13/cobra"

	"k8s.io/apiserver/pkg/util/flag"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/util/i18n"
)

type createClusterOptions struct {
	configAccess          clientcmd.ConfigAccess
	name                  string
	server                flag.StringFlag
	apiVersion            flag.StringFlag
	insecureSkipTLSVerify flag.Tristate
	certificateAuthority  flag.StringFlag
	embedCAData           flag.Tristate
}

var (
	create_cluster_long = templates.LongDesc(`
		Sets a cluster entry in kubeconfig.

		Specifying a name that already exists will merge new fields on top of existing values for those fields.`)

	create_cluster_example = templates.Examples(`
		# Set only the server field on the e2e cluster entry without touching other values.
		kubectl config set-cluster e2e --server=https://1.2.3.4

		# Embed certificate authority data for the e2e cluster entry
		kubectl config set-cluster e2e --certificate-authority=~/.kube/e2e/kubernetes.ca.crt

		# Disable cert checking for the dev cluster entry
		kubectl config set-cluster e2e --insecure-skip-tls-verify=true`)
)

func NewCmdConfigSetCluster(out io.Writer, configAccess clientcmd.ConfigAccess) *cobra.Command {
	options := &createClusterOptions{configAccess: configAccess}

	cmd := &cobra.Command{
		Use:     fmt.Sprintf("set-cluster NAME [--%v=server] [--%v=path/to/certificate/authority] [--%v=true]", clientcmd.FlagAPIServer, clientcmd.FlagCAFile, clientcmd.FlagInsecure),
		Short:   i18n.T("Sets a cluster entry in kubeconfig"),
		Long:    create_cluster_long,
		Example: create_cluster_example,
		Run: func(cmd *cobra.Command, args []string) {
			if !options.complete(cmd) {
				return
			}

			cmdutil.CheckErr(options.run())
			fmt.Fprintf(out, "Cluster %q set.\n", options.name)
		},
	}

	options.insecureSkipTLSVerify.Default(false)

	cmd.Flags().Var(&options.server, clientcmd.FlagAPIServer, clientcmd.FlagAPIServer+" for the cluster entry in kubeconfig")
	cmd.Flags().Var(&options.apiVersion, clientcmd.FlagAPIVersion, clientcmd.FlagAPIVersion+" for the cluster entry in kubeconfig")
	f := cmd.Flags().VarPF(&options.insecureSkipTLSVerify, clientcmd.FlagInsecure, "", clientcmd.FlagInsecure+" for the cluster entry in kubeconfig")
	f.NoOptDefVal = "true"
	cmd.Flags().Var(&options.certificateAuthority, clientcmd.FlagCAFile, "path to "+clientcmd.FlagCAFile+" file for the cluster entry in kubeconfig")
	cmd.MarkFlagFilename(clientcmd.FlagCAFile)
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
	cluster := o.modifyCluster(*startingStanza)
	config.Clusters[o.name] = &cluster

	if err := clientcmd.ModifyConfig(o.configAccess, *config, true); err != nil {
		return err
	}

	return nil
}

// cluster builds a Cluster object from the options
func (o *createClusterOptions) modifyCluster(existingCluster clientcmdapi.Cluster) clientcmdapi.Cluster {
	modifiedCluster := existingCluster

	if o.server.Provided() {
		modifiedCluster.Server = o.server.Value()
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
			modifiedCluster.CertificateAuthorityData, _ = ioutil.ReadFile(caPath)
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

	return modifiedCluster
}

func (o *createClusterOptions) complete(cmd *cobra.Command) bool {
	args := cmd.Flags().Args()
	if len(args) != 1 {
		cmd.Help()
		return false
	}

	o.name = args[0]
	return true
}

func (o createClusterOptions) validate() error {
	if len(o.name) == 0 {
		return errors.New("you must specify a non-empty cluster name")
	}
	if o.insecureSkipTLSVerify.Value() && o.certificateAuthority.Value() != "" {
		return errors.New("you cannot specify a certificate authority and insecure mode at the same time")
	}
	if o.embedCAData.Value() {
		caPath := o.certificateAuthority.Value()
		if caPath == "" {
			return fmt.Errorf("you must specify a --%s to embed", clientcmd.FlagCAFile)
		}
		if _, err := ioutil.ReadFile(caPath); err != nil {
			return fmt.Errorf("could not read %s data from %s: %v", clientcmd.FlagCAFile, caPath, err)
		}
	}

	return nil
}
