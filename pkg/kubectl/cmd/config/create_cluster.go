/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
	clientcmdapi "k8s.io/kubernetes/pkg/client/unversioned/clientcmd/api"
	"k8s.io/kubernetes/pkg/util"
)

type createClusterOptions struct {
	configAccess          ConfigAccess
	name                  string
	server                util.StringFlag
	apiVersion            util.StringFlag
	insecureSkipTLSVerify util.BoolFlag
	certificateAuthority  util.StringFlag
	embedCAData           util.BoolFlag
}

const (
	create_cluster_long = `Sets a cluster entry in kubeconfig.
Specifying a name that already exists will merge new fields on top of existing values for those fields.`
	create_cluster_example = `# Set only the server field on the e2e cluster entry without touching other values.
$ kubectl config set-cluster e2e --server=https://1.2.3.4

# Embed certificate authority data for the e2e cluster entry
$ kubectl config set-cluster e2e --certificate-authority=~/.kube/e2e/kubernetes.ca.crt

# Disable cert checking for the dev cluster entry
$ kubectl config set-cluster e2e --insecure-skip-tls-verify=true`
)

func NewCmdConfigSetCluster(out io.Writer, configAccess ConfigAccess) *cobra.Command {
	options := &createClusterOptions{configAccess: configAccess}

	cmd := &cobra.Command{
		Use:     fmt.Sprintf("set-cluster NAME [--%v=server] [--%v=path/to/certficate/authority] [--%v=apiversion] [--%v=true]", clientcmd.FlagAPIServer, clientcmd.FlagCAFile, clientcmd.FlagAPIVersion, clientcmd.FlagInsecure),
		Short:   "Sets a cluster entry in kubeconfig",
		Long:    create_cluster_long,
		Example: create_cluster_example,
		Run: func(cmd *cobra.Command, args []string) {
			if !options.complete(cmd) {
				return
			}

			err := options.run()
			if err != nil {
				fmt.Fprintf(out, "%v\n", err)
			}
		},
	}

	options.insecureSkipTLSVerify.Default(false)

	cmd.Flags().Var(&options.server, clientcmd.FlagAPIServer, clientcmd.FlagAPIServer+" for the cluster entry in kubeconfig")
	cmd.Flags().Var(&options.apiVersion, clientcmd.FlagAPIVersion, clientcmd.FlagAPIVersion+" for the cluster entry in kubeconfig")
	cmd.Flags().Var(&options.insecureSkipTLSVerify, clientcmd.FlagInsecure, clientcmd.FlagInsecure+" for the cluster entry in kubeconfig")
	cmd.Flags().Var(&options.certificateAuthority, clientcmd.FlagCAFile, "path to "+clientcmd.FlagCAFile+" for the cluster entry in kubeconfig")
	cmd.Flags().Var(&options.embedCAData, clientcmd.FlagEmbedCerts, clientcmd.FlagEmbedCerts+" for the cluster entry in kubeconfig")

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

	if err := ModifyConfig(o.configAccess, *config, true); err != nil {
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
	if o.apiVersion.Provided() {
		modifiedCluster.APIVersion = o.apiVersion.Value()
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
		return errors.New("You must specify a non-empty cluster name")
	}
	if o.insecureSkipTLSVerify.Value() && o.certificateAuthority.Value() != "" {
		return errors.New("You cannot specify a certificate authority and insecure mode at the same time")
	}
	if o.embedCAData.Value() {
		caPath := o.certificateAuthority.Value()
		if caPath == "" {
			return fmt.Errorf("You must specify a --%s to embed", clientcmd.FlagCAFile)
		}
		if _, err := ioutil.ReadFile(caPath); err != nil {
			return fmt.Errorf("Could not read %s data from %s: %v", clientcmd.FlagCAFile, caPath, err)
		}
	}

	return nil
}
