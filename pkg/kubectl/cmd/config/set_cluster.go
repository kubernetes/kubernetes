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
	"os"
	"path/filepath"

	"github.com/spf13/cobra"

	"k8s.io/apiserver/pkg/util/flag"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/util/i18n"
)

type setClusterOptions struct {
	configAccess          clientcmd.ConfigAccess
	name                  string
	server                flag.StringFlag
	insecureSkipTLSVerify flag.Tristate
	certificateAuthority  flag.StringFlag
	embedCAData           flag.Tristate
}

var (
	setClusterLong = templates.LongDesc(`
		Sets a cluster entry in kubeconfig.

		Specifying a name that already exists will merge new fields on top of existing values for those fields.`)

	setClusterExample = templates.Examples(`
		# Set only the server field on the e2e cluster entry without touching other values.
		kubectl config set-cluster e2e --server=https://1.2.3.4

		# Embed certificate authority data for the e2e cluster entry
		kubectl config set-cluster e2e --certificate-authority=~/.kube/e2e/kubernetes.ca.crt

		# Disable cert checking for the dev cluster entry
		kubectl config set-cluster e2e --insecure-skip-tls-verify=true`)
)

func NewCmdConfigSetCluster(out io.Writer, configAccess clientcmd.ConfigAccess) *cobra.Command {
	options := &setClusterOptions{configAccess: configAccess}

	cmd := &cobra.Command{
		Use: fmt.Sprintf("set-cluster NAME [--%v=server] [--%v=path/to/certificate/authority] [--%v=true]", clientcmd.FlagAPIServer, clientcmd.FlagCAFile, clientcmd.FlagInsecure),
		DisableFlagsInUseLine: true,
		Short:   i18n.T("Sets a cluster entry in kubeconfig"),
		Long:    setClusterLong,
		Example: setClusterExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(options.complete(cmd))
			cmdutil.CheckErr(options.run())
			fmt.Fprintf(out, "Cluster %q set.\n", options.name)
		},
	}

	options.insecureSkipTLSVerify.Default(false)

	cmd.Flags().Var(&options.server, clientcmd.FlagAPIServer, clientcmd.FlagAPIServer+" for the cluster entry in kubeconfig")
	f := cmd.Flags().VarPF(&options.insecureSkipTLSVerify, clientcmd.FlagInsecure, "", clientcmd.FlagInsecure+" for the cluster entry in kubeconfig")
	f.NoOptDefVal = "true"
	cmd.Flags().Var(&options.certificateAuthority, clientcmd.FlagCAFile, "Path to "+clientcmd.FlagCAFile+" file for the cluster entry in kubeconfig")
	cmd.MarkFlagFilename(clientcmd.FlagCAFile)
	f = cmd.Flags().VarPF(&options.embedCAData, clientcmd.FlagEmbedCerts, "", clientcmd.FlagEmbedCerts+" for the cluster entry in kubeconfig")
	f.NoOptDefVal = "true"

	return cmd
}

func (o setClusterOptions) run() error {
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
	config.Clusters[o.name] = &cluster

	if err := clientcmd.ModifyConfig(o.configAccess, *config, true); err != nil {
		return err
	}

	return nil
}

// cluster builds a Cluster object from the options
func (o *setClusterOptions) modifyCluster(existingCluster clientcmdapi.Cluster) (clientcmdapi.Cluster, error) {
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
			cad, err := ioutil.ReadFile(caPath)
			if err != nil {
				return existingCluster, err
			}
			modifiedCluster.CertificateAuthorityData = cad
			modifiedCluster.InsecureSkipTLSVerify = false
			modifiedCluster.CertificateAuthority = ""
		} else {
			caPath, err := filepath.Abs(caPath)
			if err != nil {
				return existingCluster, err
			}
			modifiedCluster.CertificateAuthority = caPath
			// Specifying a certificate authority file clears certificate authority data and insecure mode
			if caPath != "" {
				modifiedCluster.InsecureSkipTLSVerify = false
				modifiedCluster.CertificateAuthorityData = nil
			}
		}
	}

	return modifiedCluster, nil
}

func (o *setClusterOptions) complete(cmd *cobra.Command) error {
	args := cmd.Flags().Args()
	if len(args) != 1 {
		return helpErrorf(cmd, "Unexpected args: %v", args)
	}

	o.name = args[0]
	return nil
}

func (o setClusterOptions) validate() error {
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
		caFileInfo, err := os.Stat(caPath)
		if err != nil {
			return err
		}
		if caFileInfo.Size() <= 0 {
			return fmt.Errorf("you must specify a non-empty certificate authority file: %v", caPath)
		}
	}

	return nil
}
