/*
Copyright 2014 Google Inc. All rights reserved.

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

	"github.com/spf13/cobra"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/clientcmd"
	clientcmdapi "github.com/GoogleCloudPlatform/kubernetes/pkg/client/clientcmd/api"
)

type createClusterOptions struct {
	pathOptions           *pathOptions
	name                  string
	server                string
	apiVersion            string
	insecureSkipTLSVerify bool
	certificateAuthority  string
}

func NewCmdConfigSetCluster(out io.Writer, pathOptions *pathOptions) *cobra.Command {
	options := &createClusterOptions{pathOptions: pathOptions}

	cmd := &cobra.Command{
		Use:   "set-cluster name [server] [insecure-skip-tls-verify] [certificate-authority] [api-version]",
		Short: "Sets a cluster entry in .kubeconfig",
		Long: `Sets a cluster entry in .kubeconfig

		Specifying a name that already exists overwrites that cluster entry.
		`,
		Run: func(cmd *cobra.Command, args []string) {
			if !options.complete(cmd) {
				return
			}

			err := options.run()
			if err != nil {
				fmt.Printf("%v\n", err)
			}
		},
	}

	cmd.Flags().StringVar(&options.server, clientcmd.FlagAPIServer, "", clientcmd.FlagAPIServer+" for the cluster entry in .kubeconfig")
	cmd.Flags().StringVar(&options.apiVersion, clientcmd.FlagAPIVersion, "", clientcmd.FlagAPIVersion+" for the cluster entry in .kubeconfig")
	cmd.Flags().BoolVar(&options.insecureSkipTLSVerify, clientcmd.FlagInsecure, false, clientcmd.FlagInsecure+" for the cluster entry in .kubeconfig")
	cmd.Flags().StringVar(&options.certificateAuthority, clientcmd.FlagCAFile, "", clientcmd.FlagCAFile+" for the cluster entry in .kubeconfig")

	return cmd
}

func (o createClusterOptions) run() error {
	err := o.validate()
	if err != nil {
		return err
	}

	config, filename, err := o.pathOptions.getStartingConfig()
	if err != nil {
		return err
	}

	if config.Clusters == nil {
		config.Clusters = make(map[string]clientcmdapi.Cluster)
	}

	cluster := o.cluster()
	config.Clusters[o.name] = cluster

	err = clientcmd.WriteToFile(*config, filename)
	if err != nil {
		return err
	}

	return nil
}

// cluster builds a Cluster object from the options
func (o *createClusterOptions) cluster() clientcmdapi.Cluster {
	cluster := clientcmdapi.Cluster{
		Server:                o.server,
		APIVersion:            o.apiVersion,
		InsecureSkipTLSVerify: o.insecureSkipTLSVerify,
		CertificateAuthority:  o.certificateAuthority,
	}

	return cluster
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

	return nil
}
