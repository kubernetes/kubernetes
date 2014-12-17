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

type createContextOptions struct {
	pathOptions *pathOptions
	name        string
	cluster     string
	authInfo    string
	namespace   string
}

func NewCmdConfigSetContext(out io.Writer, pathOptions *pathOptions) *cobra.Command {
	options := &createContextOptions{pathOptions: pathOptions}

	cmd := &cobra.Command{
		Use:   "set-context name",
		Short: "Sets a context entry in .kubeconfig",
		Long: `Sets a context entry in .kubeconfig

		Specifying a name that already exists overwrites that context entry.
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

	cmd.Flags().StringVar(&options.cluster, clientcmd.FlagClusterName, "", clientcmd.FlagClusterName+" for the context entry in .kubeconfig")
	cmd.Flags().StringVar(&options.authInfo, clientcmd.FlagAuthInfoName, "", clientcmd.FlagAuthInfoName+" for the context entry in .kubeconfig")
	cmd.Flags().StringVar(&options.namespace, clientcmd.FlagNamespace, "", clientcmd.FlagNamespace+" for the context entry in .kubeconfig")

	return cmd
}

func (o createContextOptions) run() error {
	err := o.validate()
	if err != nil {
		return err
	}

	config, filename, err := o.pathOptions.getStartingConfig()
	if err != nil {
		return err
	}

	context := o.context()
	config.Contexts[o.name] = context

	err = clientcmd.WriteToFile(*config, filename)
	if err != nil {
		return err
	}

	return nil
}

func (o *createContextOptions) context() clientcmdapi.Context {
	context := clientcmdapi.Context{
		Cluster:   o.cluster,
		AuthInfo:  o.authInfo,
		Namespace: o.namespace,
	}

	return context
}

func (o *createContextOptions) complete(cmd *cobra.Command) bool {
	args := cmd.Flags().Args()
	if len(args) != 1 {
		cmd.Help()
		return false
	}

	o.name = args[0]
	return true
}

func (o createContextOptions) validate() error {
	if len(o.name) == 0 {
		return errors.New("You must specify a non-empty context name")
	}

	return nil
}
