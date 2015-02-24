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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

type createContextOptions struct {
	pathOptions *pathOptions
	name        string
	cluster     util.StringFlag
	authInfo    util.StringFlag
	namespace   util.StringFlag
}

func NewCmdConfigSetContext(out io.Writer, pathOptions *pathOptions) *cobra.Command {
	options := &createContextOptions{pathOptions: pathOptions}

	cmd := &cobra.Command{
		Use:   fmt.Sprintf("set-context name [--%v=cluster-nickname] [--%v=user-nickname] [--%v=namespace]", clientcmd.FlagClusterName, clientcmd.FlagAuthInfoName, clientcmd.FlagNamespace),
		Short: "Sets a context entry in .kubeconfig",
		Long: `Sets a context entry in .kubeconfig
	Specifying a name that already exists will merge new fields on top of existing values for those fields.
	e.g. 
		kubectl config set-context gce --user=cluster-admin
		only sets the user field on the gce context entry without touching other values.`,

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

	cmd.Flags().Var(&options.cluster, clientcmd.FlagClusterName, clientcmd.FlagClusterName+" for the context entry in .kubeconfig")
	cmd.Flags().Var(&options.authInfo, clientcmd.FlagAuthInfoName, clientcmd.FlagAuthInfoName+" for the context entry in .kubeconfig")
	cmd.Flags().Var(&options.namespace, clientcmd.FlagNamespace, clientcmd.FlagNamespace+" for the context entry in .kubeconfig")

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

	context := o.modifyContext(config.Contexts[o.name])
	config.Contexts[o.name] = context

	err = clientcmd.WriteToFile(*config, filename)
	if err != nil {
		return err
	}

	return nil
}

func (o *createContextOptions) modifyContext(existingContext clientcmdapi.Context) clientcmdapi.Context {
	modifiedContext := existingContext

	if o.cluster.Provided() {
		modifiedContext.Cluster = o.cluster.Value()
	}
	if o.authInfo.Provided() {
		modifiedContext.AuthInfo = o.authInfo.Value()
	}
	if o.namespace.Provided() {
		modifiedContext.Namespace = o.namespace.Value()
	}

	return modifiedContext
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
