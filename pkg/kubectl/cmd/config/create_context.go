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

	"github.com/spf13/cobra"

	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
	clientcmdapi "k8s.io/kubernetes/pkg/client/unversioned/clientcmd/api"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/util/flag"
)

type createContextOptions struct {
	configAccess clientcmd.ConfigAccess
	name         string
	cluster      flag.StringFlag
	authInfo     flag.StringFlag
	namespace    flag.StringFlag
}

var (
	create_context_long = templates.LongDesc(`
		Sets a context entry in kubeconfig

		Specifying a name that already exists will merge new fields on top of existing values for those fields.`)

	create_context_example = templates.Examples(`
		# Set the user field on the gce context entry without touching other values
		kubectl config set-context gce --user=cluster-admin`)
)

func NewCmdConfigSetContext(out io.Writer, configAccess clientcmd.ConfigAccess) *cobra.Command {
	options := &createContextOptions{configAccess: configAccess}

	cmd := &cobra.Command{
		Use:     fmt.Sprintf("set-context NAME [--%v=cluster_nickname] [--%v=user_nickname] [--%v=namespace]", clientcmd.FlagClusterName, clientcmd.FlagAuthInfoName, clientcmd.FlagNamespace),
		Short:   "Sets a context entry in kubeconfig",
		Long:    create_context_long,
		Example: create_context_example,
		Run: func(cmd *cobra.Command, args []string) {
			if !options.complete(cmd) {
				return
			}

			cmdutil.CheckErr(options.run())
			fmt.Fprintf(out, "Context %q set.\n", options.name)
		},
	}

	cmd.Flags().Var(&options.cluster, clientcmd.FlagClusterName, clientcmd.FlagClusterName+" for the context entry in kubeconfig")
	cmd.Flags().Var(&options.authInfo, clientcmd.FlagAuthInfoName, clientcmd.FlagAuthInfoName+" for the context entry in kubeconfig")
	cmd.Flags().Var(&options.namespace, clientcmd.FlagNamespace, clientcmd.FlagNamespace+" for the context entry in kubeconfig")

	return cmd
}

func (o createContextOptions) run() error {
	err := o.validate()
	if err != nil {
		return err
	}

	config, err := o.configAccess.GetStartingConfig()
	if err != nil {
		return err
	}

	startingStanza, exists := config.Contexts[o.name]
	if !exists {
		startingStanza = clientcmdapi.NewContext()
	}
	context := o.modifyContext(*startingStanza)
	config.Contexts[o.name] = &context

	if err := clientcmd.ModifyConfig(o.configAccess, *config, true); err != nil {
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
		return errors.New("you must specify a non-empty context name")
	}

	return nil
}
