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

	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	cliflag "k8s.io/component-base/cli/flag"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/templates"
)

type createContextOptions struct {
	configAccess clientcmd.ConfigAccess
	name         string
	currContext  bool
	cluster      cliflag.StringFlag
	authInfo     cliflag.StringFlag
	namespace    cliflag.StringFlag
}

var (
	createContextLong = templates.LongDesc(i18n.T(`
		Sets a context entry in kubeconfig

		Specifying a name that already exists will merge new fields on top of existing values for those fields.`))

	createContextExample = templates.Examples(`
		# Set the user field on the gce context entry without touching other values
		kubectl config set-context gce --user=cluster-admin`)
)

// NewCmdConfigSetContext returns a Command instance for 'config set-context' sub command
func NewCmdConfigSetContext(out io.Writer, configAccess clientcmd.ConfigAccess) *cobra.Command {
	options := &createContextOptions{configAccess: configAccess}

	cmd := &cobra.Command{
		Use:                   fmt.Sprintf("set-context [NAME | --current] [--%v=cluster_nickname] [--%v=user_nickname] [--%v=namespace]", clientcmd.FlagClusterName, clientcmd.FlagAuthInfoName, clientcmd.FlagNamespace),
		DisableFlagsInUseLine: true,
		Short:                 i18n.T("Sets a context entry in kubeconfig"),
		Long:                  createContextLong,
		Example:               createContextExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(options.complete(cmd))
			name, exists, err := options.run()
			cmdutil.CheckErr(err)
			if exists {
				fmt.Fprintf(out, "Context %q modified.\n", name)
			} else {
				fmt.Fprintf(out, "Context %q created.\n", name)
			}
		},
	}

	cmd.Flags().BoolVar(&options.currContext, "current", options.currContext, "Modify the current context")
	cmd.Flags().Var(&options.cluster, clientcmd.FlagClusterName, clientcmd.FlagClusterName+" for the context entry in kubeconfig")
	cmd.Flags().Var(&options.authInfo, clientcmd.FlagAuthInfoName, clientcmd.FlagAuthInfoName+" for the context entry in kubeconfig")
	cmd.Flags().Var(&options.namespace, clientcmd.FlagNamespace, clientcmd.FlagNamespace+" for the context entry in kubeconfig")

	return cmd
}

func (o createContextOptions) run() (string, bool, error) {
	err := o.validate()
	if err != nil {
		return "", false, err
	}

	config, err := o.configAccess.GetStartingConfig()
	if err != nil {
		return "", false, err
	}

	name := o.name
	if o.currContext {
		if len(config.CurrentContext) == 0 {
			return "", false, errors.New("no current context is set")
		}
		name = config.CurrentContext
	}

	startingStanza, exists := config.Contexts[name]
	if !exists {
		startingStanza = clientcmdapi.NewContext()
	}
	context := o.modifyContext(*startingStanza)
	config.Contexts[name] = &context

	if err := clientcmd.ModifyConfig(o.configAccess, *config, true); err != nil {
		return name, exists, err
	}

	return name, exists, nil
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

func (o *createContextOptions) complete(cmd *cobra.Command) error {
	args := cmd.Flags().Args()
	if len(args) > 1 {
		return helpErrorf(cmd, "Unexpected args: %v", args)
	}
	if len(args) == 1 {
		o.name = args[0]
	}
	return nil
}

func (o createContextOptions) validate() error {
	if len(o.name) == 0 && !o.currContext {
		return errors.New("you must specify a non-empty context name or --current")
	}
	if len(o.name) > 0 && o.currContext {
		return errors.New("you cannot specify both a context name and --current")
	}

	return nil
}
