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

	"github.com/spf13/cobra"

	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	cliflag "k8s.io/component-base/cli/flag"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/util/completion"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/templates"
)

var (
	setContextLong = templates.LongDesc(i18n.T(`
		Set a context entry in kubeconfig.

		Specifying a name that already exists will merge new fields on top of existing values for those fields.`))

	setContextExample = templates.Examples(`
		# Set the user field on the gce context entry without touching other values
		kubectl config set-context gce --user=cluster-admin`)
)

type SetContextFlags struct {
	authInfo    cliflag.StringFlag
	cluster     cliflag.StringFlag
	currContext bool
	namespace   cliflag.StringFlag
	name        string

	configAccess clientcmd.ConfigAccess
}

type SetContextOptions struct {
	authInfo    cliflag.StringFlag
	cluster     cliflag.StringFlag
	currContext bool
	name        string
	namespace   cliflag.StringFlag

	configAccess clientcmd.ConfigAccess
	ioStream     genericclioptions.IOStreams
}

// NewCmdConfigSetContext returns a Command instance for 'config set-context' sub command
func NewCmdConfigSetContext(streams genericclioptions.IOStreams, configAccess clientcmd.ConfigAccess) *cobra.Command {
	flags := NewSetContextFlags(configAccess)

	cmd := &cobra.Command{
		Use:                   fmt.Sprintf("set-context [NAME | --current] [--%v=cluster_nickname] [--%v=user_nickname] [--%v=namespace]", clientcmd.FlagClusterName, clientcmd.FlagAuthInfoName, clientcmd.FlagNamespace),
		DisableFlagsInUseLine: true,
		Short:                 i18n.T("Set a context entry in kubeconfig"),
		Long:                  setContextLong,
		Example:               setContextExample,
		ValidArgsFunction:     completion.ContextCompletionFunc,
		Run: func(cmd *cobra.Command, args []string) {
			options, err := flags.ToOptions(streams, args)
			cmdutil.CheckErr(err)
			cmdutil.CheckErr(options.RunSetContext())
		},
	}

	flags.AddFlags(cmd)

	return cmd
}

func NewSetContextFlags(configAccess clientcmd.ConfigAccess) *SetContextFlags {
	return &SetContextFlags{
		configAccess: configAccess,
		name:         "",
		currContext:  false,
		cluster:      cliflag.StringFlag{},
		authInfo:     cliflag.StringFlag{},
		namespace:    cliflag.StringFlag{},
	}
}

// ToOptions converts from CLI inputs to runtime inputs
func (flags *SetContextFlags) ToOptions(streams genericclioptions.IOStreams, args []string) (*SetContextOptions, error) {
	if len(args) < 1 && !flags.currContext {
		return nil, fmt.Errorf("you must specify a non-empty context name or --current")
	} else if len(args) > 1 {
		return nil, fmt.Errorf("you may only specify one context name")
	} else if len(args) == 1 && flags.currContext {
		return nil, fmt.Errorf("you cannot specify both a context name and --current")
	}

	name := ""
	if len(args) == 1 {
		name = args[0]
	}
	if flags.currContext {
		config, err := flags.configAccess.GetStartingConfig()
		if err != nil {
			return nil, err
		}
		name = config.CurrentContext
	}

	options := &SetContextOptions{
		configAccess: flags.configAccess,
		ioStream:     streams,
		name:         name,
		currContext:  flags.currContext,
		cluster:      flags.cluster,
		authInfo:     flags.authInfo,
		namespace:    flags.namespace,
	}

	return options, nil
}

// AddFlags registers flags for a cli
func (flags *SetContextFlags) AddFlags(cmd *cobra.Command) {
	cmd.Flags().BoolVar(&flags.currContext, "current", false, "Modify the current context")
	cmd.Flags().Var(&flags.cluster, clientcmd.FlagClusterName, clientcmd.FlagClusterName+" for the cluster entry in kubeconfig")
	cmd.Flags().Var(&flags.authInfo, clientcmd.FlagAuthInfoName, clientcmd.FlagAuthInfoName+" for the cluster entry in kubeconfig")
	cmd.Flags().Var(&flags.namespace, clientcmd.FlagNamespace, "Path to "+clientcmd.FlagNamespace+" file for the cluster entry in kubeconfig")
}

func (o *SetContextOptions) RunSetContext() error {
	config, _, err := loadConfig(o.configAccess)
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

	outputString := fmt.Sprintf("Context %q created.\n", o.name)
	if exists {
		outputString = fmt.Sprintf("Context %q modified.\n", o.name)
	}
	_, err = fmt.Fprint(o.ioStream.Out, outputString)
	if err != nil {
		return err
	}

	return nil
}

func (o *SetContextOptions) modifyContext(existingContext clientcmdapi.Context) clientcmdapi.Context {
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
