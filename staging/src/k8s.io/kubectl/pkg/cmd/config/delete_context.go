/*
Copyright 2016 The Kubernetes Authors.

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
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/util/completion"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/templates"
)

var (
	deleteContextExample = templates.Examples(`
		# Delete the context for the minikube cluster
		kubectl config delete-context minikube`)
)

// DeleteContextOptions holds the data needed to run the command
type DeleteContextOptions struct {
	Context string

	ConfigAccess clientcmd.ConfigAccess
	IOStreams    genericclioptions.IOStreams
}

// NewCmdConfigDeleteContext returns a Command instance for 'config delete-context' sub command
func NewCmdConfigDeleteContext(streams genericclioptions.IOStreams, configAccess clientcmd.ConfigAccess) *cobra.Command {
	cmd := &cobra.Command{
		Use:                   "delete-context NAME",
		DisableFlagsInUseLine: true,
		Short:                 i18n.T("Delete the specified context from the kubeconfig"),
		Long:                  i18n.T("Delete the specified context from the kubeconfig."),
		Example:               deleteContextExample,
		ValidArgsFunction:     completion.ContextCompletionFunc,
		Run: func(cmd *cobra.Command, args []string) {
			options := NewDeleteContextOptions(streams, configAccess)
			cmdutil.CheckErr(options.Complete(args))
			cmdutil.CheckErr(options.RunDeleteContext())
		},
	}

	return cmd
}

// NewDeleteContextOptions creates the options for the command
func NewDeleteContextOptions(ioStreams genericclioptions.IOStreams, configAccess clientcmd.ConfigAccess) *DeleteContextOptions {
	return &DeleteContextOptions{
		ConfigAccess: configAccess,
		IOStreams:    ioStreams,
	}
}

func (o *DeleteContextOptions) Complete(args []string) error {
	if len(args) != 1 {
		return fmt.Errorf("unexpected args: %v", args)
	}
	o.Context = args[0]
	return nil
}

func (o *DeleteContextOptions) RunDeleteContext() error {
	config, configFile, err := loadConfig(o.ConfigAccess)
	if err != nil {
		return err
	}

	if _, ok := config.Contexts[o.Context]; !ok {
		return fmt.Errorf("context \"%s\" does not exist in config file %s", o.Context, configFile)
	}

	if config.CurrentContext == o.Context {
		if _, err := fmt.Fprint(o.IOStreams.Out, "warning: this removed your active context, use \"kubectl config use-context\" to select a different one\n"); err != nil {
			return err
		}
	}

	delete(config.Contexts, o.Context)

	if err := clientcmd.ModifyConfig(o.ConfigAccess, *config, true); err != nil {
		return err
	}

	if _, err := fmt.Fprintf(o.IOStreams.Out, "deleted context \"%s\" from %s\n", o.Context, configFile); err != nil {
		return err
	}

	return nil
}
