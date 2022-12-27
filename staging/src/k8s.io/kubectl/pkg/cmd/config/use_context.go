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
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/util/completion"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/templates"
)

var (
	useContextExample = templates.Examples(`
		# Use the context for the minikube cluster
		kubectl config use-context minikube`)
)

type UseContextOptions struct {
	ContextName string

	ConfigAccess clientcmd.ConfigAccess
	IOStreams    genericclioptions.IOStreams
}

// NewCmdConfigUseContext returns a Command instance for 'config use-context' sub command
func NewCmdConfigUseContext(streams genericclioptions.IOStreams, ConfigAccess clientcmd.ConfigAccess) *cobra.Command {
	cmd := &cobra.Command{
		Use:                   "use-context CONTEXT_NAME",
		DisableFlagsInUseLine: true,
		Short:                 i18n.T("Set the current-context in a kubeconfig file"),
		Aliases:               []string{"use"},
		Long:                  `Set the current-context in a kubeconfig file.`,
		Example:               useContextExample,
		ValidArgsFunction:     completion.ContextCompletionFunc,
		Run: func(cmd *cobra.Command, args []string) {
			options := NewUseContextOptions(streams, ConfigAccess)
			cmdutil.CheckErr(options.Complete(args))
			cmdutil.CheckErr(options.Validate())
			cmdutil.CheckErr(options.RunUseContext())
		},
	}

	return cmd
}

// NewUseContextOptions creates the options for the command
func NewUseContextOptions(IOStreams genericclioptions.IOStreams, ConfigAccess clientcmd.ConfigAccess) *UseContextOptions {
	return &UseContextOptions{
		ConfigAccess: ConfigAccess,
		IOStreams:    IOStreams,
	}
}

func (o *UseContextOptions) Complete(args []string) error {
	if len(args) != 1 {
		return fmt.Errorf("unexpected args: %v", args)
	}

	o.ContextName = args[0]

	return nil
}

func (o *UseContextOptions) Validate() error {
	if len(o.ContextName) == 0 {
		return fmt.Errorf("empty context names are not allowed")
	}
	return nil
}

func (o *UseContextOptions) RunUseContext() error {
	config, _, err := loadConfig(o.ConfigAccess)
	if err != nil {
		return err
	}

	if _, exists := config.Contexts[o.ContextName]; !exists {
		return fmt.Errorf("can not set current-context to %q, context does not exist", o.ContextName)
	}

	config.CurrentContext = o.ContextName

	if err := clientcmd.ModifyConfig(o.ConfigAccess, *config, true); err != nil {
		return err
	}

	if _, err := fmt.Fprintf(o.IOStreams.Out, "Switched to context %q.\n", o.ContextName); err != nil {
		cmdutil.CheckErr(err)
	}

	return nil
}
