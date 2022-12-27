/*
Copyright 2017 The Kubernetes Authors.

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

	"github.com/spf13/cobra"

	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/client-go/tools/clientcmd"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/util/completion"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/templates"
)

const (
	renameContextUse = "rename-context CONTEXT_NAME NEW_NAME"
)

var (
	renameContextShort = i18n.T("Rename a context from the kubeconfig file")

	renameContextLong = templates.LongDesc(i18n.T(`
		Renames a context from the kubeconfig file.

		CONTEXT_NAME is the context name that you want to change.

		NEW_NAME is the new name you want to set.

		Note: If the context being renamed is the 'current-context', this field will also be updated.`))

	renameContextExample = templates.Examples(`
		# Rename the context 'old-name' to 'new-name' in your kubeconfig file
		kubectl config rename-context old-name new-name`)
)

// RenameContextOptions contains the options for running the rename-context cli command.
type RenameContextOptions struct {
	ContextName string
	NewName     string

	ConfigAccess clientcmd.ConfigAccess
	IOStreams    genericclioptions.IOStreams
}

// NewCmdConfigRenameContext creates a command object for the "rename-context" action
func NewCmdConfigRenameContext(streams genericclioptions.IOStreams, configAccess clientcmd.ConfigAccess) *cobra.Command {
	cmd := &cobra.Command{
		Use:                   renameContextUse,
		DisableFlagsInUseLine: true,
		Short:                 renameContextShort,
		Long:                  renameContextLong,
		Example:               renameContextExample,
		ValidArgsFunction:     completion.ContextCompletionFunc,
		Run: func(cmd *cobra.Command, args []string) {
			options := NewRenameContextOptions(streams, configAccess)
			cmdutil.CheckErr(options.Complete(args))
			cmdutil.CheckErr(options.RunRenameContext())
		},
	}

	return cmd
}

// NewRenameContextOptions creates the options for the command
func NewRenameContextOptions(ioStreams genericclioptions.IOStreams, configAccess clientcmd.ConfigAccess) *RenameContextOptions {
	return &RenameContextOptions{
		ConfigAccess: configAccess,
		IOStreams:    ioStreams,
	}
}

// Complete assigns RenameContextOptions from the args.
func (o *RenameContextOptions) Complete(args []string) error {
	if len(args) != 2 {
		return fmt.Errorf("unexpected args: %v", args)
	}

	o.ContextName = args[0]
	o.NewName = args[1]

	if len(o.ContextName) == 0 {
		return errors.New("you must specify an original non-empty context name")
	}
	if len(o.NewName) == 0 {
		return errors.New("you must specify a new non-empty context name")
	}

	return nil
}

// RunRenameContext performs the execution for 'config rename-context' sub command
func (o *RenameContextOptions) RunRenameContext() error {
	config, configFile, err := loadConfig(o.ConfigAccess)
	if err != nil {
		return err
	}

	context, exists := config.Contexts[o.ContextName]
	if !exists {
		return fmt.Errorf("cannot rename the context %q, it's not in %s", o.ContextName, configFile)
	}

	_, newExists := config.Contexts[o.NewName]
	if newExists {
		return fmt.Errorf("cannot rename the context %q, the context %q already exists in %s", o.ContextName, o.NewName, configFile)
	}

	config.Contexts[o.NewName] = context
	delete(config.Contexts, o.ContextName)

	if config.CurrentContext == o.ContextName {
		config.CurrentContext = o.NewName
	}

	if err := clientcmd.ModifyConfig(o.ConfigAccess, *config, true); err != nil {
		return err
	}

	if _, err := fmt.Fprintf(o.IOStreams.Out, "Context %q renamed to %q\n", o.ContextName, o.NewName); err != nil {
		return err
	}
	return nil
}
