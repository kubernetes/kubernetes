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
	"io"

	"github.com/spf13/cobra"

	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
)

// RenameContextOptions contains the options for running the rename-context cli command.
type RenameContextOptions struct {
	configAccess clientcmd.ConfigAccess
	contextName  string
	newName      string
}

const (
	renameContextUse = "rename-context CONTEXT_NAME NEW_NAME"

	renameContextShort = "Renames a context from the kubeconfig file."
)

var (
	renameContextLong = templates.LongDesc(`
		Renames a context from the kubeconfig file .

		CONTEXT_NAME is the context name that you wish change.

		NEW_NAME is the new name you wish to set.
		
		Note: In case the context being renamed is the 'current-context', this field will also be updated.`)

	renameContextExample = templates.Examples(`
		# Rename the context 'old-name' to 'new-name' in your kubeconfig file
		kubectl config rename-context old-name new-name`)
)

// NewCmdConfigRenameContext creates a command object for the "rename-context" action
func NewCmdConfigRenameContext(out io.Writer, configAccess clientcmd.ConfigAccess) *cobra.Command {
	options := &RenameContextOptions{configAccess: configAccess}

	cmd := &cobra.Command{
		Use:     renameContextUse,
		Short:   renameContextShort,
		Long:    renameContextLong,
		Example: renameContextExample,
		Run: func(cmd *cobra.Command, args []string) {
			if err := options.Complete(cmd, args, out); err != nil {
				cmdutil.CheckErr(err)
			}
			if err := options.Validate(); err != nil {
				cmdutil.UsageErrorf(cmd, err.Error())
			}
			if err := options.RunRenameContext(out); err != nil {
				cmdutil.CheckErr(err)
			}
		},
	}
	return cmd
}

// Complete assigns RenameContextOptions from the args.
func (o *RenameContextOptions) Complete(cmd *cobra.Command, args []string, out io.Writer) error {
	if len(args) != 2 {
		return helpErrorf(cmd, "Unexpected args: %v", args)
	}

	o.contextName = args[0]
	o.newName = args[1]
	return nil
}

func (o RenameContextOptions) Validate() error {
	if len(o.newName) == 0 {
		return errors.New("You must specify a new non-empty context name")
	}
	return nil
}

func (o RenameContextOptions) RunRenameContext(out io.Writer) error {
	config, err := o.configAccess.GetStartingConfig()
	if err != nil {
		return err
	}

	configFile := o.configAccess.GetDefaultFilename()
	if o.configAccess.IsExplicitFile() {
		configFile = o.configAccess.GetExplicitFile()
	}

	context, exists := config.Contexts[o.contextName]
	if !exists {
		return fmt.Errorf("cannot rename the context %q, it's not in %s", o.contextName, configFile)
	}

	_, newExists := config.Contexts[o.newName]
	if newExists {
		return fmt.Errorf("cannot rename the context %q, the context %q already exists in %s", o.contextName, o.newName, configFile)
	}

	config.Contexts[o.newName] = context
	delete(config.Contexts, o.contextName)

	if config.CurrentContext == o.contextName {
		config.CurrentContext = o.newName
	}

	if err := clientcmd.ModifyConfig(o.configAccess, *config, true); err != nil {
		return err
	}

	fmt.Fprintf(out, "Context %q was renamed to %q.\n", o.contextName, o.newName)
	return nil
}
