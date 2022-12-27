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
	"reflect"

	"github.com/spf13/cobra"

	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/client-go/tools/clientcmd"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/templates"
)

type UnsetOptions struct {
	PropertyName string

	ConfigAccess clientcmd.ConfigAccess
	IOStreams    genericclioptions.IOStreams
}

var (
	unsetLong = templates.LongDesc(i18n.T(`
	Unset an individual value in a kubeconfig file.

	PROPERTY_NAME is a dot delimited name where each token represents either an attribute name or a map key.  Map keys may not contain dots.`))

	unsetExample = templates.Examples(`
		# Unset the current-context
		kubectl config unset current-context

		# Unset namespace in foo context
		kubectl config unset contexts.foo.namespace`)
)

// NewCmdConfigUnset returns a Command instance for 'config unset' sub command
func NewCmdConfigUnset(streams genericclioptions.IOStreams, configAccess clientcmd.ConfigAccess) *cobra.Command {
	cmd := &cobra.Command{
		Use:                   "unset PROPERTY_NAME",
		DisableFlagsInUseLine: true,
		Short:                 i18n.T("Unset an individual value in a kubeconfig file"),
		Long:                  unsetLong,
		Example:               unsetExample,
		Run: func(cmd *cobra.Command, args []string) {
			options := NewUnsetOptions(streams, configAccess)
			cmdutil.CheckErr(options.Complete(args))
			cmdutil.CheckErr(options.RunUnset())

		},
	}

	return cmd
}

// NewUnsetOptions creates the options for the command
func NewUnsetOptions(ioStreams genericclioptions.IOStreams, configAccess clientcmd.ConfigAccess) *UnsetOptions {
	return &UnsetOptions{
		ConfigAccess: configAccess,
		IOStreams:    ioStreams,
	}
}

func (o *UnsetOptions) Complete(args []string) error {
	if len(args) != 1 {
		return fmt.Errorf("unexpected args: %v", args)
	}

	o.PropertyName = args[0]
	return nil
}

func (o *UnsetOptions) RunUnset() error {
	config, _, err := loadConfig(o.ConfigAccess)
	if err != nil {
		return err
	}

	steps, err := newNavigationSteps(o.PropertyName)
	if err != nil {
		return err
	}
	err = modifyConfig(reflect.ValueOf(config), steps, "", true, true)
	if err != nil {
		return err
	}

	if err := clientcmd.ModifyConfig(o.ConfigAccess, *config, false); err != nil {
		return err
	}
	if _, err := fmt.Fprintf(o.IOStreams.Out, "Property %q unset.\n", o.PropertyName); err != nil {
		return err
	}
	return nil
}
