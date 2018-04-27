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
	"reflect"

	"github.com/spf13/cobra"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"

	"k8s.io/client-go/tools/clientcmd"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/util/i18n"
)

type unsetOptions struct {
	configAccess clientcmd.ConfigAccess
	propertyName string
}

var (
	unsetLong = templates.LongDesc(`
	Unsets an individual value in a kubeconfig file

	PROPERTY_NAME is a dot delimited name where each token represents either an attribute name or a map key.  Map keys may not contain dots.`)

	unsetExample = templates.Examples(`
		# Unset the current-context.
		kubectl config unset current-context

		# Unset namespace in foo context.
		kubectl config unset contexts.foo.namespace`)
)

func NewCmdConfigUnset(out io.Writer, configAccess clientcmd.ConfigAccess) *cobra.Command {
	options := &unsetOptions{configAccess: configAccess}

	cmd := &cobra.Command{
		Use: "unset PROPERTY_NAME",
		DisableFlagsInUseLine: true,
		Short:   i18n.T("Unsets an individual value in a kubeconfig file"),
		Long:    unsetLong,
		Example: unsetExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(options.complete(cmd, args))
			cmdutil.CheckErr(options.run(out))

		},
	}

	return cmd
}

func (o unsetOptions) run(out io.Writer) error {
	err := o.validate()
	if err != nil {
		return err
	}

	config, err := o.configAccess.GetStartingConfig()
	if err != nil {
		return err
	}

	steps, err := newNavigationSteps(o.propertyName)
	if err != nil {
		return err
	}
	err = modifyConfig(reflect.ValueOf(config), steps, "", true, true)
	if err != nil {
		return err
	}

	if err := clientcmd.ModifyConfig(o.configAccess, *config, false); err != nil {
		return err
	}
	if _, err := fmt.Fprintf(out, "Property %q unset.\n", o.propertyName); err != nil {
		return err
	}
	return nil
}

func (o *unsetOptions) complete(cmd *cobra.Command, args []string) error {
	if len(args) != 1 {
		return helpErrorf(cmd, "Unexpected args: %v", args)
	}

	o.propertyName = args[0]
	return nil
}

func (o unsetOptions) validate() error {
	if len(o.propertyName) == 0 {
		return errors.New("you must specify a property")
	}

	return nil
}
