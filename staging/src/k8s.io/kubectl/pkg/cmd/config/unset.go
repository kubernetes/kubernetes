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
	"k8s.io/cli-runtime/pkg/genericclioptions"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	clientcmdapiv1 "k8s.io/client-go/tools/clientcmd/api/v1"
	"reflect"

	"github.com/spf13/cobra"
	"k8s.io/kubectl/pkg/util/templates"

	"k8s.io/client-go/tools/clientcmd"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/util/i18n"
)

type unsetOptions struct {
	configAccess clientcmd.ConfigAccess
	propertyName string
	jsonPath     bool

	streams genericclioptions.IOStreams
}

var (
	unsetLong = templates.LongDesc(i18n.T(`
	Unset an individual value in a kubeconfig file.

	PROPERTY_NAME is either a jsonpath query for where the property name will be, or a dot delimited name.`))

	unsetExample = templates.Examples(`
		# Unset the current-context
		kubectl config unset current-context

		# Unset namespace in foo context
		kubectl config unset contexts.foo.namespace

		# Unset cluster server using jsonpath
		kubectl config unset '{.clusters[?(@.name=="cluster-0")].cluster.server}'`)
)

// NewCmdConfigUnset returns a Command instance for 'config unset' sub command
func NewCmdConfigUnset(streams genericclioptions.IOStreams, configAccess clientcmd.ConfigAccess) *cobra.Command {
	options := &unsetOptions{configAccess: configAccess, jsonPath: false, streams: streams}

	cmd := &cobra.Command{
		Use:                   "unset PROPERTY_NAME",
		DisableFlagsInUseLine: true,
		Short:                 i18n.T("Unset an individual value in a kubeconfig file"),
		Long:                  unsetLong,
		Example:               unsetExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(options.complete(cmd, args))
			cmdutil.CheckErr(options.run())

		},
	}

	return cmd
}

func (o unsetOptions) run() error {
	err := o.validate()
	if err != nil {
		return err
	}

	config, err := o.configAccess.GetStartingConfig()
	if err != nil {
		return err
	}

	if o.jsonPath {
		// Convert api config to apiv1 config, so we can use jsonpath properly
		v1Config := &clientcmdapiv1.Config{}
		if err := clientcmdapiv1.Convert_api_Config_To_v1_Config(config, v1Config, nil); err != nil {
			return err
		}

		if err := modifyConfigJson(v1Config, o.propertyName, "", true, false, false); err != nil {
			return err
		}

		// Convert the apiv1 config back to an api config to write back out
		finalConfig := clientcmdapi.NewConfig()
		if err := clientcmdapiv1.Convert_v1_Config_To_api_Config(v1Config, finalConfig, nil); err != nil {
			return err
		}
		config = finalConfig
	} else {
		if _, err := fmt.Fprintln(o.streams.ErrOut, "Warning: usage of dot delimited path for setting config values is deprecated, please use jsonpath syntax instead."); err != nil {
			return fmt.Errorf("failed to write warning message to user")
		}
		steps, err := newNavigationSteps(o.propertyName)
		if err != nil {
			return err
		}
		err = modifyConfig(reflect.ValueOf(config), steps, "", true, true)
		if err != nil {
			return err
		}
	}

	if err := clientcmd.ModifyConfig(o.configAccess, *config, false); err != nil {
		return err
	}
	if _, err := fmt.Fprintf(o.streams.Out, "Property %q unset.\n", o.propertyName); err != nil {
		return err
	}

	return nil
}

func (o *unsetOptions) complete(cmd *cobra.Command, args []string) error {
	if len(args) != 1 {
		return helpErrorf(cmd, "Unexpected args: %v", args)
	}

	o.propertyName = args[0]

	// try to determine if we have a jsonpath from first character of first argument
	// this should only ever be a { if this is a jsonpath string, if it is not we will try to use the old dot delimited
	// syntax instead.
	if string(args[0][0]) == "{" {
		o.jsonPath = true
	}

	return nil
}

func (o unsetOptions) validate() error {
	if len(o.propertyName) == 0 {
		return errors.New("you must specify a property")
	}

	return nil
}
