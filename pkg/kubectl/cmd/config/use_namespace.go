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
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/util/i18n"
)

var (
	use_namespace_example = templates.Examples(`
		# Sets the namespace myns in current-context
		kubectl config use-ns myns`)
)

type useNamespaceOptions struct {
	configAccess clientcmd.ConfigAccess
	namespace    string
	factory      cmdutil.Factory
}

func NewCmdConfigUseNamespace(out io.Writer, configAccess clientcmd.ConfigAccess) *cobra.Command {
	options := &useNamespaceOptions{configAccess: configAccess}

	cmd := &cobra.Command{
		Use:     "use-ns NAMESPACE_NAME",
		Short:   i18n.T("Sets the namespace in current-context"),
		Long:    `Sets the namespace in current-context`,
		Example: use_namespace_example,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(options.complete(cmd))
			cmdutil.CheckErr(options.run())
			fmt.Fprintf(out, "Switched to namespace %q.\n", options.namespace)
		},
	}

	return cmd
}

func (o useNamespaceOptions) run() error {
	config, err := o.configAccess.GetStartingConfig()
	if err != nil {
		return err
	}

	err = o.validate(config)
	if err != nil {
		return err
	}

	context, exists := config.Contexts[config.CurrentContext]
	if !exists {
		context = clientcmdapi.NewContext()
	}

	context.Namespace = o.namespace

	config.Contexts[config.CurrentContext] = context

	if err := clientcmd.ModifyConfig(o.configAccess, *config, true); err != nil {
		return err
	}

	return nil
}

func (o *useNamespaceOptions) complete(cmd *cobra.Command) error {
	endingArgs := cmd.Flags().Args()
	if len(endingArgs) != 1 {
		cmd.Help()
		return fmt.Errorf("Unexpected args: %v", endingArgs)
	}

	o.namespace = endingArgs[0]
	return nil
}

func (o useNamespaceOptions) validate(config *clientcmdapi.Config) error {
	if config.CurrentContext == "" {
		return fmt.Errorf("current-context is not set, can not switch namespace\n")
	}

	if len(o.namespace) == 0 {
		return errors.New("you must specify a namespace")
	}

	return nil
}
