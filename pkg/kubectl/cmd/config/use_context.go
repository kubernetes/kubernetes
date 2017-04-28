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
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/util/i18n"
)

var (
	use_context_example = templates.Examples(`
		# Use the context for the minikube cluster
		kubectl config use-context minikube`)
)

type useContextOptions struct {
	configAccess clientcmd.ConfigAccess
	contextName  string
}

func NewCmdConfigUseContext(out io.Writer, configAccess clientcmd.ConfigAccess) *cobra.Command {
	options := &useContextOptions{configAccess: configAccess}

	cmd := &cobra.Command{
		Use:     "use-context CONTEXT_NAME",
		Short:   i18n.T("Sets the current-context in a kubeconfig file"),
		Long:    `Sets the current-context in a kubeconfig file`,
		Example: use_context_example,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(options.complete(cmd))
			cmdutil.CheckErr(options.run())
			fmt.Fprintf(out, "Switched to context %q.\n", options.contextName)
		},
	}

	return cmd
}

func (o useContextOptions) run() error {
	config, err := o.configAccess.GetStartingConfig()
	if err != nil {
		return err
	}

	err = o.validate(config)
	if err != nil {
		return err
	}

	config.CurrentContext = o.contextName

	if err := clientcmd.ModifyConfig(o.configAccess, *config, true); err != nil {
		return err
	}

	return nil
}

func (o *useContextOptions) complete(cmd *cobra.Command) error {
	endingArgs := cmd.Flags().Args()
	if len(endingArgs) != 1 {
		cmd.Help()
		return fmt.Errorf("Unexpected args: %v", endingArgs)
	}

	o.contextName = endingArgs[0]
	return nil
}

func (o useContextOptions) validate(config *clientcmdapi.Config) error {
	if len(o.contextName) == 0 {
		return errors.New("you must specify a current-context")
	}

	for name := range config.Contexts {
		if name == o.contextName {
			return nil
		}
	}

	return fmt.Errorf("no context exists with the name: %q.", o.contextName)
}
