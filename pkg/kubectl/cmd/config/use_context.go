/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

	clientcmdapi "k8s.io/kubernetes/pkg/client/unversioned/clientcmd/api"
)

type useContextOptions struct {
	configAccess ConfigAccess
	contextName  string
}

func NewCmdConfigUseContext(out io.Writer, configAccess ConfigAccess) *cobra.Command {
	options := &useContextOptions{configAccess: configAccess}

	cmd := &cobra.Command{
		Use:   "use-context CONTEXT_NAME",
		Short: "Sets the current-context in a kubeconfig file",
		Long:  `Sets the current-context in a kubeconfig file`,
		Run: func(cmd *cobra.Command, args []string) {
			if !options.complete(cmd) {
				return
			}

			err := options.run()
			if err != nil {
				fmt.Fprintf(out, "%v\n", err)
			} else {
				fmt.Fprintf(out, "switched to context %q.\n", options.contextName)
			}
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

	if err := ModifyConfig(o.configAccess, *config, true); err != nil {
		return err
	}

	return nil
}

func (o *useContextOptions) complete(cmd *cobra.Command) bool {
	endingArgs := cmd.Flags().Args()
	if len(endingArgs) != 1 {
		cmd.Help()
		return false
	}

	o.contextName = endingArgs[0]
	return true
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
