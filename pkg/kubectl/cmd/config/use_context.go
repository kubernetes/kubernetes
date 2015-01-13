/*
Copyright 2014 Google Inc. All rights reserved.

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

	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/clientcmd"
)

type useContextOptions struct {
	pathOptions *pathOptions
	contextName string
}

func NewCmdConfigUseContext(out io.Writer, pathOptions *pathOptions) *cobra.Command {
	options := &useContextOptions{pathOptions: pathOptions}

	cmd := &cobra.Command{
		Use:   "use-context context-name",
		Short: "Sets the current-context in a .kubeconfig file",
		Long:  `Sets the current-context in a .kubeconfig file`,
		Run: func(cmd *cobra.Command, args []string) {
			if !options.complete(cmd) {
				return
			}

			err := options.run()
			if err != nil {
				fmt.Printf("%v\n", err)
			}
		},
	}

	return cmd
}

func (o useContextOptions) run() error {
	err := o.validate()
	if err != nil {
		return err
	}

	config, filename, err := o.pathOptions.getStartingConfig()
	if err != nil {
		return err
	}

	if len(filename) == 0 {
		return errors.New("cannot set current-context without using a specific file")
	}

	config.CurrentContext = o.contextName

	err = clientcmd.WriteToFile(*config, filename)
	if err != nil {
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

func (o useContextOptions) validate() error {
	if len(o.contextName) == 0 {
		return errors.New("You must specify a current-context")
	}

	return nil
}
