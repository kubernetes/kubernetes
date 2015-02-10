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
	"reflect"

	"github.com/spf13/cobra"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/clientcmd"
)

type unsetOptions struct {
	pathOptions  *pathOptions
	propertyName string
}

func NewCmdConfigUnset(out io.Writer, pathOptions *pathOptions) *cobra.Command {
	options := &unsetOptions{pathOptions: pathOptions}

	cmd := &cobra.Command{
		Use:   "unset property-name",
		Short: "Unsets an individual value in a .kubeconfig file",
		Long: `Unsets an individual value in a .kubeconfig file

		property-name is a dot delimited name where each token represents either a attribute name or a map key.  Map keys may not contain dots.
		`,
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

func (o unsetOptions) run() error {
	err := o.validate()
	if err != nil {
		return err
	}

	config, filename, err := o.pathOptions.getStartingConfig()
	if err != nil {
		return err
	}

	if len(filename) == 0 {
		return errors.New("cannot set property without using a specific file")
	}

	steps, err := newNavigationSteps(o.propertyName)
	if err != nil {
		return err
	}
	err = modifyConfig(reflect.ValueOf(config), steps, "", true)
	if err != nil {
		return err
	}

	err = clientcmd.WriteToFile(*config, filename)
	if err != nil {
		return err
	}

	return nil
}

func (o *unsetOptions) complete(cmd *cobra.Command) bool {
	endingArgs := cmd.Flags().Args()
	if len(endingArgs) != 1 {
		cmd.Help()
		return false
	}

	o.propertyName = endingArgs[0]
	return true
}

func (o unsetOptions) validate() error {
	if len(o.propertyName) == 0 {
		return errors.New("You must specify a property")
	}

	return nil
}
