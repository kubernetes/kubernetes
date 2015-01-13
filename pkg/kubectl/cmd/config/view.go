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
	"fmt"
	"io"

	"github.com/ghodss/yaml"
	"github.com/spf13/cobra"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/clientcmd"
	clientcmdapi "github.com/GoogleCloudPlatform/kubernetes/pkg/client/clientcmd/api"
)

type viewOptions struct {
	pathOptions *pathOptions
	merge       bool
}

func NewCmdConfigView(out io.Writer, pathOptions *pathOptions) *cobra.Command {
	options := &viewOptions{pathOptions: pathOptions}

	cmd := &cobra.Command{
		Use:   "view",
		Short: "displays the specified .kubeconfig file or a merged result",
		Long:  `displays the specified .kubeconfig file or a merged result`,
		Run: func(cmd *cobra.Command, args []string) {
			err := options.run()
			if err != nil {
				fmt.Printf("%v\n", err)
			}
		},
	}

	cmd.Flags().BoolVar(&options.merge, "merge", false, "merge together the full hierarchy of .kubeconfig files")

	return cmd
}

func (o viewOptions) run() error {
	err := o.validate()
	if err != nil {
		return err
	}

	config, _, err := o.getStartingConfig()
	if err != nil {
		return err
	}

	content, err := yaml.Marshal(config)
	if err != nil {
		return err
	}

	fmt.Printf("%v", string(content))

	return nil
}

func (o viewOptions) validate() error {
	return nil
}

// getStartingConfig returns the Config object built from the sources specified by the options, the filename read (only if it was a single file), and an error if something goes wrong
func (o *viewOptions) getStartingConfig() (*clientcmdapi.Config, string, error) {
	switch {
	case o.merge:
		loadingRules := clientcmd.NewClientConfigLoadingRules()
		loadingRules.EnvVarPath = ""
		loadingRules.CommandLinePath = o.pathOptions.specifiedFile

		overrides := &clientcmd.ConfigOverrides{}
		clientConfig := clientcmd.NewNonInteractiveDeferredLoadingClientConfig(loadingRules, overrides)

		config, err := clientConfig.RawConfig()
		if err != nil {
			return nil, "", fmt.Errorf("Error getting config: %v", err)
		}
		return &config, "", nil
	default:
		return o.pathOptions.getStartingConfig()
	}
}
