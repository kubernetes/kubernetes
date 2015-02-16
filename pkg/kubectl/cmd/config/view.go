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

	"github.com/golang/glog"
	"github.com/spf13/cobra"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/clientcmd"
	clientcmdapi "github.com/GoogleCloudPlatform/kubernetes/pkg/client/clientcmd/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl/cmd/util"
)

type viewOptions struct {
	pathOptions *pathOptions
}

func NewCmdConfigView(out io.Writer, pathOptions *pathOptions) *cobra.Command {
	options := &viewOptions{pathOptions: pathOptions}

	cmd := &cobra.Command{
		Use:   "view",
		Short: "displays .kubeconfig settings or a specified .kubeconfig file.",
		Long: `displays .kubeconfig settings or a specified .kubeconfig file.
Examples:
  // Show settings from specified file
  $ kubectl config view --kubeconfig=path/to/my/.kubeconfig`,
		Run: func(cmd *cobra.Command, args []string) {
			printer, _, err := util.PrinterForCommand(cmd)
			if err != nil {
				glog.FatalDepth(1, err)
			}
			config, err := options.getStartingConfig()
			if err != nil {
				glog.FatalDepth(1, err)
			}
			err = printer.PrintObj(config, out)
			if err != nil {
				glog.FatalDepth(1, err)
			}
		},
	}

	util.AddPrinterFlags(cmd)
	// Default to yaml
	cmd.Flags().Set("output", "yaml")
	return cmd
}

// getStartingConfig returns the Config object built from the sources specified by the options and an error if something goes wrong
func (o *viewOptions) getStartingConfig() (*clientcmdapi.Config, error) {
	loadingOrder := clientcmd.DefaultClientConfigLoadingOrder()
	loadingOrder[0] = o.pathOptions.specifiedFile

	overrides := &clientcmd.ConfigOverrides{}
	clientConfig := clientcmd.NewNonInteractiveDeferredLoadingClientConfig(loadingOrder, overrides)

	config, err := clientConfig.RawConfig()
	if err != nil {
		return nil, fmt.Errorf("Error getting config: %v", err)
	}
	return &config, nil
}
