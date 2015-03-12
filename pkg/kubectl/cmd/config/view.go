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
	"os"

	"github.com/golang/glog"
	"github.com/spf13/cobra"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/clientcmd"
	clientcmdapi "github.com/GoogleCloudPlatform/kubernetes/pkg/client/clientcmd/api"
	cmdutil "github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl/cmd/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

type viewOptions struct {
	pathOptions *pathOptions
	merge       util.BoolFlag
}

const (
	view_long = `displays merged .kubeconfig settings or a specified .kubeconfig file.

You can use --output=template --template=TEMPLATE to extract specific values.`
	view_example = `// Show merged .kubeconfig settings.
$ kubectl config view

// Show only local ./.kubeconfig settings
$ kubectl config view --local

// Get the password for the e2e user
$ kubectl config view -o template --template='{{ index . "users" "e2e" "password" }}'`
)

func NewCmdConfigView(out io.Writer, pathOptions *pathOptions) *cobra.Command {
	options := &viewOptions{pathOptions: pathOptions}

	cmd := &cobra.Command{
		Use:     "view",
		Short:   "displays merged .kubeconfig settings or a specified .kubeconfig file.",
		Long:    view_long,
		Example: view_example,
		Run: func(cmd *cobra.Command, args []string) {
			options.complete()

			printer, _, err := cmdutil.PrinterForCommand(cmd)
			if err != nil {
				glog.FatalDepth(1, err)
			}
			config, err := options.loadConfig()
			if err != nil {
				glog.FatalDepth(1, err)
			}
			err = printer.PrintObj(config, out)
			if err != nil {
				glog.FatalDepth(1, err)
			}
		},
	}

	cmdutil.AddPrinterFlags(cmd)
	// Default to yaml
	cmd.Flags().Set("output", "yaml")

	options.merge.Default(true)
	cmd.Flags().Var(&options.merge, "merge", "merge together the full hierarchy of .kubeconfig files")
	return cmd
}

func (o *viewOptions) complete() bool {
	// if --kubeconfig, --global, or --local is specified, then merging doesn't make sense since you're declaring precise intent
	if (len(o.pathOptions.specifiedFile) > 0) || o.pathOptions.global || o.pathOptions.local {
		if !o.merge.Provided() {
			o.merge.Set("false")
		}
	}

	return true
}

func (o viewOptions) loadConfig() (*clientcmdapi.Config, error) {
	err := o.validate()
	if err != nil {
		return nil, err
	}

	config, _, err := o.getStartingConfig()
	return config, err
}

func (o viewOptions) validate() error {
	return nil
}

// getStartingConfig returns the Config object built from the sources specified by the options, the filename read (only if it was a single file), and an error if something goes wrong
func (o *viewOptions) getStartingConfig() (*clientcmdapi.Config, string, error) {
	switch {
	case o.merge.Value():
		loadingRules := clientcmd.NewClientConfigLoadingRules()
		loadingRules.EnvVarPath = os.Getenv("KUBECONFIG")
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
