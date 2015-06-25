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
	"io"

	"github.com/golang/glog"
	"github.com/spf13/cobra"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/clientcmd"
	clientcmdapi "github.com/GoogleCloudPlatform/kubernetes/pkg/client/clientcmd/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/clientcmd/api/latest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl"
	cmdutil "github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl/cmd/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

type ViewOptions struct {
	ConfigAccess ConfigAccess
	Merge        util.BoolFlag
	Flatten      bool
	Minify       bool
	RawByteData  bool
}

const (
	view_long = `displays Merged kubeconfig settings or a specified kubeconfig file.

You can use --output=template --template=TEMPLATE to extract specific values.`
	view_example = `// Show Merged kubeconfig settings.
$ kubectl config view

// Get the password for the e2e user
$ kubectl config view -o template --template='{{range .users}}{{ if eq .name "e2e" }}{{ index .user.password }}{{end}}{{end}}'`
)

func NewCmdConfigView(out io.Writer, ConfigAccess ConfigAccess) *cobra.Command {
	options := &ViewOptions{ConfigAccess: ConfigAccess}

	cmd := &cobra.Command{
		Use:     "view",
		Short:   "displays Merged kubeconfig settings or a specified kubeconfig file.",
		Long:    view_long,
		Example: view_example,
		Run: func(cmd *cobra.Command, args []string) {
			options.Complete()

			printer, _, err := cmdutil.PrinterForCommand(cmd)
			if err != nil {
				glog.FatalDepth(1, err)
			}
			version := cmdutil.OutputVersion(cmd, latest.Version)
			printer = kubectl.NewVersionedPrinter(printer, clientcmdapi.Scheme, version)

			if err := options.Run(out, printer); err != nil {
				glog.FatalDepth(1, err)
			}

		},
	}

	cmdutil.AddPrinterFlags(cmd)
	// Default to yaml
	cmd.Flags().Set("output", "yaml")

	options.Merge.Default(true)
	cmd.Flags().Var(&options.Merge, "merge", "merge together the full hierarchy of kubeconfig files")
	cmd.Flags().BoolVar(&options.RawByteData, "raw", false, "display raw byte data")
	cmd.Flags().BoolVar(&options.Flatten, "flatten", false, "flatten the resulting kubeconfig file into self contained output (useful for creating portable kubeconfig files)")
	cmd.Flags().BoolVar(&options.Minify, "minify", false, "remove all information not used by current-context from the output")
	return cmd
}

func (o ViewOptions) Run(out io.Writer, printer kubectl.ResourcePrinter) error {
	config, err := o.loadConfig()
	if err != nil {
		return err
	}

	if o.Minify {
		if err := clientcmdapi.MinifyConfig(config); err != nil {
			return err
		}
	}

	if o.Flatten {
		if err := clientcmdapi.FlattenConfig(config); err != nil {
			return err
		}
	} else if !o.RawByteData {
		clientcmdapi.ShortenConfig(config)
	}

	err = printer.PrintObj(config, out)
	if err != nil {
		return err
	}

	return nil
}

func (o *ViewOptions) Complete() bool {
	if o.ConfigAccess.IsExplicitFile() {
		if !o.Merge.Provided() {
			o.Merge.Set("false")
		}
	}

	return true
}

func (o ViewOptions) loadConfig() (*clientcmdapi.Config, error) {
	err := o.Validate()
	if err != nil {
		return nil, err
	}

	config, err := o.getStartingConfig()
	return config, err
}

func (o ViewOptions) Validate() error {
	if !o.Merge.Value() && !o.ConfigAccess.IsExplicitFile() {
		return errors.New("if merge==false a precise file must to specified")
	}

	return nil
}

// getStartingConfig returns the Config object built from the sources specified by the options, the filename read (only if it was a single file), and an error if something goes wrong
func (o *ViewOptions) getStartingConfig() (*clientcmdapi.Config, error) {
	switch {
	case !o.Merge.Value():
		return clientcmd.LoadFromFile(o.ConfigAccess.GetExplicitFile())

	default:
		return o.ConfigAccess.GetStartingConfig()
	}
}
