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

	"github.com/spf13/cobra"

	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/printers"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	"k8s.io/client-go/tools/clientcmd/api/latest"
	cliflag "k8s.io/component-base/cli/flag"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/scheme"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/templates"
)

// ViewOptions holds the command-line options for 'config view' sub command
type ViewOptions struct {
	PrintFlags  *genericclioptions.PrintFlags
	PrintObject printers.ResourcePrinterFunc

	ConfigAccess clientcmd.ConfigAccess
	Merge        cliflag.Tristate
	Flatten      bool
	Minify       bool
	RawByteData  bool

	Context      string
	OutputFormat string

	genericclioptions.IOStreams
}

var (
	viewLong = templates.LongDesc(`
		Display merged kubeconfig settings or a specified kubeconfig file.

		You can use --output jsonpath={...} to extract specific values using a jsonpath expression.`)

	viewExample = templates.Examples(`
		# Show merged kubeconfig settings.
		kubectl config view

		# Show merged kubeconfig settings and raw certificate data.
		kubectl config view --raw

		# Get the password for the e2e user
		kubectl config view -o jsonpath='{.users[?(@.name == "e2e")].user.password}'`)

	defaultOutputFormat = "yaml"
)

// NewCmdConfigView returns a Command instance for 'config view' sub command
func NewCmdConfigView(f cmdutil.Factory, streams genericclioptions.IOStreams, ConfigAccess clientcmd.ConfigAccess) *cobra.Command {
	o := &ViewOptions{
		PrintFlags:   genericclioptions.NewPrintFlags("").WithTypeSetter(scheme.Scheme).WithDefaultOutput("yaml"),
		ConfigAccess: ConfigAccess,

		IOStreams: streams,
	}

	cmd := &cobra.Command{
		Use:     "view",
		Short:   i18n.T("Display merged kubeconfig settings or a specified kubeconfig file"),
		Long:    viewLong,
		Example: viewExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Complete(cmd, args))
			cmdutil.CheckErr(o.Validate())
			cmdutil.CheckErr(o.Run())
		},
	}

	o.PrintFlags.AddFlags(cmd)

	o.Merge.Default(true)
	mergeFlag := cmd.Flags().VarPF(&o.Merge, "merge", "", "Merge the full hierarchy of kubeconfig files")
	mergeFlag.NoOptDefVal = "true"
	cmd.Flags().BoolVar(&o.RawByteData, "raw", o.RawByteData, "Display raw byte data")
	cmd.Flags().BoolVar(&o.Flatten, "flatten", o.Flatten, "Flatten the resulting kubeconfig file into self-contained output (useful for creating portable kubeconfig files)")
	cmd.Flags().BoolVar(&o.Minify, "minify", o.Minify, "Remove all information not used by current-context from the output")
	return cmd
}

// Complete completes the required command-line options
func (o *ViewOptions) Complete(cmd *cobra.Command, args []string) error {
	if len(args) != 0 {
		return cmdutil.UsageErrorf(cmd, "unexpected arguments: %v", args)
	}
	if o.ConfigAccess.IsExplicitFile() {
		if !o.Merge.Provided() {
			o.Merge.Set("false")
		}
	}

	printer, err := o.PrintFlags.ToPrinter()
	if err != nil {
		return err
	}
	o.PrintObject = printer.PrintObj
	o.Context = cmdutil.GetFlagString(cmd, "context")

	return nil
}

// Validate makes sure that provided values for command-line options are valid
func (o ViewOptions) Validate() error {
	if !o.Merge.Value() && !o.ConfigAccess.IsExplicitFile() {
		return errors.New("if merge==false a precise file must to specified")
	}

	return nil
}

// Run performs the execution of 'config view' sub command
func (o ViewOptions) Run() error {
	config, err := o.loadConfig()
	if err != nil {
		return err
	}

	if o.Minify {
		if len(o.Context) > 0 {
			config.CurrentContext = o.Context
		}
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

	convertedObj, err := latest.Scheme.ConvertToVersion(config, latest.ExternalVersion)
	if err != nil {
		return err
	}

	return o.PrintObject(convertedObj, o.Out)
}

func (o ViewOptions) loadConfig() (*clientcmdapi.Config, error) {
	err := o.Validate()
	if err != nil {
		return nil, err
	}

	config, err := o.getStartingConfig()
	return config, err
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
