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

var (
	viewLong = templates.LongDesc(i18n.T(`
		Display merged kubeconfig settings or a specified kubeconfig file.

		You can use --output jsonpath={...} to extract specific values using a jsonpath expression.`))

	viewExample = templates.Examples(`
		# Show merged kubeconfig settings
		kubectl config view

		# Show merged kubeconfig settings and raw certificate data and exposed secrets
		kubectl config view --raw

		# Get the password for the e2e user
		kubectl config view -o jsonpath='{.users[?(@.name == "e2e")].user.password}'`)
)

type ViewFlags struct {
	context      cliflag.StringFlag
	flatten      bool
	merge        bool
	minify       bool
	outputFormat cliflag.StringFlag
	printFlags   *genericclioptions.PrintFlags
	rawByteData  bool

	configAccess clientcmd.ConfigAccess
	ioStreams    genericclioptions.IOStreams
}

// ViewOptions holds the command-line options for 'config view' sub command
type ViewOptions struct {
	Context      cliflag.StringFlag
	Flatten      bool
	Merge        bool
	Minify       bool
	OutputFormat cliflag.StringFlag
	PrintObject  printers.ResourcePrinterFunc
	RawByteData  bool

	ConfigAccess clientcmd.ConfigAccess
	IOStreams    genericclioptions.IOStreams
}

// NewCmdConfigView returns a Command instance for 'config view' sub command
func NewCmdConfigView(streams genericclioptions.IOStreams, configAccess clientcmd.ConfigAccess) *cobra.Command {
	flags := NewViewFlags(streams, configAccess)

	cmd := &cobra.Command{
		Use:     "view",
		Short:   i18n.T("Display merged kubeconfig settings or a specified kubeconfig file"),
		Long:    viewLong,
		Example: viewExample,
		Run: func(cmd *cobra.Command, args []string) {
			options, err := flags.ToOptions(args)
			cmdutil.CheckErr(err)
			cmdutil.CheckErr(options.RunView())
		},
	}

	flags.AddFlags(cmd)

	return cmd
}

func NewViewFlags(streams genericclioptions.IOStreams, configAccess clientcmd.ConfigAccess) *ViewFlags {
	return &ViewFlags{
		printFlags:   genericclioptions.NewPrintFlags("").WithTypeSetter(scheme.Scheme).WithDefaultOutput("yaml"),
		configAccess: configAccess,
		ioStreams:    streams,
		merge:        true,
		flatten:      false,
		minify:       false,
		rawByteData:  false,
		context:      cliflag.StringFlag{},
		outputFormat: cliflag.StringFlag{},
	}
}

// AddFlags registers flags for a cli
func (flags *ViewFlags) AddFlags(cmd *cobra.Command) {
	cmd.Flags().Var(&flags.context, "context", "Specify context to display")
	cmd.Flags().BoolVar(&flags.flatten, "flatten", false, "Flatten the resulting kubeconfig file into self-contained output (useful for creating portable kubeconfig files)")
	cmd.Flags().BoolVar(&flags.merge, "merge", true, "Merge the full hierarchy of kubeconfig files")
	cmd.Flags().BoolVar(&flags.minify, "minify", false, "Remove all information not used by current-context from the output")
	flags.printFlags.AddFlags(cmd)
	cmd.Flags().BoolVar(&flags.rawByteData, "raw", false, "Display raw byte data")
}

// ToOptions converts from CLI inputs to runtime inputs
func (flags *ViewFlags) ToOptions(args []string) (*ViewOptions, error) {
	if len(args) != 0 {
		return nil, fmt.Errorf("received unepxected argument: %v", args)
	}

	printer, err := flags.printFlags.ToPrinter()
	if err != nil {
		return nil, err
	}

	if flags.configAccess.IsExplicitFile() {
		if !flags.merge {
			flags.merge = false
		}
	}

	if !flags.merge && !flags.configAccess.IsExplicitFile() {
		return nil, errors.New("if merge==false a precise file must be specified")
	}

	options := &ViewOptions{
		ConfigAccess: flags.configAccess,
		Context:      flags.context,
		Flatten:      flags.flatten,
		IOStreams:    flags.ioStreams,
		Merge:        flags.merge,
		Minify:       flags.minify,
		PrintObject:  printer.PrintObj,
		OutputFormat: flags.outputFormat,
		RawByteData:  flags.rawByteData,
	}

	return options, nil
}

// RunView performs the execution of 'config view' sub command
func (o *ViewOptions) RunView() error {
	config, _, err := loadConfig(o.ConfigAccess)
	if err != nil {
		return err
	}

	if o.Minify {
		if o.Context.Provided() {
			config.CurrentContext = o.Context.Value()
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
		if err := clientcmdapi.RedactSecrets(config); err != nil {
			return err
		}
		clientcmdapi.ShortenConfig(config)
	}

	convertedObj, err := latest.Scheme.ConvertToVersion(config, latest.ExternalVersion)
	if err != nil {
		return err
	}

	return o.PrintObject(convertedObj, o.IOStreams.Out)
}
