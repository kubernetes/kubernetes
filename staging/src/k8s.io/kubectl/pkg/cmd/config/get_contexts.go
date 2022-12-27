/*
Copyright 2016 The Kubernetes Authors.

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
	"sort"
	"strings"

	"github.com/liggitt/tabwriter"
	"github.com/spf13/cobra"

	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/printers"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/templates"
)

var (
	getContextsLong = templates.LongDesc(i18n.T(`Display one or many contexts from the kubeconfig file.`))

	getContextsExample = templates.Examples(`
		# List all the contexts in your kubeconfig file
		kubectl config get-contexts

		# Describe one context in your kubeconfig file
		kubectl config get-contexts my-context`)
)

// GetContextsFlags contains the assignable options from the args.
type GetContextsFlags struct {
	notShowHeaders bool
	output         string

	configAccess clientcmd.ConfigAccess
	ioStreams    genericclioptions.IOStreams
}

// GetContextsOptions contains the assignable options from the args.
type GetContextsOptions struct {
	ContextNames   []string
	NameOnly       bool
	NotShowHeaders bool

	ConfigAccess clientcmd.ConfigAccess
	IOStreams    genericclioptions.IOStreams
}

// NewCmdConfigGetContexts creates a command object for the "get-contexts" action, which
// retrieves one or more contexts from a kubeconfig.
func NewCmdConfigGetContexts(streams genericclioptions.IOStreams, configAccess clientcmd.ConfigAccess) *cobra.Command {
	flags := NewGetContextFlags(streams, configAccess)

	cmd := &cobra.Command{
		Use:                   "get-contexts [(-o|--output=)name)] [--no-headers]",
		DisableFlagsInUseLine: true,
		Short:                 i18n.T("Describe one or many contexts"),
		Long:                  getContextsLong,
		Example:               getContextsExample,
		Run: func(cmd *cobra.Command, args []string) {
			options, err := flags.ToOptions(args)
			cmdutil.CheckErr(err)
			cmdutil.CheckErr(options.RunGetContexts())
		},
	}

	flags.AddFlags(cmd)

	return cmd
}

func NewGetContextFlags(streams genericclioptions.IOStreams, configAccess clientcmd.ConfigAccess) *GetContextsFlags {
	return &GetContextsFlags{
		notShowHeaders: false,
		output:         "",
		configAccess:   configAccess,
		ioStreams:      streams,
	}
}

// ToOptions converts from CLI inputs to runtime inputs
func (flags *GetContextsFlags) ToOptions(args []string) (*GetContextsOptions, error) {
	if err := flags.validateOutputType(); err != nil {
		return nil, err
	}
	nameOnly := false
	if flags.output == "name" {
		nameOnly = true
	}
	o := &GetContextsOptions{
		ConfigAccess:   flags.configAccess,
		IOStreams:      flags.ioStreams,
		NameOnly:       nameOnly,
		NotShowHeaders: flags.notShowHeaders,
		ContextNames:   args,
	}

	return o, nil
}

// AddFlags registers flags for a cli
func (flags *GetContextsFlags) AddFlags(cmd *cobra.Command) {
	cmd.Flags().BoolVar(&flags.notShowHeaders, "no-headers", false, "When using the default or custom-column output format, don't print headers (default print headers).")
	cmd.Flags().StringVarP(&flags.output, "output", "o", "", `Output format. One of: (name).`)
}

// RunGetContexts implements all the necessary functionality for context retrieval.
func (o *GetContextsOptions) RunGetContexts() error {
	config, _, err := loadConfig(o.ConfigAccess)
	if err != nil {
		return err
	}

	out, found := o.IOStreams.Out.(*tabwriter.Writer)
	if !found {
		out = printers.GetNewTabWriter(o.IOStreams.Out)
		defer out.Flush()
	}

	// Build a list of context names to print, and warn if any requested contexts are not found.
	// Do this before printing the headers, so it doesn't look ugly.
	var allErrs []error
	var toPrint []string
	if len(o.ContextNames) == 0 {
		for name := range config.Contexts {
			toPrint = append(toPrint, name)
		}
	} else {
		for _, name := range o.ContextNames {
			_, ok := config.Contexts[name]
			if ok {
				toPrint = append(toPrint, name)
			} else {
				allErrs = append(allErrs, fmt.Errorf("context \"%v\" not found", name))
			}
		}
	}
	if !o.NotShowHeaders {
		err = printContextHeaders(out, o.NameOnly)
		if err != nil {
			allErrs = append(allErrs, err)
		}
	}

	sort.Strings(toPrint)
	for _, name := range toPrint {
		err = printContext(name, config.Contexts[name], out, o.NameOnly, config.CurrentContext == name)
		if err != nil {
			allErrs = append(allErrs, err)
		}
	}

	return utilerrors.NewAggregate(allErrs)
}

// validateOutputType ensures the type of output format
func (flags *GetContextsFlags) validateOutputType() error {
	validOutputTypes := sets.NewString("", "json", "yaml", "wide", "name", "custom-columns", "custom-columns-file", "go-template", "go-template-file", "jsonpath", "jsonpath-file")
	supportedOutputTypes := sets.NewString("", "name")
	outputFormat := flags.output
	if !validOutputTypes.Has(outputFormat) {
		return fmt.Errorf("output must be one of '' or 'name': %v", outputFormat)
	}
	if !supportedOutputTypes.Has(outputFormat) {
		return fmt.Errorf("--output %v is not available in kubectl config get-contexts", outputFormat)
	}
	return nil
}

func printContextHeaders(out io.Writer, nameOnly bool) error {
	columnNames := []string{"CURRENT", "NAME", "CLUSTER", "AUTHINFO", "NAMESPACE"}
	if nameOnly {
		columnNames = columnNames[:1]
	}
	_, err := fmt.Fprintf(out, "%s\n", strings.Join(columnNames, "\t"))
	return err
}

func printContext(name string, context *clientcmdapi.Context, w io.Writer, nameOnly, current bool) error {
	if nameOnly {
		_, err := fmt.Fprintf(w, "%s\n", name)
		return err
	}
	prefix := " "
	if current {
		prefix = "*"
	}
	_, err := fmt.Fprintf(w, "%s\t%s\t%s\t%s\t%s\n", prefix, name, context.Cluster, context.AuthInfo, context.Namespace)
	return err
}
