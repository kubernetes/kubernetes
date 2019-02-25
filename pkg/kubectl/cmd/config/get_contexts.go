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
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/util/i18n"
	"k8s.io/kubernetes/pkg/kubectl/util/printers"
	"k8s.io/kubernetes/pkg/kubectl/util/templates"
)

// GetContextsOptions contains the assignable options from the args.
type GetContextsOptions struct {
	configAccess clientcmd.ConfigAccess
	nameOnly     bool
	showHeaders  bool
	contextNames []string

	genericclioptions.IOStreams
}

var (
	getContextsLong = templates.LongDesc(`Displays one or many contexts from the kubeconfig file.`)

	getContextsExample = templates.Examples(`
		# List all the contexts in your kubeconfig file
		kubectl config get-contexts

		# Describe one context in your kubeconfig file.
		kubectl config get-contexts my-context`)
)

// NewCmdConfigGetContexts creates a command object for the "get-contexts" action, which
// retrieves one or more contexts from a kubeconfig.
func NewCmdConfigGetContexts(streams genericclioptions.IOStreams, configAccess clientcmd.ConfigAccess) *cobra.Command {
	options := &GetContextsOptions{
		configAccess: configAccess,

		IOStreams: streams,
	}

	cmd := &cobra.Command{
		Use:                   "get-contexts [(-o|--output=)name)]",
		DisableFlagsInUseLine: true,
		Short:                 i18n.T("Describe one or many contexts"),
		Long:                  getContextsLong,
		Example:               getContextsExample,
		Run: func(cmd *cobra.Command, args []string) {
			validOutputTypes := sets.NewString("", "json", "yaml", "wide", "name", "custom-columns", "custom-columns-file", "go-template", "go-template-file", "jsonpath", "jsonpath-file")
			supportedOutputTypes := sets.NewString("", "name")
			outputFormat := cmdutil.GetFlagString(cmd, "output")
			if !validOutputTypes.Has(outputFormat) {
				cmdutil.CheckErr(fmt.Errorf("output must be one of '' or 'name': %v", outputFormat))
			}
			if !supportedOutputTypes.Has(outputFormat) {
				fmt.Fprintf(options.Out, "--output %v is not available in kubectl config get-contexts; resetting to default output format\n", outputFormat)
				cmd.Flags().Set("output", "")
			}
			cmdutil.CheckErr(options.Complete(cmd, args))
			cmdutil.CheckErr(options.RunGetContexts())
		},
	}

	cmd.Flags().Bool("no-headers", false, "When using the default or custom-column output format, don't print headers (default print headers).")
	cmd.Flags().StringP("output", "o", "", "Output format. One of: name")
	return cmd
}

// Complete assigns GetContextsOptions from the args.
func (o *GetContextsOptions) Complete(cmd *cobra.Command, args []string) error {
	o.contextNames = args
	o.nameOnly = false
	if cmdutil.GetFlagString(cmd, "output") == "name" {
		o.nameOnly = true
	}
	o.showHeaders = true
	if cmdutil.GetFlagBool(cmd, "no-headers") || o.nameOnly {
		o.showHeaders = false
	}

	return nil
}

// RunGetContexts implements all the necessary functionality for context retrieval.
func (o GetContextsOptions) RunGetContexts() error {
	config, err := o.configAccess.GetStartingConfig()
	if err != nil {
		return err
	}

	out, found := o.Out.(*tabwriter.Writer)
	if !found {
		out = printers.GetNewTabWriter(o.Out)
		defer out.Flush()
	}

	// Build a list of context names to print, and warn if any requested contexts are not found.
	// Do this before printing the headers so it doesn't look ugly.
	allErrs := []error{}
	toPrint := []string{}
	if len(o.contextNames) == 0 {
		for name := range config.Contexts {
			toPrint = append(toPrint, name)
		}
	} else {
		for _, name := range o.contextNames {
			_, ok := config.Contexts[name]
			if ok {
				toPrint = append(toPrint, name)
			} else {
				allErrs = append(allErrs, fmt.Errorf("context %v not found", name))
			}
		}
	}
	if o.showHeaders {
		err = printContextHeaders(out, o.nameOnly)
		if err != nil {
			allErrs = append(allErrs, err)
		}
	}

	sort.Strings(toPrint)
	for _, name := range toPrint {
		err = printContext(name, config.Contexts[name], out, o.nameOnly, config.CurrentContext == name)
		if err != nil {
			allErrs = append(allErrs, err)
		}
	}

	return utilerrors.NewAggregate(allErrs)
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
