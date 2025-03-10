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

package top

import (
	"fmt"
	"io"
	"strings"

	"github.com/spf13/cobra"

	v1 "k8s.io/api/core/v1"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/printers"
	"k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/metricsutil"
)

// PrintFlags composes common printer flag structs used in the Top Pod command.
type PrintFlags struct {
	PodPrintFlags *metricsutil.PodPrintFlags

	OutputFormat *string
}

// EnsureWithSort ensures that flags return a printer capable of printing sorted pods list using specified field.
func (f *PrintFlags) EnsureWithSort(sortBy string) error {
	return f.PodPrintFlags.EnsureWithSort(sortBy)
}

// EnsureWithContainers ensures that flags return a printer capable of printing usage of containers within a pod.
func (f *PrintFlags) EnsureWithContainers() error {
	return f.PodPrintFlags.EnsureWithContainers()
}

// EnsureWithNamespaces ensures that flags return a printer capable of printing with a "namespace" column.
func (f *PrintFlags) EnsureWithNamespaces() error {
	return f.PodPrintFlags.EnsureWithNamespaces()
}

// EnsureWithNoHeaders ensures that flags return a printer capable of printing without headers.
func (f *PrintFlags) EnsureWithNoHeaders() error {
	return f.PodPrintFlags.EnsureWithNoHeaders()
}

// EnsureWithSum ensures that flags return a printer capable of printing with the sum of the resource usage.
func (f *PrintFlags) EnsureWithSum() error {
	return f.PodPrintFlags.EnsureWithSum()
}

// SetPodsInfo add PodsInfo for custom output.
func (f *PrintFlags) SetPodsInfo(pods *[]v1.Pod) error {
	return f.PodPrintFlags.SetPodsInfo(pods)
}

// AllowedFormats is the list of formats in which data can be displayed
func (f *PrintFlags) AllowedFormats() []string {
	formats := f.PodPrintFlags.AllowedFormats()
	return formats
}

// ToPrinter attempts to find a composed set of PrintFlags suitable for returning a printer based on current flag values.
func (f *PrintFlags) ToPrinter() (printers.ResourcePrinter, error) {
	outputFormat := ""
	if f.OutputFormat != nil {
		outputFormat = *f.OutputFormat
	}

	if p, err := f.PodPrintFlags.ToPrinter(outputFormat); !genericclioptions.IsNoCompatiblePrinterError(err) {
		return p, err
	}

	return nil, genericclioptions.NoCompatiblePrinterError{OutputFormat: &outputFormat, AllowedFormats: f.AllowedFormats()}
}

// AddFlags receives a *cobra.Command reference and binds flags related to printing.
func (f *PrintFlags) AddFlags(cmd *cobra.Command) {
	f.PodPrintFlags.AddFlags(cmd)

	if f.OutputFormat != nil {
		cmd.Flags().StringVarP(f.OutputFormat, "output", "o", *f.OutputFormat, fmt.Sprintf(`Output format. One of: (%s).`, strings.Join(f.AllowedFormats(), ", ")))
		util.CheckErr(cmd.RegisterFlagCompletionFunc(
			"output",
			func(cmd *cobra.Command, args []string, toComplete string) ([]string, cobra.ShellCompDirective) {
				var comps []string
				for _, format := range f.AllowedFormats() {
					if strings.HasPrefix(format, toComplete) {
						comps = append(comps, format)
					}
				}
				return comps, cobra.ShellCompDirectiveNoFileComp
			},
		))
	}
}

// NewPrintFlags returns flags associated with printing, with default values set.
func NewPrintFlags(out io.Writer) *PrintFlags {
	outputFormat := ""

	return &PrintFlags{
		OutputFormat: &outputFormat,

		PodPrintFlags: metricsutil.NewPodPrintFlags(out),
	}
}
