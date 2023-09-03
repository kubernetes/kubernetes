/*
Copyright 2018 The Kubernetes Authors.

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

package get

import (
	"fmt"
	"strings"

	"github.com/spf13/cobra"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/printers"
	"k8s.io/kubectl/pkg/cmd/util"
)

// PrintFlags composes common printer flag structs
// used in the Get command.
type PrintFlags struct {
	JSONYamlPrintFlags *genericclioptions.JSONYamlPrintFlags
	NamePrintFlags     *genericclioptions.NamePrintFlags
	CustomColumnsFlags *CustomColumnsPrintFlags
	HumanReadableFlags *HumanPrintFlags
	TemplateFlags      *genericclioptions.KubeTemplatePrintFlags

	NoHeaders    *bool
	OutputFormat *string
}

// SetKind sets the Kind option of humanreadable flags
func (f *PrintFlags) SetKind(kind schema.GroupKind) {
	f.HumanReadableFlags.SetKind(kind)
}

// EnsureWithNamespace ensures that humanreadable flags return
// a printer capable of printing with a "namespace" column.
func (f *PrintFlags) EnsureWithNamespace() error {
	return f.HumanReadableFlags.EnsureWithNamespace()
}

// EnsureWithKind ensures that humanreadable flags return
// a printer capable of including resource kinds.
func (f *PrintFlags) EnsureWithKind() error {
	return f.HumanReadableFlags.EnsureWithKind()
}

// Copy returns a copy of PrintFlags for mutation
func (f *PrintFlags) Copy() PrintFlags {
	printFlags := *f
	return printFlags
}

// AllowedFormats is the list of formats in which data can be displayed
func (f *PrintFlags) AllowedFormats() []string {
	formats := f.JSONYamlPrintFlags.AllowedFormats()
	formats = append(formats, f.NamePrintFlags.AllowedFormats()...)
	formats = append(formats, f.TemplateFlags.AllowedFormats()...)
	formats = append(formats, f.CustomColumnsFlags.AllowedFormats()...)
	formats = append(formats, f.HumanReadableFlags.AllowedFormats()...)
	return formats
}

// ToPrinter attempts to find a composed set of PrintFlags suitable for
// returning a printer based on current flag values.
func (f *PrintFlags) ToPrinter() (printers.ResourcePrinter, error) {
	outputFormat := ""
	if f.OutputFormat != nil {
		outputFormat = *f.OutputFormat
	}

	noHeaders := false
	if f.NoHeaders != nil {
		noHeaders = *f.NoHeaders
	}
	f.HumanReadableFlags.NoHeaders = noHeaders
	f.CustomColumnsFlags.NoHeaders = noHeaders

	// for "get.go" we want to support a --template argument given, even when no --output format is provided
	if f.TemplateFlags.TemplateArgument != nil && len(*f.TemplateFlags.TemplateArgument) > 0 && len(outputFormat) == 0 {
		outputFormat = "go-template"
	}

	if p, err := f.TemplateFlags.ToPrinter(outputFormat); !genericclioptions.IsNoCompatiblePrinterError(err) {
		return p, err
	}

	if f.TemplateFlags.TemplateArgument != nil {
		f.CustomColumnsFlags.TemplateArgument = *f.TemplateFlags.TemplateArgument
	}

	if p, err := f.JSONYamlPrintFlags.ToPrinter(outputFormat); !genericclioptions.IsNoCompatiblePrinterError(err) {
		return p, err
	}

	if p, err := f.HumanReadableFlags.ToPrinter(outputFormat); !genericclioptions.IsNoCompatiblePrinterError(err) {
		return p, err
	}

	if p, err := f.CustomColumnsFlags.ToPrinter(outputFormat); !genericclioptions.IsNoCompatiblePrinterError(err) {
		return p, err
	}

	if p, err := f.NamePrintFlags.ToPrinter(outputFormat); !genericclioptions.IsNoCompatiblePrinterError(err) {
		return p, err
	}

	return nil, genericclioptions.NoCompatiblePrinterError{OutputFormat: &outputFormat, AllowedFormats: f.AllowedFormats()}
}

// AddFlags receives a *cobra.Command reference and binds
// flags related to humanreadable and template printing.
func (f *PrintFlags) AddFlags(cmd *cobra.Command) {
	f.JSONYamlPrintFlags.AddFlags(cmd)
	f.NamePrintFlags.AddFlags(cmd)
	f.TemplateFlags.AddFlags(cmd)
	f.HumanReadableFlags.AddFlags(cmd)
	f.CustomColumnsFlags.AddFlags(cmd)

	if f.OutputFormat != nil {
		cmd.Flags().StringVarP(f.OutputFormat, "output", "o", *f.OutputFormat, fmt.Sprintf(`Output format. One of: (%s). See custom columns [https://kubernetes.io/docs/reference/kubectl/#custom-columns], golang template [http://golang.org/pkg/text/template/#pkg-overview] and jsonpath template [https://kubernetes.io/docs/reference/kubectl/jsonpath/].`, strings.Join(f.AllowedFormats(), ", ")))
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
	if f.NoHeaders != nil {
		cmd.Flags().BoolVar(f.NoHeaders, "no-headers", *f.NoHeaders, "When using the default or custom-column output format, don't print headers (default print headers).")
	}
}

// NewGetPrintFlags returns flags associated with humanreadable,
// template, and "name" printing, with default values set.
func NewGetPrintFlags() *PrintFlags {
	outputFormat := ""
	noHeaders := false

	return &PrintFlags{
		OutputFormat: &outputFormat,
		NoHeaders:    &noHeaders,

		JSONYamlPrintFlags: genericclioptions.NewJSONYamlPrintFlags(),
		NamePrintFlags:     genericclioptions.NewNamePrintFlags(""),
		TemplateFlags:      genericclioptions.NewKubeTemplatePrintFlags(),

		HumanReadableFlags: NewHumanPrintFlags(),
		CustomColumnsFlags: NewCustomColumnsPrintFlags(),
	}
}
