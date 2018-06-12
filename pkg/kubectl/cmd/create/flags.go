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

package create

import (
	"github.com/spf13/cobra"
	"k8s.io/kubernetes/pkg/kubectl/genericclioptions"

	"k8s.io/apimachinery/pkg/runtime"
	genericprinters "k8s.io/kubernetes/pkg/kubectl/genericclioptions/printers"
	"k8s.io/kubernetes/pkg/printers"
)

// PrintFlags composes common printer flag structs
// used across all create commands, and provides a method
// of retrieving a known printer based on flag values provided.
type PrintFlags struct {
	JSONYamlPrintFlags *genericclioptions.JSONYamlPrintFlags
	NamePrintFlags     *genericclioptions.NamePrintFlags
	TemplateFlags      *printers.KubeTemplatePrintFlags

	TypeSetter *genericprinters.TypeSetterPrinter

	OutputFormat *string
}

func (f *PrintFlags) AllowedFormats() []string {
	return append(append(f.JSONYamlPrintFlags.AllowedFormats(), f.NamePrintFlags.AllowedFormats()...),
		f.TemplateFlags.AllowedFormats()...)
}

func (f *PrintFlags) Complete(successTemplate string) error {
	return f.NamePrintFlags.Complete(successTemplate)
}

func (f *PrintFlags) ToPrinter() (printers.ResourcePrinter, error) {
	outputFormat := ""
	if f.OutputFormat != nil {
		outputFormat = *f.OutputFormat
	}

	if p, err := f.JSONYamlPrintFlags.ToPrinter(outputFormat); !genericclioptions.IsNoCompatiblePrinterError(err) {
		return f.TypeSetter.WrapToPrinter(p, err)
	}

	if p, err := f.NamePrintFlags.ToPrinter(outputFormat); !genericclioptions.IsNoCompatiblePrinterError(err) {
		return f.TypeSetter.WrapToPrinter(p, err)
	}

	if p, err := f.TemplateFlags.ToPrinter(outputFormat); !genericclioptions.IsNoCompatiblePrinterError(err) {
		return f.TypeSetter.WrapToPrinter(p, err)
	}

	return nil, genericclioptions.NoCompatiblePrinterError{OutputFormat: &outputFormat, AllowedFormats: f.AllowedFormats()}
}

func (f *PrintFlags) AddFlags(cmd *cobra.Command) {
	f.JSONYamlPrintFlags.AddFlags(cmd)
	f.NamePrintFlags.AddFlags(cmd)
	f.TemplateFlags.AddFlags(cmd)

	if f.OutputFormat != nil {
		cmd.Flags().StringVarP(f.OutputFormat, "output", "o", *f.OutputFormat, "Output format. One of: json|yaml|wide|name|custom-columns=...|custom-columns-file=...|go-template=...|go-template-file=...|jsonpath=...|jsonpath-file=... See custom columns [http://kubernetes.io/docs/user-guide/kubectl-overview/#custom-columns], golang template [http://golang.org/pkg/text/template/#pkg-overview] and jsonpath template [http://kubernetes.io/docs/user-guide/jsonpath].")
	}
}

func NewPrintFlags(operation string, scheme runtime.ObjectTyper) *PrintFlags {
	outputFormat := ""

	return &PrintFlags{
		OutputFormat: &outputFormat,

		JSONYamlPrintFlags: genericclioptions.NewJSONYamlPrintFlags(),
		NamePrintFlags:     genericclioptions.NewNamePrintFlags(operation),
		TemplateFlags:      printers.NewKubeTemplatePrintFlags(),

		TypeSetter: genericprinters.NewTypeSetter(scheme),
	}
}
