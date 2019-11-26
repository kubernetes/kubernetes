/*
Copyright 2019 The Kubernetes Authors.

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

package output

import (
	"fmt"
	"io"
	"strings"

	"github.com/spf13/cobra"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/printers"
)

// TextOutput describes the plain text output
const TextOutput = "text"

// TextPrintFlags is an iterface to handle custom text output
type TextPrintFlags interface {
	ToPrinter(outputFormat string) (Printer, error)
}

// PrintFlags composes common printer flag structs
// used across kubeadm commands, and provides a method
// of retrieving a known printer based on flag values provided.
type PrintFlags struct {
	// JSONYamlPrintFlags provides default flags necessary for json/yaml printing.
	JSONYamlPrintFlags *genericclioptions.JSONYamlPrintFlags
	// KubeTemplatePrintFlags composes print flags that provide both a JSONPath and a go-template printer.
	KubeTemplatePrintFlags *genericclioptions.KubeTemplatePrintFlags
	// JSONYamlPrintFlags provides default flags necessary for kubeadm text printing.
	TextPrintFlags TextPrintFlags
	// TypeSetterPrinter is an implementation of ResourcePrinter that wraps another printer with types set on the objects
	TypeSetterPrinter *printers.TypeSetterPrinter
	// OutputFormat contains currently set output format
	OutputFormat *string
}

// AllowedFormats returns list of allowed output formats
func (pf *PrintFlags) AllowedFormats() []string {
	ret := []string{TextOutput}
	ret = append(ret, pf.JSONYamlPrintFlags.AllowedFormats()...)
	ret = append(ret, pf.KubeTemplatePrintFlags.AllowedFormats()...)

	return ret
}

// ToPrinter receives an outputFormat and returns a printer capable of
// handling format printing.
// Returns error if the specified outputFormat does not match supported formats.
func (pf *PrintFlags) ToPrinter() (Printer, error) {
	outputFormat := ""
	if pf.OutputFormat != nil {
		outputFormat = *pf.OutputFormat
	}

	if pf.TextPrintFlags != nil {
		if p, err := pf.TextPrintFlags.ToPrinter(outputFormat); !genericclioptions.IsNoCompatiblePrinterError(err) {
			return p, err
		}
	}

	if pf.JSONYamlPrintFlags != nil {
		if p, err := pf.JSONYamlPrintFlags.ToPrinter(outputFormat); !genericclioptions.IsNoCompatiblePrinterError(err) {
			return NewResourcePrinterWrapper(pf.TypeSetterPrinter.WrapToPrinter(p, err))
		}
	}

	if pf.KubeTemplatePrintFlags != nil {
		if p, err := pf.KubeTemplatePrintFlags.ToPrinter(outputFormat); !genericclioptions.IsNoCompatiblePrinterError(err) {
			return NewResourcePrinterWrapper(pf.TypeSetterPrinter.WrapToPrinter(p, err))
		}
	}

	return nil, genericclioptions.NoCompatiblePrinterError{OutputFormat: pf.OutputFormat, AllowedFormats: pf.AllowedFormats()}
}

// AddFlags receives a *cobra.Command reference and binds
// flags related to Kubeadm printing to it
func (pf *PrintFlags) AddFlags(cmd *cobra.Command) {
	pf.JSONYamlPrintFlags.AddFlags(cmd)
	pf.KubeTemplatePrintFlags.AddFlags(cmd)
	cmd.Flags().StringVarP(pf.OutputFormat, "experimental-output", "o", *pf.OutputFormat, fmt.Sprintf("Output format. One of: %s.", strings.Join(pf.AllowedFormats(), "|")))
}

// WithDefaultOutput sets a default output format if one is not provided through a flag value
func (pf *PrintFlags) WithDefaultOutput(outputFormat string) *PrintFlags {
	pf.OutputFormat = &outputFormat
	return pf
}

// WithTypeSetter sets a wrapper than will surround the returned printer with a printer to type resources
func (pf *PrintFlags) WithTypeSetter(scheme *runtime.Scheme) *PrintFlags {
	pf.TypeSetterPrinter = printers.NewTypeSetter(scheme)
	return pf
}

// NewOutputFlags creates new KubeadmOutputFlags
func NewOutputFlags(textPrintFlags TextPrintFlags) *PrintFlags {
	outputFormat := ""

	pf := &PrintFlags{
		OutputFormat: &outputFormat,

		JSONYamlPrintFlags:     genericclioptions.NewJSONYamlPrintFlags(),
		KubeTemplatePrintFlags: genericclioptions.NewKubeTemplatePrintFlags(),
		TextPrintFlags:         textPrintFlags,
	}

	// disable deprecated --template option
	pf.KubeTemplatePrintFlags.TemplateArgument = nil

	return pf
}

// Printer is a common printing interface in Kubeadm
type Printer interface {
	PrintObj(obj runtime.Object, writer io.Writer) error
	Fprintf(writer io.Writer, format string, args ...interface{}) (n int, err error)
	Printf(format string, args ...interface{}) (n int, err error)
}

// TextPrinter implements Printer interface for generic text output
type TextPrinter struct {
}

// PrintObj is an implementation of ResourcePrinter.PrintObj that prints object
func (tp *TextPrinter) PrintObj(obj runtime.Object, writer io.Writer) error {
	_, err := fmt.Fprintf(writer, "%+v\n", obj)
	return err
}

// Fprintf is a wrapper around fmt.Fprintf
func (tp *TextPrinter) Fprintf(writer io.Writer, format string, args ...interface{}) (n int, err error) {
	return fmt.Fprintf(writer, format, args...)
}

// Printf is a wrapper around fmt.Printf
func (tp *TextPrinter) Printf(format string, args ...interface{}) (n int, err error) {
	return fmt.Printf(format, args...)
}

// ResourcePrinterWrapper wraps ResourcePrinter and implements Printer interface
type ResourcePrinterWrapper struct {
	Printer printers.ResourcePrinter
}

// NewResourcePrinterWrapper creates new ResourcePrinter object
func NewResourcePrinterWrapper(resourcePrinter printers.ResourcePrinter, err error) (Printer, error) {
	if err != nil {
		return nil, err
	}
	return &ResourcePrinterWrapper{Printer: resourcePrinter}, nil
}

// PrintObj is an implementation of ResourcePrinter.PrintObj that calls underlying printer API
func (rpw *ResourcePrinterWrapper) PrintObj(obj runtime.Object, writer io.Writer) error {
	return rpw.Printer.PrintObj(obj, writer)
}

// Fprintf is an empty method to satisfy Printer interface
// and silent info printing for structured output
// This method is usually redefined for the text output
func (rpw *ResourcePrinterWrapper) Fprintf(writer io.Writer, format string, args ...interface{}) (n int, err error) {
	return 0, nil
}

// Printf is an empty method to satisfy Printer interface
// and silent info printing for structured output
// This method is usually redefined for the text output
func (rpw *ResourcePrinterWrapper) Printf(format string, args ...interface{}) (n int, err error) {
	return 0, nil
}
