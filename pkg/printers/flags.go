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

package printers

import (
	"fmt"

	"github.com/spf13/cobra"
)

type NoCompatiblePrinterError struct {
	Options interface{}
}

func (e NoCompatiblePrinterError) Error() string {
	return fmt.Sprintf("unable to match a printer suitable for the options specified: %#v", e.Options)
}

func IsNoCompatiblePrinterError(err error) bool {
	if err == nil {
		return false
	}

	_, ok := err.(NoCompatiblePrinterError)
	return ok
}

// PrintFlags composes common printer flag structs
// used across all commands, and provides a method
// of retrieving a known printer based on flag values provided.
type PrintFlags struct {
	JSONYamlPrintFlags *JSONYamlPrintFlags
	NamePrintFlags     *NamePrintFlags

	OutputFormat *string
}

func (f *PrintFlags) Complete(successTemplate string) error {
	return f.NamePrintFlags.Complete(successTemplate)
}

func (f *PrintFlags) ToPrinter() (ResourcePrinter, error) {
	outputFormat := ""
	if f.OutputFormat != nil {
		outputFormat = *f.OutputFormat
	}

	if p, err := f.JSONYamlPrintFlags.ToPrinter(outputFormat); !IsNoCompatiblePrinterError(err) {
		return p, err
	}

	if p, err := f.NamePrintFlags.ToPrinter(outputFormat); !IsNoCompatiblePrinterError(err) {
		return p, err
	}

	return nil, NoCompatiblePrinterError{f}
}

func (f *PrintFlags) AddFlags(cmd *cobra.Command) {
	f.JSONYamlPrintFlags.AddFlags(cmd)
	f.NamePrintFlags.AddFlags(cmd)

	if f.OutputFormat != nil {
		cmd.Flags().StringVarP(f.OutputFormat, "output", "o", *f.OutputFormat, "Output format. One of: json|yaml|wide|name|custom-columns=...|custom-columns-file=...|go-template=...|go-template-file=...|jsonpath=...|jsonpath-file=... See custom columns [http://kubernetes.io/docs/user-guide/kubectl-overview/#custom-columns], golang template [http://golang.org/pkg/text/template/#pkg-overview] and jsonpath template [http://kubernetes.io/docs/user-guide/jsonpath].")
	}
}

func NewPrintFlags(operation string) *PrintFlags {
	outputFormat := ""

	return &PrintFlags{
		OutputFormat: &outputFormat,

		JSONYamlPrintFlags: NewJSONYamlPrintFlags(),
		NamePrintFlags:     NewNamePrintFlags(operation),
	}
}
