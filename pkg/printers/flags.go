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
	"strings"

	"github.com/spf13/cobra"

	"k8s.io/apimachinery/pkg/runtime"
)

var (
	internalObjectPrinterErr = "a versioned object must be passed to a printer"

	// disallowedPackagePrefixes contains regular expression templates
	// for object package paths that are not allowed by printers.
	disallowedPackagePrefixes = []string{
		"k8s.io/kubernetes/pkg/apis/",
	}
)

var internalObjectPreventer = &illegalPackageSourceChecker{disallowedPackagePrefixes}

type NoCompatiblePrinterError struct {
	OutputFormat *string
	Options      interface{}
}

func (e NoCompatiblePrinterError) Error() string {
	output := ""
	if e.OutputFormat != nil {
		output = *e.OutputFormat
	}

	return fmt.Sprintf("unable to match a printer suitable for the output format %q and the options specified: %#v", output, e.Options)
}

func IsNoCompatiblePrinterError(err error) bool {
	if err == nil {
		return false
	}

	_, ok := err.(NoCompatiblePrinterError)
	return ok
}

func IsInternalObjectError(err error) bool {
	if err == nil {
		return false
	}

	return err.Error() == internalObjectPrinterErr
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

	if f.JSONYamlPrintFlags != nil {
		if p, err := f.JSONYamlPrintFlags.ToPrinter(outputFormat); !IsNoCompatiblePrinterError(err) {
			return p, err
		}
	}

	if f.NamePrintFlags != nil {
		if p, err := f.NamePrintFlags.ToPrinter(outputFormat); !IsNoCompatiblePrinterError(err) {
			return p, err
		}
	}

	return nil, NoCompatiblePrinterError{Options: f, OutputFormat: f.OutputFormat}
}

func (f *PrintFlags) AddFlags(cmd *cobra.Command) {
	f.JSONYamlPrintFlags.AddFlags(cmd)
	f.NamePrintFlags.AddFlags(cmd)

	if f.OutputFormat != nil {
		cmd.Flags().StringVarP(f.OutputFormat, "output", "o", *f.OutputFormat, "Output format. One of: json|yaml|wide|name|custom-columns=...|custom-columns-file=...|go-template=...|go-template-file=...|jsonpath=...|jsonpath-file=... See custom columns [http://kubernetes.io/docs/user-guide/kubectl-overview/#custom-columns], golang template [http://golang.org/pkg/text/template/#pkg-overview] and jsonpath template [http://kubernetes.io/docs/user-guide/jsonpath].")
	}
}

// WithDefaultOutput sets a default output format if one is not provided through a flag value
func (f *PrintFlags) WithDefaultOutput(output string) *PrintFlags {
	f.OutputFormat = &output
	return f
}

func NewPrintFlags(operation string, scheme runtime.ObjectConvertor) *PrintFlags {
	outputFormat := ""

	return &PrintFlags{
		OutputFormat: &outputFormat,

		JSONYamlPrintFlags: NewJSONYamlPrintFlags(scheme),
		NamePrintFlags:     NewNamePrintFlags(operation, scheme),
	}
}

// illegalPackageSourceChecker compares a given
// object's package path, and determines if the
// object originates from a disallowed source.
type illegalPackageSourceChecker struct {
	// disallowedPrefixes is a slice of disallowed package path
	// prefixes for a given runtime.Object that we are printing.
	disallowedPrefixes []string
}

func (c *illegalPackageSourceChecker) IsForbidden(pkgPath string) bool {
	for _, forbiddenPrefix := range c.disallowedPrefixes {
		if strings.HasPrefix(pkgPath, forbiddenPrefix) {
			return true
		}
	}

	return false
}
