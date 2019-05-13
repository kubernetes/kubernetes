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
	"os"
	"strings"

	"github.com/spf13/cobra"

	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/printers"
	"k8s.io/kubernetes/pkg/kubectl/scheme"
)

var columnsFormats = map[string]bool{
	"custom-columns-file": true,
	"custom-columns":      true,
}

// CustomColumnsPrintFlags provides default flags necessary for printing
// custom resource columns from an inline-template or file.
type CustomColumnsPrintFlags struct {
	NoHeaders        bool
	TemplateArgument string
}

func (f *CustomColumnsPrintFlags) AllowedFormats() []string {
	formats := make([]string, 0, len(columnsFormats))
	for format := range columnsFormats {
		formats = append(formats, format)
	}
	return formats
}

// ToPrinter receives an templateFormat and returns a printer capable of
// handling custom-column printing.
// Returns false if the specified templateFormat does not match a supported format.
// Supported format types can be found in pkg/printers/printers.go
func (f *CustomColumnsPrintFlags) ToPrinter(templateFormat string) (printers.ResourcePrinter, error) {
	if len(templateFormat) == 0 {
		return nil, genericclioptions.NoCompatiblePrinterError{}
	}

	templateValue := ""

	if len(f.TemplateArgument) == 0 {
		for format := range columnsFormats {
			format = format + "="
			if strings.HasPrefix(templateFormat, format) {
				templateValue = templateFormat[len(format):]
				templateFormat = format[:len(format)-1]
				break
			}
		}
	} else {
		templateValue = f.TemplateArgument
	}

	if _, supportedFormat := columnsFormats[templateFormat]; !supportedFormat {
		return nil, genericclioptions.NoCompatiblePrinterError{OutputFormat: &templateFormat, AllowedFormats: f.AllowedFormats()}
	}

	if len(templateValue) == 0 {
		return nil, fmt.Errorf("custom-columns format specified but no custom columns given")
	}

	// UniversalDecoder call must specify parameter versions; otherwise it will decode to internal versions.
	decoder := scheme.Codecs.UniversalDecoder(scheme.Scheme.PrioritizedVersionsAllGroups()...)

	if templateFormat == "custom-columns-file" {
		file, err := os.Open(templateValue)
		if err != nil {
			return nil, fmt.Errorf("error reading template %s, %v\n", templateValue, err)
		}
		defer file.Close()
		p, err := NewCustomColumnsPrinterFromTemplate(file, decoder)
		return p, err
	}

	return NewCustomColumnsPrinterFromSpec(templateValue, decoder, f.NoHeaders)
}

// AddFlags receives a *cobra.Command reference and binds
// flags related to custom-columns printing
func (f *CustomColumnsPrintFlags) AddFlags(c *cobra.Command) {}

// NewCustomColumnsPrintFlags returns flags associated with
// custom-column printing, with default values set.
// NoHeaders and TemplateArgument should be set by callers.
func NewCustomColumnsPrintFlags() *CustomColumnsPrintFlags {
	return &CustomColumnsPrintFlags{
		NoHeaders:        false,
		TemplateArgument: "",
	}
}
