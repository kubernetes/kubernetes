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
	"os"
	"strings"

	"github.com/spf13/cobra"

	"k8s.io/kubernetes/pkg/kubectl/scheme"
)

// CustomColumnsPrintFlags provides default flags necessary for printing
// custom resource columns from an inline-template or file.
type CustomColumnsPrintFlags struct {
	NoHeaders        bool
	TemplateArgument string
}

// ToPrinter receives an templateFormat and returns a printer capable of
// handling custom-column printing.
// Returns false if the specified templateFormat does not match a supported format.
// Supported format types can be found in pkg/printers/printers.go
func (f *CustomColumnsPrintFlags) ToPrinter(templateFormat string) (ResourcePrinter, bool, error) {
	if len(templateFormat) == 0 {
		return nil, false, fmt.Errorf("missing output format")
	}

	templateValue := ""

	supportedFormats := map[string]bool{
		"custom-columns-file": true,
		"custom-columns":      true,
	}

	if len(f.TemplateArgument) == 0 {
		for format := range supportedFormats {
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

	if _, supportedFormat := supportedFormats[templateFormat]; !supportedFormat {
		return nil, false, nil
	}

	if len(templateValue) == 0 {
		return nil, true, fmt.Errorf("custom-columns format specified but no custom columns given")
	}

	decoder := scheme.Codecs.UniversalDecoder()

	if templateFormat == "custom-columns-file" {
		file, err := os.Open(templateValue)
		if err != nil {
			return nil, true, fmt.Errorf("error reading template %s, %v\n", templateValue, err)
		}
		defer file.Close()
		p, err := NewCustomColumnsPrinterFromTemplate(file, decoder)
		return p, true, err
	}

	p, err := NewCustomColumnsPrinterFromSpec(templateValue, decoder, f.NoHeaders)
	return p, true, err
}

// AddFlags receives a *cobra.Command reference and binds
// flags related to custom-columns printing
func (f *CustomColumnsPrintFlags) AddFlags(c *cobra.Command) {}

// NewCustomColumnsPrintFlags returns flags associated with
// custom-column printing, with default values set.
// NoHeaders and TemplateArgument should be set by callers.
func NewCustomColumnsPrintFlags(noHeaders bool, templateValue string) *CustomColumnsPrintFlags {
	return &CustomColumnsPrintFlags{
		NoHeaders:        noHeaders,
		TemplateArgument: templateValue,
	}
}
