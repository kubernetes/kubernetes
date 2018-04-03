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
	"github.com/spf13/cobra"

	"k8s.io/kubernetes/pkg/printers"
)

// KubeTemplatePrintFlags composes print flags that provide both a JSONPath and a go-template printer.
// This is necessary if dealing with cases that require support both both printers, since both sets of flags
// require overlapping flags.
type KubeTemplatePrintFlags struct {
	*GoTemplatePrintFlags
	*JSONPathPrintFlags

	// printer-specific values
	AllowMissingKeys *bool

	// must be set through the Complete method, or
	// while instantiating this struct.
	TemplateArgument string
}

func (f *KubeTemplatePrintFlags) Complete(templateValue string) {
	f.TemplateArgument = templateValue
}

func (f *KubeTemplatePrintFlags) ToPrinter(outputFormat string) (printers.ResourcePrinter, bool, error) {
	if p, match, err := f.JSONPathPrintFlags.ToPrinter(outputFormat); match {
		return p, match, err
	}
	return f.GoTemplatePrintFlags.ToPrinter(outputFormat)
}

// AddFlags receives a *cobra.Command reference and binds
// flags related to template printing to it
func (f *KubeTemplatePrintFlags) AddFlags(c *cobra.Command) {
	if f.AllowMissingKeys != nil {
		c.Flags().BoolVar(f.AllowMissingKeys, "allow-missing-template-keys", *f.AllowMissingKeys, "If true, ignore any errors in templates when a field or map key is missing in the template. Only applies to golang and jsonpath output formats.")
	}
}

// NewKubeTemplatePrintFlags returns flags associated with
// --template printing, with default values set.
func NewKubeTemplatePrintFlags(templateValue string) *KubeTemplatePrintFlags {
	allowMissingKeysPtr := true

	return &KubeTemplatePrintFlags{
		GoTemplatePrintFlags: &GoTemplatePrintFlags{
			TemplateArgument: &templateValue,
			AllowMissingKeys: &allowMissingKeysPtr,
		},
		JSONPathPrintFlags: &JSONPathPrintFlags{
			TemplateArgument: &templateValue,
			AllowMissingKeys: &allowMissingKeysPtr,
		},

		TemplateArgument: templateValue,
		AllowMissingKeys: &allowMissingKeysPtr,
	}
}
