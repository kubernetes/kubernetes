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
	"github.com/spf13/cobra"
	"k8s.io/cli-runtime/pkg/genericclioptions"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/cli-runtime/pkg/printers"
)

// HumanPrintFlags provides default flags necessary for printing.
// Given the following flag values, a printer can be requested that knows
// how to handle printing based on these values.
type HumanPrintFlags struct {
	ShowKind     *bool
	ShowLabels   *bool
	SortBy       *string
	ColumnLabels *[]string

	// get.go-specific values
	NoHeaders bool

	Kind          schema.GroupKind
	WithNamespace bool
}

// SetKind sets the Kind option
func (f *HumanPrintFlags) SetKind(kind schema.GroupKind) {
	f.Kind = kind
}

// EnsureWithKind sets the "Showkind" humanreadable option to true.
func (f *HumanPrintFlags) EnsureWithKind() error {
	showKind := true
	f.ShowKind = &showKind
	return nil
}

// EnsureWithNamespace sets the "WithNamespace" humanreadable option to true.
func (f *HumanPrintFlags) EnsureWithNamespace() error {
	f.WithNamespace = true
	return nil
}

// AllowedFormats returns more customized formatting options
func (f *HumanPrintFlags) AllowedFormats() []string {
	return []string{"wide"}
}

// ToPrinter receives an outputFormat and returns a printer capable of
// handling human-readable output.
func (f *HumanPrintFlags) ToPrinter(outputFormat string) (printers.ResourcePrinter, error) {
	if len(outputFormat) > 0 && outputFormat != "wide" {
		return nil, genericclioptions.NoCompatiblePrinterError{Options: f, AllowedFormats: f.AllowedFormats()}
	}

	showKind := false
	if f.ShowKind != nil {
		showKind = *f.ShowKind
	}

	showLabels := false
	if f.ShowLabels != nil {
		showLabels = *f.ShowLabels
	}

	columnLabels := []string{}
	if f.ColumnLabels != nil {
		columnLabels = *f.ColumnLabels
	}

	p := printers.NewTablePrinter(printers.PrintOptions{
		Kind:          f.Kind,
		WithKind:      showKind,
		NoHeaders:     f.NoHeaders,
		Wide:          outputFormat == "wide",
		WithNamespace: f.WithNamespace,
		ColumnLabels:  columnLabels,
		ShowLabels:    showLabels,
	})

	// TODO(juanvallejo): handle sorting here

	return p, nil
}

// AddFlags receives a *cobra.Command reference and binds
// flags related to human-readable printing to it
func (f *HumanPrintFlags) AddFlags(c *cobra.Command) {
	if f.ShowLabels != nil {
		c.Flags().BoolVar(f.ShowLabels, "show-labels", *f.ShowLabels, "When printing, show all labels as the last column (default hide labels column)")
	}
	if f.SortBy != nil {
		c.Flags().StringVar(f.SortBy, "sort-by", *f.SortBy, "If non-empty, sort list types using this field specification.  The field specification is expressed as a JSONPath expression (e.g. '{.metadata.name}'). The field in the API resource specified by this JSONPath expression must be an integer or a string.")
	}
	if f.ColumnLabels != nil {
		c.Flags().StringSliceVarP(f.ColumnLabels, "label-columns", "L", *f.ColumnLabels, "Accepts a comma separated list of labels that are going to be presented as columns. Names are case-sensitive. You can also use multiple flag options like -L label1 -L label2...")
	}
	if f.ShowKind != nil {
		c.Flags().BoolVar(f.ShowKind, "show-kind", *f.ShowKind, "If present, list the resource type for the requested object(s).")
	}
}

// NewHumanPrintFlags returns flags associated with
// human-readable printing, with default values set.
func NewHumanPrintFlags() *HumanPrintFlags {
	showLabels := false
	sortBy := ""
	showKind := false
	columnLabels := []string{}

	return &HumanPrintFlags{
		NoHeaders:     false,
		WithNamespace: false,
		ColumnLabels:  &columnLabels,

		Kind:       schema.GroupKind{},
		ShowLabels: &showLabels,
		SortBy:     &sortBy,
		ShowKind:   &showKind,
	}
}
