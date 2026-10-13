/*
Copyright 2025 The Kubernetes Authors.

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

package apiresources

import (
	"fmt"
	"io"
	"strings"

	"github.com/spf13/cobra"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/printers"
)

type PrintFlags struct {
	JSONYamlPrintFlags *genericclioptions.JSONYamlPrintFlags
	NamePrintFlags     NamePrintFlags
	HumanReadableFlags HumanPrintFlags

	NoHeaders    *bool
	OutputFormat *string
}

func NewPrintFlags() *PrintFlags {
	outputFormat := ""
	noHeaders := false

	return &PrintFlags{
		OutputFormat:       &outputFormat,
		NoHeaders:          &noHeaders,
		JSONYamlPrintFlags: genericclioptions.NewJSONYamlPrintFlags(),
		NamePrintFlags:     APIResourcesNewNamePrintFlags(),
		HumanReadableFlags: APIResourcesHumanReadableFlags(),
	}
}

func (f *PrintFlags) AddFlags(cmd *cobra.Command) {
	f.JSONYamlPrintFlags.AddFlags(cmd)
	f.HumanReadableFlags.AddFlags(cmd)
	f.NamePrintFlags.AddFlags(cmd)

	if f.OutputFormat != nil {
		cmd.Flags().StringVarP(f.OutputFormat, "output", "o", *f.OutputFormat, fmt.Sprintf("Output format. One of: (%s).", strings.Join(f.AllowedFormats(), ", ")))
	}
	if f.NoHeaders != nil {
		cmd.Flags().BoolVar(f.NoHeaders, "no-headers", *f.NoHeaders, "When using the default or custom-column output format, don't print headers (default print headers).")
	}
}

// PrintOptions struct defines a struct for various print options
type PrintOptions struct {
	SortBy    *string
	NoHeaders bool
	Wide      bool
}

type HumanPrintFlags struct {
	SortBy    *string
	NoHeaders bool
}

func (f *HumanPrintFlags) AllowedFormats() []string {
	return []string{"wide"}
}

// AddFlags receives a *cobra.Command reference and binds
// flags related to human-readable printing to it
func (f *HumanPrintFlags) AddFlags(c *cobra.Command) {
	if f.SortBy != nil {
		c.Flags().StringVar(f.SortBy, "sort-by", *f.SortBy, "If non-empty, sort list types using this field specification.  The field specification is expressed as a JSONPath expression (e.g. '{.metadata.name}'). The field in the API resource specified by this JSONPath expression must be an integer or a string.")
	}
}

// ToPrinter receives an outputFormat and returns a printer capable of
// handling human-readable output.
func (f *HumanPrintFlags) ToPrinter(outputFormat string) (printers.ResourcePrinter, error) {
	if len(outputFormat) > 0 && outputFormat != "wide" {
		return nil, genericclioptions.NoCompatiblePrinterError{Options: f, AllowedFormats: f.AllowedFormats()}
	}

	p := HumanReadablePrinter{
		options: PrintOptions{
			NoHeaders: f.NoHeaders,
			Wide:      outputFormat == "wide",
		},
	}

	return p, nil
}

type HumanReadablePrinter struct {
	options PrintOptions
}

func (f HumanReadablePrinter) PrintObj(obj runtime.Object, w io.Writer) error {
	flatList, ok := obj.(*metav1.APIResourceList)
	if !ok {
		return fmt.Errorf("object is not a APIResourceList")
	}
	var errs []error
	for _, r := range flatList.APIResources {
		gv, err := schema.ParseGroupVersion(strings.Join([]string{r.Group, r.Version}, "/"))
		if err != nil {
			errs = append(errs, err)
			continue
		}
		if f.options.Wide {
			if _, err := fmt.Fprintf(w, "%s\t%s\t%s\t%v\t%s\t%v\t%v\n",
				r.Name,
				strings.Join(r.ShortNames, ","),
				gv.String(),
				r.Namespaced,
				r.Kind,
				strings.Join(r.Verbs, ","),
				strings.Join(r.Categories, ",")); err != nil {
				errs = append(errs, err)
			}
			continue
		}
		if _, err := fmt.Fprintf(w, "%s\t%s\t%s\t%v\t%s\n",
			r.Name,
			strings.Join(r.ShortNames, ","),
			gv.String(),
			r.Namespaced,
			r.Kind); err != nil {
			errs = append(errs, err)
		}
	}
	return utilerrors.NewAggregate(errs)
}

type NamePrintFlags struct{}

func APIResourcesNewNamePrintFlags() NamePrintFlags {
	return NamePrintFlags{}
}

func (f *NamePrintFlags) AllowedFormats() []string {
	return []string{"name"}
}

// AddFlags receives a *cobra.Command reference and binds
// flags related to name printing to it
func (f *NamePrintFlags) AddFlags(_ *cobra.Command) {}

// ToPrinter receives an outputFormat and returns a printer capable of
// handling human-readable output.
func (f *NamePrintFlags) ToPrinter(outputFormat string) (printers.ResourcePrinter, error) {
	if outputFormat == "name" {
		return NamePrinter{}, nil
	}
	return nil, genericclioptions.NoCompatiblePrinterError{Options: f, AllowedFormats: f.AllowedFormats()}
}

type NamePrinter struct{}

func (f NamePrinter) PrintObj(obj runtime.Object, w io.Writer) error {
	flatList, ok := obj.(*metav1.APIResourceList)
	if !ok {
		return fmt.Errorf("object is not a APIResourceList")
	}
	var errs []error
	for _, r := range flatList.APIResources {
		name := r.Name
		if len(r.Group) > 0 {
			name += "." + r.Group
		}
		if _, err := fmt.Fprintf(w, "%s\n", name); err != nil {
			errs = append(errs, err)
		}
	}
	return utilerrors.NewAggregate(errs)
}

func APIResourcesHumanReadableFlags() HumanPrintFlags {
	return HumanPrintFlags{
		SortBy:    nil,
		NoHeaders: false,
	}
}

func (f *PrintFlags) AllowedFormats() []string {
	ret := []string{}
	ret = append(ret, f.JSONYamlPrintFlags.AllowedFormats()...)
	ret = append(ret, f.NamePrintFlags.AllowedFormats()...)
	ret = append(ret, f.HumanReadableFlags.AllowedFormats()...)
	return ret
}

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

	if p, err := f.JSONYamlPrintFlags.ToPrinter(outputFormat); !genericclioptions.IsNoCompatiblePrinterError(err) {
		return p, err
	}

	if p, err := f.HumanReadableFlags.ToPrinter(outputFormat); !genericclioptions.IsNoCompatiblePrinterError(err) {
		return p, err
	}

	if p, err := f.NamePrintFlags.ToPrinter(outputFormat); !genericclioptions.IsNoCompatiblePrinterError(err) {
		return p, err
	}

	return nil, genericclioptions.NoCompatiblePrinterError{OutputFormat: &outputFormat, AllowedFormats: f.AllowedFormats()}
}
