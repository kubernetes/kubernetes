/*
Copyright 2016 The Kubernetes Authors.

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

package metricsutil

import (
	"io"

	v1 "k8s.io/api/core/v1"

	"github.com/spf13/cobra"

	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/printers"
)

type PodPrintFlags struct {
	SortBy          string
	PrintContainers bool
	PrintNamespaces bool
	NoHeaders       bool
	Sum             bool

	PodsInfo *[]v1.Pod

	// Not used, just for create TopCmdPrinter.
	out io.Writer
}

// EnsureWithSort sets the "SortBy" option.
func (f *PodPrintFlags) EnsureWithSort(sortBy string) error {
	f.SortBy = sortBy
	return nil
}

// EnsureWithContainers sets the "PrintContainers" option to true.
func (f *PodPrintFlags) EnsureWithContainers() error {
	f.PrintContainers = true
	return nil
}

// EnsureWithNamespaces sets the "PrintNamespaces" option to true.
func (f *PodPrintFlags) EnsureWithNamespaces() error {
	f.PrintNamespaces = true
	return nil
}

// EnsureWithNoHeaders sets the "NoHeaders" option to true.
func (f *PodPrintFlags) EnsureWithNoHeaders() error {
	f.NoHeaders = true
	return nil
}

// EnsureWithSum sets the "Sum" option to true.
func (f *PodPrintFlags) EnsureWithSum() error {
	f.Sum = true
	return nil
}

// SetPodsInfo sets the "PodsInfo" option.
func (f *PodPrintFlags) SetPodsInfo(pods *[]v1.Pod) error {
	f.PodsInfo = pods
	return nil
}

// AllowedFormats returns more customized formating options
func (f *PodPrintFlags) AllowedFormats() []string {
	return []string{"wide"}
}

// ToPrinter receives an outputFormat and returns a printer capable of handling output.
func (f *PodPrintFlags) ToPrinter(outputFormat string) (printers.ResourcePrinter, error) {
	if len(outputFormat) > 0 && outputFormat != "wide" {
		return nil, genericclioptions.NoCompatiblePrinterError{Options: f, AllowedFormats: f.AllowedFormats()}
	}

	podPrintOptions := &PodPrintOptions{
		SortBy:          f.SortBy,
		PrintContainers: f.PrintContainers,
		PrintNamespaces: f.PrintNamespaces,
		NoHeaders:       f.NoHeaders,
		Sum:             f.Sum,
		Wide:            outputFormat == "wide",
		PodsInfo:        f.PodsInfo,
	}
	p := &TopCmdPrinter{out: f.out, PodPrintOptions: podPrintOptions}

	return p, nil
}

// AddFlags receives a *cobra.Command reference and binds flags related to printing to it
func (f *PodPrintFlags) AddFlags(c *cobra.Command) {}

// NewPodPrintFlags returns flags associated with printing, with default values set.
func NewPodPrintFlags(out io.Writer) *PodPrintFlags {
	return &PodPrintFlags{out: out}
}
