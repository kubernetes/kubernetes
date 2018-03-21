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

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"

	"k8s.io/kubernetes/pkg/kubectl/scheme"
)

// NamePrintFlags provides default flags necessary for printing
// a resource's fully-qualified Kind.group/name, or a successful
// message about that resource if an Operation is provided.
type NamePrintFlags struct {
	// DryRun indicates whether the "(dry run)" message
	// should be appended to the finalized "successful"
	// message printed about an action on an object.
	DryRun bool
	// Operation describes the name of the action that
	// took place on an object, to be included in the
	// finalized "successful" message.
	Operation string
}

// ToPrinter receives an outputFormat and returns a printer capable of
// handling --output=name printing.
// Returns false if the specified outputFormat does not match a supported format.
// Supported format types can be found in pkg/printers/printers.go
func (f *NamePrintFlags) ToPrinter(outputFormat string) (ResourcePrinter, bool, error) {
	decoders := []runtime.Decoder{scheme.Codecs.UniversalDecoder(), unstructured.UnstructuredJSONScheme}
	namePrinter := &NamePrinter{
		Operation: f.Operation,
		DryRun:    f.DryRun,
		Typer:     scheme.Scheme,
		Decoders:  decoders,
	}

	outputFormat = strings.ToLower(outputFormat)
	switch outputFormat {
	case "name":
		return namePrinter, true, nil
	case "":
		return nil, false, fmt.Errorf("missing output format")
	default:
		return nil, false, nil
	}
}

// AddFlags receives a *cobra.Command reference and binds
// flags related to name printing to it
func (f *NamePrintFlags) AddFlags(c *cobra.Command) {}

// NewNamePrintFlags returns flags associated with
// --name printing, with default values set.
func NewNamePrintFlags(operation string, dryRun bool) *NamePrintFlags {
	return &NamePrintFlags{
		Operation: operation,
		DryRun:    dryRun,
	}
}
