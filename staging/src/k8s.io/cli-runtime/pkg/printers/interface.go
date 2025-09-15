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
	"io"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

// ResourcePrinterFunc is a function that can print objects
type ResourcePrinterFunc func(runtime.Object, io.Writer) error

// PrintObj implements ResourcePrinter
func (fn ResourcePrinterFunc) PrintObj(obj runtime.Object, w io.Writer) error {
	return fn(obj, w)
}

// ResourcePrinter is an interface that knows how to print runtime objects.
type ResourcePrinter interface {
	// PrintObj receives a runtime object, formats it and prints it to a writer.
	PrintObj(runtime.Object, io.Writer) error
}

// PrintOptions struct defines a struct for various print options
type PrintOptions struct {
	NoHeaders     bool
	WithNamespace bool
	WithKind      bool
	Wide          bool
	ShowLabels    bool
	Kind          schema.GroupKind
	ColumnLabels  []string

	SortBy string

	// indicates if it is OK to ignore missing keys for rendering an output template.
	AllowMissingKeys bool
}
