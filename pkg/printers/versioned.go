/*
Copyright 2017 The Kubernetes Authors.

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
	"io"

	"github.com/golang/glog"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

// VersionedPrinter takes runtime objects and ensures they are converted to a given API version
// prior to being passed to a nested printer.
type VersionedPrinter struct {
	printer   ResourcePrinter
	converter runtime.ObjectConvertor
	typer     runtime.ObjectTyper
	versions  []schema.GroupVersion
}

// NewVersionedPrinter wraps a printer to convert objects to a known API version prior to printing.
func NewVersionedPrinter(printer ResourcePrinter, converter runtime.ObjectConvertor, typer runtime.ObjectTyper, versions ...schema.GroupVersion) ResourcePrinter {
	return &VersionedPrinter{
		printer:   printer,
		converter: converter,
		typer:     typer,
		versions:  versions,
	}
}

func (p *VersionedPrinter) AfterPrint(w io.Writer, res string) error {
	return nil
}

// PrintObj implements ResourcePrinter
func (p *VersionedPrinter) PrintObj(obj runtime.Object, w io.Writer) error {
	// if we're unstructured, no conversion necessary
	if _, ok := obj.(*unstructured.Unstructured); ok {
		return p.printer.PrintObj(obj, w)
	}
	// if we aren't a generic printer, we don't convert.  This means the printer must be aware of what it is getting.
	// The default printers fall into this category.
	// TODO eventually, all printers must be generic
	if !p.IsGeneric() {
		return p.printer.PrintObj(obj, w)
	}

	// if we're already external, no conversion necessary
	gvks, _, err := p.typer.ObjectKinds(obj)
	if err != nil {
		glog.V(1).Info("error determining type for %T, using passed object: %v", obj, err)
		return p.printer.PrintObj(obj, w)
	}
	needsConversion := false
	for _, gvk := range gvks {
		if len(gvk.Version) == 0 || gvk.Version == runtime.APIVersionInternal {
			needsConversion = true
		}
	}

	if !needsConversion {
		return p.printer.PrintObj(obj, w)
	}

	if len(p.versions) == 0 {
		return fmt.Errorf("no version specified, object cannot be converted")
	}
	converted, err := p.converter.ConvertToVersion(obj, schema.GroupVersions(p.versions))
	if err != nil {
		return err
	}
	return p.printer.PrintObj(converted, w)
}

// TODO: implement HandledResources()
func (p *VersionedPrinter) HandledResources() []string {
	return []string{}
}

func (p *VersionedPrinter) IsGeneric() bool {
	return p.printer.IsGeneric()
}
