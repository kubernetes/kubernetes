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
	"io"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

// TypeSetterPrinter is an implementation of ResourcePrinter wraps another printer with types set on the objects
type TypeSetterPrinter struct {
	Delegate ResourcePrinter

	Typer runtime.ObjectTyper
}

// NewTypeSetter constructs a wrapping printer with required params
func NewTypeSetter(typer runtime.ObjectTyper) *TypeSetterPrinter {
	return &TypeSetterPrinter{Typer: typer}
}

// PrintObj is an implementation of ResourcePrinter.PrintObj which sets type information on the obj for the duration
// of printing.  It is NOT threadsafe.
func (p *TypeSetterPrinter) PrintObj(obj runtime.Object, w io.Writer) error {
	if obj == nil {
		return p.Delegate.PrintObj(obj, w)
	}
	if !obj.GetObjectKind().GroupVersionKind().Empty() {
		return p.Delegate.PrintObj(obj, w)
	}

	// we were empty coming in, make sure we're empty going out.  This makes the call thread-unsafe
	defer func() {
		obj.GetObjectKind().SetGroupVersionKind(schema.GroupVersionKind{})
	}()

	gvks, _, err := p.Typer.ObjectKinds(obj)
	if err != nil {
		// printers wrapped by us expect to find the type information present
		return fmt.Errorf("missing apiVersion or kind and cannot assign it; %v", err)
	}

	for _, gvk := range gvks {
		if len(gvk.Kind) == 0 {
			continue
		}
		if len(gvk.Version) == 0 || gvk.Version == runtime.APIVersionInternal {
			continue
		}
		obj.GetObjectKind().SetGroupVersionKind(gvk)
		break
	}

	return p.Delegate.PrintObj(obj, w)
}

// ToPrinter returns a printer (not threadsafe!) that has been wrapped
func (p *TypeSetterPrinter) ToPrinter(delegate ResourcePrinter) ResourcePrinter {
	if p == nil {
		return delegate
	}

	p.Delegate = delegate
	return p
}

// WrapToPrinter wraps the common ToPrinter method
func (p *TypeSetterPrinter) WrapToPrinter(delegate ResourcePrinter, err error) (ResourcePrinter, error) {
	if err != nil {
		return delegate, err
	}
	if p == nil {
		return delegate, nil
	}

	p.Delegate = delegate
	return p, nil
}
