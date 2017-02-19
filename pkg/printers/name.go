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

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
)

// NamePrinter is an implementation of ResourcePrinter which outputs "resource/name" pair of an object.
type NamePrinter struct {
	Decoder runtime.Decoder
	Typer   runtime.ObjectTyper
}

func (p *NamePrinter) AfterPrint(w io.Writer, res string) error {
	return nil
}

// PrintObj is an implementation of ResourcePrinter.PrintObj which decodes the object
// and print "resource/name" pair. If the object is a List, print all items in it.
func (p *NamePrinter) PrintObj(obj runtime.Object, w io.Writer) error {
	if meta.IsListType(obj) {
		items, err := meta.ExtractList(obj)
		if err != nil {
			return err
		}
		if errs := runtime.DecodeList(items, p.Decoder, unstructured.UnstructuredJSONScheme); len(errs) > 0 {
			return utilerrors.NewAggregate(errs)
		}
		for _, obj := range items {
			if err := p.PrintObj(obj, w); err != nil {
				return err
			}
		}
		return nil
	}

	name := "<unknown>"
	if acc, err := meta.Accessor(obj); err == nil {
		if n := acc.GetName(); len(n) > 0 {
			name = n
		}
	}

	if kind := obj.GetObjectKind().GroupVersionKind(); len(kind.Kind) == 0 {
		// this is the old code.  It's unnecessary on decoded external objects, but on internal objects
		// you may have to do it.  Tests are definitely calling it with internals and I'm not sure who else
		// is
		if gvks, _, err := p.Typer.ObjectKinds(obj); err == nil {
			// TODO: this is wrong, it assumes that meta knows about all Kinds - should take a RESTMapper
			_, resource := meta.KindToResource(gvks[0])
			fmt.Fprintf(w, "%s/%s\n", resource.Resource, name)
		} else {
			fmt.Fprintf(w, "<unknown>/%s\n", name)
		}

	} else {
		// TODO: this is wrong, it assumes that meta knows about all Kinds - should take a RESTMapper
		_, resource := meta.KindToResource(kind)
		fmt.Fprintf(w, "%s/%s\n", resource.Resource, name)
	}

	return nil
}

// TODO: implement HandledResources()
func (p *NamePrinter) HandledResources() []string {
	return []string{}
}
