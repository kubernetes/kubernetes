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
	"reflect"
	"strings"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

// NamePrinter is an implementation of ResourcePrinter which outputs "resource/name" pair of an object.
type NamePrinter struct {
	// ShortOutput indicates whether an operation should be
	// printed along side the "resource/name" pair for an object.
	ShortOutput bool
	// Operation describes the name of the action that
	// took place on an object, to be included in the
	// finalized "successful" message.
	Operation string
}

// PrintObj is an implementation of ResourcePrinter.PrintObj which decodes the object
// and print "resource/name" pair. If the object is a List, print all items in it.
func (p *NamePrinter) PrintObj(obj runtime.Object, w io.Writer) error {
	// we use reflect.Indirect here in order to obtain the actual value from a pointer.
	// using reflect.Indirect indiscriminately is valid here, as all runtime.Objects are supposed to be pointers.
	// we need an actual value in order to retrieve the package path for an object.
	if InternalObjectPreventer.IsForbidden(reflect.Indirect(reflect.ValueOf(obj)).Type().PkgPath()) {
		return fmt.Errorf(InternalObjectPrinterErr)
	}

	if meta.IsListType(obj) {
		// we allow unstructured lists for now because they always contain the GVK information.  We should chase down
		// callers and stop them from passing unflattened lists
		// TODO chase the caller that is setting this and remove it.
		if _, ok := obj.(*unstructured.UnstructuredList); !ok {
			return fmt.Errorf("list types are not supported by name printing: %T", obj)
		}

		items, err := meta.ExtractList(obj)
		if err != nil {
			return err
		}
		for _, obj := range items {
			if err := p.PrintObj(obj, w); err != nil {
				return err
			}
		}
		return nil
	}

	if obj.GetObjectKind().GroupVersionKind().Empty() {
		return fmt.Errorf("missing apiVersion or kind; try GetObjectKind().SetGroupVersionKind() if you know the type")
	}

	name := "<unknown>"
	if acc, err := meta.Accessor(obj); err == nil {
		if n := acc.GetName(); len(n) > 0 {
			name = n
		}
	}

	return printObj(w, name, p.Operation, p.ShortOutput, GetObjectGroupKind(obj))
}

func GetObjectGroupKind(obj runtime.Object) schema.GroupKind {
	if obj == nil {
		return schema.GroupKind{Kind: "<unknown>"}
	}
	groupVersionKind := obj.GetObjectKind().GroupVersionKind()
	if len(groupVersionKind.Kind) > 0 {
		return groupVersionKind.GroupKind()
	}

	if uns, ok := obj.(*unstructured.Unstructured); ok {
		if len(uns.GroupVersionKind().Kind) > 0 {
			return uns.GroupVersionKind().GroupKind()
		}
	}

	return schema.GroupKind{Kind: "<unknown>"}
}

func printObj(w io.Writer, name string, operation string, shortOutput bool, groupKind schema.GroupKind) error {
	if len(groupKind.Kind) == 0 {
		return fmt.Errorf("missing kind for resource with name %v", name)
	}

	if len(operation) > 0 {
		operation = " " + operation
	}

	if shortOutput {
		operation = ""
	}

	if len(groupKind.Group) == 0 {
		fmt.Fprintf(w, "%s/%s%s\n", strings.ToLower(groupKind.Kind), name, operation)
		return nil
	}

	fmt.Fprintf(w, "%s.%s/%s%s\n", strings.ToLower(groupKind.Kind), groupKind.Group, name, operation)
	return nil
}
