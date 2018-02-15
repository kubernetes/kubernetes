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
	"strings"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
)

// NamePrinter is an implementation of ResourcePrinter which outputs "resource/name" pair of an object.
type NamePrinter struct {
	Decoders []runtime.Decoder
	Typer    runtime.ObjectTyper
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
		if errs := runtime.DecodeList(items, p.Decoders...); len(errs) > 0 {
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

	groupVersionKind := obj.GetObjectKind().GroupVersionKind()
	if len(groupVersionKind.Kind) > 0 {
		kind := groupVersionKind.Kind
		group := groupVersionKind.Group
		return printObj(w, name, group, kind)
	}

	if gvks, _, err := p.Typer.ObjectKinds(obj); err == nil {
		for _, gvk := range gvks {
			if len(gvk.Kind) == 0 {
				continue
			}

			return printObj(w, name, gvk.Group, gvk.Kind)
		}
	}

	if uns, ok := obj.(*unstructured.Unstructured); ok {
		group := uns.GroupVersionKind().Group
		kind := uns.GroupVersionKind().Kind

		if len(kind) > 0 {
			return printObj(w, name, group, kind)
		}
	}

	fmt.Fprintf(w, "<unknown>/%s\n", name)
	return nil
}

func printObj(w io.Writer, name, group, kind string) error {
	if len(kind) == 0 {
		return fmt.Errorf("missing kind for resource with name %v", name)
	}

	if len(group) == 0 {
		fmt.Fprintf(w, "%s/%s\n", strings.ToLower(kind), name)
		return nil
	}

	fmt.Fprintf(w, "%s.%s/%s\n", strings.ToLower(kind), group, name)
	return nil
}

// TODO: implement HandledResources()
func (p *NamePrinter) HandledResources() []string {
	return []string{}
}

func (p *NamePrinter) IsGeneric() bool {
	return true
}
