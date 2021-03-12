/*
Copyright 2021 The Kubernetes Authors.

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

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime"
)

// OmitManagedFieldsPrinter wraps an existing printer and omits the managed fields from the object
// before printing it.
type OmitManagedFieldsPrinter struct {
	Delegate ResourcePrinter
}

var _ ResourcePrinter = (*OmitManagedFieldsPrinter)(nil)

func omitManagedFields(o runtime.Object) runtime.Object {
	a, err := meta.Accessor(o)
	if err != nil {
		// The object is not a `metav1.Object`, ignore it.
		return o
	}
	a.SetManagedFields(nil)
	return o
}

// PrintObj copies the object and omits the managed fields from the copied object before printing it.
func (p *OmitManagedFieldsPrinter) PrintObj(obj runtime.Object, w io.Writer) error {
	if obj == nil {
		return p.Delegate.PrintObj(obj, w)
	}
	if meta.IsListType(obj) {
		obj = obj.DeepCopyObject()
		_ = meta.EachListItem(obj, func(item runtime.Object) error {
			omitManagedFields(item)
			return nil
		})
	} else if _, err := meta.Accessor(obj); err == nil {
		obj = omitManagedFields(obj.DeepCopyObject())
	}
	return p.Delegate.PrintObj(obj, w)
}
