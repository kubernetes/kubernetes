/*
Copyright The Kubernetes Authors.

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

// Package managedfields provides a helper shared by the JSON, Protobuf, and
// CBOR serializers to strip metadata.managedFields from objects before
// encoding. It is internal so that only the sibling serializer packages can
// depend on it.
package managedfields

import (
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime"
)

// RemoveInPlace clears metadata.managedFields from obj — and, for list objects,
// from each item. obj is mutated, so it must be safe to mutate, e.g. a private
// copy such as the one cachingObject.GetObject() returns to encoders. Objects
// that do not expose object metadata (e.g. *metav1.Status, runtime.Unknown) are
// left untouched.
func RemoveInPlace(obj runtime.Object) {
	if meta.IsListType(obj) {
		_ = meta.EachListItem(obj, func(item runtime.Object) error {
			clearManagedFields(item)
			return nil
		})
		return
	}
	clearManagedFields(obj)
}

func clearManagedFields(obj runtime.Object) {
	if accessor, err := meta.Accessor(obj); err == nil {
		accessor.SetManagedFields(nil)
	}
}

// Remove returns a copy of obj with metadata.managedFields cleared; obj itself
// is never mutated. Use RemoveInPlace instead when obj is already a private copy
// (e.g. on the cachingObject path) to avoid copying twice.
func Remove(obj runtime.Object) runtime.Object {
	if obj == nil {
		return obj
	}
	obj = obj.DeepCopyObject()
	RemoveInPlace(obj)
	return obj
}
