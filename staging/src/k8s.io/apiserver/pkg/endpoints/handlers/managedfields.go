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

package handlers

import (
	"maps"
	"reflect"

	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
)

// dropManagedFields returns obj with metadata.managedFields cleared. Lists are
// cleared in place, since list objects are never cached and own the items they
// hold by value. Anything else is copied only along the path to managedFields
// and shares the rest with obj, so the result must not be mutated otherwise.
func dropManagedFields(obj runtime.Object) runtime.Object {
	switch t := obj.(type) {
	case *unstructured.Unstructured:
		if t.IsList() {
			return deepCopyWithoutManagedFields(t)
		}
		return &unstructured.Unstructured{Object: contentWithoutManagedFields(t.Object)}
	case *unstructured.UnstructuredList:
		for i := range t.Items {
			t.Items[i].Object = contentWithoutManagedFields(t.Items[i].Object)
		}
		return t
	}
	if meta.IsListType(obj) {
		return listWithoutManagedFields(obj)
	}
	return itemCopyWithoutManagedFields(obj)
}

// contentWithoutManagedFields copies only the top-level and metadata maps.
// Unstructured.SetManagedFields would delete the key from the shared map.
func contentWithoutManagedFields(content map[string]interface{}) map[string]interface{} {
	metadata, ok := content["metadata"].(map[string]interface{})
	if !ok {
		return content
	}
	if _, ok := metadata["managedFields"]; !ok {
		return content
	}
	newMetadata := maps.Clone(metadata)
	delete(newMetadata, "managedFields")
	newContent := maps.Clone(content)
	newContent["metadata"] = newMetadata
	return newContent
}

func itemCopyWithoutManagedFields(obj runtime.Object) runtime.Object {
	acc, err := meta.Accessor(obj)
	if err != nil || len(acc.GetManagedFields()) == 0 {
		return obj
	}
	v := reflect.ValueOf(obj)
	if v.Kind() != reflect.Pointer || !hasValueObjectMeta(v.Type().Elem()) {
		return deepCopyWithoutManagedFields(obj)
	}
	cp := reflect.New(v.Type().Elem())
	cp.Elem().Set(v.Elem())
	out := cp.Interface().(runtime.Object)
	if acc, err = meta.Accessor(out); err == nil {
		acc.SetManagedFields(nil)
	}
	return out
}

func listWithoutManagedFields(list runtime.Object) runtime.Object {
	v := reflect.ValueOf(list)
	if v.Kind() != reflect.Pointer || v.Elem().Kind() != reflect.Struct {
		return deepCopyWithoutManagedFields(list)
	}
	items := v.Elem().FieldByName("Items")
	// Items reached through pointers may be shared; they take the fallback.
	if items.Kind() != reflect.Slice || !hasValueObjectMeta(items.Type().Elem()) {
		return deepCopyWithoutManagedFields(list)
	}
	for i := 0; i < items.Len(); i++ {
		if acc, err := meta.Accessor(items.Index(i).Addr().Interface()); err == nil {
			acc.SetManagedFields(nil)
		}
	}
	return list
}

// deepCopyWithoutManagedFields is the always-correct fallback for other shapes.
func deepCopyWithoutManagedFields(obj runtime.Object) runtime.Object {
	out := obj.DeepCopyObject()
	if !meta.IsListType(out) {
		if acc, err := meta.Accessor(out); err == nil {
			acc.SetManagedFields(nil)
		}
		return out
	}
	// Items that are not objects have no managedFields to clear.
	_ = meta.EachListItem(out, func(item runtime.Object) error {
		if acc, err := meta.Accessor(item); err == nil {
			acc.SetManagedFields(nil)
		}
		return nil
	})
	return out
}

var objectMetaType = reflect.TypeFor[metav1.ObjectMeta]()

// hasValueObjectMeta reports whether t has a metav1.ObjectMeta field by value,
// which a shallow copy of t owns; metadata behind a pointer would be shared.
func hasValueObjectMeta(t reflect.Type) bool {
	if t.Kind() != reflect.Struct {
		return false
	}
	for i := range t.NumField() { //nolint:modernize // t.Fields() allocates on every call.
		if t.Field(i).Type == objectMetaType {
			return true
		}
	}
	return false
}
