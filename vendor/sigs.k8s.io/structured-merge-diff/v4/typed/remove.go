/*
Copyright 2019 The Kubernetes Authors.
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

package typed

import (
	"sigs.k8s.io/structured-merge-diff/v4/fieldpath"
	"sigs.k8s.io/structured-merge-diff/v4/schema"
	"sigs.k8s.io/structured-merge-diff/v4/value"
)

type removingWalker struct {
	value     value.Value
	out       interface{}
	schema    *schema.Schema
	toRemove  *fieldpath.Set
	allocator value.Allocator
}

func removeItemsWithSchema(val value.Value, toRemove *fieldpath.Set, schema *schema.Schema, typeRef schema.TypeRef) value.Value {
	w := &removingWalker{
		value:     val,
		schema:    schema,
		toRemove:  toRemove,
		allocator: value.NewFreelistAllocator(),
	}
	resolveSchema(schema, typeRef, val, w)
	return value.NewValueInterface(w.out)
}

func (w *removingWalker) doScalar(t *schema.Scalar) ValidationErrors {
	w.out = w.value.Unstructured()
	return nil
}

func (w *removingWalker) doList(t *schema.List) (errs ValidationErrors) {
	l := w.value.AsListUsing(w.allocator)
	defer w.allocator.Free(l)
	// If list is null, empty, or atomic just return
	if l == nil || l.Length() == 0 || t.ElementRelationship == schema.Atomic {
		return nil
	}

	var newItems []interface{}
	iter := l.RangeUsing(w.allocator)
	defer w.allocator.Free(iter)
	for iter.Next() {
		i, item := iter.Item()
		// Ignore error because we have already validated this list
		pe, _ := listItemToPathElement(w.allocator, t, i, item)
		path, _ := fieldpath.MakePath(pe)
		if w.toRemove.Has(path) {
			continue
		}
		if subset := w.toRemove.WithPrefix(pe); !subset.Empty() {
			item = removeItemsWithSchema(item, subset, w.schema, t.ElementType)
		}
		newItems = append(newItems, item.Unstructured())
	}
	if len(newItems) > 0 {
		w.out = newItems
	}
	return nil
}

func (w *removingWalker) doMap(t *schema.Map) ValidationErrors {
	m := w.value.AsMapUsing(w.allocator)
	if m != nil {
		defer w.allocator.Free(m)
	}
	// If map is null, empty, or atomic just return
	if m == nil || m.Empty() || t.ElementRelationship == schema.Atomic {
		return nil
	}

	fieldTypes := map[string]schema.TypeRef{}
	for _, structField := range t.Fields {
		fieldTypes[structField.Name] = structField.Type
	}

	newMap := map[string]interface{}{}
	m.Iterate(func(k string, val value.Value) bool {
		pe := fieldpath.PathElement{FieldName: &k}
		path, _ := fieldpath.MakePath(pe)
		fieldType := t.ElementType
		if ft, ok := fieldTypes[k]; ok {
			fieldType = ft
		}
		if w.toRemove.Has(path) {
			return true
		}
		if subset := w.toRemove.WithPrefix(pe); !subset.Empty() {
			val = removeItemsWithSchema(val, subset, w.schema, fieldType)
		}
		newMap[k] = val.Unstructured()
		return true
	})
	if len(newMap) > 0 {
		w.out = newMap
	}
	return nil
}
