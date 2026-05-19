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
	"sigs.k8s.io/structured-merge-diff/v6/fieldpath"
	"sigs.k8s.io/structured-merge-diff/v6/schema"
	"sigs.k8s.io/structured-merge-diff/v6/value"
)

type removingWalker struct {
	value         value.Value
	out           interface{}
	schema        *schema.Schema
	toRemove      *fieldpath.Set
	allocator     value.Allocator
	shouldExtract bool
}

// removeItemsWithSchema will walk the given value and look for items from the toRemove set.
// Depending on whether shouldExtract is set true or false, it will return a modified version
// of the input value with either:
// 1. only the items in the toRemove set (when shouldExtract is true) or
// 2. the items from the toRemove set removed from the value (when shouldExtract is false).
func removeItemsWithSchema(val value.Value, toRemove *fieldpath.Set, schema *schema.Schema, typeRef schema.TypeRef, shouldExtract bool) value.Value {
	w := &removingWalker{
		value:         val,
		schema:        schema,
		toRemove:      toRemove,
		allocator:     value.NewFreelistAllocator(),
		shouldExtract: shouldExtract,
	}
	resolveSchema(schema, typeRef, val, w)
	return value.NewValueInterface(w.out)
}

func (w *removingWalker) doScalar(t *schema.Scalar) ValidationErrors {
	w.out = w.value.Unstructured()
	return nil
}

func (w *removingWalker) doList(t *schema.List) (errs ValidationErrors) {
	if !w.value.IsList() {
		return nil
	}
	l := w.value.AsListUsing(w.allocator)
	defer w.allocator.Free(l)
	// If list is null or empty just return
	if l == nil || l.Length() == 0 {
		return nil
	}

	// atomic lists should return everything in the case of extract
	// and nothing in the case of remove (!w.shouldExtract)
	if t.ElementRelationship == schema.Atomic {
		if w.shouldExtract {
			w.out = w.value.Unstructured()
		}
		return nil
	}

	var newItems []interface{}
	iter := l.RangeUsing(w.allocator)
	defer w.allocator.Free(iter)
	for iter.Next() {
		_, item := iter.Item()
		// Ignore error because we have already validated this list
		pe, _ := listItemToPathElement(w.allocator, w.schema, t, item)
		path, _ := fieldpath.MakePath(pe)
		// save items on the path when we shouldExtract
		// but ignore them when we are removing (i.e. !w.shouldExtract)
		if w.toRemove.Has(path) {
			if w.shouldExtract {
				newItems = append(newItems, removeItemsWithSchema(item, w.toRemove, w.schema, t.ElementType, w.shouldExtract).Unstructured())
			} else {
				continue
			}
		}
		if subset := w.toRemove.WithPrefix(pe); !subset.Empty() {
			item = removeItemsWithSchema(item, subset, w.schema, t.ElementType, w.shouldExtract)
		} else {
			// don't save items not on the path when we shouldExtract.
			if w.shouldExtract {
				continue
			}
		}
		newItems = append(newItems, item.Unstructured())
	}
	if len(newItems) > 0 {
		w.out = newItems
	}
	return nil
}

func (w *removingWalker) doMap(t *schema.Map) ValidationErrors {
	if !w.value.IsMap() {
		return nil
	}
	m := w.value.AsMapUsing(w.allocator)
	if m != nil {
		defer w.allocator.Free(m)
	}
	// If map is null or empty just return
	if m == nil || m.Empty() {
		return nil
	}

	// atomic maps should return everything in the case of extract
	// and nothing in the case of remove (!w.shouldExtract)
	if t.ElementRelationship == schema.Atomic {
		if w.shouldExtract {
			w.out = w.value.Unstructured()
		}
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
		// save values on the path when we shouldExtract
		// but ignore them when we are removing (i.e. !w.shouldExtract)
		if w.toRemove.Has(path) {
			if w.shouldExtract {
				newMap[k] = removeItemsWithSchema(val, w.toRemove, w.schema, fieldType, w.shouldExtract).Unstructured()

			}
			return true
		}
		if subset := w.toRemove.WithPrefix(pe); !subset.Empty() {
			val = removeItemsWithSchema(val, subset, w.schema, fieldType, w.shouldExtract)
		} else {
			// don't save values not on the path when we shouldExtract.
			if w.shouldExtract {
				return true
			}
		}
		newMap[k] = val.Unstructured()
		return true
	})
	if len(newMap) > 0 {
		w.out = newMap
	}
	return nil
}
