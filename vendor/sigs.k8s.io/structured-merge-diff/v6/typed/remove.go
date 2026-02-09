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
		// For extraction, we just return the value as is (which is nil or empty). For extraction the difference matters.
		if w.shouldExtract {
			w.out = w.value.Unstructured()
		}
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
	hadMatches := false
	iter := l.RangeUsing(w.allocator)
	defer w.allocator.Free(iter)
	for iter.Next() {
		_, item := iter.Item()
		// Ignore error because we have already validated this list
		pe, _ := listItemToPathElement(w.allocator, w.schema, t, item)
		path, _ := fieldpath.MakePath(pe)
		// save items on the path when we shouldExtract
		// but ignore them when we are removing (i.e. !w.shouldExtract)
		isExactPathMatch := w.toRemove.Has(path)
		isPrefixMatch := !w.toRemove.WithPrefix(pe).Empty()
		if w.shouldExtract {
			if isPrefixMatch {
				item = removeItemsWithSchema(item, w.toRemove.WithPrefix(pe), w.schema, t.ElementType, w.shouldExtract)
			}
			if isExactPathMatch || isPrefixMatch {
				newItems = append(newItems, item.Unstructured())
			}
		} else {
			if isExactPathMatch {
				continue
			}
			if isPrefixMatch {
				// Removing nested items within this list item and preserve if it becomes empty
				hadMatches = true
				wasMap := item.IsMap()
				wasList := item.IsList()
				item = removeItemsWithSchema(item, w.toRemove.WithPrefix(pe), w.schema, t.ElementType, w.shouldExtract)
				// If item returned null but we're removing items within the structure(not the item itself),
				// preserve the empty container structure
				if item.IsNull() && !w.shouldExtract {
					if wasMap {
						item = value.NewValueInterface(map[string]interface{}{})
					} else if wasList {
						item = value.NewValueInterface([]interface{}{})
					}
				}
			}
			newItems = append(newItems, item.Unstructured())
		}
	}
	// Preserve empty lists (non-nil) instead of converting to null when items were matched and removed
	if len(newItems) > 0 || (hadMatches && !w.shouldExtract) {
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
		// For extraction, we just return the value as is (which is nil or empty). For extraction the difference matters.
		if w.shouldExtract {
			w.out = w.value.Unstructured()
		}
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
	hadMatches := false
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
			hadMatches = true
			wasMap := val.IsMap()
			wasList := val.IsList()
			val = removeItemsWithSchema(val, subset, w.schema, fieldType, w.shouldExtract)
			// If val returned null but we're removing items within the structure (not the field itself),
			// preserve the empty container structure
			if val.IsNull() && !w.shouldExtract {
				if wasMap {
					val = value.NewValueInterface(map[string]interface{}{})
				} else if wasList {
					val = value.NewValueInterface([]interface{}{})
				}
			}
		} else {
			// don't save values not on the path when we shouldExtract.
			if w.shouldExtract {
				return true
			}
		}
		newMap[k] = val.Unstructured()
		return true
	})
	// Preserve empty maps (non-nil) instead of converting to null when items were matched and removed
	if len(newMap) > 0 || (hadMatches && !w.shouldExtract) {
		w.out = newMap
	}
	return nil
}
