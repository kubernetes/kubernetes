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

package value

import (
	"fmt"
	"reflect"
)

type structReflect struct {
	valueReflect
}

func (r structReflect) Length() int {
	i := 0
	eachStructField(r.Value, func(_ *TypeReflectCacheEntry, s string, value reflect.Value) bool {
		i++
		return true
	})
	return i
}

func (r structReflect) Empty() bool {
	return eachStructField(r.Value, func(_ *TypeReflectCacheEntry, s string, value reflect.Value) bool {
		return false // exit early if the struct is non-empty
	})
}

func (r structReflect) Get(key string) (Value, bool) {
	return r.GetUsing(HeapAllocator, key)
}

func (r structReflect) GetUsing(a Allocator, key string) (Value, bool) {
	if val, ok := r.findJsonNameField(key); ok {
		return a.allocValueReflect().mustReuse(val, nil, nil, nil), true
	}
	return nil, false
}

func (r structReflect) Has(key string) bool {
	_, ok := r.findJsonNameField(key)
	return ok
}

func (r structReflect) Set(key string, val Value) {
	fieldEntry, ok := TypeReflectEntryOf(r.Value.Type()).Fields()[key]
	if !ok {
		panic(fmt.Sprintf("key %s may not be set on struct %T: field does not exist", key, r.Value.Interface()))
	}
	oldVal := fieldEntry.GetFrom(r.Value)
	newVal := reflect.ValueOf(val.Unstructured())
	r.update(fieldEntry, key, oldVal, newVal)
}

func (r structReflect) Delete(key string) {
	fieldEntry, ok := TypeReflectEntryOf(r.Value.Type()).Fields()[key]
	if !ok {
		panic(fmt.Sprintf("key %s may not be deleted on struct %T: field does not exist", key, r.Value.Interface()))
	}
	oldVal := fieldEntry.GetFrom(r.Value)
	if oldVal.Kind() != reflect.Ptr && !fieldEntry.isOmitEmpty {
		panic(fmt.Sprintf("key %s may not be deleted on struct: %T: value is neither a pointer nor an omitempty field", key, r.Value.Interface()))
	}
	r.update(fieldEntry, key, oldVal, reflect.Zero(oldVal.Type()))
}

func (r structReflect) update(fieldEntry *FieldCacheEntry, key string, oldVal, newVal reflect.Value) {
	if oldVal.CanSet() {
		oldVal.Set(newVal)
		return
	}

	// map items are not addressable, so if a struct is contained in a map, the only way to modify it is
	// to write a replacement fieldEntry into the map.
	if r.ParentMap != nil {
		if r.ParentMapKey == nil {
			panic("ParentMapKey must not be nil if ParentMap is not nil")
		}
		replacement := reflect.New(r.Value.Type()).Elem()
		fieldEntry.GetFrom(replacement).Set(newVal)
		r.ParentMap.SetMapIndex(*r.ParentMapKey, replacement)
		return
	}

	// This should never happen since NewValueReflect ensures that the root object reflected on is a pointer and map
	// item replacement is handled above.
	panic(fmt.Sprintf("key %s may not be modified on struct: %T: struct is not settable", key, r.Value.Interface()))
}

func (r structReflect) Iterate(fn func(string, Value) bool) bool {
	return r.IterateUsing(HeapAllocator, fn)
}

func (r structReflect) IterateUsing(a Allocator, fn func(string, Value) bool) bool {
	vr := a.allocValueReflect()
	defer a.Free(vr)
	return eachStructField(r.Value, func(e *TypeReflectCacheEntry, s string, value reflect.Value) bool {
		return fn(s, vr.mustReuse(value, e, nil, nil))
	})
}

func eachStructField(structVal reflect.Value, fn func(*TypeReflectCacheEntry, string, reflect.Value) bool) bool {
	for _, fieldCacheEntry := range TypeReflectEntryOf(structVal.Type()).OrderedFields() {
		fieldVal := fieldCacheEntry.GetFrom(structVal)
		if fieldCacheEntry.CanOmit(fieldVal) {
			// omit it
			continue
		}
		ok := fn(fieldCacheEntry.TypeEntry, fieldCacheEntry.JsonName, fieldVal)
		if !ok {
			return false
		}
	}
	return true
}

func (r structReflect) Unstructured() interface{} {
	// Use number of struct fields as a cheap way to rough estimate map size
	result := make(map[string]interface{}, r.Value.NumField())
	r.Iterate(func(s string, value Value) bool {
		result[s] = value.Unstructured()
		return true
	})
	return result
}

func (r structReflect) Equals(m Map) bool {
	return r.EqualsUsing(HeapAllocator, m)
}

func (r structReflect) EqualsUsing(a Allocator, m Map) bool {
	// MapEquals uses zip and is fairly efficient for structReflect
	return MapEqualsUsing(a, &r, m)
}

func (r structReflect) findJsonNameFieldAndNotEmpty(jsonName string) (reflect.Value, bool) {
	structCacheEntry, ok := TypeReflectEntryOf(r.Value.Type()).Fields()[jsonName]
	if !ok {
		return reflect.Value{}, false
	}
	fieldVal := structCacheEntry.GetFrom(r.Value)
	return fieldVal, !structCacheEntry.CanOmit(fieldVal)
}

func (r structReflect) findJsonNameField(jsonName string) (val reflect.Value, ok bool) {
	structCacheEntry, ok := TypeReflectEntryOf(r.Value.Type()).Fields()[jsonName]
	if !ok {
		return reflect.Value{}, false
	}
	fieldVal := structCacheEntry.GetFrom(r.Value)
	return fieldVal, !structCacheEntry.CanOmit(fieldVal)
}

func (r structReflect) Zip(other Map, order MapTraverseOrder, fn func(key string, lhs, rhs Value) bool) bool {
	return r.ZipUsing(HeapAllocator, other, order, fn)
}

func (r structReflect) ZipUsing(a Allocator, other Map, order MapTraverseOrder, fn func(key string, lhs, rhs Value) bool) bool {
	if otherStruct, ok := other.(*structReflect); ok && r.Value.Type() == otherStruct.Value.Type() {
		lhsvr, rhsvr := a.allocValueReflect(), a.allocValueReflect()
		defer a.Free(lhsvr)
		defer a.Free(rhsvr)
		return r.structZip(otherStruct, lhsvr, rhsvr, fn)
	}
	return defaultMapZip(a, &r, other, order, fn)
}

// structZip provides an optimized zip for structReflect types. The zip is always lexical key ordered since there is
// no additional cost to ordering the zip for structured types.
func (r structReflect) structZip(other *structReflect, lhsvr, rhsvr *valueReflect, fn func(key string, lhs, rhs Value) bool) bool {
	lhsVal := r.Value
	rhsVal := other.Value

	for _, fieldCacheEntry := range TypeReflectEntryOf(lhsVal.Type()).OrderedFields() {
		lhsFieldVal := fieldCacheEntry.GetFrom(lhsVal)
		rhsFieldVal := fieldCacheEntry.GetFrom(rhsVal)
		lhsOmit := fieldCacheEntry.CanOmit(lhsFieldVal)
		rhsOmit := fieldCacheEntry.CanOmit(rhsFieldVal)
		if lhsOmit && rhsOmit {
			continue
		}
		var lhsVal, rhsVal Value
		if !lhsOmit {
			lhsVal = lhsvr.mustReuse(lhsFieldVal, fieldCacheEntry.TypeEntry, nil, nil)
		}
		if !rhsOmit {
			rhsVal = rhsvr.mustReuse(rhsFieldVal, fieldCacheEntry.TypeEntry, nil, nil)
		}
		if !fn(fieldCacheEntry.JsonName, lhsVal, rhsVal) {
			return false
		}
	}
	return true
}
