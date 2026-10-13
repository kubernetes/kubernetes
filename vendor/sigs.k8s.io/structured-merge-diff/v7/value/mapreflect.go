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
	"reflect"
)

type mapReflect struct {
	valueReflect
}

func (r mapReflect) Length() int {
	val := r.Value
	return val.Len()
}

func (r mapReflect) Empty() bool {
	val := r.Value
	return val.Len() == 0
}

func (r mapReflect) Get(key string) (Value, bool) {
	return r.GetUsing(HeapAllocator, key)
}

func (r mapReflect) GetUsing(a Allocator, key string) (Value, bool) {
	k, v, ok := r.get(key)
	if !ok {
		return nil, false
	}
	return a.allocValueReflect().mustReuse(v, nil, &r.Value, &k), true
}

func (r mapReflect) get(k string) (key, value reflect.Value, ok bool) {
	mapKey := r.toMapKey(k)
	val := r.Value.MapIndex(mapKey)
	return mapKey, val, val.IsValid() && val != reflect.Value{}
}

func (r mapReflect) Has(key string) bool {
	var val reflect.Value
	val = r.Value.MapIndex(r.toMapKey(key))
	if !val.IsValid() {
		return false
	}
	return val != reflect.Value{}
}

func (r mapReflect) Set(key string, val Value) {
	r.Value.SetMapIndex(r.toMapKey(key), reflect.ValueOf(val.Unstructured()))
}

func (r mapReflect) Delete(key string) {
	val := r.Value
	val.SetMapIndex(r.toMapKey(key), reflect.Value{})
}

// TODO: Do we need to support types that implement json.Marshaler and are used as string keys?
func (r mapReflect) toMapKey(key string) reflect.Value {
	val := r.Value
	return reflect.ValueOf(key).Convert(val.Type().Key())
}

func (r mapReflect) Iterate(fn func(string, Value) bool) bool {
	return r.IterateUsing(HeapAllocator, fn)
}

func (r mapReflect) IterateUsing(a Allocator, fn func(string, Value) bool) bool {
	if r.Value.Len() == 0 {
		return true
	}
	v := a.allocValueReflect()
	defer a.Free(v)
	return eachMapEntry(r.Value, func(e *TypeReflectCacheEntry, key reflect.Value, value reflect.Value) bool {
		return fn(key.String(), v.mustReuse(value, e, &r.Value, &key))
	})
}

func eachMapEntry(val reflect.Value, fn func(*TypeReflectCacheEntry, reflect.Value, reflect.Value) bool) bool {
	iter := val.MapRange()
	entry := TypeReflectEntryOf(val.Type().Elem())
	for iter.Next() {
		next := iter.Value()
		if !next.IsValid() {
			continue
		}
		if !fn(entry, iter.Key(), next) {
			return false
		}
	}
	return true
}

func (r mapReflect) Unstructured() interface{} {
	result := make(map[string]interface{}, r.Length())
	r.Iterate(func(s string, value Value) bool {
		result[s] = value.Unstructured()
		return true
	})
	return result
}

func (r mapReflect) Equals(m Map) bool {
	return r.EqualsUsing(HeapAllocator, m)
}

func (r mapReflect) EqualsUsing(a Allocator, m Map) bool {
	lhsLength := r.Length()
	rhsLength := m.Length()
	if lhsLength != rhsLength {
		return false
	}
	if lhsLength == 0 {
		return true
	}
	vr := a.allocValueReflect()
	defer a.Free(vr)
	entry := TypeReflectEntryOf(r.Value.Type().Elem())
	return m.Iterate(func(key string, value Value) bool {
		_, lhsVal, ok := r.get(key)
		if !ok {
			return false
		}
		return EqualsUsing(a, vr.mustReuse(lhsVal, entry, nil, nil), value)
	})
}

func (r mapReflect) Zip(other Map, order MapTraverseOrder, fn func(key string, lhs, rhs Value) bool) bool {
	return r.ZipUsing(HeapAllocator, other, order, fn)
}

func (r mapReflect) ZipUsing(a Allocator, other Map, order MapTraverseOrder, fn func(key string, lhs, rhs Value) bool) bool {
	if otherMapReflect, ok := other.(*mapReflect); ok && order == Unordered {
		return r.unorderedReflectZip(a, otherMapReflect, fn)
	}
	return defaultMapZip(a, &r, other, order, fn)
}

// unorderedReflectZip provides an optimized unordered zip for mapReflect types.
func (r mapReflect) unorderedReflectZip(a Allocator, other *mapReflect, fn func(key string, lhs, rhs Value) bool) bool {
	if r.Empty() && (other == nil || other.Empty()) {
		return true
	}

	lhs := r.Value
	lhsEntry := TypeReflectEntryOf(lhs.Type().Elem())

	// map lookup via reflection is expensive enough that it is better to keep track of visited keys
	visited := map[string]struct{}{}

	vlhs, vrhs := a.allocValueReflect(), a.allocValueReflect()
	defer a.Free(vlhs)
	defer a.Free(vrhs)

	if other != nil {
		rhs := other.Value
		rhsEntry := TypeReflectEntryOf(rhs.Type().Elem())
		iter := rhs.MapRange()

		for iter.Next() {
			key := iter.Key()
			keyString := key.String()
			next := iter.Value()
			if !next.IsValid() {
				continue
			}
			rhsVal := vrhs.mustReuse(next, rhsEntry, &rhs, &key)
			visited[keyString] = struct{}{}
			var lhsVal Value
			if _, v, ok := r.get(keyString); ok {
				lhsVal = vlhs.mustReuse(v, lhsEntry, &lhs, &key)
			}
			if !fn(keyString, lhsVal, rhsVal) {
				return false
			}
		}
	}

	iter := lhs.MapRange()
	for iter.Next() {
		key := iter.Key()
		if _, ok := visited[key.String()]; ok {
			continue
		}
		next := iter.Value()
		if !next.IsValid() {
			continue
		}
		if !fn(key.String(), vlhs.mustReuse(next, lhsEntry, &lhs, &key), nil) {
			return false
		}
	}
	return true
}
