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

import "reflect"

type mapReflect struct {
	valueReflect
}

func (r mapReflect) Length() int {
	val := r.Value
	return val.Len()
}

func (r mapReflect) Get(key string) (Value, bool) {
	mapKey := r.toMapKey(key)
	val := r.Value.MapIndex(mapKey)
	if !val.IsValid() {
		return nil, false
	}
	return mustWrapValueReflectMapItem(&r.Value, &mapKey, val), val != reflect.Value{}
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
	return eachMapEntry(r.Value, func(s string, value reflect.Value) bool {
		mapVal := mustWrapValueReflect(value)
		defer mapVal.Recycle()
		return fn(s, mapVal)
	})
}

func eachMapEntry(val reflect.Value, fn func(string, reflect.Value) bool) bool {
	iter := val.MapRange()
	for iter.Next() {
		next := iter.Value()
		if !next.IsValid() {
			continue
		}
		if !fn(iter.Key().String(), next) {
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
	if r.Length() != m.Length() {
		return false
	}

	// TODO: Optimize to avoid Iterate looping here by using r.Value.MapRange or similar if it improves performance.
	return m.Iterate(func(key string, value Value) bool {
		lhsVal, ok := r.Get(key)
		if !ok {
			return false
		}
		return Equals(lhsVal, value)
	})
}
