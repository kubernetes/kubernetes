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

type listReflect struct {
	Value reflect.Value
}

func (r listReflect) Length() int {
	val := r.Value
	return val.Len()
}

func (r listReflect) At(i int) Value {
	val := r.Value
	return mustWrapValueReflect(val.Index(i), nil, nil)
}

func (r listReflect) AtUsing(a Allocator, i int) Value {
	val := r.Value
	return a.allocValueReflect().mustReuse(val.Index(i), nil, nil, nil)
}

func (r listReflect) Unstructured() interface{} {
	l := r.Length()
	result := make([]interface{}, l)
	for i := 0; i < l; i++ {
		result[i] = r.At(i).Unstructured()
	}
	return result
}

func (r listReflect) Range() ListRange {
	return r.RangeUsing(HeapAllocator)
}

func (r listReflect) RangeUsing(a Allocator) ListRange {
	length := r.Value.Len()
	if length == 0 {
		return EmptyRange
	}
	rr := a.allocListReflectRange()
	rr.list = r.Value
	rr.i = -1
	rr.entry = TypeReflectEntryOf(r.Value.Type().Elem())
	return rr
}

func (r listReflect) Equals(other List) bool {
	return r.EqualsUsing(HeapAllocator, other)
}
func (r listReflect) EqualsUsing(a Allocator, other List) bool {
	if otherReflectList, ok := other.(*listReflect); ok {
		return reflect.DeepEqual(r.Value.Interface(), otherReflectList.Value.Interface())
	}
	return ListEqualsUsing(a, &r, other)
}

type listReflectRange struct {
	list  reflect.Value
	vr    *valueReflect
	i     int
	entry *TypeReflectCacheEntry
}

func (r *listReflectRange) Next() bool {
	r.i += 1
	return r.i < r.list.Len()
}

func (r *listReflectRange) Item() (index int, value Value) {
	if r.i < 0 {
		panic("Item() called before first calling Next()")
	}
	if r.i >= r.list.Len() {
		panic("Item() called on ListRange with no more items")
	}
	v := r.list.Index(r.i)
	return r.i, r.vr.mustReuse(v, r.entry, nil, nil)
}
