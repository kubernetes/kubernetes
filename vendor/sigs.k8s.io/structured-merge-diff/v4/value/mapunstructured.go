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

type mapUnstructuredInterface map[interface{}]interface{}

func (m mapUnstructuredInterface) Set(key string, val Value) {
	m[key] = val.Unstructured()
}

func (m mapUnstructuredInterface) Get(key string) (Value, bool) {
	return m.GetUsing(HeapAllocator, key)
}

func (m mapUnstructuredInterface) GetUsing(a Allocator, key string) (Value, bool) {
	if v, ok := m[key]; !ok {
		return nil, false
	} else {
		return a.allocValueUnstructured().reuse(v), true
	}
}

func (m mapUnstructuredInterface) Has(key string) bool {
	_, ok := m[key]
	return ok
}

func (m mapUnstructuredInterface) Delete(key string) {
	delete(m, key)
}

func (m mapUnstructuredInterface) Iterate(fn func(key string, value Value) bool) bool {
	return m.IterateUsing(HeapAllocator, fn)
}

func (m mapUnstructuredInterface) IterateUsing(a Allocator, fn func(key string, value Value) bool) bool {
	if len(m) == 0 {
		return true
	}
	vv := a.allocValueUnstructured()
	defer a.Free(vv)
	for k, v := range m {
		if ks, ok := k.(string); !ok {
			continue
		} else {
			if !fn(ks, vv.reuse(v)) {
				return false
			}
		}
	}
	return true
}

func (m mapUnstructuredInterface) Length() int {
	return len(m)
}

func (m mapUnstructuredInterface) Empty() bool {
	return len(m) == 0
}

func (m mapUnstructuredInterface) Equals(other Map) bool {
	return m.EqualsUsing(HeapAllocator, other)
}

func (m mapUnstructuredInterface) EqualsUsing(a Allocator, other Map) bool {
	lhsLength := m.Length()
	rhsLength := other.Length()
	if lhsLength != rhsLength {
		return false
	}
	if lhsLength == 0 {
		return true
	}
	vv := a.allocValueUnstructured()
	defer a.Free(vv)
	return other.Iterate(func(key string, value Value) bool {
		lhsVal, ok := m[key]
		if !ok {
			return false
		}
		return Equals(vv.reuse(lhsVal), value)
	})
}

func (m mapUnstructuredInterface) Zip(other Map, order MapTraverseOrder, fn func(key string, lhs, rhs Value) bool) bool {
	return m.ZipUsing(HeapAllocator, other, order, fn)
}

func (m mapUnstructuredInterface) ZipUsing(a Allocator, other Map, order MapTraverseOrder, fn func(key string, lhs, rhs Value) bool) bool {
	return defaultMapZip(a, m, other, order, fn)
}

type mapUnstructuredString map[string]interface{}

func (m mapUnstructuredString) Set(key string, val Value) {
	m[key] = val.Unstructured()
}

func (m mapUnstructuredString) Get(key string) (Value, bool) {
	return m.GetUsing(HeapAllocator, key)
}
func (m mapUnstructuredString) GetUsing(a Allocator, key string) (Value, bool) {
	if v, ok := m[key]; !ok {
		return nil, false
	} else {
		return a.allocValueUnstructured().reuse(v), true
	}
}

func (m mapUnstructuredString) Has(key string) bool {
	_, ok := m[key]
	return ok
}

func (m mapUnstructuredString) Delete(key string) {
	delete(m, key)
}

func (m mapUnstructuredString) Iterate(fn func(key string, value Value) bool) bool {
	return m.IterateUsing(HeapAllocator, fn)
}

func (m mapUnstructuredString) IterateUsing(a Allocator, fn func(key string, value Value) bool) bool {
	if len(m) == 0 {
		return true
	}
	vv := a.allocValueUnstructured()
	defer a.Free(vv)
	for k, v := range m {
		if !fn(k, vv.reuse(v)) {
			return false
		}
	}
	return true
}

func (m mapUnstructuredString) Length() int {
	return len(m)
}

func (m mapUnstructuredString) Equals(other Map) bool {
	return m.EqualsUsing(HeapAllocator, other)
}

func (m mapUnstructuredString) EqualsUsing(a Allocator, other Map) bool {
	lhsLength := m.Length()
	rhsLength := other.Length()
	if lhsLength != rhsLength {
		return false
	}
	if lhsLength == 0 {
		return true
	}
	vv := a.allocValueUnstructured()
	defer a.Free(vv)
	return other.Iterate(func(key string, value Value) bool {
		lhsVal, ok := m[key]
		if !ok {
			return false
		}
		return Equals(vv.reuse(lhsVal), value)
	})
}

func (m mapUnstructuredString) Zip(other Map, order MapTraverseOrder, fn func(key string, lhs, rhs Value) bool) bool {
	return m.ZipUsing(HeapAllocator, other, order, fn)
}

func (m mapUnstructuredString) ZipUsing(a Allocator, other Map, order MapTraverseOrder, fn func(key string, lhs, rhs Value) bool) bool {
	return defaultMapZip(a, m, other, order, fn)
}

func (m mapUnstructuredString) Empty() bool {
	return len(m) == 0
}
