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

type listUnstructured []interface{}

func (l listUnstructured) Length() int {
	return len(l)
}

func (l listUnstructured) At(i int) Value {
	return NewValueInterface(l[i])
}

func (l listUnstructured) AtUsing(a Allocator, i int) Value {
	return a.allocValueUnstructured().reuse(l[i])
}

func (l listUnstructured) Equals(other List) bool {
	return l.EqualsUsing(HeapAllocator, other)
}

func (l listUnstructured) EqualsUsing(a Allocator, other List) bool {
	return ListEqualsUsing(a, &l, other)
}

func (l listUnstructured) Range() ListRange {
	return l.RangeUsing(HeapAllocator)
}

func (l listUnstructured) RangeUsing(a Allocator) ListRange {
	if len(l) == 0 {
		return EmptyRange
	}
	r := a.allocListUnstructuredRange()
	r.list = l
	r.i = -1
	return r
}

type listUnstructuredRange struct {
	list listUnstructured
	vv   *valueUnstructured
	i    int
}

func (r *listUnstructuredRange) Next() bool {
	r.i += 1
	return r.i < len(r.list)
}

func (r *listUnstructuredRange) Item() (index int, value Value) {
	if r.i < 0 {
		panic("Item() called before first calling Next()")
	}
	if r.i >= len(r.list) {
		panic("Item() called on ListRange with no more items")
	}
	return r.i, r.vv.reuse(r.list[r.i])
}
