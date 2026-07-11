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

// List represents a list object.
type List interface {
	// Length returns how many items can be found in the map.
	Length() int
	// At returns the item at the given position in the map. It will
	// panic if the index is out of range.
	At(int) Value
	// AtUsing uses the provided allocator and returns the item at the given
	// position in the map. It will panic if the index is out of range.
	// The returned Value should be given back to the Allocator when no longer needed
	// by calling Allocator.Free(Value).
	AtUsing(Allocator, int) Value
	// Range returns a ListRange for iterating over the items in the list.
	Range() ListRange
	// RangeUsing uses the provided allocator and returns a ListRange for
	// iterating over the items in the list.
	// The returned Range should be given back to the Allocator when no longer needed
	// by calling Allocator.Free(Value).
	RangeUsing(Allocator) ListRange
	// Equals compares the two lists, and return true if they are the same, false otherwise.
	// Implementations can use ListEquals as a general implementation for this methods.
	Equals(List) bool
	// EqualsUsing uses the provided allocator and compares the two lists, and return true if
	// they are the same, false otherwise. Implementations can use ListEqualsUsing as a general
	// implementation for this methods.
	EqualsUsing(Allocator, List) bool
}

// ListRange represents a single iteration across the items of a list.
type ListRange interface {
	// Next increments to the next item in the range, if there is one, and returns true, or returns false if there are no more items.
	Next() bool
	// Item returns the index and value of the current item in the range. or panics if there is no current item.
	// For efficiency, Item may reuse the values returned by previous Item calls. Callers should be careful avoid holding
	// pointers to the value returned by Item() that escape the iteration loop since they become invalid once either
	// Item() or Allocator.Free() is called.
	Item() (index int, value Value)
}

var EmptyRange = &emptyRange{}

type emptyRange struct{}

func (_ *emptyRange) Next() bool {
	return false
}

func (_ *emptyRange) Item() (index int, value Value) {
	panic("Item called on empty ListRange")
}

// ListEquals compares two lists lexically.
// WARN: This is a naive implementation, calling lhs.Equals(rhs) is typically the most efficient.
func ListEquals(lhs, rhs List) bool {
	return ListEqualsUsing(HeapAllocator, lhs, rhs)
}

// ListEqualsUsing uses the provided allocator and compares two lists lexically.
// WARN: This is a naive implementation, calling lhs.EqualsUsing(allocator, rhs) is typically the most efficient.
func ListEqualsUsing(a Allocator, lhs, rhs List) bool {
	if lhs.Length() != rhs.Length() {
		return false
	}

	lhsRange := lhs.RangeUsing(a)
	defer a.Free(lhsRange)
	rhsRange := rhs.RangeUsing(a)
	defer a.Free(rhsRange)

	for lhsRange.Next() && rhsRange.Next() {
		_, lv := lhsRange.Item()
		_, rv := rhsRange.Item()
		if !EqualsUsing(a, lv, rv) {
			return false
		}
	}
	return true
}

// ListLess compares two lists lexically.
func ListLess(lhs, rhs List) bool {
	return ListCompare(lhs, rhs) == -1
}

// ListCompare compares two lists lexically. The result will be 0 if l==rhs, -1
// if l < rhs, and +1 if l > rhs.
func ListCompare(lhs, rhs List) int {
	return ListCompareUsing(HeapAllocator, lhs, rhs)
}

// ListCompareUsing uses the provided allocator and compares two lists lexically. The result will be 0 if l==rhs, -1
// if l < rhs, and +1 if l > rhs.
func ListCompareUsing(a Allocator, lhs, rhs List) int {
	lhsRange := lhs.RangeUsing(a)
	defer a.Free(lhsRange)
	rhsRange := rhs.RangeUsing(a)
	defer a.Free(rhsRange)

	for {
		lhsOk := lhsRange.Next()
		rhsOk := rhsRange.Next()
		if !lhsOk && !rhsOk {
			// Lists are the same length and all items are equal.
			return 0
		}
		if !lhsOk {
			// LHS is shorter.
			return -1
		}
		if !rhsOk {
			// RHS is shorter.
			return 1
		}
		_, lv := lhsRange.Item()
		_, rv := rhsRange.Item()
		if c := CompareUsing(a, lv, rv); c != 0 {
			return c
		}
		// The items are equal; continue.
	}
}
