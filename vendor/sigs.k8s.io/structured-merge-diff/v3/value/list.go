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
}

// ListEquals compares two lists lexically.
func ListEquals(lhs, rhs List) bool {
	if lhs.Length() != rhs.Length() {
		return false
	}

	for i := 0; i < lhs.Length(); i++ {
		lv := lhs.At(i)
		rv := rhs.At(i)
		if !Equals(lv, rv) {
			lv.Recycle()
			rv.Recycle()
			return false
		}
		lv.Recycle()
		rv.Recycle()
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
	i := 0
	for {
		if i >= lhs.Length() && i >= rhs.Length() {
			// Lists are the same length and all items are equal.
			return 0
		}
		if i >= lhs.Length() {
			// LHS is shorter.
			return -1
		}
		if i >= rhs.Length() {
			// RHS is shorter.
			return 1
		}
		lv := lhs.At(i)
		rv := rhs.At(i)
		if c := Compare(lv, rv); c != 0 {
			return c
		}
		lv.Recycle()
		rv.Recycle()
		// The items are equal; continue.
		i++
	}
}
