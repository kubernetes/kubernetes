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
	"sort"
	"strings"
)

// Map represents a Map or go structure.
type Map interface {
	// Set changes or set the value of the given key.
	Set(key string, val Value)
	// Get returns the value for the given key, if present, or (nil, false) otherwise.
	Get(key string) (Value, bool)
	// Has returns true if the key is present, or false otherwise.
	Has(key string) bool
	// Delete removes the key from the map.
	Delete(key string)
	// Equals compares the two maps, and return true if they are the same, false otherwise.
	// Implementations can use MapEquals as a general implementation for this methods.
	Equals(other Map) bool
	// Iterate runs the given function for each key/value in the
	// map. Returning false in the closure prematurely stops the
	// iteration.
	Iterate(func(key string, value Value) bool) bool
	// Length returns the number of items in the map.
	Length() int
}

// MapLess compares two maps lexically.
func MapLess(lhs, rhs Map) bool {
	return MapCompare(lhs, rhs) == -1
}

// MapCompare compares two maps lexically.
func MapCompare(lhs, rhs Map) int {
	lorder := make([]string, 0, lhs.Length())
	lhs.Iterate(func(key string, _ Value) bool {
		lorder = append(lorder, key)
		return true
	})
	sort.Strings(lorder)
	rorder := make([]string, 0, rhs.Length())
	rhs.Iterate(func(key string, _ Value) bool {
		rorder = append(rorder, key)
		return true
	})
	sort.Strings(rorder)

	i := 0
	for {
		if i >= len(lorder) && i >= len(rorder) {
			// Maps are the same length and all items are equal.
			return 0
		}
		if i >= len(lorder) {
			// LHS is shorter.
			return -1
		}
		if i >= len(rorder) {
			// RHS is shorter.
			return 1
		}
		if c := strings.Compare(lorder[i], rorder[i]); c != 0 {
			return c
		}
		litem, _ := lhs.Get(lorder[i])
		ritem, _ := rhs.Get(rorder[i])
		if c := Compare(litem, ritem); c != 0 {
			return c
		}
		litem.Recycle()
		ritem.Recycle()
		// The items are equal; continue.
		i++
	}
}

// MapEquals returns true if lhs == rhs, false otherwise. This function
// acts on generic types and should not be used by callers, but can help
// implement Map.Equals.
func MapEquals(lhs, rhs Map) bool {
	if lhs.Length() != rhs.Length() {
		return false
	}
	return lhs.Iterate(func(k string, v Value) bool {
		vo, ok := rhs.Get(k)
		if !ok {
			return false
		}
		if !Equals(v, vo) {
			vo.Recycle()
			return false
		}
		vo.Recycle()
		return true
	})
}
