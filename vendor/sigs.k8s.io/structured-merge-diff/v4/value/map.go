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
)

// Map represents a Map or go structure.
type Map interface {
	// Set changes or set the value of the given key.
	Set(key string, val Value)
	// Get returns the value for the given key, if present, or (nil, false) otherwise.
	Get(key string) (Value, bool)
	// GetUsing uses the provided allocator and returns the value for the given key,
	// if present, or (nil, false) otherwise.
	// The returned Value should be given back to the Allocator when no longer needed
	// by calling Allocator.Free(Value).
	GetUsing(a Allocator, key string) (Value, bool)
	// Has returns true if the key is present, or false otherwise.
	Has(key string) bool
	// Delete removes the key from the map.
	Delete(key string)
	// Equals compares the two maps, and return true if they are the same, false otherwise.
	// Implementations can use MapEquals as a general implementation for this methods.
	Equals(other Map) bool
	// EqualsUsing uses the provided allocator and compares the two maps, and return true if
	// they are the same, false otherwise. Implementations can use MapEqualsUsing as a general
	// implementation for this methods.
	EqualsUsing(a Allocator, other Map) bool
	// Iterate runs the given function for each key/value in the
	// map. Returning false in the closure prematurely stops the
	// iteration.
	Iterate(func(key string, value Value) bool) bool
	// IterateUsing uses the provided allocator and runs the given function for each key/value
	// in the map. Returning false in the closure prematurely stops the iteration.
	IterateUsing(Allocator, func(key string, value Value) bool) bool
	// Length returns the number of items in the map.
	Length() int
	// Empty returns true if the map is empty.
	Empty() bool
	// Zip iterates over the entries of two maps together. If both maps contain a value for a given key, fn is called
	// with the values from both maps, otherwise it is called with the value of the map that contains the key and nil
	// for the map that does not contain the key. Returning false in the closure prematurely stops the iteration.
	Zip(other Map, order MapTraverseOrder, fn func(key string, lhs, rhs Value) bool) bool
	// ZipUsing uses the provided allocator and iterates over the entries of two maps together. If both maps
	// contain a value for a given key, fn is called with the values from both maps, otherwise it is called with
	// the value of the map that contains the key and nil for the map that does not contain the key. Returning
	// false in the closure prematurely stops the iteration.
	ZipUsing(a Allocator, other Map, order MapTraverseOrder, fn func(key string, lhs, rhs Value) bool) bool
}

// MapTraverseOrder defines the map traversal ordering available.
type MapTraverseOrder int

const (
	// Unordered indicates that the map traversal has no ordering requirement.
	Unordered = iota
	// LexicalKeyOrder indicates that the map traversal is ordered by key, lexically.
	LexicalKeyOrder
)

// MapZip iterates over the entries of two maps together. If both maps contain a value for a given key, fn is called
// with the values from both maps, otherwise it is called with the value of the map that contains the key and nil
// for the other map. Returning false in the closure prematurely stops the iteration.
func MapZip(lhs, rhs Map, order MapTraverseOrder, fn func(key string, lhs, rhs Value) bool) bool {
	return MapZipUsing(HeapAllocator, lhs, rhs, order, fn)
}

// MapZipUsing uses the provided allocator and iterates over the entries of two maps together. If both maps
// contain a value for a given key, fn is called with the values from both maps, otherwise it is called with
// the value of the map that contains the key and nil for the other map. Returning false in the closure
// prematurely stops the iteration.
func MapZipUsing(a Allocator, lhs, rhs Map, order MapTraverseOrder, fn func(key string, lhs, rhs Value) bool) bool {
	if lhs != nil {
		return lhs.ZipUsing(a, rhs, order, fn)
	}
	if rhs != nil {
		return rhs.ZipUsing(a, lhs, order, func(key string, rhs, lhs Value) bool { // arg positions of lhs and rhs deliberately swapped
			return fn(key, lhs, rhs)
		})
	}
	return true
}

// defaultMapZip provides a default implementation of Zip for implementations that do not need to provide
// their own optimized implementation.
func defaultMapZip(a Allocator, lhs, rhs Map, order MapTraverseOrder, fn func(key string, lhs, rhs Value) bool) bool {
	switch order {
	case Unordered:
		return unorderedMapZip(a, lhs, rhs, fn)
	case LexicalKeyOrder:
		return lexicalKeyOrderedMapZip(a, lhs, rhs, fn)
	default:
		panic("Unsupported map order")
	}
}

func unorderedMapZip(a Allocator, lhs, rhs Map, fn func(key string, lhs, rhs Value) bool) bool {
	if (lhs == nil || lhs.Empty()) && (rhs == nil || rhs.Empty()) {
		return true
	}

	if lhs != nil {
		ok := lhs.IterateUsing(a, func(key string, lhsValue Value) bool {
			var rhsValue Value
			if rhs != nil {
				if item, ok := rhs.GetUsing(a, key); ok {
					rhsValue = item
					defer a.Free(rhsValue)
				}
			}
			return fn(key, lhsValue, rhsValue)
		})
		if !ok {
			return false
		}
	}
	if rhs != nil {
		return rhs.IterateUsing(a, func(key string, rhsValue Value) bool {
			if lhs == nil || !lhs.Has(key) {
				return fn(key, nil, rhsValue)
			}
			return true
		})
	}
	return true
}

func lexicalKeyOrderedMapZip(a Allocator, lhs, rhs Map, fn func(key string, lhs, rhs Value) bool) bool {
	var lhsLength, rhsLength int
	var orderedLength int // rough estimate of length of union of map keys
	if lhs != nil {
		lhsLength = lhs.Length()
		orderedLength = lhsLength
	}
	if rhs != nil {
		rhsLength = rhs.Length()
		if rhsLength > orderedLength {
			orderedLength = rhsLength
		}
	}
	if lhsLength == 0 && rhsLength == 0 {
		return true
	}

	ordered := make([]string, 0, orderedLength)
	if lhs != nil {
		lhs.IterateUsing(a, func(key string, _ Value) bool {
			ordered = append(ordered, key)
			return true
		})
	}
	if rhs != nil {
		rhs.IterateUsing(a, func(key string, _ Value) bool {
			if lhs == nil || !lhs.Has(key) {
				ordered = append(ordered, key)
			}
			return true
		})
	}
	sort.Strings(ordered)
	for _, key := range ordered {
		var litem, ritem Value
		if lhs != nil {
			litem, _ = lhs.GetUsing(a, key)
		}
		if rhs != nil {
			ritem, _ = rhs.GetUsing(a, key)
		}
		ok := fn(key, litem, ritem)
		if litem != nil {
			a.Free(litem)
		}
		if ritem != nil {
			a.Free(ritem)
		}
		if !ok {
			return false
		}
	}
	return true
}

// MapLess compares two maps lexically.
func MapLess(lhs, rhs Map) bool {
	return MapCompare(lhs, rhs) == -1
}

// MapCompare compares two maps lexically.
func MapCompare(lhs, rhs Map) int {
	return MapCompareUsing(HeapAllocator, lhs, rhs)
}

// MapCompareUsing uses the provided allocator and compares two maps lexically.
func MapCompareUsing(a Allocator, lhs, rhs Map) int {
	c := 0
	var llength, rlength int
	if lhs != nil {
		llength = lhs.Length()
	}
	if rhs != nil {
		rlength = rhs.Length()
	}
	if llength == 0 && rlength == 0 {
		return 0
	}
	i := 0
	MapZipUsing(a, lhs, rhs, LexicalKeyOrder, func(key string, lhs, rhs Value) bool {
		switch {
		case i == llength:
			c = -1
		case i == rlength:
			c = 1
		case lhs == nil:
			c = 1
		case rhs == nil:
			c = -1
		default:
			c = CompareUsing(a, lhs, rhs)
		}
		i++
		return c == 0
	})
	return c
}

// MapEquals returns true if lhs == rhs, false otherwise. This function
// acts on generic types and should not be used by callers, but can help
// implement Map.Equals.
// WARN: This is a naive implementation, calling lhs.Equals(rhs) is typically the most efficient.
func MapEquals(lhs, rhs Map) bool {
	return MapEqualsUsing(HeapAllocator, lhs, rhs)
}

// MapEqualsUsing uses the provided allocator and returns true if lhs == rhs,
// false otherwise. This function acts on generic types and should not be used
// by callers, but can help implement Map.Equals.
// WARN: This is a naive implementation, calling lhs.EqualsUsing(allocator, rhs) is typically the most efficient.
func MapEqualsUsing(a Allocator, lhs, rhs Map) bool {
	if lhs == nil && rhs == nil {
		return true
	}
	if lhs == nil || rhs == nil {
		return false
	}
	if lhs.Length() != rhs.Length() {
		return false
	}
	return MapZipUsing(a, lhs, rhs, Unordered, func(key string, lhs, rhs Value) bool {
		if lhs == nil || rhs == nil {
			return false
		}
		return EqualsUsing(a, lhs, rhs)
	})
}
