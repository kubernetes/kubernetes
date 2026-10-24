/*
Copyright 2023 The Kubernetes Authors.

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

package set

import (
	"sort"
)

// Empty is public since it is used by some internal API objects for conversions between external
// string arrays and internal sets, and conversion logic requires public types today.
type Empty struct{}

// Set is a set of the same type elements, implemented via map[ordered]struct{} for minimal memory consumption.
type Set[E ordered] map[E]Empty

// New creates a new set.
func New[E ordered](items ...E) Set[E] {
	ss := Set[E]{}
	ss.Insert(items...)
	return ss
}

// Clear empties the set.
// It is preferable to replace the set with a newly constructed set,
// but not all callers can do that (when there are other references to the map).
func (s Set[T]) Clear() Set[T] {
	clear(s)
	return s
}

// KeySet creates a Set[E] from a keys of a map[E](? extends interface{}).
func KeySet[E ordered, A any](theMap map[E]A) Set[E] {
	ret := Set[E]{}
	for key := range theMap {
		ret.Insert(key)
	}
	return ret
}

// Insert adds items to the set.
func (s Set[E]) Insert(items ...E) Set[E] {
	for _, item := range items {
		s[item] = Empty{}
	}
	return s
}

// Delete removes all items from the set.
func (s Set[E]) Delete(items ...E) Set[E] {
	for _, item := range items {
		delete(s, item)
	}
	return s
}

// Has returns true if and only if item is contained in the set.
func (s Set[E]) Has(item E) bool {
	_, contained := s[item]
	return contained
}

// HasAll returns true if and only if all items are contained in the set.
func (s Set[E]) HasAll(items ...E) bool {
	for _, item := range items {
		if !s.Has(item) {
			return false
		}
	}
	return true
}

// HasAny returns true if any items are contained in the set.
func (s Set[E]) HasAny(items ...E) bool {
	for _, item := range items {
		if s.Has(item) {
			return true
		}
	}
	return false
}

// Union returns a new set which includes items in either s1 or s2.
// For example:
// s1 = {a1, a2}
// s2 = {a3, a4}
// s1.Union(s2) = {a1, a2, a3, a4}
// s2.Union(s1) = {a1, a2, a3, a4}
func (s Set[E]) Union(s2 Set[E]) Set[E] {
	result := Set[E]{}
	result.Insert(s.UnsortedList()...)
	result.Insert(s2.UnsortedList()...)
	return result
}

// Len returns the number of elements in the set.
func (s Set[E]) Len() int {
	return len(s)
}

// Intersection returns a new set which includes the item in BOTH s1 and s2
// For example:
// s1 = {a1, a2}
// s2 = {a2, a3}
// s1.Intersection(s2) = {a2}
func (s Set[E]) Intersection(s2 Set[E]) Set[E] {
	var walk, other Set[E]
	result := Set[E]{}
	if s.Len() < s2.Len() {
		walk = s
		other = s2
	} else {
		walk = s2
		other = s
	}
	for key := range walk {
		if other.Has(key) {
			result.Insert(key)
		}
	}
	return result
}

// IsSuperset returns true if and only if s1 is a superset of s2.
func (s Set[E]) IsSuperset(s2 Set[E]) bool {
	for item := range s2 {
		if !s.Has(item) {
			return false
		}
	}
	return true
}

// Difference returns a set of objects that are not in s2
// For example:
// s1 = {a1, a2, a3}
// s2 = {a1, a2, a4, a5}
// s1.Difference(s2) = {a3}
// s2.Difference(s1) = {a4, a5}
func (s Set[E]) Difference(s2 Set[E]) Set[E] {
	result := Set[E]{}
	for key := range s {
		if !s2.Has(key) {
			result.Insert(key)
		}
	}
	return result
}

// Equal returns true if and only if s1 is equal (as a set) to s2.
// Two sets are equal if their membership is identical.
func (s Set[E]) Equal(s2 Set[E]) bool {
	return s.Len() == s2.Len() && s.IsSuperset(s2)
}

type sortableSlice[E ordered] []E

func (s sortableSlice[E]) Len() int {
	return len(s)
}
func (s sortableSlice[E]) Less(i, j int) bool { return s[i] < s[j] }
func (s sortableSlice[E]) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }

// SortedList returns the contents as a sorted slice.
func (s Set[E]) SortedList() []E {
	res := make(sortableSlice[E], 0, s.Len())
	for key := range s {
		res = append(res, key)
	}
	sort.Sort(res)
	return res
}

// UnsortedList returns the slice with contents in random order.
func (s Set[E]) UnsortedList() []E {
	res := make([]E, 0, len(s))
	for key := range s {
		res = append(res, key)
	}
	return res
}

// PopAny returns a single element from the set.
func (s Set[E]) PopAny() (E, bool) {
	for key := range s {
		s.Delete(key)
		return key, true
	}
	var zeroValue E
	return zeroValue, false
}

// Clone returns a new set which is a copy of the current set.
func (s Set[T]) Clone() Set[T] {
	result := make(Set[T], len(s))
	for key := range s {
		result.Insert(key)
	}
	return result
}

// SymmetricDifference returns a set of elements which are in either of the sets, but not in their intersection.
// For example:
// s1 = {a1, a2, a3}
// s2 = {a1, a2, a4, a5}
// s1.SymmetricDifference(s2) = {a3, a4, a5}
// s2.SymmetricDifference(s1) = {a3, a4, a5}
func (s Set[T]) SymmetricDifference(s2 Set[T]) Set[T] {
	return s.Difference(s2).Union(s2.Difference(s))
}
