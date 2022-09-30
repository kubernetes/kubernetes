/*
Copyright 2022 The Kubernetes Authors.

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

package sets

import (
	"sort"

	"golang.org/x/exp/constraints"
)

// Generic is a set of items, implemented via map[T]struct{} for minimal memory consumption.
type Generic[T comparable] map[T]Empty

// NewGeneric creates a Generic from a list of values.
func NewGeneric[T comparable](items ...T) Generic[T] {
	g := make(Generic[T], len(items))
	g.Insert(items...)
	return g
}

// GenericKeySet creates a generic set from the keys of a map[T].
func GenericKeySet[T comparable, V any](theMap map[T]V) Generic[T] {
	g := make(Generic[T], len(theMap))
	for key := range theMap {
		g.Insert(key)
	}
	return g
}

// Insert adds items to the set.
func (g Generic[T]) Insert(items ...T) Generic[T] {
	for _, item := range items {
		g[item] = Empty{}
	}
	return g
}

// Delete removes all items from the set.
func (g Generic[T]) Delete(items ...T) Generic[T] {
	for _, item := range items {
		delete(g, item)
	}
	return g
}

// Has returns true if and only if item is contained in the set.
func (g Generic[T]) Has(item T) bool {
	_, contained := g[item]
	return contained
}

// HasAll returns true if and only if all items are contained in the set.
func (g Generic[T]) HasAll(items ...T) bool {
	for _, item := range items {
		if !g.Has(item) {
			return false
		}
	}
	return true
}

// HasAny returns true if any items are contained in the set.
func (g Generic[T]) HasAny(items ...T) bool {
	for _, item := range items {
		if g.Has(item) {
			return true
		}
	}
	return false
}

// Clone returns a new set which is a copy of the current set.
func (g Generic[T]) Clone() Generic[T] {
	result := make(Generic[T], len(g))
	for key := range g {
		result.Insert(key)
	}
	return result
}

// Difference returns a set of objects that are not in s2.
// For example:
// s1 = {a1, a2, a3}
// s2 = {a1, a2, a4, a5}
// s1.Difference(s2) = {a3}
// s2.Difference(s1) = {a4, a5}
func (s1 Generic[T]) Difference(s2 Generic[T]) Generic[T] {
	result := NewGeneric[T]()
	for key := range s1 {
		if !s2.Has(key) {
			result.Insert(key)
		}
	}
	return result
}

// SymmetricDifference returns a set of elements which are in either of the sets, but not in their intersection.
// For example:
// s1 = {a1, a2, a3}
// s2 = {a1, a2, a4, a5}
// s1.SymmetricDifference(s2) = {a3, a4, a5}
// s2.SymmetricDifference(s1) = {a3, a4, a5}
func (s1 Generic[T]) SymmetricDifference(s2 Generic[T]) Generic[T] {
	return s1.Difference(s2).Union(s2.Difference(s1))
}

// Union returns a new set which includes items in either s1 or s2.
// For example:
// s1 = {a1, a2}
// s2 = {a3, a4}
// s1.Union(s2) = {a1, a2, a3, a4}
// s2.Union(s1) = {a1, a2, a3, a4}
func (s1 Generic[T]) Union(s2 Generic[T]) Generic[T] {
	result := s1.Clone()
	for key := range s2 {
		result.Insert(key)
	}
	return result
}

// Intersection returns a new set which includes the item in BOTH s1 and s2
// For example:
// s1 = {a1, a2}
// s2 = {a2, a3}
// s1.Intersection(s2) = {a2}
func (s1 Generic[T]) Intersection(s2 Generic[T]) Generic[T] {
	var walk, other Generic[T]
	result := NewGeneric[T]()
	if s1.Len() < s2.Len() {
		walk = s1
		other = s2
	} else {
		walk = s2
		other = s1
	}
	for key := range walk {
		if other.Has(key) {
			result.Insert(key)
		}
	}
	return result
}

// IsSuperset returns true if and only if s1 is a superset of s2.
func (s1 Generic[T]) IsSuperset(s2 Generic[T]) bool {
	for item := range s2 {
		if !s1.Has(item) {
			return false
		}
	}
	return true
}

// Equal returns true if and only if s1 is equal (as a set) to s2.
// Two sets are equal if their membership is identical.
// (In practice, this means same elements, order doesn't matter)
func (s1 Generic[T]) Equal(s2 Generic[T]) bool {
	return len(s1) == len(s2) && s1.IsSuperset(s2)
}

type sortableSliceOfGeneric[T constraints.Ordered] []T

func (g sortableSliceOfGeneric[T]) Len() int           { return len(g) }
func (g sortableSliceOfGeneric[T]) Less(i, j int) bool { return lessGeneric[T](g[i], g[j]) }
func (g sortableSliceOfGeneric[T]) Swap(i, j int)      { g[i], g[j] = g[j], g[i] }

// List returns the contents as a sorted T slice.
//
// This is a separate function and not a method because not all types supported
// by Generic are ordered and only those can be sorted.
func List[T constraints.Ordered](g Generic[T]) []T {
	res := make(sortableSliceOfGeneric[T], 0, len(g))
	for key := range g {
		res = append(res, key)
	}
	sort.Sort(res)
	return []T(res)
}

// UnsortedList returns the slice with contents in random order.
func (g Generic[T]) UnsortedList() []T {
	res := make([]T, 0, len(g))
	for key := range g {
		res = append(res, key)
	}
	return res
}

// Returns a single element from the set.
func (g Generic[T]) PopAny() (T, bool) {
	for key := range g {
		g.Delete(key)
		return key, true
	}
	var zeroValue T
	return zeroValue, false
}

// Len returns the size of the set.
func (g Generic[T]) Len() int {
	return len(g)
}

func lessGeneric[T constraints.Ordered](lhs, rhs T) bool {
	return lhs < rhs
}
