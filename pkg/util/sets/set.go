/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"reflect"
	"sort"
)

// Empty is public since it is used by some internal API objects for conversions between external
// string arrays and internal sets, and conversion logic requires public types today.
type Empty struct{}

// StringSet is a set of strings, implemented via map[string]struct{} for minimal memory consumption.
type String map[string]Empty

// New creates a StringSet from a list of values.
func NewString(items ...string) String {
	ss := String{}
	ss.Insert(items...)
	return ss
}

// KeySet creates a StringSet from a keys of a map[string](? extends interface{}).  Since you can't describe that map type in the Go type system
// the reflected value is required.
func KeySet(theMap reflect.Value) String {
	ret := String{}

	for _, keyValue := range theMap.MapKeys() {
		ret.Insert(keyValue.String())
	}

	return ret
}

// Insert adds items to the set.
func (s String) Insert(items ...string) {
	for _, item := range items {
		s[item] = Empty{}
	}
}

// Delete removes all items from the set.
func (s String) Delete(items ...string) {
	for _, item := range items {
		delete(s, item)
	}
}

// Has returns true iff item is contained in the set.
func (s String) Has(item string) bool {
	_, contained := s[item]
	return contained
}

// HasAll returns true iff all items are contained in the set.
func (s String) HasAll(items ...string) bool {
	for _, item := range items {
		if !s.Has(item) {
			return false
		}
	}
	return true
}

// HasAny returns true if any items are contained in the set.
func (s String) HasAny(items ...string) bool {
	for _, item := range items {
		if s.Has(item) {
			return true
		}
	}
	return false
}

// Difference returns a set of objects that are not in s2
// For example:
// s1 = {1, 2, 3}
// s2 = {1, 2, 4, 5}
// s1.Difference(s2) = {3}
// s2.Difference(s1) = {4, 5}
func (s String) Difference(s2 String) String {
	result := NewString()
	for key := range s {
		if !s2.Has(key) {
			result.Insert(key)
		}
	}
	return result
}

// Union returns a new set which includes items in either s1 or s2.
// vof objects that are not in s2
// For example:
// s1 = {1, 2}
// s2 = {3, 4}
// s1.Union(s2) = {1, 2, 3, 4}
// s2.Union(s1) = {1, 2, 3, 4}
func (s1 String) Union(s2 String) String {
	result := NewString()
	for key := range s1 {
		result.Insert(key)
	}
	for key := range s2 {
		result.Insert(key)
	}
	return result
}

// IsSuperset returns true iff s1 is a superset of s2.
func (s1 String) IsSuperset(s2 String) bool {
	for item := range s2 {
		if !s1.Has(item) {
			return false
		}
	}
	return true
}

// Equal returns true iff s1 is equal (as a set) to s2.
// Two sets are equal if their membership is identical.
// (In practice, this means same elements, order doesn't matter)
func (s1 String) Equal(s2 String) bool {
	if len(s1) != len(s2) {
		return false
	}
	for item := range s2 {
		if !s1.Has(item) {
			return false
		}
	}
	return true
}

// List returns the contents as a sorted string slice.
func (s String) List() []string {
	res := make([]string, 0, len(s))
	for key := range s {
		res = append(res, key)
	}
	sort.StringSlice(res).Sort()
	return res
}

// Returns a single element from the set.
func (s String) PopAny() (string, bool) {
	for key := range s {
		s.Delete(key)
		return key, true
	}
	return "", false
}

// Len returns the size of the set.
func (s String) Len() int {
	return len(s)
}
