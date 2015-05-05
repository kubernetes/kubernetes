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

package util

import (
	"reflect"
	"sort"
)

type empty struct{}

// StringSet is a set of strings, implemented via map[string]struct{} for minimal memory consumption.
type StringSet map[string]empty

// NewStringSet creates a StringSet from a list of values.
func NewStringSet(items ...string) StringSet {
	ss := StringSet{}
	ss.Insert(items...)
	return ss
}

// KeySet creates a StringSet from a keys of a map[string](? extends interface{}).  Since you can't describe that map type in the Go type system
// the reflected value is required.
func KeySet(theMap reflect.Value) StringSet {
	ret := StringSet{}

	for _, keyValue := range theMap.MapKeys() {
		ret.Insert(keyValue.String())
	}

	return ret
}

// Insert adds items to the set.
func (s StringSet) Insert(items ...string) {
	for _, item := range items {
		s[item] = empty{}
	}
}

// Delete removes all items from the set.
func (s StringSet) Delete(items ...string) {
	for _, item := range items {
		delete(s, item)
	}
}

// Has returns true iff item is contained in the set.
func (s StringSet) Has(item string) bool {
	_, contained := s[item]
	return contained
}

// HasAll returns true iff all items are contained in the set.
func (s StringSet) HasAll(items ...string) bool {
	for _, item := range items {
		if !s.Has(item) {
			return false
		}
	}
	return true
}

// HasAny returns true if any items are contained in the set.
func (s StringSet) HasAny(items ...string) bool {
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
func (s StringSet) Difference(s2 StringSet) StringSet {
	result := NewStringSet()
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
func (s1 StringSet) Union(s2 StringSet) StringSet {
	result := NewStringSet()
	for key := range s1 {
		result.Insert(key)
	}
	for key := range s2 {
		result.Insert(key)
	}
	return result
}

// IsSuperset returns true iff s1 is a superset of s2.
func (s1 StringSet) IsSuperset(s2 StringSet) bool {
	for item := range s2 {
		if !s1.Has(item) {
			return false
		}
	}
	return true
}

// List returns the contents as a sorted string slice.
func (s StringSet) List() []string {
	res := make([]string, 0, len(s))
	for key := range s {
		res = append(res, key)
	}
	sort.StringSlice(res).Sort()
	return res
}

// Returns a single element from the set.
func (s StringSet) PopAny() (string, bool) {
	for key := range s {
		s.Delete(key)
		return key, true
	}
	return "", false
}

// Len returns the size of the set.
func (s StringSet) Len() int {
	return len(s)
}
