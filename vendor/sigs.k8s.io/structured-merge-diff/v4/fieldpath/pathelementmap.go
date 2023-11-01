/*
Copyright 2018 The Kubernetes Authors.

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

package fieldpath

import (
	"sort"

	"sigs.k8s.io/structured-merge-diff/v4/value"
)

// PathElementValueMap is a map from PathElement to value.Value.
//
// TODO(apelisse): We have multiple very similar implementation of this
// for PathElementSet and SetNodeMap, so we could probably share the
// code.
type PathElementValueMap struct {
	valueMap PathElementMap
}

func MakePathElementValueMap(size int) PathElementValueMap {
	return PathElementValueMap{
		valueMap: MakePathElementMap(size),
	}
}

type sortedPathElementValues []pathElementValue

// Implement the sort interface; this would permit bulk creation, which would
// be faster than doing it one at a time via Insert.
func (spev sortedPathElementValues) Len() int { return len(spev) }
func (spev sortedPathElementValues) Less(i, j int) bool {
	return spev[i].PathElement.Less(spev[j].PathElement)
}
func (spev sortedPathElementValues) Swap(i, j int) { spev[i], spev[j] = spev[j], spev[i] }

// Insert adds the pathelement and associated value in the map.
// If insert is called twice with the same PathElement, the value is replaced.
func (s *PathElementValueMap) Insert(pe PathElement, v value.Value) {
	s.valueMap.Insert(pe, v)
}

// Get retrieves the value associated with the given PathElement from the map.
// (nil, false) is returned if there is no such PathElement.
func (s *PathElementValueMap) Get(pe PathElement) (value.Value, bool) {
	v, ok := s.valueMap.Get(pe)
	if !ok {
		return nil, false
	}
	return v.(value.Value), true
}

// PathElementValueMap is a map from PathElement to interface{}.
type PathElementMap struct {
	members sortedPathElementValues
}

type pathElementValue struct {
	PathElement PathElement
	Value       interface{}
}

func MakePathElementMap(size int) PathElementMap {
	return PathElementMap{
		members: make(sortedPathElementValues, 0, size),
	}
}

// Insert adds the pathelement and associated value in the map.
// If insert is called twice with the same PathElement, the value is replaced.
func (s *PathElementMap) Insert(pe PathElement, v interface{}) {
	loc := sort.Search(len(s.members), func(i int) bool {
		return !s.members[i].PathElement.Less(pe)
	})
	if loc == len(s.members) {
		s.members = append(s.members, pathElementValue{pe, v})
		return
	}
	if s.members[loc].PathElement.Equals(pe) {
		s.members[loc].Value = v
		return
	}
	s.members = append(s.members, pathElementValue{})
	copy(s.members[loc+1:], s.members[loc:])
	s.members[loc] = pathElementValue{pe, v}
}

// Get retrieves the value associated with the given PathElement from the map.
// (nil, false) is returned if there is no such PathElement.
func (s *PathElementMap) Get(pe PathElement) (interface{}, bool) {
	loc := sort.Search(len(s.members), func(i int) bool {
		return !s.members[i].PathElement.Less(pe)
	})
	if loc == len(s.members) {
		return nil, false
	}
	if s.members[loc].PathElement.Equals(pe) {
		return s.members[loc].Value, true
	}
	return nil, false
}
