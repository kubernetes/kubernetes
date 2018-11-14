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
	"fmt"
	"sort"
	"strings"

	"sigs.k8s.io/structured-merge-diff/value"
)

// PathElement describes how to select a child field given a containing object.
type PathElement struct {
	// Exactly one of the following fields should be non-nil.

	// FieldName selects a single field from a map (reminder: this is also
	// how structs are represented). The containing object must be a map.
	FieldName *string

	// Key selects the list element which has fields matching those given.
	// The containing object must be an associative list with map typed
	// elements.
	Key []value.Field

	// Value selects the list element with the given value. The containing
	// object must be an associative list with a primitive typed element
	// (i.e., a set).
	Value *value.Value

	// Index selects a list element by its index number. The containing
	// object must be an atomic list.
	Index *int
}

// String presents the path element as a human-readable string.
func (e PathElement) String() string {
	switch {
	case e.FieldName != nil:
		return "." + *e.FieldName
	case len(e.Key) > 0:
		strs := make([]string, len(e.Key))
		for i, k := range e.Key {
			strs[i] = fmt.Sprintf("%v=%v", k.Name, k.Value)
		}
		// The order must be canonical, since we use the string value
		// in a set structure.
		sort.Strings(strs)
		return "[" + strings.Join(strs, ",") + "]"
	case e.Value != nil:
		return fmt.Sprintf("[=%v]", e.Value)
	case e.Index != nil:
		return fmt.Sprintf("[%v]", *e.Index)
	default:
		return "{{invalid path element}}"
	}
}

// KeyByFields is a helper function which constructs a key for an associative
// list type. `nameValues` must have an even number of entries, alternating
// names (type must be string) with values (type must be value.Value). If these
// conditions are not met, KeyByFields will panic--it's intended for static
// construction and shouldn't have user-produced values passed to it.
func KeyByFields(nameValues ...interface{}) []value.Field {
	if len(nameValues)%2 != 0 {
		panic("must have a value for every name")
	}
	out := []value.Field{}
	for i := 0; i < len(nameValues)-1; i += 2 {
		out = append(out, value.Field{
			Name:  nameValues[i].(string),
			Value: nameValues[i+1].(value.Value),
		})
	}
	return out
}

// PathElementSet is a set of path elements.
// TODO: serialize as a list.
type PathElementSet struct {
	// The strange construction is because there's no way to test
	// PathElements for equality (it can't be used as a key for a map).
	members map[string]PathElement
}

// Insert adds pe to the set.
func (s *PathElementSet) Insert(pe PathElement) {
	serialized := pe.String()
	if s.members == nil {
		s.members = map[string]PathElement{
			serialized: pe,
		}
		return
	}
	if _, ok := s.members[serialized]; !ok {
		s.members[serialized] = pe
	}
}

// Union returns a set containing elements that appear in either s or s2.
func (s *PathElementSet) Union(s2 *PathElementSet) *PathElementSet {
	out := &PathElementSet{
		members: map[string]PathElement{},
	}
	for k, v := range s.members {
		out.members[k] = v
	}
	for k, v := range s2.members {
		out.members[k] = v
	}
	return out
}

// Intersection returns a set containing elements which appear in both s and s2.
func (s *PathElementSet) Intersection(s2 *PathElementSet) *PathElementSet {
	out := &PathElementSet{
		members: map[string]PathElement{},
	}
	for k, v := range s.members {
		if _, ok := s2.members[k]; ok {
			out.members[k] = v
		}
	}
	return out
}

// Difference returns a set containing elements which appear in s but not in s2.
func (s *PathElementSet) Difference(s2 *PathElementSet) *PathElementSet {
	out := &PathElementSet{
		members: map[string]PathElement{},
	}
	for k, v := range s.members {
		if _, ok := s2.members[k]; !ok {
			out.members[k] = v
		}
	}
	return out
}

// Size retuns the number of elements in the set.
func (s *PathElementSet) Size() int { return len(s.members) }

// Has returns true if pe is a member of the set.
func (s *PathElementSet) Has(pe PathElement) bool {
	if s.members == nil {
		return false
	}
	_, ok := s.members[pe.String()]
	return ok
}

// Equals returns true if s and s2 have exactly the same members.
func (s *PathElementSet) Equals(s2 *PathElementSet) bool {
	if len(s.members) != len(s2.members) {
		return false
	}
	for k := range s.members {
		if _, ok := s2.members[k]; !ok {
			return false
		}
	}
	return true
}

// Iterate calls f for each PathElement in the set.
func (s *PathElementSet) Iterate(f func(PathElement)) {
	for _, pe := range s.members {
		f(pe)
	}
}
