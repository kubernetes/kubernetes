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
	"iter"
	"sort"
	"strings"

	"sigs.k8s.io/structured-merge-diff/v6/value"
)

// PathElement describes how to select a child field given a containing object.
type PathElement struct {
	// Exactly one of the following fields should be non-nil.

	// FieldName selects a single field from a map (reminder: this is also
	// how structs are represented). The containing object must be a map.
	FieldName *string

	// Key selects the list element which has fields matching those given.
	// The containing object must be an associative list with map typed
	// elements. They are sorted alphabetically.
	Key *value.FieldList

	// Value selects the list element with the given value. The containing
	// object must be an associative list with a primitive typed element
	// (i.e., a set).
	Value *value.Value

	// Index selects a list element by its index number. The containing
	// object must be an atomic list.
	Index *int
}

// FieldNameElement creates a new FieldName PathElement.
func FieldNameElement(name string) PathElement {
	return PathElement{FieldName: &name}
}

// KeyElement creates a new Key PathElement with the key fields.
func KeyElement(fields ...value.Field) PathElement {
	l := value.FieldList(fields)
	return PathElement{Key: &l}
}

// KeyElementByFields creates a new Key PathElement from names and values.
// `nameValues` must have an even number of entries, alternating
// names (type must be string) with values (type must be value.Value). If these
// conditions are not met, KeyByFields will panic--it's intended for static
// construction and shouldn't have user-produced values passed to it.
func KeyElementByFields(nameValues ...any) PathElement {
	return PathElement{Key: KeyByFields(nameValues...)}
}

// ValueElement creates a new Value PathElement.
func ValueElement(value value.Value) PathElement {
	return PathElement{Value: &value}
}

// IndexElement creates a new Index PathElement.
func IndexElement(index int) PathElement {
	return PathElement{Index: &index}
}

// Less provides an order for path elements.
func (e PathElement) Less(rhs PathElement) bool {
	return e.Compare(rhs) < 0
}

// Compare provides an order for path elements.
func (e PathElement) Compare(rhs PathElement) int {
	if e.FieldName != nil {
		if rhs.FieldName == nil {
			return -1
		}
		return strings.Compare(*e.FieldName, *rhs.FieldName)
	} else if rhs.FieldName != nil {
		return 1
	}

	if e.Key != nil {
		if rhs.Key == nil {
			return -1
		}
		return e.Key.Compare(*rhs.Key)
	} else if rhs.Key != nil {
		return 1
	}

	if e.Value != nil {
		if rhs.Value == nil {
			return -1
		}
		return value.Compare(*e.Value, *rhs.Value)
	} else if rhs.Value != nil {
		return 1
	}

	if e.Index != nil {
		if rhs.Index == nil {
			return -1
		}
		if *e.Index < *rhs.Index {
			return -1
		} else if *e.Index == *rhs.Index {
			return 0
		}
		return 1
	} else if rhs.Index != nil {
		return 1
	}

	return 0
}

// Equals returns true if both path elements are equal.
func (e PathElement) Equals(rhs PathElement) bool {
	if e.FieldName != nil {
		if rhs.FieldName == nil {
			return false
		}
		return *e.FieldName == *rhs.FieldName
	} else if rhs.FieldName != nil {
		return false
	}
	if e.Key != nil {
		if rhs.Key == nil {
			return false
		}
		return e.Key.Equals(*rhs.Key)
	} else if rhs.Key != nil {
		return false
	}
	if e.Value != nil {
		if rhs.Value == nil {
			return false
		}
		return value.Equals(*e.Value, *rhs.Value)
	} else if rhs.Value != nil {
		return false
	}
	if e.Index != nil {
		if rhs.Index == nil {
			return false
		}
		return *e.Index == *rhs.Index
	} else if rhs.Index != nil {
		return false
	}
	return true
}

// String presents the path element as a human-readable string.
func (e PathElement) String() string {
	switch {
	case e.FieldName != nil:
		return "." + *e.FieldName
	case e.Key != nil:
		strs := make([]string, len(*e.Key))
		for i, k := range *e.Key {
			strs[i] = fmt.Sprintf("%v=%v", k.Name, value.ToString(k.Value))
		}
		// Keys are supposed to be sorted.
		return "[" + strings.Join(strs, ",") + "]"
	case e.Value != nil:
		return fmt.Sprintf("[=%v]", value.ToString(*e.Value))
	case e.Index != nil:
		return fmt.Sprintf("[%v]", *e.Index)
	default:
		return "{{invalid path element}}"
	}
}

// Copy returns a copy of the PathElement.
// This is not a full deep copy as any contained value.Value is not copied.
func (e PathElement) Copy() PathElement {
	if e.FieldName != nil {
		return PathElement{FieldName: e.FieldName}
	}
	if e.Key != nil {
		c := e.Key.Copy()
		return PathElement{Key: &c}
	}
	if e.Value != nil {
		return PathElement{Value: e.Value}
	}
	if e.Index != nil {
		return PathElement{Index: e.Index}
	}
	return e // zero value
}

// KeyByFields is a helper function which constructs a key for an associative
// list type. `nameValues` must have an even number of entries, alternating
// names (type must be string) with values (type must be value.Value). If these
// conditions are not met, KeyByFields will panic--it's intended for static
// construction and shouldn't have user-produced values passed to it.
func KeyByFields(nameValues ...interface{}) *value.FieldList {
	if len(nameValues)%2 != 0 {
		panic("must have a value for every name")
	}
	out := value.FieldList{}
	for i := 0; i < len(nameValues)-1; i += 2 {
		out = append(out, value.Field{Name: nameValues[i].(string), Value: value.NewValueInterface(nameValues[i+1])})
	}
	out.Sort()
	return &out
}

// PathElementSet is a set of path elements.
// TODO: serialize as a list.
type PathElementSet struct {
	members sortedPathElements
}

func MakePathElementSet(size int) PathElementSet {
	return PathElementSet{
		members: make(sortedPathElements, 0, size),
	}
}

type sortedPathElements []PathElement

// Implement the sort interface; this would permit bulk creation, which would
// be faster than doing it one at a time via Insert.
func (spe sortedPathElements) Len() int           { return len(spe) }
func (spe sortedPathElements) Less(i, j int) bool { return spe[i].Less(spe[j]) }
func (spe sortedPathElements) Swap(i, j int)      { spe[i], spe[j] = spe[j], spe[i] }

// Copy returns a copy of the PathElementSet.
// This is not a full deep copy as any contained value.Value is not copied.
func (s PathElementSet) Copy() PathElementSet {
	out := make(sortedPathElements, len(s.members))
	for i := range s.members {
		out[i] = s.members[i].Copy()
	}
	return PathElementSet{members: out}
}

// Insert adds pe to the set.
func (s *PathElementSet) Insert(pe PathElement) {
	loc := sort.Search(len(s.members), func(i int) bool {
		return !s.members[i].Less(pe)
	})
	if loc == len(s.members) {
		s.members = append(s.members, pe)
		return
	}
	if s.members[loc].Equals(pe) {
		return
	}
	s.members = append(s.members, PathElement{})
	copy(s.members[loc+1:], s.members[loc:])
	s.members[loc] = pe
}

// Union returns a set containing elements that appear in either s or s2.
func (s *PathElementSet) Union(s2 *PathElementSet) *PathElementSet {
	out := &PathElementSet{}

	i, j := 0, 0
	for i < len(s.members) && j < len(s2.members) {
		if s.members[i].Less(s2.members[j]) {
			out.members = append(out.members, s.members[i])
			i++
		} else {
			out.members = append(out.members, s2.members[j])
			if !s2.members[j].Less(s.members[i]) {
				i++
			}
			j++
		}
	}

	if i < len(s.members) {
		out.members = append(out.members, s.members[i:]...)
	}
	if j < len(s2.members) {
		out.members = append(out.members, s2.members[j:]...)
	}
	return out
}

// Intersection returns a set containing elements which appear in both s and s2.
func (s *PathElementSet) Intersection(s2 *PathElementSet) *PathElementSet {
	out := &PathElementSet{}

	i, j := 0, 0
	for i < len(s.members) && j < len(s2.members) {
		if s.members[i].Less(s2.members[j]) {
			i++
		} else {
			if !s2.members[j].Less(s.members[i]) {
				out.members = append(out.members, s.members[i])
				i++
			}
			j++
		}
	}

	return out
}

// Difference returns a set containing elements which appear in s but not in s2.
func (s *PathElementSet) Difference(s2 *PathElementSet) *PathElementSet {
	out := &PathElementSet{}

	i, j := 0, 0
	for i < len(s.members) && j < len(s2.members) {
		if s.members[i].Less(s2.members[j]) {
			out.members = append(out.members, s.members[i])
			i++
		} else {
			if !s2.members[j].Less(s.members[i]) {
				i++
			}
			j++
		}
	}
	if i < len(s.members) {
		out.members = append(out.members, s.members[i:]...)
	}
	return out
}

// Size retuns the number of elements in the set.
func (s *PathElementSet) Size() int { return len(s.members) }

// Has returns true if pe is a member of the set.
func (s *PathElementSet) Has(pe PathElement) bool {
	loc := sort.Search(len(s.members), func(i int) bool {
		return !s.members[i].Less(pe)
	})
	if loc == len(s.members) {
		return false
	}
	if s.members[loc].Equals(pe) {
		return true
	}
	return false
}

// Equals returns true if s and s2 have exactly the same members.
func (s *PathElementSet) Equals(s2 *PathElementSet) bool {
	if len(s.members) != len(s2.members) {
		return false
	}
	for k := range s.members {
		if !s.members[k].Equals(s2.members[k]) {
			return false
		}
	}
	return true
}

// Iterate calls f for each PathElement in the set. The order is deterministic.
func (s *PathElementSet) Iterate(f func(PathElement)) {
	for _, pe := range s.members {
		f(pe)
	}
}

// All iterates over each PathElement in the set. The order is deterministic.
func (s *PathElementSet) All() iter.Seq[PathElement] {
	return func(yield func(element PathElement) bool) {
		for _, pe := range s.members {
			if !yield(pe) {
				return
			}
		}
	}
}
