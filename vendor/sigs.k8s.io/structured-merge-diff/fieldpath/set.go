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
	"strings"
)

// Set identifies a set of fields.
type Set struct {
	// Members lists fields that are part of the set.
	// TODO: will be serialized as a list of path elements.
	Members PathElementSet

	// Children lists child fields which themselves have children that are
	// members of the set. Appearance in this list does not imply membership.
	// Note: this is a tree, not an arbitrary graph.
	Children SetNodeMap
}

// NewSet makes a set from a list of paths.
func NewSet(paths ...Path) *Set {
	s := &Set{}
	for _, p := range paths {
		s.Insert(p)
	}
	return s
}

// Insert adds the field identified by `p` to the set. Important: parent fields
// are NOT added to the set; if that is desired, they must be added separately.
func (s *Set) Insert(p Path) {
	if len(p) == 0 {
		// Zero-length path identifies the entire object; we don't
		// track top-level ownership.
		return
	}
	for {
		if len(p) == 1 {
			s.Members.Insert(p[0])
			return
		}
		s = s.Children.Descend(p[0])
		p = p[1:]
	}
}

// Union returns a Set containing elements which appear in either s or s2.
func (s *Set) Union(s2 *Set) *Set {
	return &Set{
		Members:  *s.Members.Union(&s2.Members),
		Children: *s.Children.Union(&s2.Children),
	}
}

// Intersection returns a Set containing leaf elements which appear in both s
// and s2. Intersection can be constructed from Union and Difference operations
// (example in the tests) but it's much faster to do it in one pass.
func (s *Set) Intersection(s2 *Set) *Set {
	return &Set{
		Members:  *s.Members.Intersection(&s2.Members),
		Children: *s.Children.Intersection(&s2.Children),
	}
}

// Difference returns a Set containing elements which:
// * appear in s
// * do not appear in s2
// * and are not children of elements that appear in s2.
//
// In other words, for leaf fields, this acts like a regular set difference
// operation. When non leaf fields are compared with leaf fields ("parents"
// which contain "children"), the effect is:
// * parent - child = parent
// * child - parent = {empty set}
func (s *Set) Difference(s2 *Set) *Set {
	return &Set{
		Members:  *s.Members.Difference(&s2.Members),
		Children: *s.Children.Difference(s2),
	}
}

// Size returns the number of members of the set.
func (s *Set) Size() int {
	return s.Members.Size() + s.Children.Size()
}

// Empty returns true if there are no members of the set. It is a separate
// function from Size since it's common to check whether size > 0, and
// potentially much faster to return as soon as a single element is found.
func (s *Set) Empty() bool {
	if s.Members.Size() > 0 {
		return false
	}
	return s.Children.Empty()
}

// Has returns true if the field referenced by `p` is a member of the set.
func (s *Set) Has(p Path) bool {
	if len(p) == 0 {
		// No one owns "the entire object"
		return false
	}
	for {
		if len(p) == 1 {
			return s.Members.Has(p[0])
		}
		var ok bool
		s, ok = s.Children.Get(p[0])
		if !ok {
			return false
		}
		p = p[1:]
	}
}

// Equals returns true if s and s2 have exactly the same members.
func (s *Set) Equals(s2 *Set) bool {
	return s.Members.Equals(&s2.Members) && s.Children.Equals(&s2.Children)
}

// String returns the set one element per line.
func (s *Set) String() string {
	elements := []string{}
	s.Iterate(func(p Path) {
		elements = append(elements, p.String())
	})
	return strings.Join(elements, "\n")
}

// Iterate calls f once for each field that is a member of the set (preorder
// DFS). The path passed to f will be reused so make a copy if you wish to keep
// it.
func (s *Set) Iterate(f func(Path)) {
	s.iteratePrefix(Path{}, f)
}

func (s *Set) iteratePrefix(prefix Path, f func(Path)) {
	s.Members.Iterate(func(pe PathElement) { f(append(prefix, pe)) })
	s.Children.iteratePrefix(prefix, f)
}

// WithPrefix returns the subset of paths which begin with the given prefix,
// with the prefix not included.
func (s *Set) WithPrefix(pe PathElement) *Set {
	subset, ok := s.Children.Get(pe)
	if !ok {
		return NewSet()
	}
	return subset
}

// setNode is a pair of PathElement / Set, for the purpose of expressing
// nested set membership.
type setNode struct {
	pathElement PathElement
	set         *Set
}

// SetNodeMap is a map of PathElement to subset.
type SetNodeMap struct {
	members map[string]setNode
}

// Descend adds pe to the set if necessary, returning the associated subset.
func (s *SetNodeMap) Descend(pe PathElement) *Set {
	serialized := pe.String()
	if s.members == nil {
		s.members = map[string]setNode{}
	}
	if n, ok := s.members[serialized]; ok {
		return n.set
	}
	ss := &Set{}
	s.members[serialized] = setNode{
		pathElement: pe,
		set:         ss,
	}
	return ss
}

// Size returns the sum of the number of members of all subsets.
func (s *SetNodeMap) Size() int {
	count := 0
	for _, v := range s.members {
		count += v.set.Size()
	}
	return count
}

// Empty returns false if there's at least one member in some child set.
func (s *SetNodeMap) Empty() bool {
	for _, n := range s.members {
		if !n.set.Empty() {
			return false
		}
	}
	return true
}

// Get returns (the associated set, true) or (nil, false) if there is none.
func (s *SetNodeMap) Get(pe PathElement) (*Set, bool) {
	if s.members == nil {
		return nil, false
	}
	serialized := pe.String()
	if n, ok := s.members[serialized]; ok {
		return n.set, true
	}
	return nil, false
}

// Equals returns true if s and s2 have the same structure (same nested
// child sets).
func (s *SetNodeMap) Equals(s2 *SetNodeMap) bool {
	if len(s.members) != len(s2.members) {
		return false
	}
	for k, v := range s.members {
		v2, ok := s2.members[k]
		if !ok {
			return false
		}
		if !v.set.Equals(v2.set) {
			return false
		}
	}
	return true
}

// Union returns a SetNodeMap with members that appear in either s or s2.
func (s *SetNodeMap) Union(s2 *SetNodeMap) *SetNodeMap {
	out := &SetNodeMap{}
	for k, sn := range s.members {
		pe := sn.pathElement
		if sn2, ok := s2.members[k]; ok {
			*out.Descend(pe) = *sn.set.Union(sn2.set)
		} else {
			*out.Descend(pe) = *sn.set
		}
	}
	for k, sn2 := range s2.members {
		pe := sn2.pathElement
		if _, ok := s.members[k]; ok {
			// already handled
			continue
		}
		*out.Descend(pe) = *sn2.set
	}
	return out
}

// Intersection returns a SetNodeMap with members that appear in both s and s2.
func (s *SetNodeMap) Intersection(s2 *SetNodeMap) *SetNodeMap {
	out := &SetNodeMap{}
	for k, sn := range s.members {
		pe := sn.pathElement
		if sn2, ok := s2.members[k]; ok {
			i := *sn.set.Intersection(sn2.set)
			if !i.Empty() {
				*out.Descend(pe) = i
			}
		}
	}
	return out
}

// Difference returns a SetNodeMap with members that appear in s but not in s2.
func (s *SetNodeMap) Difference(s2 *Set) *SetNodeMap {
	out := &SetNodeMap{}
	for k, sn := range s.members {
		pe := sn.pathElement
		if s2.Members.Has(pe) {
			continue
		}
		if sn2, ok := s2.Children.members[k]; ok {
			diff := *sn.set.Difference(sn2.set)
			// We aren't permitted to add nodes with no elements.
			if !diff.Empty() {
				*out.Descend(pe) = diff
			}
		} else {
			*out.Descend(pe) = *sn.set
		}
	}
	return out
}

// Iterate calls f for each PathElement in the set.
func (s *SetNodeMap) Iterate(f func(PathElement)) {
	for _, n := range s.members {
		f(n.pathElement)
	}
}

func (s *SetNodeMap) iteratePrefix(prefix Path, f func(Path)) {
	for _, n := range s.members {
		pe := n.pathElement
		n.set.iteratePrefix(append(prefix, pe), f)
	}
}
