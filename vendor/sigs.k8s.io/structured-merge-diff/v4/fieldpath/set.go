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
	"sigs.k8s.io/structured-merge-diff/v4/value"
	"sort"
	"strings"

	"sigs.k8s.io/structured-merge-diff/v4/schema"
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

// RecursiveDifference returns a Set containing elements which:
// * appear in s
// * do not appear in s2
//
// Compared to a regular difference,
// this removes every field **and its children** from s that is contained in s2.
//
// For example, with s containing `a.b.c` and s2 containing `a.b`,
// a RecursiveDifference will result in `a`, as the entire node `a.b` gets removed.
func (s *Set) RecursiveDifference(s2 *Set) *Set {
	return &Set{
		Members:  *s.Members.Difference(&s2.Members),
		Children: *s.Children.RecursiveDifference(s2),
	}
}

// EnsureNamedFieldsAreMembers returns a Set that contains all the
// fields in s, as well as all the named fields that are typically not
// included. For example, a set made of "a.b.c" will end-up also owning
// "a" if it's a named fields but not "a.b" if it's a map.
func (s *Set) EnsureNamedFieldsAreMembers(sc *schema.Schema, tr schema.TypeRef) *Set {
	members := PathElementSet{
		members: make(sortedPathElements, 0, s.Members.Size()+len(s.Children.members)),
	}
	atom, _ := sc.Resolve(tr)
	members.members = append(members.members, s.Members.members...)
	for _, node := range s.Children.members {
		// Only insert named fields.
		if node.pathElement.FieldName != nil && atom.Map != nil {
			if _, has := atom.Map.FindField(*node.pathElement.FieldName); has {
				members.Insert(node.pathElement)
			}
		}
	}
	return &Set{
		Members:  members,
		Children: *s.Children.EnsureNamedFieldsAreMembers(sc, tr),
	}
}

// MakePrefixMatcherOrDie is the same as PrefixMatcher except it panics if parts can't be
// turned into a SetMatcher.
func MakePrefixMatcherOrDie(parts ...interface{}) *SetMatcher {
	result, err := PrefixMatcher(parts...)
	if err != nil {
		panic(err)
	}
	return result
}

// PrefixMatcher creates a SetMatcher that matches all field paths prefixed by the given list of matcher path parts.
// The matcher parts may any of:
//
//   - PathElementMatcher - for wildcards, `MatchAnyPathElement()` can be used as well.
//   - PathElement - for any path element
//   - value.FieldList - for listMap keys
//   - value.Value - for scalar list elements
//   - string - For field names
//   - int - for array indices
func PrefixMatcher(parts ...interface{}) (*SetMatcher, error) {
	current := MatchAnySet() // match all field path suffixes
	for i := len(parts) - 1; i >= 0; i-- {
		part := parts[i]
		var pattern PathElementMatcher
		switch t := part.(type) {
		case PathElementMatcher:
			// any path matcher, including wildcard
			pattern = t
		case PathElement:
			// any path element
			pattern = PathElementMatcher{PathElement: t}
		case *value.FieldList:
			// a listMap key
			if len(*t) == 0 {
				return nil, fmt.Errorf("associative list key type path elements must have at least one key (got zero)")
			}
			pattern = PathElementMatcher{PathElement: PathElement{Key: t}}
		case value.Value:
			// a scalar or set-type list element
			pattern = PathElementMatcher{PathElement: PathElement{Value: &t}}
		case string:
			// a plain field name
			pattern = PathElementMatcher{PathElement: PathElement{FieldName: &t}}
		case int:
			// a plain list index
			pattern = PathElementMatcher{PathElement: PathElement{Index: &t}}
		default:
			return nil, fmt.Errorf("unexpected type %T", t)
		}
		current = &SetMatcher{
			members: []*SetMemberMatcher{{
				Path:  pattern,
				Child: current,
			}},
		}
	}
	return current, nil
}

// MatchAnyPathElement returns a PathElementMatcher that matches any path element.
func MatchAnyPathElement() PathElementMatcher {
	return PathElementMatcher{Wildcard: true}
}

// MatchAnySet returns a SetMatcher that matches any set.
func MatchAnySet() *SetMatcher {
	return &SetMatcher{wildcard: true}
}

// NewSetMatcher returns a new SetMatcher.
// Wildcard members take precedent over non-wildcard members;
// all non-wildcard members are ignored if there is a wildcard members.
func NewSetMatcher(wildcard bool, members ...*SetMemberMatcher) *SetMatcher {
	sort.Sort(sortedMemberMatcher(members))
	return &SetMatcher{wildcard: wildcard, members: members}
}

// SetMatcher defines a matcher that matches fields in a Set.
// SetMatcher is structured much like a Set but with wildcard support.
type SetMatcher struct {
	// wildcard indicates that all members and children are included in the match.
	// If set, the members field is ignored.
	wildcard bool
	// members provides patterns to match the members of a Set.
	// Wildcard members are sorted before non-wildcards and take precedent over
	// non-wildcard members.
	members sortedMemberMatcher
}

type sortedMemberMatcher []*SetMemberMatcher

func (s sortedMemberMatcher) Len() int           { return len(s) }
func (s sortedMemberMatcher) Less(i, j int) bool { return s[i].Path.Less(s[j].Path) }
func (s sortedMemberMatcher) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }
func (s sortedMemberMatcher) Find(p PathElementMatcher) (location int, ok bool) {
	return sort.Find(len(s), func(i int) int {
		return s[i].Path.Compare(p)
	})
}

// Merge merges s and s2 and returns a SetMatcher that matches all field paths matched by either s or s2.
// During the merge, members of s and s2 with the same PathElementMatcher merged into a single member
// with the children of each merged by calling this function recursively.
func (s *SetMatcher) Merge(s2 *SetMatcher) *SetMatcher {
	if s.wildcard || s2.wildcard {
		return NewSetMatcher(true)
	}
	merged := make(sortedMemberMatcher, len(s.members), len(s.members)+len(s2.members))
	copy(merged, s.members)
	for _, m := range s2.members {
		if i, ok := s.members.Find(m.Path); ok {
			// since merged is a shallow copy, do not modify elements in place
			merged[i] = &SetMemberMatcher{
				Path:  merged[i].Path,
				Child: merged[i].Child.Merge(m.Child),
			}
		} else {
			merged = append(merged, m)
		}
	}
	return NewSetMatcher(false, merged...) // sort happens here
}

// SetMemberMatcher defines a matcher that matches the members of a Set.
// SetMemberMatcher is structured much like the elements of a SetNodeMap, but
// with wildcard support.
type SetMemberMatcher struct {
	// Path provides a matcher to match members of a Set.
	// If Path is a wildcard, all members of a Set are included in the match.
	// Otherwise, if any Path is Equal to a member of a Set, that member is
	// included in the match and the children of that member are matched
	// against the Child matcher.
	Path PathElementMatcher

	// Child provides a matcher to use for the children of matched members of a Set.
	Child *SetMatcher
}

// PathElementMatcher defined a path matcher for a PathElement.
type PathElementMatcher struct {
	// Wildcard indicates that all PathElements are matched by this matcher.
	// If set, PathElement is ignored.
	Wildcard bool

	// PathElement indicates that a PathElement is matched if it is Equal
	// to this PathElement.
	PathElement
}

func (p PathElementMatcher) Equals(p2 PathElementMatcher) bool {
	return p.Wildcard != p2.Wildcard && p.PathElement.Equals(p2.PathElement)
}

func (p PathElementMatcher) Less(p2 PathElementMatcher) bool {
	if p.Wildcard && !p2.Wildcard {
		return true
	} else if p2.Wildcard {
		return false
	}
	return p.PathElement.Less(p2.PathElement)
}

func (p PathElementMatcher) Compare(p2 PathElementMatcher) int {
	if p.Wildcard && !p2.Wildcard {
		return -1
	} else if p2.Wildcard {
		return 1
	}
	return p.PathElement.Compare(p2.PathElement)
}

// FilterIncludeMatches returns a Set with only the field paths that match.
func (s *Set) FilterIncludeMatches(pattern *SetMatcher) *Set {
	if pattern.wildcard {
		return s
	}

	members := PathElementSet{}
	for _, m := range s.Members.members {
		for _, pm := range pattern.members {
			if pm.Path.Wildcard || pm.Path.PathElement.Equals(m) {
				members.Insert(m)
				break
			}
		}
	}
	return &Set{
		Members:  members,
		Children: *s.Children.FilterIncludeMatches(pattern),
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

// Leaves returns a set containing only the leaf paths
// of a set.
func (s *Set) Leaves() *Set {
	leaves := PathElementSet{}
	im := 0
	ic := 0

	// any members that are not also children are leaves
outer:
	for im < len(s.Members.members) {
		member := s.Members.members[im]

		for ic < len(s.Children.members) {
			d := member.Compare(s.Children.members[ic].pathElement)
			if d == 0 {
				ic++
				im++
				continue outer
			} else if d < 0 {
				break
			} else /* if d > 0 */ {
				ic++
			}
		}
		leaves.members = append(leaves.members, member)
		im++
	}

	return &Set{
		Members:  leaves,
		Children: *s.Children.Leaves(),
	}
}

// setNode is a pair of PathElement / Set, for the purpose of expressing
// nested set membership.
type setNode struct {
	pathElement PathElement
	set         *Set
}

// SetNodeMap is a map of PathElement to subset.
type SetNodeMap struct {
	members sortedSetNode
}

type sortedSetNode []setNode

// Implement the sort interface; this would permit bulk creation, which would
// be faster than doing it one at a time via Insert.
func (s sortedSetNode) Len() int           { return len(s) }
func (s sortedSetNode) Less(i, j int) bool { return s[i].pathElement.Less(s[j].pathElement) }
func (s sortedSetNode) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }

// Descend adds pe to the set if necessary, returning the associated subset.
func (s *SetNodeMap) Descend(pe PathElement) *Set {
	loc := sort.Search(len(s.members), func(i int) bool {
		return !s.members[i].pathElement.Less(pe)
	})
	if loc == len(s.members) {
		s.members = append(s.members, setNode{pathElement: pe, set: &Set{}})
		return s.members[loc].set
	}
	if s.members[loc].pathElement.Equals(pe) {
		return s.members[loc].set
	}
	s.members = append(s.members, setNode{})
	copy(s.members[loc+1:], s.members[loc:])
	s.members[loc] = setNode{pathElement: pe, set: &Set{}}
	return s.members[loc].set
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
	loc := sort.Search(len(s.members), func(i int) bool {
		return !s.members[i].pathElement.Less(pe)
	})
	if loc == len(s.members) {
		return nil, false
	}
	if s.members[loc].pathElement.Equals(pe) {
		return s.members[loc].set, true
	}
	return nil, false
}

// Equals returns true if s and s2 have the same structure (same nested
// child sets).
func (s *SetNodeMap) Equals(s2 *SetNodeMap) bool {
	if len(s.members) != len(s2.members) {
		return false
	}
	for i := range s.members {
		if !s.members[i].pathElement.Equals(s2.members[i].pathElement) {
			return false
		}
		if !s.members[i].set.Equals(s2.members[i].set) {
			return false
		}
	}
	return true
}

// Union returns a SetNodeMap with members that appear in either s or s2.
func (s *SetNodeMap) Union(s2 *SetNodeMap) *SetNodeMap {
	out := &SetNodeMap{}

	i, j := 0, 0
	for i < len(s.members) && j < len(s2.members) {
		if s.members[i].pathElement.Less(s2.members[j].pathElement) {
			out.members = append(out.members, s.members[i])
			i++
		} else {
			if !s2.members[j].pathElement.Less(s.members[i].pathElement) {
				out.members = append(out.members, setNode{pathElement: s.members[i].pathElement, set: s.members[i].set.Union(s2.members[j].set)})
				i++
			} else {
				out.members = append(out.members, s2.members[j])
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

// Intersection returns a SetNodeMap with members that appear in both s and s2.
func (s *SetNodeMap) Intersection(s2 *SetNodeMap) *SetNodeMap {
	out := &SetNodeMap{}

	i, j := 0, 0
	for i < len(s.members) && j < len(s2.members) {
		if s.members[i].pathElement.Less(s2.members[j].pathElement) {
			i++
		} else {
			if !s2.members[j].pathElement.Less(s.members[i].pathElement) {
				res := s.members[i].set.Intersection(s2.members[j].set)
				if !res.Empty() {
					out.members = append(out.members, setNode{pathElement: s.members[i].pathElement, set: res})
				}
				i++
			}
			j++
		}
	}
	return out
}

// Difference returns a SetNodeMap with members that appear in s but not in s2.
func (s *SetNodeMap) Difference(s2 *Set) *SetNodeMap {
	out := &SetNodeMap{}

	i, j := 0, 0
	for i < len(s.members) && j < len(s2.Children.members) {
		if s.members[i].pathElement.Less(s2.Children.members[j].pathElement) {
			out.members = append(out.members, setNode{pathElement: s.members[i].pathElement, set: s.members[i].set})
			i++
		} else {
			if !s2.Children.members[j].pathElement.Less(s.members[i].pathElement) {

				diff := s.members[i].set.Difference(s2.Children.members[j].set)
				// We aren't permitted to add nodes with no elements.
				if !diff.Empty() {
					out.members = append(out.members, setNode{pathElement: s.members[i].pathElement, set: diff})
				}

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

// RecursiveDifference returns a SetNodeMap with members that appear in s but not in s2.
//
// Compared to a regular difference,
// this removes every field **and its children** from s that is contained in s2.
//
// For example, with s containing `a.b.c` and s2 containing `a.b`,
// a RecursiveDifference will result in `a`, as the entire node `a.b` gets removed.
func (s *SetNodeMap) RecursiveDifference(s2 *Set) *SetNodeMap {
	out := &SetNodeMap{}

	i, j := 0, 0
	for i < len(s.members) && j < len(s2.Children.members) {
		if s.members[i].pathElement.Less(s2.Children.members[j].pathElement) {
			if !s2.Members.Has(s.members[i].pathElement) {
				out.members = append(out.members, setNode{pathElement: s.members[i].pathElement, set: s.members[i].set})
			}
			i++
		} else {
			if !s2.Children.members[j].pathElement.Less(s.members[i].pathElement) {
				if !s2.Members.Has(s.members[i].pathElement) {
					diff := s.members[i].set.RecursiveDifference(s2.Children.members[j].set)
					if !diff.Empty() {
						out.members = append(out.members, setNode{pathElement: s.members[i].pathElement, set: diff})
					}
				}
				i++
			}
			j++
		}
	}

	if i < len(s.members) {
		for _, c := range s.members[i:] {
			if !s2.Members.Has(c.pathElement) {
				out.members = append(out.members, c)
			}
		}
	}

	return out
}

// EnsureNamedFieldsAreMembers returns a set that contains all the named fields along with the leaves.
func (s *SetNodeMap) EnsureNamedFieldsAreMembers(sc *schema.Schema, tr schema.TypeRef) *SetNodeMap {
	out := make(sortedSetNode, 0, s.Size())
	atom, _ := sc.Resolve(tr)
	for _, member := range s.members {
		tr := schema.TypeRef{}
		if member.pathElement.FieldName != nil && atom.Map != nil {
			tr = atom.Map.ElementType
			if sf, ok := atom.Map.FindField(*member.pathElement.FieldName); ok {
				tr = sf.Type
			}
		} else if member.pathElement.Key != nil && atom.List != nil {
			tr = atom.List.ElementType
		}
		out = append(out, setNode{
			pathElement: member.pathElement,
			set:         member.set.EnsureNamedFieldsAreMembers(sc, tr),
		})
	}

	return &SetNodeMap{
		members: out,
	}
}

// FilterIncludeMatches returns a SetNodeMap with only the field paths that match the matcher.
func (s *SetNodeMap) FilterIncludeMatches(pattern *SetMatcher) *SetNodeMap {
	if pattern.wildcard {
		return s
	}

	var out sortedSetNode
	for _, member := range s.members {
		for _, c := range pattern.members {
			if c.Path.Wildcard || c.Path.PathElement.Equals(member.pathElement) {
				childSet := member.set.FilterIncludeMatches(c.Child)
				if childSet.Size() > 0 {
					out = append(out, setNode{
						pathElement: member.pathElement,
						set:         childSet,
					})
				}
				break
			}
		}
	}

	return &SetNodeMap{
		members: out,
	}
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

// Leaves returns a SetNodeMap containing
// only setNodes with leaf PathElements.
func (s *SetNodeMap) Leaves() *SetNodeMap {
	out := &SetNodeMap{}
	out.members = make(sortedSetNode, len(s.members))
	for i, n := range s.members {
		out.members[i] = setNode{
			pathElement: n.pathElement,
			set:         n.set.Leaves(),
		}
	}
	return out
}

// Filter defines an interface for excluding field paths from a set.
// NewExcludeSetFilter can be used to create a filter that removes
// specific field paths and all of their children.
// NewIncludeMatcherFilter can be used to create a filter that removes all fields except
// the fields that match a field path matcher. PrefixMatcher and MakePrefixMatcherOrDie
// can be used to define field path patterns.
type Filter interface {
	// Filter returns a filtered copy of the set.
	Filter(*Set) *Set
}

// NewExcludeSetFilter returns a filter that removes field paths in the exclude set.
func NewExcludeSetFilter(exclude *Set) Filter {
	return excludeFilter{exclude}
}

// NewExcludeFilterSetMap converts a map of APIVersion to exclude set to a map of APIVersion to exclude filters.
func NewExcludeFilterSetMap(resetFields map[APIVersion]*Set) map[APIVersion]Filter {
	result := make(map[APIVersion]Filter)
	for k, v := range resetFields {
		result[k] = excludeFilter{v}
	}
	return result
}

type excludeFilter struct {
	excludeSet *Set
}

func (t excludeFilter) Filter(set *Set) *Set {
	return set.RecursiveDifference(t.excludeSet)
}

// NewIncludeMatcherFilter returns a filter that only includes field paths that match.
// If no matchers are provided, the filter includes all field paths.
// PrefixMatcher and MakePrefixMatcherOrDie can help create basic matcher.
func NewIncludeMatcherFilter(matchers ...*SetMatcher) Filter {
	if len(matchers) == 0 {
		return includeMatcherFilter{&SetMatcher{wildcard: true}}
	}
	matcher := matchers[0]
	for i := 1; i < len(matchers); i++ {
		matcher = matcher.Merge(matchers[i])
	}

	return includeMatcherFilter{matcher}
}

type includeMatcherFilter struct {
	matcher *SetMatcher
}

func (pf includeMatcherFilter) Filter(set *Set) *Set {
	return set.FilterIncludeMatches(pf.matcher)
}
