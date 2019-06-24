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
)

type LeveledElement struct {
	Level   int
	Element PathElement
}

// This implements the SetIterator
type SetAsListIterator struct {
	list    []LeveledElement
	current int

	nextPath Path
}

// Create a new set from a list of path element and their level.
func NewSetAsListIterator(list []LeveledElement) *SetAsListIterator {
	s := &SetAsListIterator{
		list:     list,
		current:  0,
		nextPath: Path{},
	}
	s.nextPath, _ = s.getNext()
	return s
}

// Peek takes a look at the next item in the list.
// Returns nil if there are no other items.
func (s *SetAsListIterator) Peek() Path {
	return s.nextPath
}

func (s *SetAsListIterator) Empty() bool {
	return s.Peek() == nil
}

// Next returns the next items in the list, and points to the next item.
func (s *SetAsListIterator) Next() Path {
	ret := s.nextPath
	s.nextPath, _ = s.getNext()
	return ret
}

// Clone the set in a new set.
func (s SetAsListIterator) Clone() *SetAsListIterator {
	return &s
}

// Validate that the set is
func (s SetAsListIterator) Validate() error {
	for {
		p, err := s.getNext()
		if err != nil {
			return err
		}
		if p == nil {
			return nil
		}
	}
}

// [1 a, 2 b, -1, 3 c, 2 d, 2 e, -1, 3 f]

// ab
// abc
// ad
// ae
// aef

// getNext returns a pointer to the next element. Returns nil if there
// are no other element.
func (s *SetAsListIterator) getNext() (Path, error) {
	if s.current == len(s.list) {
		return nil, nil
	}
	path := s.nextPath.Copy()

	// We're diverging to a different path, first step back.
	nextLevel := s.list[s.current].Level
	if nextLevel < len(path) {
		path = path[:nextLevel]
	}

	// Keep following the path until either we have to diverge or stop.
	for len(path) <= nextLevel {
		if nextLevel > len(path)+1 {
			return nil, fmt.Errorf("Invalid jump from %d to %d", len(path), nextLevel)
		}
		path = append(path, s.list[s.current].Element)
		s.current++
		// We need to find the end and just return
		if s.current == len(s.list) {
			return path, nil
		}
		nextLevel = s.list[s.current].Level
	}

	// If we had a stop, pass the stop sign.
	if s.current != len(s.list) && s.list[s.current].Level == -1 {
		s.current++
	}
	return path, nil
}

type SetAsList struct {
	list []LeveledElement
	last Path
}

func NewSetAsListFromList(lpes []LeveledElement) *SetAsList {
	return &SetAsList{
		list: lpes,
	}
}

func NewSetAsList(paths ...Path) *SetAsList {
	s := &SetAsList{
		list: []LeveledElement{},
		last: nil,
	}

	sort.Slice(paths, func(i, j int) bool {
		return paths[i].Compare(paths[j]) < 0
	})
	for _, path := range paths {
		s.Insert(path)
	}

	return s
}

func (s *SetAsList) List() []LeveledElement {
	return s.list
}

func (s *SetAsList) Empty() bool {
	return len(s.list) == 0
}

func (s *SetAsList) String() string {
	elements := []string{}
	for _, element := range s.list {
		elements = append(elements, fmt.Sprintf("%d:%s", element.Level, element.Element))
	}

	it := s.Iterator()
	path := it.Next()
	for path != nil {
		elements = append(elements, path.String())
		path = it.Next()
	}
	return strings.Join(elements, "\n")
}

func (s *SetAsList) Equals(other *SetAsList) bool {
	if len(s.list) != len(other.list) {
		return false
	}

	for i, le := range s.list {
		other := other.list[i]
		if le.Level != other.Level {
			return false
		}
		if le.Level == -1 {
			continue
		}
		if comp := le.Element.Compare(other.Element); comp != 0 {
			return false
		}
	}
	return true
}

// Insert path into the set. Path MUST BE inserted in sorted order.
func (s *SetAsList) Insert(p Path) {
	level := 0
	for _, pe := range s.last {
		if level == len(p) {
			panic(fmt.Errorf("Path inserted out-of-order: %v then %v", s.last, p))
		}
		if comp := pe.Compare(p[level]); comp != 0 {
			if comp > 0 {
				panic(fmt.Errorf("Path inserted out-of-order: %v then %v", s.last, p))
			}
			// PathElement are different, push the rest of the Path
			break
		}
		level++
	}
	// If we are a super-set of the last path, we need to mark the end of the previous path.
	if level == len(s.last) && len(s.last) != 0 {
		if len(p) == len(s.last) {
			// Don't insert -1 for duplicates.
			return
		}
		// -1 indicates that there is an item that ends here,
		// even though the next item is following on that
		// exactly, e.g. foo.bar followed by foo.bar.fuz
		s.list = append(s.list, LeveledElement{
			Level: -1,
		})
	}

	for i, pe := range p[level:] {
		s.list = append(s.list, LeveledElement{
			Element: pe,
			Level:   level + i,
		})
	}
	s.last = p.Copy()
}

func (s *SetAsList) Iterator() SetIterator {
	return NewSetAsListIterator(s.list)
}

type SetIterator interface {
	// Returns the next path until there are none (then it returns nil)
	Next() Path
	// Returns empty if there are no more items in the list, by peeking.
	Empty() bool
}

func Union(as, bs *SetAsList) *SetAsList {
	set := NewSetAsList()

	a, b := as.Iterator(), bs.Iterator()
	apath, bpath := a.Next(), b.Next()

	for apath != nil && bpath != nil {
		comp := apath.Compare(bpath)
		if comp == 0 {
			set.Insert(apath)
			apath = a.Next()
			bpath = b.Next()
		} else if comp < 0 {
			set.Insert(apath)
			apath = a.Next()
		} else {
			set.Insert(bpath)
			bpath = b.Next()
		}
	}

	// Insert all of a or b now.
	for apath != nil {
		set.Insert(apath)
		apath = a.Next()
	}
	for bpath != nil {
		set.Insert(bpath)
		bpath = b.Next()
	}

	return set
}

func Intersection(as, bs *SetAsList) *SetAsList {
	set := NewSetAsList()

	a, b := as.Iterator(), bs.Iterator()
	apath, bpath := a.Next(), b.Next()

	for apath != nil && bpath != nil {
		comp := apath.Compare(bpath)
		if comp == 0 {
			set.Insert(apath)
			apath = a.Next()
			bpath = b.Next()
		} else if comp < 0 {
			apath = a.Next()
		} else {
			bpath = b.Next()
		}
	}

	return set
}

func Difference(as, bs *SetAsList) *SetAsList {
	set := NewSetAsList()

	a, b := as.Iterator(), bs.Iterator()
	apath, bpath := a.Next(), b.Next()
	for apath != nil && bpath != nil {
		comp := apath.Compare(bpath)
		if comp == 0 {
			apath = a.Next()
			bpath = b.Next()
		} else if comp < 0 {
			set.Insert(apath)
			apath = a.Next()
		} else {
			bpath = b.Next()
		}
	}

	// Push everything left in A into the set
	for apath != nil {
		set.Insert(apath)
		apath = a.Next()
	}

	return set
}

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

// Empty returns true if there are no members of the set. It is a separate
// function from Size since it's common to check whether size > 0, and
// potentially much faster to return as soon as a single element is found.
func (s *Set) Empty() bool {
	return s.Members.Empty() && s.Children.Empty()
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

type setIterator struct {
	c    chan Path
	next Path
}

func (s *setIterator) Peek() Path {
	return s.next
}

func (s *setIterator) Next() Path {
	current := s.next
	next, ok := <-s.c
	if !ok {
		s.next = nil
		return current
	}
	if comp := current.Compare(next); comp > 0 {
		panic(fmt.Errorf("Unsorted path: %v should be before %v", next, current))
	}
	s.next = next
	return current
}

func (s *setIterator) Empty() bool {
	return s.next == nil
}

// If you don't want the goroutine to leak, you should consume the iterator entirely.
// TODO: This is not returning the things in the right order.
func (s *Set) Iterator() SetIterator {
	i := setIterator{
		c: make(chan Path),
	}
	go func() {
		s.Iterate(func(p Path) {
			i.c <- p.Copy()
		})
		close(i.c)
	}()

	// First call doesn't return anything.
	i.Next()
	return &i
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
	pes := []PathElement{}
	for _, n := range s.members {
		pes = append(pes, n.pathElement)
	}
	sort.Slice(pes, func(i, j int) bool {
		return pes[i].Compare(pes[j]) < 0
	})
	for _, pe := range pes {
		f(pe)
	}
}

func (s *SetNodeMap) iteratePrefix(prefix Path, f func(Path)) {
	for _, n := range s.members {
		pe := n.pathElement
		n.set.iteratePrefix(append(prefix, pe), f)
	}
}
