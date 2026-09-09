/*
Copyright 2017 The Kubernetes Authors.

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

// Package cpuset represents a collection of CPUs in a 'set' data structure.
//
// It can be used to represent core IDs, hyper thread siblings, CPU nodes, or processor IDs.
//
// The only special thing about this package is that
// methods are provided to convert back and forth from Linux 'list' syntax.
// See http://man7.org/linux/man-pages/man7/cpuset.7.html#FORMATS for details.
//
// Future work can migrate this to use a 'set' library, and relax the dubious 'immutable' property.
//
// This package was originally developed in the 'kubernetes' repository.
package cpuset

import (
	"bytes"
	"fmt"
	"reflect"
	"sort"
	"strconv"
	"strings"
)

// CPUSet is a thread-safe, immutable set-like data structure for CPU IDs.
type CPUSet struct {
	elems map[int]struct{}
}

// New returns a new CPUSet containing the supplied elements.
func New(cpus ...int) CPUSet {
	s := CPUSet{
		elems: map[int]struct{}{},
	}
	for _, c := range cpus {
		s.add(c)
	}
	return s
}

// add adds the supplied elements to the CPUSet.
// It is intended for internal use only, since it mutates the CPUSet.
func (s CPUSet) add(elems ...int) {
	for _, elem := range elems {
		s.elems[elem] = struct{}{}
	}
}

// Size returns the number of elements in this set.
func (s CPUSet) Size() int {
	return len(s.elems)
}

// IsEmpty returns true if there are zero elements in this set.
func (s CPUSet) IsEmpty() bool {
	return s.Size() == 0
}

// Contains returns true if the supplied element is present in this set.
func (s CPUSet) Contains(cpu int) bool {
	_, found := s.elems[cpu]
	return found
}

// Equals returns true if the supplied set contains exactly the same elements
// as this set (s IsSubsetOf s2 and s2 IsSubsetOf s).
func (s CPUSet) Equals(s2 CPUSet) bool {
	return reflect.DeepEqual(s.elems, s2.elems)
}

// filter returns a new CPU set that contains all of the elements from this
// set that match the supplied predicate, without mutating the source set.
func (s CPUSet) filter(predicate func(int) bool) CPUSet {
	r := New()
	for cpu := range s.elems {
		if predicate(cpu) {
			r.add(cpu)
		}
	}
	return r
}

// IsSubsetOf returns true if the supplied set contains all the elements
func (s CPUSet) IsSubsetOf(s2 CPUSet) bool {
	result := true
	for cpu := range s.elems {
		if !s2.Contains(cpu) {
			result = false
			break
		}
	}
	return result
}

// Union returns a new CPU set that contains all of the elements from this
// set and all of the elements from the supplied sets, without mutating
// either source set.
func (s CPUSet) Union(s2 ...CPUSet) CPUSet {
	r := New()
	for cpu := range s.elems {
		r.add(cpu)
	}
	for _, cs := range s2 {
		for cpu := range cs.elems {
			r.add(cpu)
		}
	}
	return r
}

// Intersection returns a new CPU set that contains all of the elements
// that are present in both this set and the supplied set, without mutating
// either source set.
func (s CPUSet) Intersection(s2 CPUSet) CPUSet {
	return s.filter(func(cpu int) bool { return s2.Contains(cpu) })
}

// Difference returns a new CPU set that contains all of the elements that
// are present in this set and not the supplied set, without mutating either
// source set.
func (s CPUSet) Difference(s2 CPUSet) CPUSet {
	return s.filter(func(cpu int) bool { return !s2.Contains(cpu) })
}

// List returns a slice of integers that contains all elements from
// this set. The list is sorted.
func (s CPUSet) List() []int {
	result := s.UnsortedList()
	sort.Ints(result)
	return result
}

// UnsortedList returns a slice of integers that contains all elements from
// this set.
func (s CPUSet) UnsortedList() []int {
	result := make([]int, 0, len(s.elems))
	for cpu := range s.elems {
		result = append(result, cpu)
	}
	return result
}

// String returns a new string representation of the elements in this CPU set
// in canonical linux CPU list format.
//
// See: http://man7.org/linux/man-pages/man7/cpuset.7.html#FORMATS
func (s CPUSet) String() string {
	if s.IsEmpty() {
		return ""
	}

	elems := s.List()

	type rng struct {
		start int
		end   int
	}

	ranges := []rng{{elems[0], elems[0]}}

	for i := 1; i < len(elems); i++ {
		lastRange := &ranges[len(ranges)-1]
		// if this element is adjacent to the high end of the last range
		if elems[i] == lastRange.end+1 {
			// then extend the last range to include this element
			lastRange.end = elems[i]
			continue
		}
		// otherwise, start a new range beginning with this element
		ranges = append(ranges, rng{elems[i], elems[i]})
	}

	// construct string from ranges
	var result bytes.Buffer
	for _, r := range ranges {
		if r.start == r.end {
			result.WriteString(strconv.Itoa(r.start))
		} else {
			result.WriteString(fmt.Sprintf("%d-%d", r.start, r.end))
		}
		result.WriteString(",")
	}
	return strings.TrimRight(result.String(), ",")
}

// Parse CPUSet constructs a new CPU set from a Linux CPU list formatted string.
//
// See: http://man7.org/linux/man-pages/man7/cpuset.7.html#FORMATS
func Parse(s string) (CPUSet, error) {
	// Handle empty string.
	if s == "" {
		return New(), nil
	}

	result := New()

	// Split CPU list string:
	// "0-5,34,46-48" => ["0-5", "34", "46-48"]
	ranges := strings.Split(s, ",")

	for _, r := range ranges {
		boundaries := strings.SplitN(r, "-", 2)
		if len(boundaries) == 1 {
			// Handle ranges that consist of only one element like "34".
			elem, err := strconv.Atoi(boundaries[0])
			if err != nil {
				return New(), err
			}
			result.add(elem)
		} else if len(boundaries) == 2 {
			// Handle multi-element ranges like "0-5".
			start, err := strconv.Atoi(boundaries[0])
			if err != nil {
				return New(), err
			}
			end, err := strconv.Atoi(boundaries[1])
			if err != nil {
				return New(), err
			}
			if start > end {
				return New(), fmt.Errorf("invalid range %q (%d > %d)", r, start, end)
			}
			// start == end is acceptable (1-1 -> 1)

			// Add all elements to the result.
			// e.g. "0-5", "46-48" => [0, 1, 2, 3, 4, 5, 46, 47, 48].
			for e := start; e <= end; e++ {
				result.add(e)
			}
		}
	}
	return result, nil
}

// Clone returns a copy of this CPU set.
func (s CPUSet) Clone() CPUSet {
	r := New()
	for elem := range s.elems {
		r.add(elem)
	}
	return r
}
