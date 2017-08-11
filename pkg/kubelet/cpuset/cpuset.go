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

package cpuset

import (
	"bytes"
	"fmt"
	"github.com/golang/glog"
	"reflect"
	"sort"
	"strconv"
	"strings"
)

// CPUSet is a set-like data structure for CPU IDs.
type CPUSet map[int]struct{}

// NewCPUSet return CPUSet based on provided cpu id's
func NewCPUSet(cpus ...int) CPUSet {
	res := CPUSet{}
	for _, c := range cpus {
		res.Add(c)
	}
	return res
}

// Size returns the number of elements in this set.
func (s CPUSet) Size() int {
	return len(s)
}

// IsEmpty returns true if there are zero elements in this set.
func (s CPUSet) IsEmpty() bool {
	return s.Size() == 0
}

// Contains returns true if the supplied element is present in this set.
func (s CPUSet) Contains(cpu int) bool {
	_, found := s[cpu]
	return found
}

// Add mutates this set to contain the supplied elements.
func (s CPUSet) Add(cpus ...int) {
	for _, cpu := range cpus {
		s[cpu] = struct{}{}
	}
}

// Remove mutates this set to not contain the supplied elements, if they
// exists.
func (s CPUSet) Remove(cpus ...int) {
	for _, cpu := range cpus {
		delete(s, cpu)
	}
}

// Equals returns true if the supplied set contains exactly the same elements
// as this set (s IsSubsetOf s2 and s2 IsSubsetOf s).
func (s CPUSet) Equals(s2 CPUSet) bool {
	return reflect.DeepEqual(s, s2)
}

// Filter returns a new CPU set that contains all of the elements from this
// set that match the supplied predicate, without mutating the source set.
func (s CPUSet) Filter(predicate func(int) bool) CPUSet {
	result := NewCPUSet()
	for cpu := range s {
		if predicate(cpu) {
			result.Add(cpu)
		}
	}
	return result
}

// FilterNot returns a new CPU set that contains all of the elements from this
// set that do not match the supplied predicate, without mutating the source
// set.
func (s CPUSet) FilterNot(predicate func(int) bool) CPUSet {
	result := NewCPUSet()
	for cpu := range s {
		if !predicate(cpu) {
			result.Add(cpu)
		}
	}
	return result
}

// IsSubsetOf returns true if the supplied set contains all the elements
func (s CPUSet) IsSubsetOf(s2 CPUSet) bool {
	result := true
	for cpu := range s {
		if !s2.Contains(cpu) {
			result = false
			break
		}
	}
	return result
}

// Union returns a new CPU set that contains all of the elements from this
// set and all of the elements from the supplied set, without mutating
// either source set.
func (s CPUSet) Union(s2 CPUSet) CPUSet {
	result := NewCPUSet()
	for cpu := range s {
		result.Add(cpu)
	}
	for cpu := range s2 {
		result.Add(cpu)
	}
	return result
}

// Intersection returns a new CPU set that contains all of the elements
// that are present in both this set and the supplied set, without mutating
// either source set.
func (s CPUSet) Intersection(s2 CPUSet) CPUSet {
	return s.Filter(func(cpu int) bool { return s2.Contains(cpu) })
}

// Difference returns a new CPU set that contains all of the elements that
// are present in this set and not the supplied set, without mutating either
// source set.
func (s CPUSet) Difference(s2 CPUSet) CPUSet {
	return s.FilterNot(func(cpu int) bool { return s2.Contains(cpu) })
}

// AsSlice returns a slice of integers that contains all elements from
// this set.
func (s CPUSet) AsSlice() []int {
	result := []int{}
	for cpu := range s {
		result = append(result, cpu)
	}
	sort.Ints(result)
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

	elems := s.AsSlice()
	sort.Ints(elems)

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

// MustParse CPUSet constructs a new CPU set from a Linux CPU list formatted
// string. Unlike Parse, it does not return an error but rather panics if the
// input cannot be used to construct a CPU set.
func MustParse(s string) CPUSet {
	res, err := Parse(s)
	if err != nil {
		glog.Fatalf("unable to parse [%s] as CPUSet: %v", s, err)
	}
	return res
}

// Parse CPUSet constructs a new CPU set from a Linux CPU list formatted string.
//
// See: http://man7.org/linux/man-pages/man7/cpuset.7.html#FORMATS
func Parse(s string) (CPUSet, error) {
	result := NewCPUSet()

	// Handle empty string.
	if s == "" {
		return result, nil
	}

	// Split CPU list string:
	// "0-5,34,46-48 => ["0-5", "34", "46-48"]
	ranges := strings.Split(s, ",")

	for _, r := range ranges {
		boundaries := strings.Split(r, "-")
		if len(boundaries) == 1 {
			// Handle ranges that consist of only one element like "34".
			elem, err := strconv.Atoi(boundaries[0])
			if err != nil {
				return nil, err
			}
			result.Add(elem)
		} else if len(boundaries) == 2 {
			// Handle multi-element ranges like "0-5".
			start, err := strconv.Atoi(boundaries[0])
			if err != nil {
				return nil, err
			}
			end, err := strconv.Atoi(boundaries[1])
			if err != nil {
				return nil, err
			}
			// Add all elements to the result.
			// e.g. "0-5", "46-48" => [0, 1, 2, 3, 4, 5, 46, 47, 48].
			for e := start; e <= end; e++ {
				result.Add(e)
			}
		}
	}
	return result, nil
}

// Clone returns a copy of this CPU set.
func (s CPUSet) Clone() CPUSet {
	res := NewCPUSet()
	for k, v := range s {
		res[k] = v
	}
	return res
}
