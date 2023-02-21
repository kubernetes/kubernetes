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
	"reflect"
	"sort"
	"testing"
)

func TestCPUSetSize(t *testing.T) {
	testCases := []struct {
		cpuset   CPUSet
		expected int
	}{
		{New(), 0},
		{New(5), 1},
		{New(1, 2, 3, 4, 5), 5},
	}

	for _, c := range testCases {
		actual := c.cpuset.Size()
		if actual != c.expected {
			t.Errorf("expected: %d, actual: %d, cpuset: [%v]", c.expected, actual, c.cpuset)
		}
	}
}

func TestCPUSetIsEmpty(t *testing.T) {
	testCases := []struct {
		cpuset   CPUSet
		expected bool
	}{
		{New(), true},
		{New(5), false},
		{New(1, 2, 3, 4, 5), false},
	}

	for _, c := range testCases {
		actual := c.cpuset.IsEmpty()
		if actual != c.expected {
			t.Errorf("expected: %t, IsEmpty() returned: %t, cpuset: [%v]", c.expected, actual, c.cpuset)
		}
	}
}

func TestCPUSetContains(t *testing.T) {
	testCases := []struct {
		cpuset         CPUSet
		mustContain    []int
		mustNotContain []int
	}{
		{New(), []int{}, []int{1, 2, 3, 4, 5}},
		{New(5), []int{5}, []int{1, 2, 3, 4}},
		{New(1, 2, 4, 5), []int{1, 2, 4, 5}, []int{0, 3, 6}},
	}

	for _, c := range testCases {
		for _, elem := range c.mustContain {
			if !c.cpuset.Contains(elem) {
				t.Errorf("expected cpuset to contain element %d: [%v]", elem, c.cpuset)
			}
		}
		for _, elem := range c.mustNotContain {
			if c.cpuset.Contains(elem) {
				t.Errorf("expected cpuset not to contain element %d: [%v]", elem, c.cpuset)
			}
		}
	}
}

func TestCPUSetEqual(t *testing.T) {
	shouldEqual := []struct {
		s1 CPUSet
		s2 CPUSet
	}{
		{New(), New()},
		{New(5), New(5)},
		{New(1, 2, 3, 4, 5), New(1, 2, 3, 4, 5)},
		{New(5, 4, 3, 2, 1), New(1, 2, 3, 4, 5)},
	}

	shouldNotEqual := []struct {
		s1 CPUSet
		s2 CPUSet
	}{
		{New(), New(5)},
		{New(5), New()},
		{New(), New(1, 2, 3, 4, 5)},
		{New(1, 2, 3, 4, 5), New()},
		{New(5), New(1, 2, 3, 4, 5)},
		{New(1, 2, 3, 4, 5), New(5)},
	}

	for _, c := range shouldEqual {
		if !c.s1.Equals(c.s2) {
			t.Errorf("expected cpusets to be equal: s1: [%v], s2: [%v]", c.s1, c.s2)
		}
	}
	for _, c := range shouldNotEqual {
		if c.s1.Equals(c.s2) {
			t.Errorf("expected cpusets to not be equal: s1: [%v], s2: [%v]", c.s1, c.s2)
		}
	}
}

func TestCPUSetIsSubsetOf(t *testing.T) {
	shouldBeSubset := []struct {
		s1 CPUSet
		s2 CPUSet
	}{
		// A set is a subset of itself
		{New(), New()},
		{New(5), New(5)},
		{New(1, 2, 3, 4, 5), New(1, 2, 3, 4, 5)},

		// Empty set is a subset of every set
		{New(), New(5)},
		{New(), New(1, 2, 3, 4, 5)},

		{New(5), New(1, 2, 3, 4, 5)},
		{New(1, 2, 3), New(1, 2, 3, 4, 5)},
		{New(4, 5), New(1, 2, 3, 4, 5)},
		{New(2, 3), New(1, 2, 3, 4, 5)},
	}

	shouldNotBeSubset := []struct {
		s1 CPUSet
		s2 CPUSet
	}{
		// A set with more elements is not a subset.
		{New(5), New()},

		// Disjoint set is not a subset.
		{New(6), New(5)},
	}

	for _, c := range shouldBeSubset {
		if !c.s1.IsSubsetOf(c.s2) {
			t.Errorf("expected s1 to be a subset of s2: s1: [%v], s2: [%v]", c.s1, c.s2)
		}
	}
	for _, c := range shouldNotBeSubset {
		if c.s1.IsSubsetOf(c.s2) {
			t.Errorf("expected s1 to not be a subset of s2: s1: [%v], s2: [%v]", c.s1, c.s2)
		}
	}
}

func TestCPUSetUnion(t *testing.T) {
	testCases := []struct {
		s1       CPUSet
		others   []CPUSet
		expected CPUSet
	}{
		{New(5), []CPUSet{}, New(5)},

		{New(), []CPUSet{New()}, New()},

		{New(), []CPUSet{New(5)}, New(5)},
		{New(5), []CPUSet{New()}, New(5)},
		{New(5), []CPUSet{New(5)}, New(5)},

		{New(), []CPUSet{New(1, 2, 3, 4, 5)}, New(1, 2, 3, 4, 5)},
		{New(1, 2, 3, 4, 5), []CPUSet{New()}, New(1, 2, 3, 4, 5)},
		{New(1, 2, 3, 4, 5), []CPUSet{New(1, 2, 3, 4, 5)}, New(1, 2, 3, 4, 5)},

		{New(5), []CPUSet{New(1, 2, 3, 4, 5)}, New(1, 2, 3, 4, 5)},
		{New(1, 2, 3, 4, 5), []CPUSet{New(5)}, New(1, 2, 3, 4, 5)},

		{New(1, 2), []CPUSet{New(3, 4, 5)}, New(1, 2, 3, 4, 5)},
		{New(1, 2, 3), []CPUSet{New(3, 4, 5)}, New(1, 2, 3, 4, 5)},

		{New(), []CPUSet{New(1, 2, 3, 4, 5), New(4, 5)}, New(1, 2, 3, 4, 5)},
		{New(1, 2, 3, 4, 5), []CPUSet{New(), New(4)}, New(1, 2, 3, 4, 5)},
		{New(1, 2, 3, 4, 5), []CPUSet{New(1, 2, 3, 4, 5), New(1, 5)}, New(1, 2, 3, 4, 5)},
	}

	for _, c := range testCases {
		result := c.s1.Union(c.others...)
		if !result.Equals(c.expected) {
			t.Errorf("expected the union of s1 and s2 to be [%v] (got [%v]), others: [%v]", c.expected, result, c.others)
		}
	}
}

func TestCPUSetIntersection(t *testing.T) {
	testCases := []struct {
		s1       CPUSet
		s2       CPUSet
		expected CPUSet
	}{
		{New(), New(), New()},

		{New(), New(5), New()},
		{New(5), New(), New()},
		{New(5), New(5), New(5)},

		{New(), New(1, 2, 3, 4, 5), New()},
		{New(1, 2, 3, 4, 5), New(), New()},
		{New(1, 2, 3, 4, 5), New(1, 2, 3, 4, 5), New(1, 2, 3, 4, 5)},

		{New(5), New(1, 2, 3, 4, 5), New(5)},
		{New(1, 2, 3, 4, 5), New(5), New(5)},

		{New(1, 2), New(3, 4, 5), New()},
		{New(1, 2, 3), New(3, 4, 5), New(3)},
	}

	for _, c := range testCases {
		result := c.s1.Intersection(c.s2)
		if !result.Equals(c.expected) {
			t.Errorf("expected the intersection of s1 and s2 to be [%v] (got [%v]), s1: [%v], s2: [%v]", c.expected, result, c.s1, c.s2)
		}
	}
}

func TestCPUSetDifference(t *testing.T) {
	testCases := []struct {
		s1       CPUSet
		s2       CPUSet
		expected CPUSet
	}{
		{New(), New(), New()},

		{New(), New(5), New()},
		{New(5), New(), New(5)},
		{New(5), New(5), New()},

		{New(), New(1, 2, 3, 4, 5), New()},
		{New(1, 2, 3, 4, 5), New(), New(1, 2, 3, 4, 5)},
		{New(1, 2, 3, 4, 5), New(1, 2, 3, 4, 5), New()},

		{New(5), New(1, 2, 3, 4, 5), New()},
		{New(1, 2, 3, 4, 5), New(5), New(1, 2, 3, 4)},

		{New(1, 2), New(3, 4, 5), New(1, 2)},
		{New(1, 2, 3), New(3, 4, 5), New(1, 2)},
	}

	for _, c := range testCases {
		result := c.s1.Difference(c.s2)
		if !result.Equals(c.expected) {
			t.Errorf("expected the difference of s1 and s2 to be [%v] (got [%v]), s1: [%v], s2: [%v]", c.expected, result, c.s1, c.s2)
		}
	}
}

func TestCPUSetList(t *testing.T) {
	testCases := []struct {
		set      CPUSet
		expected []int // must be sorted
	}{
		{New(), []int{}},
		{New(5), []int{5}},
		{New(1, 2, 3, 4, 5), []int{1, 2, 3, 4, 5}},
		{New(5, 4, 3, 2, 1), []int{1, 2, 3, 4, 5}},
	}

	for _, c := range testCases {
		result := c.set.List()
		if !reflect.DeepEqual(result, c.expected) {
			t.Errorf("unexpected List() contents. got [%v] want [%v] (set: [%v])", result, c.expected, c.set)
		}

		// We cannot rely on internal storage order details for a unit test.
		// The best we can do is to sort the output of 'UnsortedList'.
		result = c.set.UnsortedList()
		sort.Ints(result)
		if !reflect.DeepEqual(result, c.expected) {
			t.Errorf("unexpected UnsortedList() contents. got [%v] want [%v] (set: [%v])", result, c.expected, c.set)
		}
	}
}

func TestCPUSetString(t *testing.T) {
	testCases := []struct {
		set      CPUSet
		expected string
	}{
		{New(), ""},
		{New(5), "5"},
		{New(1, 2, 3, 4, 5), "1-5"},
		{New(1, 2, 3, 5, 6, 8), "1-3,5-6,8"},
	}

	for _, c := range testCases {
		result := c.set.String()
		if result != c.expected {
			t.Errorf("expected set as string to be %s (got \"%s\"), s: [%v]", c.expected, result, c.set)
		}
	}
}

func TestParse(t *testing.T) {
	positiveTestCases := []struct {
		cpusetString string
		expected     CPUSet
	}{
		{"", New()},
		{"5", New(5)},
		{"1,2,3,4,5", New(1, 2, 3, 4, 5)},
		{"1-5", New(1, 2, 3, 4, 5)},
		{"1-2,3-5", New(1, 2, 3, 4, 5)},
		{"5,4,3,2,1", New(1, 2, 3, 4, 5)},  // Range ordering
		{"3-6,1-5", New(1, 2, 3, 4, 5, 6)}, // Overlapping ranges
		{"3-3,5-5", New(3, 5)},             // Very short ranges
	}

	for _, c := range positiveTestCases {
		result, err := Parse(c.cpusetString)
		if err != nil {
			t.Errorf("expected error not to have occurred: %v", err)
		}
		if !result.Equals(c.expected) {
			t.Errorf("expected string \"%s\" to parse as [%v] (got [%v])", c.cpusetString, c.expected, result)
		}
	}

	negativeTestCases := []string{
		// Non-numeric entries
		"nonnumeric", "non-numeric", "no,numbers", "0-a", "a-0", "0,a", "a,0", "1-2,a,3-5",
		// Incomplete sequences
		"0,", "0,,", ",3", ",,3", "0,,3",
		// Incomplete ranges and/or negative numbers
		"-1", "1-", "1,2-,3", "1,-2,3", "-1--2", "--1", "1--",
		// Reversed ranges
		"3-0", "0--3"}
	for _, c := range negativeTestCases {
		result, err := Parse(c)
		if err == nil {
			t.Errorf("expected parse failure of \"%s\", but it succeeded as \"%s\"", c, result.String())
		}
	}
}

func TestClone(t *testing.T) {
	original := New(1, 2, 3, 4, 5)
	clone := original.Clone()

	if !original.Equals(clone) {
		t.Errorf("expected clone [%v] to equal original [%v]", clone, original)
	}
}
