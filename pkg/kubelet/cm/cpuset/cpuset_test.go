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
	"testing"
)

func TestCPUSetBuilder(t *testing.T) {
	b := NewBuilder()
	elems := []int{1, 2, 3, 4, 5}
	for _, elem := range elems {
		b.Add(elem)
	}
	result := b.Result()
	for _, elem := range elems {
		if !result.Contains(elem) {
			t.Fatalf("expected cpuset to contain element %d: [%v]", elem, result)
		}
	}
	if len(elems) != result.Size() {
		t.Fatalf("expected cpuset %s to have the same size as %v", result, elems)
	}
}

func TestCPUSetSize(t *testing.T) {
	testCases := []struct {
		cpuset   CPUSet
		expected int
	}{
		{NewCPUSet(), 0},
		{NewCPUSet(5), 1},
		{NewCPUSet(1, 2, 3, 4, 5), 5},
	}

	for _, c := range testCases {
		actual := c.cpuset.Size()
		if actual != c.expected {
			t.Fatalf("expected: %d, actual: %d, cpuset: [%v]", c.expected, actual, c.cpuset)
		}
	}
}

func TestCPUSetIsEmpty(t *testing.T) {
	testCases := []struct {
		cpuset   CPUSet
		expected bool
	}{
		{NewCPUSet(), true},
		{NewCPUSet(5), false},
		{NewCPUSet(1, 2, 3, 4, 5), false},
	}

	for _, c := range testCases {
		actual := c.cpuset.IsEmpty()
		if actual != c.expected {
			t.Fatalf("expected: %t, IsEmpty() returned: %t, cpuset: [%v]", c.expected, actual, c.cpuset)
		}
	}
}

func TestCPUSetContains(t *testing.T) {
	testCases := []struct {
		cpuset         CPUSet
		mustContain    []int
		mustNotContain []int
	}{
		{NewCPUSet(), []int{}, []int{1, 2, 3, 4, 5}},
		{NewCPUSet(5), []int{5}, []int{1, 2, 3, 4}},
		{NewCPUSet(1, 2, 4, 5), []int{1, 2, 4, 5}, []int{0, 3, 6}},
	}

	for _, c := range testCases {
		for _, elem := range c.mustContain {
			if !c.cpuset.Contains(elem) {
				t.Fatalf("expected cpuset to contain element %d: [%v]", elem, c.cpuset)
			}
		}
		for _, elem := range c.mustNotContain {
			if c.cpuset.Contains(elem) {
				t.Fatalf("expected cpuset not to contain element %d: [%v]", elem, c.cpuset)
			}
		}
	}
}

func TestCPUSetEqual(t *testing.T) {
	shouldEqual := []struct {
		s1 CPUSet
		s2 CPUSet
	}{
		{NewCPUSet(), NewCPUSet()},
		{NewCPUSet(5), NewCPUSet(5)},
		{NewCPUSet(1, 2, 3, 4, 5), NewCPUSet(1, 2, 3, 4, 5)},
	}

	shouldNotEqual := []struct {
		s1 CPUSet
		s2 CPUSet
	}{
		{NewCPUSet(), NewCPUSet(5)},
		{NewCPUSet(5), NewCPUSet()},
		{NewCPUSet(), NewCPUSet(1, 2, 3, 4, 5)},
		{NewCPUSet(1, 2, 3, 4, 5), NewCPUSet()},
		{NewCPUSet(5), NewCPUSet(1, 2, 3, 4, 5)},
		{NewCPUSet(1, 2, 3, 4, 5), NewCPUSet(5)},
	}

	for _, c := range shouldEqual {
		if !c.s1.Equals(c.s2) {
			t.Fatalf("expected cpusets to be equal: s1: [%v], s2: [%v]", c.s1, c.s2)
		}
	}
	for _, c := range shouldNotEqual {
		if c.s1.Equals(c.s2) {
			t.Fatalf("expected cpusets to not be equal: s1: [%v], s2: [%v]", c.s1, c.s2)
		}
	}
}

func TestCPUSetIsSubsetOf(t *testing.T) {
	shouldBeSubset := []struct {
		s1 CPUSet
		s2 CPUSet
	}{
		// A set is a subset of itself
		{NewCPUSet(), NewCPUSet()},
		{NewCPUSet(5), NewCPUSet(5)},
		{NewCPUSet(1, 2, 3, 4, 5), NewCPUSet(1, 2, 3, 4, 5)},

		// Empty set is a subset of every set
		{NewCPUSet(), NewCPUSet(5)},
		{NewCPUSet(), NewCPUSet(1, 2, 3, 4, 5)},

		{NewCPUSet(5), NewCPUSet(1, 2, 3, 4, 5)},
		{NewCPUSet(1, 2, 3), NewCPUSet(1, 2, 3, 4, 5)},
		{NewCPUSet(4, 5), NewCPUSet(1, 2, 3, 4, 5)},
		{NewCPUSet(2, 3), NewCPUSet(1, 2, 3, 4, 5)},
	}

	shouldNotBeSubset := []struct {
		s1 CPUSet
		s2 CPUSet
	}{}

	for _, c := range shouldBeSubset {
		if !c.s1.IsSubsetOf(c.s2) {
			t.Fatalf("expected s1 to be a subset of s2: s1: [%v], s2: [%v]", c.s1, c.s2)
		}
	}
	for _, c := range shouldNotBeSubset {
		if c.s1.IsSubsetOf(c.s2) {
			t.Fatalf("expected s1 to not be a subset of s2: s1: [%v], s2: [%v]", c.s1, c.s2)
		}
	}
}

func TestCPUSetUnionAll(t *testing.T) {
	testCases := []struct {
		s1       CPUSet
		s2       CPUSet
		s3       CPUSet
		expected CPUSet
	}{
		{NewCPUSet(), NewCPUSet(1, 2, 3, 4, 5), NewCPUSet(4, 5), NewCPUSet(1, 2, 3, 4, 5)},
		{NewCPUSet(1, 2, 3, 4, 5), NewCPUSet(), NewCPUSet(4), NewCPUSet(1, 2, 3, 4, 5)},
		{NewCPUSet(1, 2, 3, 4, 5), NewCPUSet(1, 2, 3, 4, 5), NewCPUSet(1, 5), NewCPUSet(1, 2, 3, 4, 5)},
	}
	for _, c := range testCases {
		s := []CPUSet{}
		s = append(s, c.s2)
		s = append(s, c.s3)
		result := c.s1.UnionAll(s)
		if !result.Equals(c.expected) {
			t.Fatalf("expected the union of s1 and s2 to be [%v] (got [%v]), s1: [%v], s2: [%v]", c.expected, result, c.s1, c.s2)
		}
	}
}

func TestCPUSetUnion(t *testing.T) {
	testCases := []struct {
		s1       CPUSet
		s2       CPUSet
		expected CPUSet
	}{
		{NewCPUSet(), NewCPUSet(), NewCPUSet()},

		{NewCPUSet(), NewCPUSet(5), NewCPUSet(5)},
		{NewCPUSet(5), NewCPUSet(), NewCPUSet(5)},
		{NewCPUSet(5), NewCPUSet(5), NewCPUSet(5)},

		{NewCPUSet(), NewCPUSet(1, 2, 3, 4, 5), NewCPUSet(1, 2, 3, 4, 5)},
		{NewCPUSet(1, 2, 3, 4, 5), NewCPUSet(), NewCPUSet(1, 2, 3, 4, 5)},
		{NewCPUSet(1, 2, 3, 4, 5), NewCPUSet(1, 2, 3, 4, 5), NewCPUSet(1, 2, 3, 4, 5)},

		{NewCPUSet(5), NewCPUSet(1, 2, 3, 4, 5), NewCPUSet(1, 2, 3, 4, 5)},
		{NewCPUSet(1, 2, 3, 4, 5), NewCPUSet(5), NewCPUSet(1, 2, 3, 4, 5)},

		{NewCPUSet(1, 2), NewCPUSet(3, 4, 5), NewCPUSet(1, 2, 3, 4, 5)},
		{NewCPUSet(1, 2, 3), NewCPUSet(3, 4, 5), NewCPUSet(1, 2, 3, 4, 5)},
	}

	for _, c := range testCases {
		result := c.s1.Union(c.s2)
		if !result.Equals(c.expected) {
			t.Fatalf("expected the union of s1 and s2 to be [%v] (got [%v]), s1: [%v], s2: [%v]", c.expected, result, c.s1, c.s2)
		}
	}
}

func TestCPUSetIntersection(t *testing.T) {
	testCases := []struct {
		s1       CPUSet
		s2       CPUSet
		expected CPUSet
	}{
		{NewCPUSet(), NewCPUSet(), NewCPUSet()},

		{NewCPUSet(), NewCPUSet(5), NewCPUSet()},
		{NewCPUSet(5), NewCPUSet(), NewCPUSet()},
		{NewCPUSet(5), NewCPUSet(5), NewCPUSet(5)},

		{NewCPUSet(), NewCPUSet(1, 2, 3, 4, 5), NewCPUSet()},
		{NewCPUSet(1, 2, 3, 4, 5), NewCPUSet(), NewCPUSet()},
		{NewCPUSet(1, 2, 3, 4, 5), NewCPUSet(1, 2, 3, 4, 5), NewCPUSet(1, 2, 3, 4, 5)},

		{NewCPUSet(5), NewCPUSet(1, 2, 3, 4, 5), NewCPUSet(5)},
		{NewCPUSet(1, 2, 3, 4, 5), NewCPUSet(5), NewCPUSet(5)},

		{NewCPUSet(1, 2), NewCPUSet(3, 4, 5), NewCPUSet()},
		{NewCPUSet(1, 2, 3), NewCPUSet(3, 4, 5), NewCPUSet(3)},
	}

	for _, c := range testCases {
		result := c.s1.Intersection(c.s2)
		if !result.Equals(c.expected) {
			t.Fatalf("expected the intersection of s1 and s2 to be [%v] (got [%v]), s1: [%v], s2: [%v]", c.expected, result, c.s1, c.s2)
		}
	}
}

func TestCPUSetDifference(t *testing.T) {
	testCases := []struct {
		s1       CPUSet
		s2       CPUSet
		expected CPUSet
	}{
		{NewCPUSet(), NewCPUSet(), NewCPUSet()},

		{NewCPUSet(), NewCPUSet(5), NewCPUSet()},
		{NewCPUSet(5), NewCPUSet(), NewCPUSet(5)},
		{NewCPUSet(5), NewCPUSet(5), NewCPUSet()},

		{NewCPUSet(), NewCPUSet(1, 2, 3, 4, 5), NewCPUSet()},
		{NewCPUSet(1, 2, 3, 4, 5), NewCPUSet(), NewCPUSet(1, 2, 3, 4, 5)},
		{NewCPUSet(1, 2, 3, 4, 5), NewCPUSet(1, 2, 3, 4, 5), NewCPUSet()},

		{NewCPUSet(5), NewCPUSet(1, 2, 3, 4, 5), NewCPUSet()},
		{NewCPUSet(1, 2, 3, 4, 5), NewCPUSet(5), NewCPUSet(1, 2, 3, 4)},

		{NewCPUSet(1, 2), NewCPUSet(3, 4, 5), NewCPUSet(1, 2)},
		{NewCPUSet(1, 2, 3), NewCPUSet(3, 4, 5), NewCPUSet(1, 2)},
	}

	for _, c := range testCases {
		result := c.s1.Difference(c.s2)
		if !result.Equals(c.expected) {
			t.Fatalf("expected the difference of s1 and s2 to be [%v] (got [%v]), s1: [%v], s2: [%v]", c.expected, result, c.s1, c.s2)
		}
	}
}

func TestCPUSetToSlice(t *testing.T) {
	testCases := []struct {
		set      CPUSet
		expected []int
	}{
		{NewCPUSet(), []int{}},
		{NewCPUSet(5), []int{5}},
		{NewCPUSet(1, 2, 3, 4, 5), []int{1, 2, 3, 4, 5}},
	}

	for _, c := range testCases {
		result := c.set.ToSlice()
		if !reflect.DeepEqual(result, c.expected) {
			t.Fatalf("expected set as slice to be [%v] (got [%v]), s: [%v]", c.expected, result, c.set)
		}
	}
}

func TestCPUSetString(t *testing.T) {
	testCases := []struct {
		set      CPUSet
		expected string
	}{
		{NewCPUSet(), ""},
		{NewCPUSet(5), "5"},
		{NewCPUSet(1, 2, 3, 4, 5), "1-5"},
		{NewCPUSet(1, 2, 3, 5, 6, 8), "1-3,5-6,8"},
	}

	for _, c := range testCases {
		result := c.set.String()
		if result != c.expected {
			t.Fatalf("expected set as string to be %s (got \"%s\"), s: [%v]", c.expected, result, c.set)
		}
	}
}

func TestParse(t *testing.T) {
	testCases := []struct {
		cpusetString string
		expected     CPUSet
	}{
		{"", NewCPUSet()},
		{"5", NewCPUSet(5)},
		{"1,2,3,4,5", NewCPUSet(1, 2, 3, 4, 5)},
		{"1-5", NewCPUSet(1, 2, 3, 4, 5)},
		{"1-2,3-5", NewCPUSet(1, 2, 3, 4, 5)},
	}

	for _, c := range testCases {
		result, err := Parse(c.cpusetString)
		if err != nil {
			t.Fatalf("expected error not to have occurred: %v", err)
		}
		if !result.Equals(c.expected) {
			t.Fatalf("expected string \"%s\" to parse as [%v] (got [%v])", c.cpusetString, c.expected, result)
		}
	}
}
