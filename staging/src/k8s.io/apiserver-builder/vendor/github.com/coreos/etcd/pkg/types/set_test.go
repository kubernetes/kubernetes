// Copyright 2015 The etcd Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package types

import (
	"reflect"
	"sort"
	"testing"
)

func TestUnsafeSet(t *testing.T) {
	driveSetTests(t, NewUnsafeSet())
}

func TestThreadsafeSet(t *testing.T) {
	driveSetTests(t, NewThreadsafeSet())
}

// Check that two slices contents are equal; order is irrelevant
func equal(a, b []string) bool {
	as := sort.StringSlice(a)
	bs := sort.StringSlice(b)
	as.Sort()
	bs.Sort()
	return reflect.DeepEqual(as, bs)
}

func driveSetTests(t *testing.T, s Set) {
	// Verify operations on an empty set
	eValues := []string{}
	values := s.Values()
	if !reflect.DeepEqual(values, eValues) {
		t.Fatalf("Expect values=%v got %v", eValues, values)
	}
	if l := s.Length(); l != 0 {
		t.Fatalf("Expected length=0, got %d", l)
	}
	for _, v := range []string{"foo", "bar", "baz"} {
		if s.Contains(v) {
			t.Fatalf("Expect s.Contains(%q) to be fale, got true", v)
		}
	}

	// Add three items, ensure they show up
	s.Add("foo")
	s.Add("bar")
	s.Add("baz")

	eValues = []string{"foo", "bar", "baz"}
	values = s.Values()
	if !equal(values, eValues) {
		t.Fatalf("Expect values=%v got %v", eValues, values)
	}

	for _, v := range eValues {
		if !s.Contains(v) {
			t.Fatalf("Expect s.Contains(%q) to be true, got false", v)
		}
	}

	if l := s.Length(); l != 3 {
		t.Fatalf("Expected length=3, got %d", l)
	}

	// Add the same item a second time, ensuring it is not duplicated
	s.Add("foo")

	values = s.Values()
	if !equal(values, eValues) {
		t.Fatalf("Expect values=%v got %v", eValues, values)
	}
	if l := s.Length(); l != 3 {
		t.Fatalf("Expected length=3, got %d", l)
	}

	// Remove all items, ensure they are gone
	s.Remove("foo")
	s.Remove("bar")
	s.Remove("baz")

	eValues = []string{}
	values = s.Values()
	if !equal(values, eValues) {
		t.Fatalf("Expect values=%v got %v", eValues, values)
	}

	if l := s.Length(); l != 0 {
		t.Fatalf("Expected length=0, got %d", l)
	}

	// Create new copies of the set, and ensure they are unlinked to the
	// original Set by making modifications
	s.Add("foo")
	s.Add("bar")
	cp1 := s.Copy()
	cp2 := s.Copy()
	s.Remove("foo")
	cp3 := s.Copy()
	cp1.Add("baz")

	for i, tt := range []struct {
		want []string
		got  []string
	}{
		{[]string{"bar"}, s.Values()},
		{[]string{"foo", "bar", "baz"}, cp1.Values()},
		{[]string{"foo", "bar"}, cp2.Values()},
		{[]string{"bar"}, cp3.Values()},
	} {
		if !equal(tt.want, tt.got) {
			t.Fatalf("case %d: expect values=%v got %v", i, tt.want, tt.got)
		}
	}

	for i, tt := range []struct {
		want bool
		got  bool
	}{
		{true, s.Equals(cp3)},
		{true, cp3.Equals(s)},
		{false, s.Equals(cp2)},
		{false, s.Equals(cp1)},
		{false, cp1.Equals(s)},
		{false, cp2.Equals(s)},
		{false, cp2.Equals(cp1)},
	} {
		if tt.got != tt.want {
			t.Fatalf("case %d: want %t, got %t", i, tt.want, tt.got)

		}
	}

	// Subtract values from a Set, ensuring a new Set is created and
	// the original Sets are unmodified
	sub1 := cp1.Sub(s)
	sub2 := cp2.Sub(cp1)

	for i, tt := range []struct {
		want []string
		got  []string
	}{
		{[]string{"foo", "bar", "baz"}, cp1.Values()},
		{[]string{"foo", "bar"}, cp2.Values()},
		{[]string{"bar"}, s.Values()},
		{[]string{"foo", "baz"}, sub1.Values()},
		{[]string{}, sub2.Values()},
	} {
		if !equal(tt.want, tt.got) {
			t.Fatalf("case %d: expect values=%v got %v", i, tt.want, tt.got)
		}
	}
}

func TestUnsafeSetContainsAll(t *testing.T) {
	vals := []string{"foo", "bar", "baz"}
	s := NewUnsafeSet(vals...)

	tests := []struct {
		strs     []string
		wcontain bool
	}{
		{[]string{}, true},
		{vals[:1], true},
		{vals[:2], true},
		{vals, true},
		{[]string{"cuz"}, false},
		{[]string{vals[0], "cuz"}, false},
	}
	for i, tt := range tests {
		if g := s.ContainsAll(tt.strs); g != tt.wcontain {
			t.Errorf("#%d: ok = %v, want %v", i, g, tt.wcontain)
		}
	}
}
