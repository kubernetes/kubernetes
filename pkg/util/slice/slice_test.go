/*
Copyright 2015 The Kubernetes Authors.

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

package slice

import (
	"reflect"
	"testing"
)

func TestCopyStrings(t *testing.T) {
	var src1 []string
	dest1 := CopyStrings(src1)

	if !reflect.DeepEqual(src1, dest1) {
		t.Errorf("%v and %v are not equal", src1, dest1)
	}

	src2 := []string{}
	dest2 := CopyStrings(src2)

	if !reflect.DeepEqual(src2, dest2) {
		t.Errorf("%v and %v are not equal", src2, dest2)
	}

	src3 := []string{"a", "c", "b"}
	dest3 := CopyStrings(src3)

	if !reflect.DeepEqual(src3, dest3) {
		t.Errorf("%v and %v are not equal", src3, dest3)
	}

	src3[0] = "A"
	if reflect.DeepEqual(src3, dest3) {
		t.Errorf("CopyStrings didn't make a copy")
	}
}

func TestSortStrings(t *testing.T) {
	src := []string{"a", "c", "b"}
	dest := SortStrings(src)
	expected := []string{"a", "b", "c"}

	if !reflect.DeepEqual(dest, expected) {
		t.Errorf("SortString didn't sort the strings")
	}

	if !reflect.DeepEqual(src, expected) {
		t.Errorf("SortString didn't sort in place")
	}
}

func TestContainsString(t *testing.T) {
	src := []string{"aa", "bb", "cc"}
	if !ContainsString(src, "bb", nil) {
		t.Errorf("ContainsString didn't find the string as expected")
	}

	modifier := func(s string) string {
		if s == "cc" {
			return "ee"
		}
		return s
	}
	if !ContainsString(src, "ee", modifier) {
		t.Errorf("ContainsString didn't find the string by modifier")
	}
}

func TestRemoveString(t *testing.T) {
	modifier := func(s string) string {
		if s == "ab" {
			return "ee"
		}
		return s
	}
	tests := []struct {
		testName string
		input    []string
		remove   string
		modifier func(s string) string
		want     []string
	}{
		{
			testName: "Nil input slice",
			input:    nil,
			remove:   "",
			modifier: nil,
			want:     nil,
		},
		{
			testName: "Slice doesn't contain the string",
			input:    []string{"a", "ab", "cdef"},
			remove:   "NotPresentInSlice",
			modifier: nil,
			want:     []string{"a", "ab", "cdef"},
		},
		{
			testName: "All strings removed, result is nil",
			input:    []string{"a"},
			remove:   "a",
			modifier: nil,
			want:     nil,
		},
		{
			testName: "No modifier func, one string removed",
			input:    []string{"a", "ab", "cdef"},
			remove:   "ab",
			modifier: nil,
			want:     []string{"a", "cdef"},
		},
		{
			testName: "No modifier func, all(three) strings removed",
			input:    []string{"ab", "a", "ab", "cdef", "ab"},
			remove:   "ab",
			modifier: nil,
			want:     []string{"a", "cdef"},
		},
		{
			testName: "Removed both the string and the modifier func result",
			input:    []string{"a", "cd", "ab", "ee"},
			remove:   "ee",
			modifier: modifier,
			want:     []string{"a", "cd"},
		},
	}
	for _, tt := range tests {
		if got := RemoveString(tt.input, tt.remove, tt.modifier); !reflect.DeepEqual(got, tt.want) {
			t.Errorf("%v: RemoveString(%v, %q, %T) = %v WANT %v", tt.testName, tt.input, tt.remove, tt.modifier, got, tt.want)
		}
	}
}
