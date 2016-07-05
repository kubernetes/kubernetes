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
	src := []string{"a", "c", "b"}
	dest := CopyStrings(src)

	if !reflect.DeepEqual(src, dest) {
		t.Errorf("%v and %v are not equal", src, dest)
	}

	src[0] = "A"
	if reflect.DeepEqual(src, dest) {
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

func TestShuffleStrings(t *testing.T) {
	src := []string{"a", "b", "c", "d", "e", "f"}
	dest := ShuffleStrings(src)

	if len(src) != len(dest) {
		t.Errorf("Shuffled slice is wrong length, expected %v got %v", len(src), len(dest))
	}

	m := make(map[string]bool, len(dest))
	for _, s := range dest {
		m[s] = true
	}

	for _, k := range src {
		if _, exists := m[k]; !exists {
			t.Errorf("Element %v missing from shuffled slice", k)
		}
	}
}

func TestSortInts64(t *testing.T) {
	src := []int64{10, 1, 2, 3, 4, 5, 6}
	expected := []int64{1, 2, 3, 4, 5, 6, 10}
	SortInts64(src)
	if !reflect.DeepEqual(src, expected) {
		t.Errorf("func Ints64 didnt sort correctly, %v !- %v", src, expected)
	}
}
