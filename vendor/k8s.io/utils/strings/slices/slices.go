/*
Copyright 2021 The Kubernetes Authors.

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

// Package slices defines various functions useful with slices of string type.
// The goal is to be as close as possible to
// https://github.com/golang/go/issues/45955. Ideal would be if we can just
// replace "stringslices" if the "slices" package becomes standard.
package slices

// Equal reports whether two slices are equal: the same length and all
// elements equal. If the lengths are different, Equal returns false.
// Otherwise, the elements are compared in index order, and the
// comparison stops at the first unequal pair.
func Equal(s1, s2 []string) bool {
	if len(s1) != len(s2) {
		return false
	}
	for i, n := range s1 {
		if n != s2[i] {
			return false
		}
	}
	return true
}

// Filter appends to d each element e of s for which keep(e) returns true.
// It returns the modified d. d may be s[:0], in which case the kept
// elements will be stored in the same slice.
// if the slices overlap in some other way, the results are unspecified.
// To create a new slice with the filtered results, pass nil for d.
func Filter(d, s []string, keep func(string) bool) []string {
	for _, n := range s {
		if keep(n) {
			d = append(d, n)
		}
	}
	return d
}

// Contains reports whether v is present in s.
func Contains(s []string, v string) bool {
	return Index(s, v) >= 0
}

// Index returns the index of the first occurrence of v in s, or -1 if
// not present.
func Index(s []string, v string) int {
	// "Contains" may be replaced with "Index(s, v) >= 0":
	// https://github.com/golang/go/issues/45955#issuecomment-873377947
	for i, n := range s {
		if n == v {
			return i
		}
	}
	return -1
}

// Functions below are not in https://github.com/golang/go/issues/45955

// Clone returns a new clone of s.
func Clone(s []string) []string {
	// https://github.com/go101/go101/wiki/There-is-not-a-perfect-way-to-clone-slices-in-Go
	if s == nil {
		return nil
	}
	c := make([]string, len(s))
	copy(c, s)
	return c
}
