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

// Package slice provides utility methods for common operations on slices.
package slice

import (
	"sort"

	utilrand "k8s.io/kubernetes/pkg/util/rand"
)

// CopyStrings copies the contents of the specified string slice
// into a new slice.
func CopyStrings(s []string) []string {
	c := make([]string, len(s))
	copy(c, s)
	return c
}

// SortStrings sorts the specified string slice in place. It returns the same
// slice that was provided in order to facilitate method chaining.
func SortStrings(s []string) []string {
	sort.Strings(s)
	return s
}

// ShuffleStrings copies strings from the specified slice into a copy in random
// order. It returns a new slice.
func ShuffleStrings(s []string) []string {
	shuffled := make([]string, len(s))
	perm := utilrand.Perm(len(s))
	for i, j := range perm {
		shuffled[j] = s[i]
	}
	return shuffled
}

// Int64Slice attaches the methods of Interface to []int64,
// sorting in increasing order.
type Int64Slice []int64

func (p Int64Slice) Len() int           { return len(p) }
func (p Int64Slice) Less(i, j int) bool { return p[i] < p[j] }
func (p Int64Slice) Swap(i, j int)      { p[i], p[j] = p[j], p[i] }

// Sorts []int64 in increasing order
func SortInts64(a []int64) { sort.Sort(Int64Slice(a)) }
