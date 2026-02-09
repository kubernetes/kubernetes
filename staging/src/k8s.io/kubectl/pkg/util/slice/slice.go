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

package slice

import (
	"sort"
)

// SortInts64 sorts []int64 in increasing order
func SortInts64(a []int64) { sort.Slice(a, func(i, j int) bool { return a[i] < a[j] }) }

// Contains checks if a given slice of type T contains the provided item.
// If a modifier func is provided, it is called with the slice item before the comparation.
func Contains[T comparable](slice []T, s T, modifier func(s T) T) bool {
	for _, item := range slice {
		if item == s {
			return true
		}
		if modifier != nil && modifier(item) == s {
			return true
		}
	}
	return false
}

// ContainsString checks if a given slice of strings contains the provided string.
// If a modifier func is provided, it is called with the slice item before the comparation.
// Deprecated: Use Contains[T] instead
func ContainsString(slice []string, s string, modifier func(s string) string) bool {
	for _, item := range slice {
		if item == s {
			return true
		}
		if modifier != nil && modifier(item) == s {
			return true
		}
	}
	return false
}

// ToSet returns a single slice containing the unique values from one or more slices. The order of the items in the
// result is not guaranteed.
func ToSet[T comparable](slices ...[]T) []T {
	if len(slices) == 0 {
		return nil
	}
	m := map[T]struct{}{}
	for _, slice := range slices {
		for _, value := range slice {
			m[value] = struct{}{}
		}
	}
	result := []T{}
	for k := range m {
		result = append(result, k)
	}
	return result
}
