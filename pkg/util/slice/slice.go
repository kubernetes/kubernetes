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
	"slices"
)

// CopyStrings copies the contents of the specified string slice
// into a new slice.
//
// Deprecated: Use slices.Clone instead.
var CopyStrings = slices.Clone[[]string]

// SortStrings sorts the specified string slice in place. It returns the same
// slice that was provided in order to facilitate method chaining.
//
// Deprecated: Use slices.Sort instead.
func SortStrings(s []string) []string {
	slices.Sort(s)
	return s
}

// ContainsString checks if a given slice of strings contains the provided string.
// If a modifier func is provided, it is called with the slice item before the comparation.
//
// Deprecated: Use slices.ContainsFunc or slices.Contains instead.
func ContainsString(slice []string, s string, modifier func(s string) string) bool {
	if slices.Contains(slice, s) {
		return true
	}
	if modifier != nil {
		return slices.ContainsFunc(slice, func(item string) bool { return modifier(item) == s })
	}
	return false
}

// RemoveString returns a newly created []string that contains all items from slice that
// are not equal to s and modifier(s) in case modifier func is provided.
func RemoveString(slice []string, s string, modifier func(s string) string) []string {
	newSlice := make([]string, 0)
	for _, item := range slice {
		if item == s {
			continue
		}
		if modifier != nil && modifier(item) == s {
			continue
		}
		newSlice = append(newSlice, item)
	}
	if len(newSlice) == 0 {
		// Sanitize for unit tests so we don't need to distinguish empty array
		// and nil.
		newSlice = nil
	}
	return newSlice
}
