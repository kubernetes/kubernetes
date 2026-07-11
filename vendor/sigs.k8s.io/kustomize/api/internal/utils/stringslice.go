// Copyright 2020 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package utils

// StringSliceIndex returns the index of the str, else -1.
func StringSliceIndex(slice []string, str string) int {
	for i := range slice {
		if slice[i] == str {
			return i
		}
	}
	return -1
}

// StringSliceContains returns true if the slice has the string.
func StringSliceContains(slice []string, str string) bool {
	for _, s := range slice {
		if s == str {
			return true
		}
	}
	return false
}

// SameEndingSubSlice returns true if the slices end the same way, e.g.
// {"a", "b", "c"}, {"b", "c"} => true
// {"a", "b", "c"}, {"a", "b"} => false
// If one slice is empty and the other is not, return false.
func SameEndingSubSlice(shortest, longest []string) bool {
	if len(shortest) > len(longest) {
		longest, shortest = shortest, longest
	}
	diff := len(longest) - len(shortest)
	if len(shortest) == 0 {
		return diff == 0
	}
	for i := len(shortest) - 1; i >= 0; i-- {
		if longest[i+diff] != shortest[i] {
			return false
		}
	}
	return true
}
