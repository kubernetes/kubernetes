// Copyright 2021 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package sliceutil

// Contains return true if string e is present in slice s
func Contains(s []string, e string) bool {
	for _, a := range s {
		if a == e {
			return true
		}
	}
	return false
}

// Remove removes the first occurrence of r in slice s
// and returns remaining slice
func Remove(s []string, r string) []string {
	for i, v := range s {
		if v == r {
			return append(s[:i], s[i+1:]...)
		}
	}
	return s
}
