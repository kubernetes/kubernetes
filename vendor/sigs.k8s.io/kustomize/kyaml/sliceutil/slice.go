// Copyright 2021 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package sliceutil

import "slices"

// Contains return true if string e is present in slice s
//
// Deprecated: use slices.Contains instead.
//
//go:fix inline
func Contains(s []string, e string) bool {
	return slices.Contains(s, e)
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
