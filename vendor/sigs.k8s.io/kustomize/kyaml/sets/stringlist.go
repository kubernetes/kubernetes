// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package sets

// StringList is a set, where each element of
// the set is a string slice.
type StringList [][]string

func (s StringList) Len() int {
	return len(s)
}

func (s StringList) Insert(val []string) StringList {
	if !s.Has(val) {
		return append(s, val)
	}
	return s
}

func (s StringList) Has(val []string) bool {
	if len(s) == 0 {
		return false
	}

	for i := range s {
		if isStringSliceEqual(s[i], val) {
			return true
		}
	}
	return false
}

func isStringSliceEqual(s []string, t []string) bool {
	if len(s) != len(t) {
		return false
	}
	for i := range s {
		if s[i] != t[i] {
			return false
		}
	}
	return true
}
