// SPDX-FileCopyrightText: Copyright 2015-2025 go-swagger maintainers
// SPDX-License-Identifier: Apache-2.0

package stringutils

import (
	"slices"
	"strings"
)

// ContainsStrings searches a slice of strings for a case-sensitive match
//
// Now equivalent to the standard library [slice.Contains].
func ContainsStrings(coll []string, item string) bool {
	return slices.Contains(coll, item)
}

// ContainsStringsCI searches a slice of strings for a case-insensitive match
func ContainsStringsCI(coll []string, item string) bool {
	return slices.ContainsFunc(coll, func(e string) bool {
		return strings.EqualFold(e, item)
	})
}
