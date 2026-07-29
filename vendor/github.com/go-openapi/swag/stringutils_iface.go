// SPDX-FileCopyrightText: Copyright 2015-2025 go-swagger maintainers
// SPDX-License-Identifier: Apache-2.0

package swag

import "github.com/go-openapi/swag/stringutils"

// ContainsStrings searches a slice of strings for a case-sensitive match.
//
// Deprecated: use [slices.Contains] or [stringutils.ContainsStrings] instead.
func ContainsStrings(coll []string, item string) bool {
	return stringutils.ContainsStrings(coll, item)
}

// ContainsStringsCI searches a slice of strings for a case-insensitive match.
//
// Deprecated: use [stringutils.ContainsStringsCI] instead.
func ContainsStringsCI(coll []string, item string) bool {
	return stringutils.ContainsStringsCI(coll, item)
}

// JoinByFormat joins a string array by a known format (e.g. swagger's collectionFormat attribute).
//
// Deprecated: use [stringutils.JoinByFormat] instead.
func JoinByFormat(data []string, format string) []string {
	return stringutils.JoinByFormat(data, format)
}

// SplitByFormat splits a string by a known format.
//
// Deprecated: use [stringutils.SplitByFormat] instead.
func SplitByFormat(data, format string) []string {
	return stringutils.SplitByFormat(data, format)
}
