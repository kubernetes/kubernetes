//go:build go1.18
// +build go1.18

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

package azcore

import (
	"strings"
)

// ETag is a property used for optimistic concurrency during updates
// ETag is a validator based on https://tools.ietf.org/html/rfc7232#section-2.3.2
// An ETag can be empty ("").
type ETag string

// ETagAny is an ETag that represents everything, the value is "*"
const ETagAny ETag = "*"

// Equals does a strong comparison of two ETags. Equals returns true when both
// ETags are not weak and the values of the underlying strings are equal.
func (e ETag) Equals(other ETag) bool {
	return !e.IsWeak() && !other.IsWeak() && e == other
}

// WeakEquals does a weak comparison of two ETags. Two ETags are equivalent if their opaque-tags match
// character-by-character, regardless of either or both being tagged as "weak".
func (e ETag) WeakEquals(other ETag) bool {
	getStart := func(e1 ETag) int {
		if e1.IsWeak() {
			return 2
		}
		return 0
	}
	aStart := getStart(e)
	bStart := getStart(other)

	aVal := e[aStart:]
	bVal := other[bStart:]

	return aVal == bVal
}

// IsWeak specifies whether the ETag is strong or weak.
func (e ETag) IsWeak() bool {
	return len(e) >= 4 && strings.HasPrefix(string(e), "W/\"") && strings.HasSuffix(string(e), "\"")
}
