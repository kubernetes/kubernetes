// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gps

import (
	"bytes"
	"unicode"
	"unicode/utf8"
)

// toFold returns a string with the property that strings.EqualFold(s, t) iff
// ToFold(s) == ToFold(t) This lets us test a large set of strings for
// fold-equivalent duplicates without making a quadratic number of calls to
// EqualFold. Note that strings.ToUpper and strings.ToLower do not have the
// desired property in some corner cases.
//
// This is hoisted from toolchain internals: src/cmd/go/internal/str/str.go
func toFold(s string) string {
	// Fast path: all ASCII, no upper case.
	// Most paths look like this already.
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c >= utf8.RuneSelf || 'A' <= c && c <= 'Z' {
			goto Slow
		}
	}
	return s

Slow:
	var buf bytes.Buffer
	for _, r := range s {
		// SimpleFold(x) cycles to the next equivalent rune > x
		// or wraps around to smaller values. Iterate until it wraps,
		// and we've found the minimum value.
		for {
			r0 := r
			r = unicode.SimpleFold(r0)
			if r <= r0 {
				break
			}
		}
		// Exception to allow fast path above: A-Z => a-z
		if 'A' <= r && r <= 'Z' {
			r += 'a' - 'A'
		}
		buf.WriteRune(r)
	}
	return buf.String()
}
