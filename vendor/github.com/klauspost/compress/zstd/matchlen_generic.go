//go:build !amd64 || appengine || !gc || noasm
// +build !amd64 appengine !gc noasm

// Copyright 2019+ Klaus Post. All rights reserved.
// License information can be found in the LICENSE file.

package zstd

import (
	"math/bits"

	"github.com/klauspost/compress/internal/le"
)

// matchLen returns the maximum common prefix length of a and b.
// a must be the shortest of the two.
func matchLen(a, b []byte) (n int) {
	left := len(a)
	for left >= 8 {
		diff := le.Load64(a, n) ^ le.Load64(b, n)
		if diff != 0 {
			return n + bits.TrailingZeros64(diff)>>3
		}
		n += 8
		left -= 8
	}
	a = a[n:]
	b = b[n:]

	for i := range a {
		if a[i] != b[i] {
			break
		}
		n++
	}
	return n

}
