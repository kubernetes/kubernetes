// Copyright 2014-2017 Ulrich Kunitz. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lzma

/* Naming conventions follows the CodeReviewComments in the Go Wiki. */

// ntz32Const is used by the functions NTZ and NLZ.
const ntz32Const = 0x04d7651f

// ntz32Table is a helper table for de Bruijn algorithm by Danny Dubé.
// See Henry S. Warren, Jr. "Hacker's Delight" section 5-1 figure 5-26.
var ntz32Table = [32]int8{
	0, 1, 2, 24, 3, 19, 6, 25,
	22, 4, 20, 10, 16, 7, 12, 26,
	31, 23, 18, 5, 21, 9, 15, 11,
	30, 17, 8, 14, 29, 13, 28, 27,
}

// ntz32 computes the number of trailing zeros for an unsigned 32-bit integer.
func ntz32(x uint32) int {
	if x == 0 {
		return 32
	}
	x = (x & -x) * ntz32Const
	return int(ntz32Table[x>>27])
}

// nlz32 computes the number of leading zeros for an unsigned 32-bit integer.
func nlz32(x uint32) int {
	// Smear left most bit to the right
	x |= x >> 1
	x |= x >> 2
	x |= x >> 4
	x |= x >> 8
	x |= x >> 16
	// Use ntz mechanism to calculate nlz.
	x++
	if x == 0 {
		return 0
	}
	x *= ntz32Const
	return 32 - int(ntz32Table[x>>27])
}
