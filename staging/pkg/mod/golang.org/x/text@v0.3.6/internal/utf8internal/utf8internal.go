// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package utf8internal contains low-level utf8-related constants, tables, etc.
// that are used internally by the text package.
package utf8internal

// The default lowest and highest continuation byte.
const (
	LoCB = 0x80 // 1000 0000
	HiCB = 0xBF // 1011 1111
)

// Constants related to getting information of first bytes of UTF-8 sequences.
const (
	// ASCII identifies a UTF-8 byte as ASCII.
	ASCII = as

	// FirstInvalid indicates a byte is invalid as a first byte of a UTF-8
	// sequence.
	FirstInvalid = xx

	// SizeMask is a mask for the size bits. Use use x&SizeMask to get the size.
	SizeMask = 7

	// AcceptShift is the right-shift count for the first byte info byte to get
	// the index into the AcceptRanges table. See AcceptRanges.
	AcceptShift = 4

	// The names of these constants are chosen to give nice alignment in the
	// table below. The first nibble is an index into acceptRanges or F for
	// special one-byte cases. The second nibble is the Rune length or the
	// Status for the special one-byte case.
	xx = 0xF1 // invalid: size 1
	as = 0xF0 // ASCII: size 1
	s1 = 0x02 // accept 0, size 2
	s2 = 0x13 // accept 1, size 3
	s3 = 0x03 // accept 0, size 3
	s4 = 0x23 // accept 2, size 3
	s5 = 0x34 // accept 3, size 4
	s6 = 0x04 // accept 0, size 4
	s7 = 0x44 // accept 4, size 4
)

// First is information about the first byte in a UTF-8 sequence.
var First = [256]uint8{
	//   1   2   3   4   5   6   7   8   9   A   B   C   D   E   F
	as, as, as, as, as, as, as, as, as, as, as, as, as, as, as, as, // 0x00-0x0F
	as, as, as, as, as, as, as, as, as, as, as, as, as, as, as, as, // 0x10-0x1F
	as, as, as, as, as, as, as, as, as, as, as, as, as, as, as, as, // 0x20-0x2F
	as, as, as, as, as, as, as, as, as, as, as, as, as, as, as, as, // 0x30-0x3F
	as, as, as, as, as, as, as, as, as, as, as, as, as, as, as, as, // 0x40-0x4F
	as, as, as, as, as, as, as, as, as, as, as, as, as, as, as, as, // 0x50-0x5F
	as, as, as, as, as, as, as, as, as, as, as, as, as, as, as, as, // 0x60-0x6F
	as, as, as, as, as, as, as, as, as, as, as, as, as, as, as, as, // 0x70-0x7F
	//   1   2   3   4   5   6   7   8   9   A   B   C   D   E   F
	xx, xx, xx, xx, xx, xx, xx, xx, xx, xx, xx, xx, xx, xx, xx, xx, // 0x80-0x8F
	xx, xx, xx, xx, xx, xx, xx, xx, xx, xx, xx, xx, xx, xx, xx, xx, // 0x90-0x9F
	xx, xx, xx, xx, xx, xx, xx, xx, xx, xx, xx, xx, xx, xx, xx, xx, // 0xA0-0xAF
	xx, xx, xx, xx, xx, xx, xx, xx, xx, xx, xx, xx, xx, xx, xx, xx, // 0xB0-0xBF
	xx, xx, s1, s1, s1, s1, s1, s1, s1, s1, s1, s1, s1, s1, s1, s1, // 0xC0-0xCF
	s1, s1, s1, s1, s1, s1, s1, s1, s1, s1, s1, s1, s1, s1, s1, s1, // 0xD0-0xDF
	s2, s3, s3, s3, s3, s3, s3, s3, s3, s3, s3, s3, s3, s4, s3, s3, // 0xE0-0xEF
	s5, s6, s6, s6, s7, xx, xx, xx, xx, xx, xx, xx, xx, xx, xx, xx, // 0xF0-0xFF
}

// AcceptRange gives the range of valid values for the second byte in a UTF-8
// sequence for any value for First that is not ASCII or FirstInvalid.
type AcceptRange struct {
	Lo uint8 // lowest value for second byte.
	Hi uint8 // highest value for second byte.
}

// AcceptRanges is a slice of AcceptRange values. For a given byte sequence b
//
//		AcceptRanges[First[b[0]]>>AcceptShift]
//
// will give the value of AcceptRange for the multi-byte UTF-8 sequence starting
// at b[0].
var AcceptRanges = [...]AcceptRange{
	0: {LoCB, HiCB},
	1: {0xA0, HiCB},
	2: {LoCB, 0x9F},
	3: {0x90, HiCB},
	4: {LoCB, 0x8F},
}
