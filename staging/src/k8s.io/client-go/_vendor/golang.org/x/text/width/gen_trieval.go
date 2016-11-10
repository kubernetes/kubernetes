// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

package main

// elem is an entry of the width trie. The high byte is used to encode the type
// of the rune. The low byte is used to store the index to a mapping entry in
// the inverseData array.
type elem uint16

const (
	tagNeutral elem = iota << typeShift
	tagAmbiguous
	tagWide
	tagNarrow
	tagFullwidth
	tagHalfwidth
)

const (
	numTypeBits = 3
	typeShift   = 16 - numTypeBits

	// tagNeedsFold is true for all fullwidth and halfwidth runes except for
	// the Won sign U+20A9.
	tagNeedsFold = 0x1000

	// The Korean Won sign is halfwidth, but SHOULD NOT be mapped to a wide
	// variant.
	wonSign rune = 0x20A9
)
