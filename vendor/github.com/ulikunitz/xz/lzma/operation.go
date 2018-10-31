// Copyright 2014-2017 Ulrich Kunitz. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lzma

import (
	"errors"
	"fmt"
	"unicode"
)

// operation represents an operation on the dictionary during encoding or
// decoding.
type operation interface {
	Len() int
}

// rep represents a repetition at the given distance and the given length
type match struct {
	// supports all possible distance values, including the eos marker
	distance int64
	// length
	n int
}

// verify checks whether the match is valid. If that is not the case an
// error is returned.
func (m match) verify() error {
	if !(minDistance <= m.distance && m.distance <= maxDistance) {
		return errors.New("distance out of range")
	}
	if !(1 <= m.n && m.n <= maxMatchLen) {
		return errors.New("length out of range")
	}
	return nil
}

// l return the l-value for the match, which is the difference of length
// n and 2.
func (m match) l() uint32 {
	return uint32(m.n - minMatchLen)
}

// dist returns the dist value for the match, which is one less of the
// distance stored in the match.
func (m match) dist() uint32 {
	return uint32(m.distance - minDistance)
}

// Len returns the number of bytes matched.
func (m match) Len() int {
	return m.n
}

// String returns a string representation for the repetition.
func (m match) String() string {
	return fmt.Sprintf("M{%d,%d}", m.distance, m.n)
}

// lit represents a single byte literal.
type lit struct {
	b byte
}

// Len returns 1 for the single byte literal.
func (l lit) Len() int {
	return 1
}

// String returns a string representation for the literal.
func (l lit) String() string {
	var c byte
	if unicode.IsPrint(rune(l.b)) {
		c = l.b
	} else {
		c = '.'
	}
	return fmt.Sprintf("L{%c/%02x}", c, l.b)
}
