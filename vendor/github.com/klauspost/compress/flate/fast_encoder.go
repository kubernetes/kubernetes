// Copyright 2011 The Snappy-Go Authors. All rights reserved.
// Modified for deflate by Klaus Post (c) 2015.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package flate

import (
	"fmt"
	"math/bits"
)

type fastEnc interface {
	Encode(dst *tokens, src []byte)
	Reset()
}

func newFastEnc(level int) fastEnc {
	switch level {
	case 1:
		return &fastEncL1{fastGen: fastGen{cur: maxStoreBlockSize}}
	case 2:
		return &fastEncL2{fastGen: fastGen{cur: maxStoreBlockSize}}
	case 3:
		return &fastEncL3{fastGen: fastGen{cur: maxStoreBlockSize}}
	case 4:
		return &fastEncL4{fastGen: fastGen{cur: maxStoreBlockSize}}
	case 5:
		return &fastEncL5{fastGen: fastGen{cur: maxStoreBlockSize}}
	case 6:
		return &fastEncL6{fastGen: fastGen{cur: maxStoreBlockSize}}
	default:
		panic("invalid level specified")
	}
}

const (
	tableBits       = 15             // Bits used in the table
	tableSize       = 1 << tableBits // Size of the table
	tableShift      = 32 - tableBits // Right-shift to get the tableBits most significant bits of a uint32.
	baseMatchOffset = 1              // The smallest match offset
	baseMatchLength = 3              // The smallest match length per the RFC section 3.2.5
	maxMatchOffset  = 1 << 15        // The largest match offset

	bTableBits   = 17                                               // Bits used in the big tables
	bTableSize   = 1 << bTableBits                                  // Size of the table
	allocHistory = maxStoreBlockSize * 10                           // Size to preallocate for history.
	bufferReset  = (1 << 31) - allocHistory - maxStoreBlockSize - 1 // Reset the buffer offset when reaching this.
)

const (
	prime3bytes = 506832829
	prime4bytes = 2654435761
	prime5bytes = 889523592379
	prime6bytes = 227718039650203
	prime7bytes = 58295818150454627
	prime8bytes = 0xcf1bbcdcb7a56463
)

func load32(b []byte, i int) uint32 {
	// Help the compiler eliminate bounds checks on the read so it can be done in a single read.
	b = b[i:]
	b = b[:4]
	return uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24
}

func load64(b []byte, i int) uint64 {
	// Help the compiler eliminate bounds checks on the read so it can be done in a single read.
	b = b[i:]
	b = b[:8]
	return uint64(b[0]) | uint64(b[1])<<8 | uint64(b[2])<<16 | uint64(b[3])<<24 |
		uint64(b[4])<<32 | uint64(b[5])<<40 | uint64(b[6])<<48 | uint64(b[7])<<56
}

func load3232(b []byte, i int32) uint32 {
	// Help the compiler eliminate bounds checks on the read so it can be done in a single read.
	b = b[i:]
	b = b[:4]
	return uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24
}

func load6432(b []byte, i int32) uint64 {
	// Help the compiler eliminate bounds checks on the read so it can be done in a single read.
	b = b[i:]
	b = b[:8]
	return uint64(b[0]) | uint64(b[1])<<8 | uint64(b[2])<<16 | uint64(b[3])<<24 |
		uint64(b[4])<<32 | uint64(b[5])<<40 | uint64(b[6])<<48 | uint64(b[7])<<56
}

func hash(u uint32) uint32 {
	return (u * 0x1e35a7bd) >> tableShift
}

type tableEntry struct {
	offset int32
}

// fastGen maintains the table for matches,
// and the previous byte block for level 2.
// This is the generic implementation.
type fastGen struct {
	hist []byte
	cur  int32
}

func (e *fastGen) addBlock(src []byte) int32 {
	// check if we have space already
	if len(e.hist)+len(src) > cap(e.hist) {
		if cap(e.hist) == 0 {
			e.hist = make([]byte, 0, allocHistory)
		} else {
			if cap(e.hist) < maxMatchOffset*2 {
				panic("unexpected buffer size")
			}
			// Move down
			offset := int32(len(e.hist)) - maxMatchOffset
			copy(e.hist[0:maxMatchOffset], e.hist[offset:])
			e.cur += offset
			e.hist = e.hist[:maxMatchOffset]
		}
	}
	s := int32(len(e.hist))
	e.hist = append(e.hist, src...)
	return s
}

// hash4 returns the hash of u to fit in a hash table with h bits.
// Preferably h should be a constant and should always be <32.
func hash4u(u uint32, h uint8) uint32 {
	return (u * prime4bytes) >> ((32 - h) & reg8SizeMask32)
}

type tableEntryPrev struct {
	Cur  tableEntry
	Prev tableEntry
}

// hash4x64 returns the hash of the lowest 4 bytes of u to fit in a hash table with h bits.
// Preferably h should be a constant and should always be <32.
func hash4x64(u uint64, h uint8) uint32 {
	return (uint32(u) * prime4bytes) >> ((32 - h) & reg8SizeMask32)
}

// hash7 returns the hash of the lowest 7 bytes of u to fit in a hash table with h bits.
// Preferably h should be a constant and should always be <64.
func hash7(u uint64, h uint8) uint32 {
	return uint32(((u << (64 - 56)) * prime7bytes) >> ((64 - h) & reg8SizeMask64))
}

// hash8 returns the hash of u to fit in a hash table with h bits.
// Preferably h should be a constant and should always be <64.
func hash8(u uint64, h uint8) uint32 {
	return uint32((u * prime8bytes) >> ((64 - h) & reg8SizeMask64))
}

// hash6 returns the hash of the lowest 6 bytes of u to fit in a hash table with h bits.
// Preferably h should be a constant and should always be <64.
func hash6(u uint64, h uint8) uint32 {
	return uint32(((u << (64 - 48)) * prime6bytes) >> ((64 - h) & reg8SizeMask64))
}

// matchlen will return the match length between offsets and t in src.
// The maximum length returned is maxMatchLength - 4.
// It is assumed that s > t, that t >=0 and s < len(src).
func (e *fastGen) matchlen(s, t int32, src []byte) int32 {
	if debugDecode {
		if t >= s {
			panic(fmt.Sprint("t >=s:", t, s))
		}
		if int(s) >= len(src) {
			panic(fmt.Sprint("s >= len(src):", s, len(src)))
		}
		if t < 0 {
			panic(fmt.Sprint("t < 0:", t))
		}
		if s-t > maxMatchOffset {
			panic(fmt.Sprint(s, "-", t, "(", s-t, ") > maxMatchLength (", maxMatchOffset, ")"))
		}
	}
	s1 := int(s) + maxMatchLength - 4
	if s1 > len(src) {
		s1 = len(src)
	}

	// Extend the match to be as long as possible.
	return int32(matchLen(src[s:s1], src[t:]))
}

// matchlenLong will return the match length between offsets and t in src.
// It is assumed that s > t, that t >=0 and s < len(src).
func (e *fastGen) matchlenLong(s, t int32, src []byte) int32 {
	if debugDecode {
		if t >= s {
			panic(fmt.Sprint("t >=s:", t, s))
		}
		if int(s) >= len(src) {
			panic(fmt.Sprint("s >= len(src):", s, len(src)))
		}
		if t < 0 {
			panic(fmt.Sprint("t < 0:", t))
		}
		if s-t > maxMatchOffset {
			panic(fmt.Sprint(s, "-", t, "(", s-t, ") > maxMatchLength (", maxMatchOffset, ")"))
		}
	}
	// Extend the match to be as long as possible.
	return int32(matchLen(src[s:], src[t:]))
}

// Reset the encoding table.
func (e *fastGen) Reset() {
	if cap(e.hist) < allocHistory {
		e.hist = make([]byte, 0, allocHistory)
	}
	// We offset current position so everything will be out of reach.
	// If we are above the buffer reset it will be cleared anyway since len(hist) == 0.
	if e.cur <= bufferReset {
		e.cur += maxMatchOffset + int32(len(e.hist))
	}
	e.hist = e.hist[:0]
}

// matchLen returns the maximum length.
// 'a' must be the shortest of the two.
func matchLen(a, b []byte) int {
	b = b[:len(a)]
	var checked int
	if len(a) > 4 {
		// Try 4 bytes first
		if diff := load32(a, 0) ^ load32(b, 0); diff != 0 {
			return bits.TrailingZeros32(diff) >> 3
		}
		// Switch to 8 byte matching.
		checked = 4
		a = a[4:]
		b = b[4:]
		for len(a) >= 8 {
			b = b[:len(a)]
			if diff := load64(a, 0) ^ load64(b, 0); diff != 0 {
				return checked + (bits.TrailingZeros64(diff) >> 3)
			}
			checked += 8
			a = a[8:]
			b = b[8:]
		}
	}
	b = b[:len(a)]
	for i := range a {
		if a[i] != b[i] {
			return int(i) + checked
		}
	}
	return len(a) + checked
}
