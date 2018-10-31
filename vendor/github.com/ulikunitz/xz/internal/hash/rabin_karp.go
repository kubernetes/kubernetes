// Copyright 2014-2017 Ulrich Kunitz. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hash

// A is the default constant for Robin-Karp rolling hash. This is a random
// prime.
const A = 0x97b548add41d5da1

// RabinKarp supports the computation of a rolling hash.
type RabinKarp struct {
	A uint64
	// a^n
	aOldest uint64
	h       uint64
	p       []byte
	i       int
}

// NewRabinKarp creates a new RabinKarp value. The argument n defines the
// length of the byte sequence to be hashed. The default constant will will be
// used.
func NewRabinKarp(n int) *RabinKarp {
	return NewRabinKarpConst(n, A)
}

// NewRabinKarpConst creates a new RabinKarp value. The argument n defines the
// length of the byte sequence to be hashed. The argument a provides the
// constant used to compute the hash.
func NewRabinKarpConst(n int, a uint64) *RabinKarp {
	if n <= 0 {
		panic("number of bytes n must be positive")
	}
	aOldest := uint64(1)
	// There are faster methods. For the small n required by the LZMA
	// compressor O(n) is sufficient.
	for i := 0; i < n; i++ {
		aOldest *= a
	}
	return &RabinKarp{
		A: a, aOldest: aOldest,
		p: make([]byte, 0, n),
	}
}

// Len returns the length of the byte sequence.
func (r *RabinKarp) Len() int {
	return cap(r.p)
}

// RollByte computes the hash after x has been added.
func (r *RabinKarp) RollByte(x byte) uint64 {
	if len(r.p) < cap(r.p) {
		r.h += uint64(x)
		r.h *= r.A
		r.p = append(r.p, x)
	} else {
		r.h -= uint64(r.p[r.i]) * r.aOldest
		r.h += uint64(x)
		r.h *= r.A
		r.p[r.i] = x
		r.i = (r.i + 1) % cap(r.p)
	}
	return r.h
}
