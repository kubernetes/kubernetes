// Package xxhash implements the 64-bit variant of xxHash (XXH64) as described
// at http://cyan4973.github.io/xxHash/.
package xxhash

import (
	"encoding/binary"
	"hash"
)

const (
	prime1 uint64 = 11400714785074694791
	prime2 uint64 = 14029467366897019727
	prime3 uint64 = 1609587929392839161
	prime4 uint64 = 9650029242287828579
	prime5 uint64 = 2870177450012600261
)

// NOTE(caleb): I'm using both consts and vars of the primes. Using consts where
// possible in the Go code is worth a small (but measurable) performance boost
// by avoiding some MOVQs. Vars are needed for the asm and also are useful for
// convenience in the Go code in a few places where we need to intentionally
// avoid constant arithmetic (e.g., v1 := prime1 + prime2 fails because the
// result overflows a uint64).
var (
	prime1v = prime1
	prime2v = prime2
	prime3v = prime3
	prime4v = prime4
	prime5v = prime5
)

type xxh struct {
	v1    uint64
	v2    uint64
	v3    uint64
	v4    uint64
	total int
	mem   [32]byte
	n     int // how much of mem is used
}

// New creates a new hash.Hash64 that implements the 64-bit xxHash algorithm.
func New() hash.Hash64 {
	var x xxh
	x.Reset()
	return &x
}

func (x *xxh) Reset() {
	x.n = 0
	x.total = 0
	x.v1 = prime1v + prime2
	x.v2 = prime2
	x.v3 = 0
	x.v4 = -prime1v
}

func (x *xxh) Size() int      { return 8 }
func (x *xxh) BlockSize() int { return 32 }

// Write adds more data to x. It always returns len(b), nil.
func (x *xxh) Write(b []byte) (n int, err error) {
	n = len(b)
	x.total += len(b)

	if x.n+len(b) < 32 {
		// This new data doesn't even fill the current block.
		copy(x.mem[x.n:], b)
		x.n += len(b)
		return
	}

	if x.n > 0 {
		// Finish off the partial block.
		copy(x.mem[x.n:], b)
		x.v1 = round(x.v1, u64(x.mem[0:8]))
		x.v2 = round(x.v2, u64(x.mem[8:16]))
		x.v3 = round(x.v3, u64(x.mem[16:24]))
		x.v4 = round(x.v4, u64(x.mem[24:32]))
		b = b[32-x.n:]
		x.n = 0
	}

	if len(b) >= 32 {
		// One or more full blocks left.
		b = writeBlocks(x, b)
	}

	// Store any remaining partial block.
	copy(x.mem[:], b)
	x.n = len(b)

	return
}

func (x *xxh) Sum(b []byte) []byte {
	s := x.Sum64()
	return append(
		b,
		byte(s>>56),
		byte(s>>48),
		byte(s>>40),
		byte(s>>32),
		byte(s>>24),
		byte(s>>16),
		byte(s>>8),
		byte(s),
	)
}

func (x *xxh) Sum64() uint64 {
	var h uint64

	if x.total >= 32 {
		v1, v2, v3, v4 := x.v1, x.v2, x.v3, x.v4
		h = rol1(v1) + rol7(v2) + rol12(v3) + rol18(v4)
		h = mergeRound(h, v1)
		h = mergeRound(h, v2)
		h = mergeRound(h, v3)
		h = mergeRound(h, v4)
	} else {
		h = x.v3 + prime5
	}

	h += uint64(x.total)

	i, end := 0, x.n
	for ; i+8 <= end; i += 8 {
		k1 := round(0, u64(x.mem[i:i+8]))
		h ^= k1
		h = rol27(h)*prime1 + prime4
	}
	if i+4 <= end {
		h ^= uint64(u32(x.mem[i:i+4])) * prime1
		h = rol23(h)*prime2 + prime3
		i += 4
	}
	for i < end {
		h ^= uint64(x.mem[i]) * prime5
		h = rol11(h) * prime1
		i++
	}

	h ^= h >> 33
	h *= prime2
	h ^= h >> 29
	h *= prime3
	h ^= h >> 32

	return h
}

func u64(b []byte) uint64 { return binary.LittleEndian.Uint64(b) }
func u32(b []byte) uint32 { return binary.LittleEndian.Uint32(b) }

func round(acc, input uint64) uint64 {
	acc += input * prime2
	acc = rol31(acc)
	acc *= prime1
	return acc
}

func mergeRound(acc, val uint64) uint64 {
	val = round(0, val)
	acc ^= val
	acc = acc*prime1 + prime4
	return acc
}
