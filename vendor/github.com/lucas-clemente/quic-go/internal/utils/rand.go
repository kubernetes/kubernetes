package utils

import (
	"crypto/rand"
	"encoding/binary"
)

// Rand is a wrapper around crypto/rand that adds some convenience functions known from math/rand.
type Rand struct {
	buf [4]byte
}

func (r *Rand) Int31() int32 {
	rand.Read(r.buf[:])
	return int32(binary.BigEndian.Uint32(r.buf[:]) & ^uint32(1<<31))
}

// copied from the standard library math/rand implementation of Int63n
func (r *Rand) Int31n(n int32) int32 {
	if n&(n-1) == 0 { // n is power of two, can mask
		return r.Int31() & (n - 1)
	}
	max := int32((1 << 31) - 1 - (1<<31)%uint32(n))
	v := r.Int31()
	for v > max {
		v = r.Int31()
	}
	return v % n
}
