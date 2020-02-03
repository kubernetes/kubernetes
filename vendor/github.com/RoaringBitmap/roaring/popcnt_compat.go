// +build !go1.9

package roaring

// bit population count, take from
// https://code.google.com/p/go/issues/detail?id=4988#c11
// credit: https://code.google.com/u/arnehormann/
// credit: https://play.golang.org/p/U7SogJ7psJ
// credit: http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
func popcount(x uint64) uint64 {
	x -= (x >> 1) & 0x5555555555555555
	x = (x>>2)&0x3333333333333333 + x&0x3333333333333333
	x += x >> 4
	x &= 0x0f0f0f0f0f0f0f0f
	x *= 0x0101010101010101
	return x >> 56
}
