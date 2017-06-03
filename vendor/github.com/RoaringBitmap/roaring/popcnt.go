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

func popcntSliceGo(s []uint64) uint64 {
	cnt := uint64(0)
	for _, x := range s {
		cnt += popcount(x)
	}
	return cnt
}

func popcntMaskSliceGo(s, m []uint64) uint64 {
	cnt := uint64(0)
	for i := range s {
		cnt += popcount(s[i] &^ m[i])
	}
	return cnt
}

func popcntAndSliceGo(s, m []uint64) uint64 {
	cnt := uint64(0)
	for i := range s {
		cnt += popcount(s[i] & m[i])
	}
	return cnt
}

func popcntOrSliceGo(s, m []uint64) uint64 {
	cnt := uint64(0)
	for i := range s {
		cnt += popcount(s[i] | m[i])
	}
	return cnt
}

func popcntXorSliceGo(s, m []uint64) uint64 {
	cnt := uint64(0)
	for i := range s {
		cnt += popcount(s[i] ^ m[i])
	}
	return cnt
}
