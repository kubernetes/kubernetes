// +build !amd64 appengine

package bitset

func popcntSlice(s []uint64) uint64 {
	return popcntSliceGo(s)
}

func popcntMaskSlice(s, m []uint64) uint64 {
	return popcntMaskSliceGo(s, m)
}

func popcntAndSlice(s, m []uint64) uint64 {
	return popcntAndSliceGo(s, m)
}

func popcntOrSlice(s, m []uint64) uint64 {
	return popcntOrSliceGo(s, m)
}

func popcntXorSlice(s, m []uint64) uint64 {
	return popcntXorSliceGo(s, m)
}
