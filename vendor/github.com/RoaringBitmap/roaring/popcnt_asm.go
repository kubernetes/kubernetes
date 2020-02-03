// +build amd64,!appengine,!go1.9

package roaring

// *** the following functions are defined in popcnt_amd64.s

//go:noescape

func hasAsm() bool

// useAsm is a flag used to select the GO or ASM implementation of the popcnt function
var useAsm = hasAsm()

//go:noescape

func popcntSliceAsm(s []uint64) uint64

//go:noescape

func popcntMaskSliceAsm(s, m []uint64) uint64

//go:noescape

func popcntAndSliceAsm(s, m []uint64) uint64

//go:noescape

func popcntOrSliceAsm(s, m []uint64) uint64

//go:noescape

func popcntXorSliceAsm(s, m []uint64) uint64

func popcntSlice(s []uint64) uint64 {
	if useAsm {
		return popcntSliceAsm(s)
	}
	return popcntSliceGo(s)
}

func popcntMaskSlice(s, m []uint64) uint64 {
	if useAsm {
		return popcntMaskSliceAsm(s, m)
	}
	return popcntMaskSliceGo(s, m)
}

func popcntAndSlice(s, m []uint64) uint64 {
	if useAsm {
		return popcntAndSliceAsm(s, m)
	}
	return popcntAndSliceGo(s, m)
}

func popcntOrSlice(s, m []uint64) uint64 {
	if useAsm {
		return popcntOrSliceAsm(s, m)
	}
	return popcntOrSliceGo(s, m)
}

func popcntXorSlice(s, m []uint64) uint64 {
	if useAsm {
		return popcntXorSliceAsm(s, m)
	}
	return popcntXorSliceGo(s, m)
}
