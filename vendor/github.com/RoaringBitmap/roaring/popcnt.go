// +build go1.9
// "go1.9", from Go version 1.9 onward
// See https://golang.org/pkg/go/build/#hdr-Build_Constraints

package roaring

import "math/bits"

func popcount(x uint64) uint64 {
	return uint64(bits.OnesCount64(x))
}
