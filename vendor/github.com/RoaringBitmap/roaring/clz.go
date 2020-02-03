// +build go1.9
// "go1.9", from Go version 1.9 onward
// See https://golang.org/pkg/go/build/#hdr-Build_Constraints

package roaring

import "math/bits"

func countLeadingZeros(x uint64) int {
	return bits.LeadingZeros64(x)
}
