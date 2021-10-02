// +build go1.9

package bitset

import "math/bits"

func trailingZeroes64(v uint64) uint {
	return uint(bits.TrailingZeros64(v))
}
