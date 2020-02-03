// +build !go1.9

package roaring

// LeadingZeroBits returns the number of consecutive most significant zero
// bits of x.
func countLeadingZeros(i uint64) int {
	if i == 0 {
		return 64
	}
	n := 1
	x := uint32(i >> 32)
	if x == 0 {
		n += 32
		x = uint32(i)
	}
	if (x >> 16) == 0 {
		n += 16
		x <<= 16
	}
	if (x >> 24) == 0 {
		n += 8
		x <<= 8
	}
	if x>>28 == 0 {
		n += 4
		x <<= 4
	}
	if x>>30 == 0 {
		n += 2
		x <<= 2

	}
	n -= int(x >> 31)
	return n
}
