// +build !go1.10

package sdkmath

import "math"

// Copied from the Go standard library's (Go 1.12) math/floor.go for use in
// Go version prior to Go 1.10.
const (
	uvone    = 0x3FF0000000000000
	mask     = 0x7FF
	shift    = 64 - 11 - 1
	bias     = 1023
	signMask = 1 << 63
	fracMask = 1<<shift - 1
)

// Round returns the nearest integer, rounding half away from zero.
//
// Special cases are:
//	Round(±0) = ±0
//	Round(±Inf) = ±Inf
//	Round(NaN) = NaN
//
// Copied from the Go standard library's (Go 1.12) math/floor.go for use in
// Go version prior to Go 1.10.
func Round(x float64) float64 {
	// Round is a faster implementation of:
	//
	// func Round(x float64) float64 {
	//   t := Trunc(x)
	//   if Abs(x-t) >= 0.5 {
	//     return t + Copysign(1, x)
	//   }
	//   return t
	// }
	bits := math.Float64bits(x)
	e := uint(bits>>shift) & mask
	if e < bias {
		// Round abs(x) < 1 including denormals.
		bits &= signMask // +-0
		if e == bias-1 {
			bits |= uvone // +-1
		}
	} else if e < bias+shift {
		// Round any abs(x) >= 1 containing a fractional component [0,1).
		//
		// Numbers with larger exponents are returned unchanged since they
		// must be either an integer, infinity, or NaN.
		const half = 1 << (shift - 1)
		e -= bias
		bits += half >> e
		bits &^= fracMask >> e
	}
	return math.Float64frombits(bits)
}
