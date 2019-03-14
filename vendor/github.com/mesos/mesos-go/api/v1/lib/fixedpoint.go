package mesos

// fixed point scalar math from mesos:src/common/values.cpp
// --
// We manipulate scalar values by converting them from floating point to a
// fixed point representation, doing a calculation, and then converting
// the result back to floating point. We deliberately only preserve three
// decimal digits of precision in the fixed point representation. This
// ensures that client applications see predictable numerical behavior, at
// the expense of sacrificing some precision.

import "math"

func convertToFloat64(f int64) float64 {
	// NOTE: We do the conversion from fixed point via integer division
	// and then modulus, rather than a single floating point division.
	// This ensures that we only apply floating point division to inputs
	// in the range [0,999], which is easier to check for correctness.
	var (
		quotient  = float64(f / 1000)
		remainder = float64(f%1000) / 1000.0
	)
	return quotient + remainder
}

func convertToFixed64(f float64) int64 {
	return round64(f * 1000)
}

func round64(f float64) int64 {
	if math.Abs(f) < 0.5 {
		return 0
	}
	return int64(f + math.Copysign(0.5, f))
}
