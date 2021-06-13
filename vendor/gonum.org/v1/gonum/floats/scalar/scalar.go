// Copyright ©2013 The Gonum Authors. All rights reserved.
// Use of this code is governed by a BSD-style
// license that can be found in the LICENSE file.

package scalar

import (
	"math"
	"strconv"
)

// EqualWithinAbs returns true when a and b have an absolute difference
// not greater than tol.
func EqualWithinAbs(a, b, tol float64) bool {
	return a == b || math.Abs(a-b) <= tol
}

// minNormalFloat64 is the smallest normal number. For 64 bit IEEE-754
// floats this is 2^{-1022}.
const minNormalFloat64 = 0x1p-1022

// EqualWithinRel returns true when the difference between a and b
// is not greater than tol times the greater absolute value of a and b,
//  abs(a-b) <= tol * max(abs(a), abs(b)).
func EqualWithinRel(a, b, tol float64) bool {
	if a == b {
		return true
	}
	delta := math.Abs(a - b)
	if delta <= minNormalFloat64 {
		return delta <= tol*minNormalFloat64
	}
	// We depend on the division in this relationship to identify
	// infinities (we rely on the NaN to fail the test) otherwise
	// we compare Infs of the same sign and evaluate Infs as equal
	// independent of sign.
	return delta/math.Max(math.Abs(a), math.Abs(b)) <= tol
}

// EqualWithinAbsOrRel returns true when a and b are equal to within
// the absolute or relative tolerances. See EqualWithinAbs and
// EqualWithinRel for details.
func EqualWithinAbsOrRel(a, b, absTol, relTol float64) bool {
	return EqualWithinAbs(a, b, absTol) || EqualWithinRel(a, b, relTol)
}

// EqualWithinULP returns true when a and b are equal to within
// the specified number of floating point units in the last place.
func EqualWithinULP(a, b float64, ulp uint) bool {
	if a == b {
		return true
	}
	if math.IsNaN(a) || math.IsNaN(b) {
		return false
	}
	if math.Signbit(a) != math.Signbit(b) {
		return math.Float64bits(math.Abs(a))+math.Float64bits(math.Abs(b)) <= uint64(ulp)
	}
	return ulpDiff(math.Float64bits(a), math.Float64bits(b)) <= uint64(ulp)
}

func ulpDiff(a, b uint64) uint64 {
	if a > b {
		return a - b
	}
	return b - a
}

const (
	nanBits = 0x7ff8000000000000
	nanMask = 0xfff8000000000000
)

// NaNWith returns an IEEE 754 "quiet not-a-number" value with the
// payload specified in the low 51 bits of payload.
// The NaN returned by math.NaN has a bit pattern equal to NaNWith(1).
func NaNWith(payload uint64) float64 {
	return math.Float64frombits(nanBits | (payload &^ nanMask))
}

// NaNPayload returns the lowest 51 bits payload of an IEEE 754 "quiet
// not-a-number". For values of f other than quiet-NaN, NaNPayload
// returns zero and false.
func NaNPayload(f float64) (payload uint64, ok bool) {
	b := math.Float64bits(f)
	if b&nanBits != nanBits {
		return 0, false
	}
	return b &^ nanMask, true
}

// ParseWithNA converts the string s to a float64 in value.
// If s equals missing, weight is returned as 0, otherwise 1.
func ParseWithNA(s, missing string) (value, weight float64, err error) {
	if s == missing {
		return 0, 0, nil
	}
	value, err = strconv.ParseFloat(s, 64)
	if err == nil {
		weight = 1
	}
	return value, weight, err
}

// Round returns the half away from zero rounded value of x with prec precision.
//
// Special cases are:
// 	Round(±0) = +0
// 	Round(±Inf) = ±Inf
// 	Round(NaN) = NaN
func Round(x float64, prec int) float64 {
	if x == 0 {
		// Make sure zero is returned
		// without the negative bit set.
		return 0
	}
	// Fast path for positive precision on integers.
	if prec >= 0 && x == math.Trunc(x) {
		return x
	}
	pow := math.Pow10(prec)
	intermed := x * pow
	if math.IsInf(intermed, 0) {
		return x
	}
	if x < 0 {
		x = math.Ceil(intermed - 0.5)
	} else {
		x = math.Floor(intermed + 0.5)
	}

	if x == 0 {
		return 0
	}

	return x / pow
}

// RoundEven returns the half even rounded value of x with prec precision.
//
// Special cases are:
// 	RoundEven(±0) = +0
// 	RoundEven(±Inf) = ±Inf
// 	RoundEven(NaN) = NaN
func RoundEven(x float64, prec int) float64 {
	if x == 0 {
		// Make sure zero is returned
		// without the negative bit set.
		return 0
	}
	// Fast path for positive precision on integers.
	if prec >= 0 && x == math.Trunc(x) {
		return x
	}
	pow := math.Pow10(prec)
	intermed := x * pow
	if math.IsInf(intermed, 0) {
		return x
	}
	if isHalfway(intermed) {
		correction, _ := math.Modf(math.Mod(intermed, 2))
		intermed += correction
		if intermed > 0 {
			x = math.Floor(intermed)
		} else {
			x = math.Ceil(intermed)
		}
	} else {
		if x < 0 {
			x = math.Ceil(intermed - 0.5)
		} else {
			x = math.Floor(intermed + 0.5)
		}
	}

	if x == 0 {
		return 0
	}

	return x / pow
}

func isHalfway(x float64) bool {
	_, frac := math.Modf(x)
	frac = math.Abs(frac)
	return frac == 0.5 || (math.Nextafter(frac, math.Inf(-1)) < 0.5 && math.Nextafter(frac, math.Inf(1)) > 0.5)
}

// Same returns true when the inputs have the same value, allowing NaN equality.
func Same(a, b float64) bool {
	return a == b || (math.IsNaN(a) && math.IsNaN(b))
}
