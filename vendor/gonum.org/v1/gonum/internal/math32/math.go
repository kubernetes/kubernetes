// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package math32

import (
	"math"
)

const (
	unan    = 0x7fc00000
	uinf    = 0x7f800000
	uneginf = 0xff800000
	mask    = 0x7f8 >> 3
	shift   = 32 - 8 - 1
	bias    = 127
)

// Abs returns the absolute value of x.
//
// Special cases are:
//	Abs(±Inf) = +Inf
//	Abs(NaN) = NaN
func Abs(x float32) float32 {
	switch {
	case x < 0:
		return -x
	case x == 0:
		return 0 // return correctly abs(-0)
	}
	return x
}

// Copysign returns a value with the magnitude
// of x and the sign of y.
func Copysign(x, y float32) float32 {
	const sign = 1 << 31
	return math.Float32frombits(math.Float32bits(x)&^sign | math.Float32bits(y)&sign)
}

// Hypot returns Sqrt(p*p + q*q), taking care to avoid
// unnecessary overflow and underflow.
//
// Special cases are:
//	Hypot(±Inf, q) = +Inf
//	Hypot(p, ±Inf) = +Inf
//	Hypot(NaN, q) = NaN
//	Hypot(p, NaN) = NaN
func Hypot(p, q float32) float32 {
	// special cases
	switch {
	case IsInf(p, 0) || IsInf(q, 0):
		return Inf(1)
	case IsNaN(p) || IsNaN(q):
		return NaN()
	}
	if p < 0 {
		p = -p
	}
	if q < 0 {
		q = -q
	}
	if p < q {
		p, q = q, p
	}
	if p == 0 {
		return 0
	}
	q = q / p
	return p * Sqrt(1+q*q)
}

// Inf returns positive infinity if sign >= 0, negative infinity if sign < 0.
func Inf(sign int) float32 {
	var v uint32
	if sign >= 0 {
		v = uinf
	} else {
		v = uneginf
	}
	return math.Float32frombits(v)
}

// IsInf reports whether f is an infinity, according to sign.
// If sign > 0, IsInf reports whether f is positive infinity.
// If sign < 0, IsInf reports whether f is negative infinity.
// If sign == 0, IsInf reports whether f is either infinity.
func IsInf(f float32, sign int) bool {
	// Test for infinity by comparing against maximum float.
	// To avoid the floating-point hardware, could use:
	//	x := math.Float32bits(f);
	//	return sign >= 0 && x == uinf || sign <= 0 && x == uneginf;
	return sign >= 0 && f > math.MaxFloat32 || sign <= 0 && f < -math.MaxFloat32
}

// IsNaN reports whether f is an IEEE 754 ``not-a-number'' value.
func IsNaN(f float32) (is bool) {
	// IEEE 754 says that only NaNs satisfy f != f.
	// To avoid the floating-point hardware, could use:
	//	x := math.Float32bits(f);
	//	return uint32(x>>shift)&mask == mask && x != uinf && x != uneginf
	return f != f
}

// NaN returns an IEEE 754 ``not-a-number'' value.
func NaN() float32 { return math.Float32frombits(unan) }
