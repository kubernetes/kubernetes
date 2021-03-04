// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !amd64 noasm appengine safe

package math32

import (
	"math"
)

// Sqrt returns the square root of x.
//
// Special cases are:
//	Sqrt(+Inf) = +Inf
//	Sqrt(±0) = ±0
//	Sqrt(x < 0) = NaN
//	Sqrt(NaN) = NaN
func Sqrt(x float32) float32 {
	// FIXME(kortschak): Direct translation of the math package
	// asm code for 386 fails to build. No test hardware is available
	// for arm, so using conversion instead.
	return float32(math.Sqrt(float64(x)))
}
