// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import "math"

// Dlassq updates a sum of squares in scaled form. The input parameters scale and
// sumsq represent the current scale and total sum of squares. These values are
// updated with the information in the first n elements of the vector specified
// by x and incX.
//
// Dlassq is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dlassq(n int, x []float64, incx int, scale float64, sumsq float64) (scl, smsq float64) {
	switch {
	case n < 0:
		panic(nLT0)
	case incx <= 0:
		panic(badIncX)
	case len(x) < 1+(n-1)*incx:
		panic(shortX)
	}

	if n == 0 {
		return scale, sumsq
	}

	for ix := 0; ix <= (n-1)*incx; ix += incx {
		absxi := math.Abs(x[ix])
		if absxi > 0 || math.IsNaN(absxi) {
			if scale < absxi {
				sumsq = 1 + sumsq*(scale/absxi)*(scale/absxi)
				scale = absxi
			} else {
				sumsq += (absxi / scale) * (absxi / scale)
			}
		}
	}
	return scale, sumsq
}
