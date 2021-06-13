// Copyright ©2019 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !amd64 noasm gccgo safe

package f64

import "math"

// L2NormUnitary returns the L2-norm of x.
func L2NormUnitary(x []float64) (norm float64) {
	var scale float64
	sumSquares := 1.0
	for _, v := range x {
		if v == 0 {
			continue
		}
		absxi := math.Abs(v)
		if math.IsNaN(absxi) {
			return math.NaN()
		}
		if scale < absxi {
			s := scale / absxi
			sumSquares = 1 + sumSquares*s*s
			scale = absxi
		} else {
			s := absxi / scale
			sumSquares += s * s
		}
	}
	if math.IsInf(scale, 1) {
		return math.Inf(1)
	}
	return scale * math.Sqrt(sumSquares)
}

// L2NormInc returns the L2-norm of x.
func L2NormInc(x []float64, n, incX uintptr) (norm float64) {
	var scale float64
	sumSquares := 1.0
	for ix := uintptr(0); ix < n*incX; ix += incX {
		val := x[ix]
		if val == 0 {
			continue
		}
		absxi := math.Abs(val)
		if math.IsNaN(absxi) {
			return math.NaN()
		}
		if scale < absxi {
			s := scale / absxi
			sumSquares = 1 + sumSquares*s*s
			scale = absxi
		} else {
			s := absxi / scale
			sumSquares += s * s
		}
	}
	if math.IsInf(scale, 1) {
		return math.Inf(1)
	}
	return scale * math.Sqrt(sumSquares)
}

// L2DistanceUnitary returns the L2-norm of x-y.
func L2DistanceUnitary(x, y []float64) (norm float64) {
	var scale float64
	sumSquares := 1.0
	for i, v := range x {
		v -= y[i]
		if v == 0 {
			continue
		}
		absxi := math.Abs(v)
		if math.IsNaN(absxi) {
			return math.NaN()
		}
		if scale < absxi {
			s := scale / absxi
			sumSquares = 1 + sumSquares*s*s
			scale = absxi
		} else {
			s := absxi / scale
			sumSquares += s * s
		}
	}
	if math.IsInf(scale, 1) {
		return math.Inf(1)
	}
	return scale * math.Sqrt(sumSquares)
}
