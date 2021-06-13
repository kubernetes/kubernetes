// Copyright ©2019 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package f32

import "gonum.org/v1/gonum/internal/math32"

// L2NormUnitary is the level 2 norm of x.
func L2NormUnitary(x []float32) (sum float32) {
	var scale float32
	var sumSquares float32 = 1
	for _, v := range x {
		if v == 0 {
			continue
		}
		absxi := math32.Abs(v)
		if math32.IsNaN(absxi) {
			return math32.NaN()
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
	if math32.IsInf(scale, 1) {
		return math32.Inf(1)
	}
	return scale * math32.Sqrt(sumSquares)
}

// L2NormInc is the level 2 norm of x.
func L2NormInc(x []float32, n, incX uintptr) (sum float32) {
	var scale float32
	var sumSquares float32 = 1
	for ix := uintptr(0); ix < n*incX; ix += incX {
		val := x[ix]
		if val == 0 {
			continue
		}
		absxi := math32.Abs(val)
		if math32.IsNaN(absxi) {
			return math32.NaN()
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
	if math32.IsInf(scale, 1) {
		return math32.Inf(1)
	}
	return scale * math32.Sqrt(sumSquares)
}

// L2DistanceUnitary is the L2 norm of x-y.
func L2DistanceUnitary(x, y []float32) (sum float32) {
	var scale float32
	var sumSquares float32 = 1
	for i, v := range x {
		v -= y[i]
		if v == 0 {
			continue
		}
		absxi := math32.Abs(v)
		if math32.IsNaN(absxi) {
			return math32.NaN()
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
	if math32.IsInf(scale, 1) {
		return math32.Inf(1)
	}
	return scale * math32.Sqrt(sumSquares)
}
