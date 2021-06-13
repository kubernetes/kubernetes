// Copyright ©2020 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package c128

import (
	"math"
	"math/cmplx"
)

// Add is
//  for i, v := range s {
//  	dst[i] += v
//  }
func Add(dst, s []complex128) {
	for i, v := range s {
		dst[i] += v
	}
}

// AddConst is
//  for i := range x {
//  	x[i] += alpha
//  }
func AddConst(alpha complex128, x []complex128) {
	for i := range x {
		x[i] += alpha
	}
}

// CumSum is
//  if len(s) == 0 {
//  	return dst
//  }
//  dst[0] = s[0]
//  for i, v := range s[1:] {
//  	dst[i+1] = dst[i] + v
//  }
//  return dst
func CumSum(dst, s []complex128) []complex128 {
	if len(s) == 0 {
		return dst
	}
	dst[0] = s[0]
	for i, v := range s[1:] {
		dst[i+1] = dst[i] + v
	}
	return dst
}

// CumProd is
//  if len(s) == 0 {
//  	return dst
//  }
//  dst[0] = s[0]
//  for i, v := range s[1:] {
//  	dst[i+1] = dst[i] * v
//  }
//  return dst
func CumProd(dst, s []complex128) []complex128 {
	if len(s) == 0 {
		return dst
	}
	dst[0] = s[0]
	for i, v := range s[1:] {
		dst[i+1] = dst[i] * v
	}
	return dst
}

// Div is
//  for i, v := range s {
//  	dst[i] /= v
//  }
func Div(dst, s []complex128) {
	for i, v := range s {
		dst[i] /= v
	}
}

// DivTo is
//  for i, v := range s {
//  	dst[i] = v / t[i]
//  }
//  return dst
func DivTo(dst, s, t []complex128) []complex128 {
	for i, v := range s {
		dst[i] = v / t[i]
	}
	return dst
}

// DotUnitary is
//  for i, v := range x {
//  	sum += cmplx.Conj(v) * y[i]
//  }
//  return sum
func DotUnitary(x, y []complex128) (sum complex128) {
	for i, v := range x {
		sum += cmplx.Conj(v) * y[i]
	}
	return sum
}

// L2DistanceUnitary returns the L2-norm of x-y.
func L2DistanceUnitary(x, y []complex128) (norm float64) {
	var scale float64
	sumSquares := 1.0
	for i, v := range x {
		v -= y[i]
		if v == 0 {
			continue
		}
		absxi := cmplx.Abs(v)
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

// L2NormUnitary returns the L2-norm of x.
func L2NormUnitary(x []complex128) (norm float64) {
	var scale float64
	sumSquares := 1.0
	for _, v := range x {
		if v == 0 {
			continue
		}
		absxi := cmplx.Abs(v)
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

// Sum is
//  var sum complex128
//  for i := range x {
//      sum += x[i]
//  }
func Sum(x []complex128) complex128 {
	var sum complex128
	for _, v := range x {
		sum += v
	}
	return sum
}
