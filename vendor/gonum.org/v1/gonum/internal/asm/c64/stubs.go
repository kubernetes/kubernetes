// Copyright ©2020 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package c64

import (
	"gonum.org/v1/gonum/internal/cmplx64"
	"gonum.org/v1/gonum/internal/math32"
)

// Add is
//  for i, v := range s {
//  	dst[i] += v
//  }
func Add(dst, s []complex64) {
	for i, v := range s {
		dst[i] += v
	}
}

// AddConst is
//  for i := range x {
//  	x[i] += alpha
//  }
func AddConst(alpha complex64, x []complex64) {
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
func CumSum(dst, s []complex64) []complex64 {
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
func CumProd(dst, s []complex64) []complex64 {
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
func Div(dst, s []complex64) {
	for i, v := range s {
		dst[i] /= v
	}
}

// DivTo is
//  for i, v := range s {
//  	dst[i] = v / t[i]
//  }
//  return dst
func DivTo(dst, s, t []complex64) []complex64 {
	for i, v := range s {
		dst[i] = v / t[i]
	}
	return dst
}

// DotUnitary is
//  for i, v := range x {
//  	sum += conj(v) * y[i]
//  }
//  return sum
func DotUnitary(x, y []complex64) (sum complex64) {
	for i, v := range x {
		sum += cmplx64.Conj(v) * y[i]
	}
	return sum
}

// L2DistanceUnitary returns the L2-norm of x-y.
func L2DistanceUnitary(x, y []complex64) (norm float32) {
	var scale float32
	sumSquares := float32(1.0)
	for i, v := range x {
		v -= y[i]
		if v == 0 {
			continue
		}
		absxi := cmplx64.Abs(v)
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

// L2NormUnitary returns the L2-norm of x.
func L2NormUnitary(x []complex64) (norm float32) {
	var scale float32
	sumSquares := float32(1.0)
	for _, v := range x {
		if v == 0 {
			continue
		}
		absxi := cmplx64.Abs(v)
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

// Sum is
//  var sum complex64
//  for i := range x {
//      sum += x[i]
//  }
func Sum(x []complex64) complex64 {
	var sum complex64
	for _, v := range x {
		sum += v
	}
	return sum
}
