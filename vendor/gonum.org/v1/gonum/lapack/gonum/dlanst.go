// Copyright ©2016 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"math"

	"gonum.org/v1/gonum/lapack"
)

// Dlanst computes the specified norm of a symmetric tridiagonal matrix A.
// The diagonal elements of A are stored in d and the off-diagonal elements
// are stored in e.
func (impl Implementation) Dlanst(norm lapack.MatrixNorm, n int, d, e []float64) float64 {
	switch {
	case norm != lapack.MaxRowSum && norm != lapack.MaxColumnSum && norm != lapack.Frobenius && norm != lapack.MaxAbs:
		panic(badNorm)
	case n < 0:
		panic(nLT0)
	}
	if n == 0 {
		return 0
	}
	switch {
	case len(d) < n:
		panic(shortD)
	case len(e) < n-1:
		panic(shortE)
	}

	switch norm {
	default:
		panic(badNorm)
	case lapack.MaxAbs:
		anorm := math.Abs(d[n-1])
		for i := 0; i < n-1; i++ {
			sum := math.Abs(d[i])
			if anorm < sum || math.IsNaN(sum) {
				anorm = sum
			}
			sum = math.Abs(e[i])
			if anorm < sum || math.IsNaN(sum) {
				anorm = sum
			}
		}
		return anorm
	case lapack.MaxColumnSum, lapack.MaxRowSum:
		if n == 1 {
			return math.Abs(d[0])
		}
		anorm := math.Abs(d[0]) + math.Abs(e[0])
		sum := math.Abs(e[n-2]) + math.Abs(d[n-1])
		if anorm < sum || math.IsNaN(sum) {
			anorm = sum
		}
		for i := 1; i < n-1; i++ {
			sum := math.Abs(d[i]) + math.Abs(e[i]) + math.Abs(e[i-1])
			if anorm < sum || math.IsNaN(sum) {
				anorm = sum
			}
		}
		return anorm
	case lapack.Frobenius:
		var scale float64
		sum := 1.0
		if n > 1 {
			scale, sum = impl.Dlassq(n-1, e, 1, scale, sum)
			sum = 2 * sum
		}
		scale, sum = impl.Dlassq(n, d, 1, scale, sum)
		return scale * math.Sqrt(sum)
	}
}
