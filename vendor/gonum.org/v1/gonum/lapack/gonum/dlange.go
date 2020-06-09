// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"math"

	"gonum.org/v1/gonum/lapack"
)

// Dlange returns the value of the specified norm of a general m×n matrix A:
//  lapack.MaxAbs:       the maximum absolute value of any element.
//  lapack.MaxColumnSum: the maximum column sum of the absolute values of the elements (1-norm).
//  lapack.MaxRowSum:    the maximum row sum of the absolute values of the elements (infinity-norm).
//  lapack.Frobenius:    the square root of the sum of the squares of the elements (Frobenius norm).
// If norm == lapack.MaxColumnSum, work must be of length n, and this function will
// panic otherwise. There are no restrictions on work for the other matrix norms.
func (impl Implementation) Dlange(norm lapack.MatrixNorm, m, n int, a []float64, lda int, work []float64) float64 {
	// TODO(btracey): These should probably be refactored to use BLAS calls.
	switch {
	case norm != lapack.MaxRowSum && norm != lapack.MaxColumnSum && norm != lapack.Frobenius && norm != lapack.MaxAbs:
		panic(badNorm)
	case m < 0:
		panic(mLT0)
	case n < 0:
		panic(nLT0)
	case lda < max(1, n):
		panic(badLdA)
	}

	// Quick return if possible.
	if m == 0 || n == 0 {
		return 0
	}

	switch {
	case len(a) < (m-1)*lda+n:
		panic(badLdA)
	case norm == lapack.MaxColumnSum && len(work) < n:
		panic(shortWork)
	}

	switch norm {
	case lapack.MaxAbs:
		var value float64
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				value = math.Max(value, math.Abs(a[i*lda+j]))
			}
		}
		return value
	case lapack.MaxColumnSum:
		for i := 0; i < n; i++ {
			work[i] = 0
		}
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				work[j] += math.Abs(a[i*lda+j])
			}
		}
		var value float64
		for i := 0; i < n; i++ {
			value = math.Max(value, work[i])
		}
		return value
	case lapack.MaxRowSum:
		var value float64
		for i := 0; i < m; i++ {
			var sum float64
			for j := 0; j < n; j++ {
				sum += math.Abs(a[i*lda+j])
			}
			value = math.Max(value, sum)
		}
		return value
	default:
		// lapack.Frobenius
		scale := 0.0
		sum := 1.0
		for i := 0; i < m; i++ {
			rowscale, rowsum := impl.Dlassq(n, a[i*lda:], 1, 0, 1)
			scale, sum = impl.Dcombssq(scale, sum, rowscale, rowsum)
		}
		return scale * math.Sqrt(sum)
	}
}
