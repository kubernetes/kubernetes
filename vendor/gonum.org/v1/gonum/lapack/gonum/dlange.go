// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"math"

	"gonum.org/v1/gonum/lapack"
)

// Dlange computes the matrix norm of the general m×n matrix a. The input norm
// specifies the norm computed.
//  lapack.MaxAbs: the maximum absolute value of an element.
//  lapack.MaxColumnSum: the maximum column sum of the absolute values of the entries.
//  lapack.MaxRowSum: the maximum row sum of the absolute values of the entries.
//  lapack.Frobenius: the square root of the sum of the squares of the entries.
// If norm == lapack.MaxColumnSum, work must be of length n, and this function will panic otherwise.
// There are no restrictions on work for the other matrix norms.
func (impl Implementation) Dlange(norm lapack.MatrixNorm, m, n int, a []float64, lda int, work []float64) float64 {
	// TODO(btracey): These should probably be refactored to use BLAS calls.
	switch {
	case norm != lapack.MaxRowSum && norm != lapack.MaxColumnSum && norm != lapack.Frobenius && norm != lapack.MaxAbs:
		panic(badNorm)
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

	if norm == lapack.MaxAbs {
		var value float64
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				value = math.Max(value, math.Abs(a[i*lda+j]))
			}
		}
		return value
	}
	if norm == lapack.MaxColumnSum {
		if len(work) < n {
			panic(shortWork)
		}
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
	}
	if norm == lapack.MaxRowSum {
		var value float64
		for i := 0; i < m; i++ {
			var sum float64
			for j := 0; j < n; j++ {
				sum += math.Abs(a[i*lda+j])
			}
			value = math.Max(value, sum)
		}
		return value
	}
	// norm == lapack.Frobenius
	var value float64
	scale := 0.0
	sum := 1.0
	for i := 0; i < m; i++ {
		scale, sum = impl.Dlassq(n, a[i*lda:], 1, scale, sum)
	}
	value = scale * math.Sqrt(sum)
	return value
}
