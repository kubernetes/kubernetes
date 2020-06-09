// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"math"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/lapack"
)

// Dlansy returns the value of the specified norm of an n×n symmetric matrix. If
// norm == lapack.MaxColumnSum or norm == lapack.MaxRowSum, work must have length
// at least n, otherwise work is unused.
func (impl Implementation) Dlansy(norm lapack.MatrixNorm, uplo blas.Uplo, n int, a []float64, lda int, work []float64) float64 {
	switch {
	case norm != lapack.MaxRowSum && norm != lapack.MaxColumnSum && norm != lapack.Frobenius && norm != lapack.MaxAbs:
		panic(badNorm)
	case uplo != blas.Upper && uplo != blas.Lower:
		panic(badUplo)
	case n < 0:
		panic(nLT0)
	case lda < max(1, n):
		panic(badLdA)
	}

	// Quick return if possible.
	if n == 0 {
		return 0
	}

	switch {
	case len(a) < (n-1)*lda+n:
		panic(shortA)
	case (norm == lapack.MaxColumnSum || norm == lapack.MaxRowSum) && len(work) < n:
		panic(shortWork)
	}

	switch norm {
	case lapack.MaxAbs:
		if uplo == blas.Upper {
			var max float64
			for i := 0; i < n; i++ {
				for j := i; j < n; j++ {
					v := math.Abs(a[i*lda+j])
					if math.IsNaN(v) {
						return math.NaN()
					}
					if v > max {
						max = v
					}
				}
			}
			return max
		}
		var max float64
		for i := 0; i < n; i++ {
			for j := 0; j <= i; j++ {
				v := math.Abs(a[i*lda+j])
				if math.IsNaN(v) {
					return math.NaN()
				}
				if v > max {
					max = v
				}
			}
		}
		return max
	case lapack.MaxRowSum, lapack.MaxColumnSum:
		// A symmetric matrix has the same 1-norm and ∞-norm.
		for i := 0; i < n; i++ {
			work[i] = 0
		}
		if uplo == blas.Upper {
			for i := 0; i < n; i++ {
				work[i] += math.Abs(a[i*lda+i])
				for j := i + 1; j < n; j++ {
					v := math.Abs(a[i*lda+j])
					work[i] += v
					work[j] += v
				}
			}
		} else {
			for i := 0; i < n; i++ {
				for j := 0; j < i; j++ {
					v := math.Abs(a[i*lda+j])
					work[i] += v
					work[j] += v
				}
				work[i] += math.Abs(a[i*lda+i])
			}
		}
		var max float64
		for i := 0; i < n; i++ {
			v := work[i]
			if math.IsNaN(v) {
				return math.NaN()
			}
			if v > max {
				max = v
			}
		}
		return max
	default:
		// lapack.Frobenius:
		scale := 0.0
		ssq := 1.0
		// Sum off-diagonals.
		if uplo == blas.Upper {
			for i := 0; i < n-1; i++ {
				rowscale, rowssq := impl.Dlassq(n-i-1, a[i*lda+i+1:], 1, 0, 1)
				scale, ssq = impl.Dcombssq(scale, ssq, rowscale, rowssq)
			}
		} else {
			for i := 1; i < n; i++ {
				rowscale, rowssq := impl.Dlassq(i, a[i*lda:], 1, 0, 1)
				scale, ssq = impl.Dcombssq(scale, ssq, rowscale, rowssq)
			}
		}
		ssq *= 2
		// Sum diagonal.
		dscale, dssq := impl.Dlassq(n, a, lda+1, 0, 1)
		scale, ssq = impl.Dcombssq(scale, ssq, dscale, dssq)
		return scale * math.Sqrt(ssq)
	}
}
