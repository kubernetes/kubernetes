// Copyright ©2019 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"math"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/lapack"
)

// Dlansb returns the given norm of an n×n symmetric band matrix with kd
// super-diagonals.
//
// When norm is lapack.MaxColumnSum or lapack.MaxRowSum, the length of work must
// be at least n.
func (impl Implementation) Dlansb(norm lapack.MatrixNorm, uplo blas.Uplo, n, kd int, ab []float64, ldab int, work []float64) float64 {
	switch {
	case norm != lapack.MaxAbs && norm != lapack.MaxRowSum && norm != lapack.MaxColumnSum && norm != lapack.Frobenius:
		panic(badNorm)
	case uplo != blas.Upper && uplo != blas.Lower:
		panic(badUplo)
	case n < 0:
		panic(nLT0)
	case kd < 0:
		panic(kdLT0)
	case ldab < kd+1:
		panic(badLdA)
	}

	// Quick return if possible.
	if n == 0 {
		return 0
	}

	switch {
	case len(ab) < (n-1)*ldab+kd+1:
		panic(shortAB)
	case len(work) < n && (norm == lapack.MaxColumnSum || norm == lapack.MaxRowSum):
		panic(shortWork)
	}

	var value float64
	switch norm {
	case lapack.MaxAbs:
		if uplo == blas.Upper {
			for i := 0; i < n; i++ {
				for j := 0; j < min(n-i, kd+1); j++ {
					aij := math.Abs(ab[i*ldab+j])
					if aij > value || math.IsNaN(aij) {
						value = aij
					}
				}
			}
		} else {
			for i := 0; i < n; i++ {
				for j := max(0, kd-i); j < kd+1; j++ {
					aij := math.Abs(ab[i*ldab+j])
					if aij > value || math.IsNaN(aij) {
						value = aij
					}
				}
			}
		}
	case lapack.MaxColumnSum, lapack.MaxRowSum:
		work = work[:n]
		var sum float64
		if uplo == blas.Upper {
			for i := range work {
				work[i] = 0
			}
			for i := 0; i < n; i++ {
				sum := work[i] + math.Abs(ab[i*ldab])
				for j := i + 1; j < min(i+kd+1, n); j++ {
					aij := math.Abs(ab[i*ldab+j-i])
					sum += aij
					work[j] += aij
				}
				if sum > value || math.IsNaN(sum) {
					value = sum
				}
			}
		} else {
			for i := 0; i < n; i++ {
				sum = 0
				for j := max(0, i-kd); j < i; j++ {
					aij := math.Abs(ab[i*ldab+kd+j-i])
					sum += aij
					work[j] += aij
				}
				work[i] = sum + math.Abs(ab[i*ldab+kd])
			}
			for _, sum := range work {
				if sum > value || math.IsNaN(sum) {
					value = sum
				}
			}
		}
	case lapack.Frobenius:
		scale := 0.0
		ssq := 1.0
		if uplo == blas.Upper {
			if kd > 0 {
				// Sum off-diagonals.
				for i := 0; i < n-1; i++ {
					ilen := min(n-i-1, kd)
					rowscale, rowssq := impl.Dlassq(ilen, ab[i*ldab+1:], 1, 0, 1)
					scale, ssq = impl.Dcombssq(scale, ssq, rowscale, rowssq)
				}
				ssq *= 2
			}
			// Sum diagonal.
			dscale, dssq := impl.Dlassq(n, ab, ldab, 0, 1)
			scale, ssq = impl.Dcombssq(scale, ssq, dscale, dssq)
		} else {
			if kd > 0 {
				// Sum off-diagonals.
				for i := 1; i < n; i++ {
					ilen := min(i, kd)
					rowscale, rowssq := impl.Dlassq(ilen, ab[i*ldab+kd-ilen:], 1, 0, 1)
					scale, ssq = impl.Dcombssq(scale, ssq, rowscale, rowssq)
				}
				ssq *= 2
			}
			// Sum diagonal.
			dscale, dssq := impl.Dlassq(n, ab[kd:], ldab, 0, 1)
			scale, ssq = impl.Dcombssq(scale, ssq, dscale, dssq)
		}
		value = scale * math.Sqrt(ssq)
	}

	return value
}
