// Copyright ©2020 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"math"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/lapack"
)

// Dlantb returns the value of the given norm of an n×n triangular band matrix A
// with k+1 diagonals.
//
// When norm is lapack.MaxColumnSum, the length of work must be at least n.
func (impl Implementation) Dlantb(norm lapack.MatrixNorm, uplo blas.Uplo, diag blas.Diag, n, k int, a []float64, lda int, work []float64) float64 {
	switch {
	case norm != lapack.MaxAbs && norm != lapack.MaxRowSum && norm != lapack.MaxColumnSum && norm != lapack.Frobenius:
		panic(badNorm)
	case uplo != blas.Upper && uplo != blas.Lower:
		panic(badUplo)
	case n < 0:
		panic(nLT0)
	case k < 0:
		panic(kdLT0)
	case lda < k+1:
		panic(badLdA)
	}

	// Quick return if possible.
	if n == 0 {
		return 0
	}

	switch {
	case len(a) < (n-1)*lda+k+1:
		panic(shortAB)
	case len(work) < n && norm == lapack.MaxColumnSum:
		panic(shortWork)
	}

	var value float64
	switch norm {
	case lapack.MaxAbs:
		if uplo == blas.Upper {
			var jfirst int
			if diag == blas.Unit {
				value = 1
				jfirst = 1
			}
			for i := 0; i < n; i++ {
				for _, aij := range a[i*lda+jfirst : i*lda+min(n-i, k+1)] {
					if math.IsNaN(aij) {
						return aij
					}
					aij = math.Abs(aij)
					if aij > value {
						value = aij
					}
				}
			}
		} else {
			jlast := k + 1
			if diag == blas.Unit {
				value = 1
				jlast = k
			}
			for i := 0; i < n; i++ {
				for _, aij := range a[i*lda+max(0, k-i) : i*lda+jlast] {
					if math.IsNaN(aij) {
						return math.NaN()
					}
					aij = math.Abs(aij)
					if aij > value {
						value = aij
					}
				}
			}
		}
	case lapack.MaxRowSum:
		var sum float64
		if uplo == blas.Upper {
			var jfirst int
			if diag == blas.Unit {
				jfirst = 1
			}
			for i := 0; i < n; i++ {
				sum = 0
				if diag == blas.Unit {
					sum = 1
				}
				for _, aij := range a[i*lda+jfirst : i*lda+min(n-i, k+1)] {
					sum += math.Abs(aij)
				}
				if math.IsNaN(sum) {
					return math.NaN()
				}
				if sum > value {
					value = sum
				}
			}
		} else {
			jlast := k + 1
			if diag == blas.Unit {
				jlast = k
			}
			for i := 0; i < n; i++ {
				sum = 0
				if diag == blas.Unit {
					sum = 1
				}
				for _, aij := range a[i*lda+max(0, k-i) : i*lda+jlast] {
					sum += math.Abs(aij)
				}
				if math.IsNaN(sum) {
					return math.NaN()
				}
				if sum > value {
					value = sum
				}
			}
		}
	case lapack.MaxColumnSum:
		work = work[:n]
		if diag == blas.Unit {
			for i := range work {
				work[i] = 1
			}
		} else {
			for i := range work {
				work[i] = 0
			}
		}
		if uplo == blas.Upper {
			var jfirst int
			if diag == blas.Unit {
				jfirst = 1
			}
			for i := 0; i < n; i++ {
				for j, aij := range a[i*lda+jfirst : i*lda+min(n-i, k+1)] {
					work[i+jfirst+j] += math.Abs(aij)
				}
			}
		} else {
			jlast := k + 1
			if diag == blas.Unit {
				jlast = k
			}
			for i := 0; i < n; i++ {
				off := max(0, k-i)
				for j, aij := range a[i*lda+off : i*lda+jlast] {
					work[i+j+off-k] += math.Abs(aij)
				}
			}
		}
		for _, wi := range work {
			if math.IsNaN(wi) {
				return math.NaN()
			}
			if wi > value {
				value = wi
			}
		}
	case lapack.Frobenius:
		var scale, ssq float64
		switch uplo {
		case blas.Upper:
			if diag == blas.Unit {
				scale = 1
				ssq = float64(n)
				if k > 0 {
					for i := 0; i < n-1; i++ {
						ilen := min(n-i-1, k)
						rowscale, rowssq := impl.Dlassq(ilen, a[i*lda+1:], 1, 0, 1)
						scale, ssq = impl.Dcombssq(scale, ssq, rowscale, rowssq)
					}
				}
			} else {
				scale = 0
				ssq = 1
				for i := 0; i < n; i++ {
					ilen := min(n-i, k+1)
					rowscale, rowssq := impl.Dlassq(ilen, a[i*lda:], 1, 0, 1)
					scale, ssq = impl.Dcombssq(scale, ssq, rowscale, rowssq)
				}
			}
		case blas.Lower:
			if diag == blas.Unit {
				scale = 1
				ssq = float64(n)
				if k > 0 {
					for i := 1; i < n; i++ {
						ilen := min(i, k)
						rowscale, rowssq := impl.Dlassq(ilen, a[i*lda+k-ilen:], 1, 0, 1)
						scale, ssq = impl.Dcombssq(scale, ssq, rowscale, rowssq)
					}
				}
			} else {
				scale = 0
				ssq = 1
				for i := 0; i < n; i++ {
					ilen := min(i, k) + 1
					rowscale, rowssq := impl.Dlassq(ilen, a[i*lda+k+1-ilen:], 1, 0, 1)
					scale, ssq = impl.Dcombssq(scale, ssq, rowscale, rowssq)
				}
			}
		}
		value = scale * math.Sqrt(ssq)
	}
	return value
}
