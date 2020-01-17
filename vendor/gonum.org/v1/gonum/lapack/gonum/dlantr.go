// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"math"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/lapack"
)

// Dlantr computes the specified norm of an m×n trapezoidal matrix A. If
// norm == lapack.MaxColumnSum work must have length at least n, otherwise work
// is unused.
func (impl Implementation) Dlantr(norm lapack.MatrixNorm, uplo blas.Uplo, diag blas.Diag, m, n int, a []float64, lda int, work []float64) float64 {
	switch {
	case norm != lapack.MaxRowSum && norm != lapack.MaxColumnSum && norm != lapack.Frobenius && norm != lapack.MaxAbs:
		panic(badNorm)
	case uplo != blas.Upper && uplo != blas.Lower:
		panic(badUplo)
	case diag != blas.Unit && diag != blas.NonUnit:
		panic(badDiag)
	case m < 0:
		panic(mLT0)
	case n < 0:
		panic(nLT0)
	case lda < max(1, n):
		panic(badLdA)
	}

	// Quick return if possible.
	minmn := min(m, n)
	if minmn == 0 {
		return 0
	}

	switch {
	case len(a) < (m-1)*lda+n:
		panic(shortA)
	case norm == lapack.MaxColumnSum && len(work) < n:
		panic(shortWork)
	}

	switch norm {
	case lapack.MaxAbs:
		if diag == blas.Unit {
			value := 1.0
			if uplo == blas.Upper {
				for i := 0; i < m; i++ {
					for j := i + 1; j < n; j++ {
						tmp := math.Abs(a[i*lda+j])
						if math.IsNaN(tmp) {
							return tmp
						}
						if tmp > value {
							value = tmp
						}
					}
				}
				return value
			}
			for i := 1; i < m; i++ {
				for j := 0; j < min(i, n); j++ {
					tmp := math.Abs(a[i*lda+j])
					if math.IsNaN(tmp) {
						return tmp
					}
					if tmp > value {
						value = tmp
					}
				}
			}
			return value
		}
		var value float64
		if uplo == blas.Upper {
			for i := 0; i < m; i++ {
				for j := i; j < n; j++ {
					tmp := math.Abs(a[i*lda+j])
					if math.IsNaN(tmp) {
						return tmp
					}
					if tmp > value {
						value = tmp
					}
				}
			}
			return value
		}
		for i := 0; i < m; i++ {
			for j := 0; j <= min(i, n-1); j++ {
				tmp := math.Abs(a[i*lda+j])
				if math.IsNaN(tmp) {
					return tmp
				}
				if tmp > value {
					value = tmp
				}
			}
		}
		return value
	case lapack.MaxColumnSum:
		if diag == blas.Unit {
			for i := 0; i < minmn; i++ {
				work[i] = 1
			}
			for i := minmn; i < n; i++ {
				work[i] = 0
			}
			if uplo == blas.Upper {
				for i := 0; i < m; i++ {
					for j := i + 1; j < n; j++ {
						work[j] += math.Abs(a[i*lda+j])
					}
				}
			} else {
				for i := 1; i < m; i++ {
					for j := 0; j < min(i, n); j++ {
						work[j] += math.Abs(a[i*lda+j])
					}
				}
			}
		} else {
			for i := 0; i < n; i++ {
				work[i] = 0
			}
			if uplo == blas.Upper {
				for i := 0; i < m; i++ {
					for j := i; j < n; j++ {
						work[j] += math.Abs(a[i*lda+j])
					}
				}
			} else {
				for i := 0; i < m; i++ {
					for j := 0; j <= min(i, n-1); j++ {
						work[j] += math.Abs(a[i*lda+j])
					}
				}
			}
		}
		var max float64
		for _, v := range work[:n] {
			if math.IsNaN(v) {
				return math.NaN()
			}
			if v > max {
				max = v
			}
		}
		return max
	case lapack.MaxRowSum:
		var maxsum float64
		if diag == blas.Unit {
			if uplo == blas.Upper {
				for i := 0; i < m; i++ {
					var sum float64
					if i < minmn {
						sum = 1
					}
					for j := i + 1; j < n; j++ {
						sum += math.Abs(a[i*lda+j])
					}
					if math.IsNaN(sum) {
						return math.NaN()
					}
					if sum > maxsum {
						maxsum = sum
					}
				}
				return maxsum
			} else {
				for i := 0; i < m; i++ {
					var sum float64
					if i < minmn {
						sum = 1
					}
					for j := 0; j < min(i, n); j++ {
						sum += math.Abs(a[i*lda+j])
					}
					if math.IsNaN(sum) {
						return math.NaN()
					}
					if sum > maxsum {
						maxsum = sum
					}
				}
				return maxsum
			}
		} else {
			if uplo == blas.Upper {
				for i := 0; i < m; i++ {
					var sum float64
					for j := i; j < n; j++ {
						sum += math.Abs(a[i*lda+j])
					}
					if math.IsNaN(sum) {
						return sum
					}
					if sum > maxsum {
						maxsum = sum
					}
				}
				return maxsum
			} else {
				for i := 0; i < m; i++ {
					var sum float64
					for j := 0; j <= min(i, n-1); j++ {
						sum += math.Abs(a[i*lda+j])
					}
					if math.IsNaN(sum) {
						return sum
					}
					if sum > maxsum {
						maxsum = sum
					}
				}
				return maxsum
			}
		}
	default:
		// lapack.Frobenius:
		var scale, ssq float64
		if diag == blas.Unit {
			scale = 1
			ssq = float64(min(m, n))
			if uplo == blas.Upper {
				for i := 0; i < min(m, n); i++ {
					rowscale, rowssq := impl.Dlassq(n-i-1, a[i*lda+i+1:], 1, 0, 1)
					scale, ssq = impl.Dcombssq(scale, ssq, rowscale, rowssq)
				}
			} else {
				for i := 1; i < m; i++ {
					rowscale, rowssq := impl.Dlassq(min(i, n), a[i*lda:], 1, 0, 1)
					scale, ssq = impl.Dcombssq(scale, ssq, rowscale, rowssq)
				}
			}
		} else {
			scale = 0
			ssq = 1
			if uplo == blas.Upper {
				for i := 0; i < min(m, n); i++ {
					rowscale, rowssq := impl.Dlassq(n-i, a[i*lda+i:], 1, 0, 1)
					scale, ssq = impl.Dcombssq(scale, ssq, rowscale, rowssq)
				}
			} else {
				for i := 0; i < m; i++ {
					rowscale, rowssq := impl.Dlassq(min(i+1, n), a[i*lda:], 1, 0, 1)
					scale, ssq = impl.Dcombssq(scale, ssq, rowscale, rowssq)
				}
			}
		}
		return scale * math.Sqrt(ssq)
	}
}
