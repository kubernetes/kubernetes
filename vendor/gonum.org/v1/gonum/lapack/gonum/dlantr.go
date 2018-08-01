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
	checkMatrix(m, n, a, lda)
	switch norm {
	case lapack.MaxRowSum, lapack.MaxColumnSum, lapack.NormFrob, lapack.MaxAbs:
	default:
		panic(badNorm)
	}
	if uplo != blas.Upper && uplo != blas.Lower {
		panic(badUplo)
	}
	if diag != blas.Unit && diag != blas.NonUnit {
		panic(badDiag)
	}
	if norm == lapack.MaxColumnSum && len(work) < n {
		panic(badWork)
	}
	if min(m, n) == 0 {
		return 0
	}
	switch norm {
	default:
		panic("unreachable")
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
			for i := 0; i < min(m, n); i++ {
				work[i] = 1
			}
			for i := min(m, n); i < n; i++ {
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
					if i < min(m, n) {
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
				for i := 1; i < m; i++ {
					var sum float64
					if i < min(m, n) {
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
	case lapack.NormFrob:
		var nrm float64
		if diag == blas.Unit {
			if uplo == blas.Upper {
				for i := 0; i < m; i++ {
					for j := i + 1; j < n; j++ {
						tmp := a[i*lda+j]
						nrm += tmp * tmp
					}
				}
			} else {
				for i := 1; i < m; i++ {
					for j := 0; j < min(i, n); j++ {
						tmp := a[i*lda+j]
						nrm += tmp * tmp
					}
				}
			}
			nrm += float64(min(m, n))
		} else {
			if uplo == blas.Upper {
				for i := 0; i < m; i++ {
					for j := i; j < n; j++ {
						tmp := math.Abs(a[i*lda+j])
						nrm += tmp * tmp
					}
				}
			} else {
				for i := 0; i < m; i++ {
					for j := 0; j <= min(i, n-1); j++ {
						tmp := math.Abs(a[i*lda+j])
						nrm += tmp * tmp
					}
				}
			}
		}
		return math.Sqrt(nrm)
	}
}
