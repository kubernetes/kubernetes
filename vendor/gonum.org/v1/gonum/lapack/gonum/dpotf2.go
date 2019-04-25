// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"math"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
)

// Dpotf2 computes the Cholesky decomposition of the symmetric positive definite
// matrix a. If ul == blas.Upper, then a is stored as an upper-triangular matrix,
// and a = U^T U is stored in place into a. If ul == blas.Lower, then a = L L^T
// is computed and stored in-place into a. If a is not positive definite, false
// is returned. This is the unblocked version of the algorithm.
//
// Dpotf2 is an internal routine. It is exported for testing purposes.
func (Implementation) Dpotf2(ul blas.Uplo, n int, a []float64, lda int) (ok bool) {
	switch {
	case ul != blas.Upper && ul != blas.Lower:
		panic(badUplo)
	case n < 0:
		panic(nLT0)
	case lda < max(1, n):
		panic(badLdA)
	}

	// Quick return if possible.
	if n == 0 {
		return true
	}

	if len(a) < (n-1)*lda+n {
		panic(shortA)
	}

	bi := blas64.Implementation()

	if ul == blas.Upper {
		for j := 0; j < n; j++ {
			ajj := a[j*lda+j]
			if j != 0 {
				ajj -= bi.Ddot(j, a[j:], lda, a[j:], lda)
			}
			if ajj <= 0 || math.IsNaN(ajj) {
				a[j*lda+j] = ajj
				return false
			}
			ajj = math.Sqrt(ajj)
			a[j*lda+j] = ajj
			if j < n-1 {
				bi.Dgemv(blas.Trans, j, n-j-1,
					-1, a[j+1:], lda, a[j:], lda,
					1, a[j*lda+j+1:], 1)
				bi.Dscal(n-j-1, 1/ajj, a[j*lda+j+1:], 1)
			}
		}
		return true
	}
	for j := 0; j < n; j++ {
		ajj := a[j*lda+j]
		if j != 0 {
			ajj -= bi.Ddot(j, a[j*lda:], 1, a[j*lda:], 1)
		}
		if ajj <= 0 || math.IsNaN(ajj) {
			a[j*lda+j] = ajj
			return false
		}
		ajj = math.Sqrt(ajj)
		a[j*lda+j] = ajj
		if j < n-1 {
			bi.Dgemv(blas.NoTrans, n-j-1, j,
				-1, a[(j+1)*lda:], lda, a[j*lda:], 1,
				1, a[(j+1)*lda+j:], lda)
			bi.Dscal(n-j-1, 1/ajj, a[(j+1)*lda+j:], lda)
		}
	}
	return true
}
