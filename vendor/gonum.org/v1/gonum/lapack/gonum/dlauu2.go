// Copyright ©2018 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
)

// Dlauu2 computes the product
//  U * U^T  if uplo is blas.Upper
//  L^T * L  if uplo is blas.Lower
// where U or L is stored in the upper or lower triangular part of A.
// Only the upper or lower triangle of the result is stored, overwriting
// the corresponding factor in A.
func (impl Implementation) Dlauu2(uplo blas.Uplo, n int, a []float64, lda int) {
	switch {
	case uplo != blas.Upper && uplo != blas.Lower:
		panic(badUplo)
	case n < 0:
		panic(nLT0)
	case lda < max(1, n):
		panic(badLdA)
	}

	// Quick return if possible.
	if n == 0 {
		return
	}

	if len(a) < (n-1)*lda+n {
		panic(shortA)
	}

	bi := blas64.Implementation()

	if uplo == blas.Upper {
		// Compute the product U*U^T.
		for i := 0; i < n; i++ {
			aii := a[i*lda+i]
			if i < n-1 {
				a[i*lda+i] = bi.Ddot(n-i, a[i*lda+i:], 1, a[i*lda+i:], 1)
				bi.Dgemv(blas.NoTrans, i, n-i-1, 1, a[i+1:], lda, a[i*lda+i+1:], 1,
					aii, a[i:], lda)
			} else {
				bi.Dscal(i+1, aii, a[i:], lda)
			}
		}
	} else {
		// Compute the product L^T*L.
		for i := 0; i < n; i++ {
			aii := a[i*lda+i]
			if i < n-1 {
				a[i*lda+i] = bi.Ddot(n-i, a[i*lda+i:], lda, a[i*lda+i:], lda)
				bi.Dgemv(blas.Trans, n-i-1, i, 1, a[(i+1)*lda:], lda, a[(i+1)*lda+i:], lda,
					aii, a[i*lda:], 1)
			} else {
				bi.Dscal(i+1, aii, a[i*lda:], 1)
			}
		}
	}
}
