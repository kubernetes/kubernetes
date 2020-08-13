// Copyright ©2018 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
)

// Dlauum computes the product
//  U * Uᵀ  if uplo is blas.Upper
//  Lᵀ * L  if uplo is blas.Lower
// where U or L is stored in the upper or lower triangular part of A.
// Only the upper or lower triangle of the result is stored, overwriting
// the corresponding factor in A.
func (impl Implementation) Dlauum(uplo blas.Uplo, n int, a []float64, lda int) {
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

	// Determine the block size.
	opts := "U"
	if uplo == blas.Lower {
		opts = "L"
	}
	nb := impl.Ilaenv(1, "DLAUUM", opts, n, -1, -1, -1)

	if nb <= 1 || n <= nb {
		// Use unblocked code.
		impl.Dlauu2(uplo, n, a, lda)
		return
	}

	// Use blocked code.
	bi := blas64.Implementation()
	if uplo == blas.Upper {
		// Compute the product U*Uᵀ.
		for i := 0; i < n; i += nb {
			ib := min(nb, n-i)
			bi.Dtrmm(blas.Right, blas.Upper, blas.Trans, blas.NonUnit,
				i, ib, 1, a[i*lda+i:], lda, a[i:], lda)
			impl.Dlauu2(blas.Upper, ib, a[i*lda+i:], lda)
			if n-i-ib > 0 {
				bi.Dgemm(blas.NoTrans, blas.Trans, i, ib, n-i-ib,
					1, a[i+ib:], lda, a[i*lda+i+ib:], lda, 1, a[i:], lda)
				bi.Dsyrk(blas.Upper, blas.NoTrans, ib, n-i-ib,
					1, a[i*lda+i+ib:], lda, 1, a[i*lda+i:], lda)
			}
		}
	} else {
		// Compute the product Lᵀ*L.
		for i := 0; i < n; i += nb {
			ib := min(nb, n-i)
			bi.Dtrmm(blas.Left, blas.Lower, blas.Trans, blas.NonUnit,
				ib, i, 1, a[i*lda+i:], lda, a[i*lda:], lda)
			impl.Dlauu2(blas.Lower, ib, a[i*lda+i:], lda)
			if n-i-ib > 0 {
				bi.Dgemm(blas.Trans, blas.NoTrans, ib, i, n-i-ib,
					1, a[(i+ib)*lda+i:], lda, a[(i+ib)*lda:], lda, 1, a[i*lda:], lda)
				bi.Dsyrk(blas.Lower, blas.Trans, ib, n-i-ib,
					1, a[(i+ib)*lda+i:], lda, 1, a[i*lda+i:], lda)
			}
		}
	}
}
