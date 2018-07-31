// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
)

// Dgetri computes the inverse of the matrix A using the LU factorization computed
// by Dgetrf. On entry, a contains the PLU decomposition of A as computed by
// Dgetrf and on exit contains the reciprocal of the original matrix.
//
// Dgetri will not perform the inversion if the matrix is singular, and returns
// a boolean indicating whether the inversion was successful.
//
// work is temporary storage, and lwork specifies the usable memory length.
// At minimum, lwork >= n and this function will panic otherwise.
// Dgetri is a blocked inversion, but the block size is limited
// by the temporary space available. If lwork == -1, instead of performing Dgetri,
// the optimal work length will be stored into work[0].
func (impl Implementation) Dgetri(n int, a []float64, lda int, ipiv []int, work []float64, lwork int) (ok bool) {
	checkMatrix(n, n, a, lda)
	if len(ipiv) < n {
		panic(badIpiv)
	}
	nb := impl.Ilaenv(1, "DGETRI", " ", n, -1, -1, -1)
	if lwork == -1 {
		work[0] = float64(n * nb)
		return true
	}
	if lwork < n {
		panic(badWork)
	}
	if len(work) < lwork {
		panic(badWork)
	}
	if n == 0 {
		return true
	}
	ok = impl.Dtrtri(blas.Upper, blas.NonUnit, n, a, lda)
	if !ok {
		return false
	}
	nbmin := 2
	ldwork := nb
	if nb > 1 && nb < n {
		iws := max(ldwork*n, 1)
		if lwork < iws {
			nb = lwork / ldwork
			nbmin = max(2, impl.Ilaenv(2, "DGETRI", " ", n, -1, -1, -1))
		}
	}
	bi := blas64.Implementation()
	// TODO(btracey): Replace this with a more row-major oriented algorithm.
	if nb < nbmin || nb >= n {
		// Unblocked code.
		for j := n - 1; j >= 0; j-- {
			for i := j + 1; i < n; i++ {
				work[i*ldwork] = a[i*lda+j]
				a[i*lda+j] = 0
			}
			if j < n {
				bi.Dgemv(blas.NoTrans, n, n-j-1, -1, a[(j+1):], lda, work[(j+1)*ldwork:], ldwork, 1, a[j:], lda)
			}
		}
	} else {
		nn := ((n - 1) / nb) * nb
		for j := nn; j >= 0; j -= nb {
			jb := min(nb, n-j)
			for jj := j; jj < j+jb-1; jj++ {
				for i := jj + 1; i < n; i++ {
					work[i*ldwork+(jj-j)] = a[i*lda+jj]
					a[i*lda+jj] = 0
				}
			}
			if j+jb < n {
				bi.Dgemm(blas.NoTrans, blas.NoTrans, n, jb, n-j-jb, -1, a[(j+jb):], lda, work[(j+jb)*ldwork:], ldwork, 1, a[j:], lda)
				bi.Dtrsm(blas.Right, blas.Lower, blas.NoTrans, blas.Unit, n, jb, 1, work[j*ldwork:], ldwork, a[j:], lda)
			}
		}
	}
	for j := n - 2; j >= 0; j-- {
		jp := ipiv[j]
		if jp != j {
			bi.Dswap(n, a[j:], lda, a[jp:], lda)
		}
	}
	return true
}
