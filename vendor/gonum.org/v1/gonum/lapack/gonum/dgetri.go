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
	iws := max(1, n)
	switch {
	case n < 0:
		panic(nLT0)
	case lda < max(1, n):
		panic(badLdA)
	case lwork < iws && lwork != -1:
		panic(badLWork)
	case len(work) < max(1, lwork):
		panic(shortWork)
	}

	if n == 0 {
		work[0] = 1
		return true
	}

	nb := impl.Ilaenv(1, "DGETRI", " ", n, -1, -1, -1)
	if lwork == -1 {
		work[0] = float64(n * nb)
		return true
	}

	switch {
	case len(a) < (n-1)*lda+n:
		panic(shortA)
	case len(ipiv) != n:
		panic(badLenIpiv)
	}

	// Form inv(U).
	ok = impl.Dtrtri(blas.Upper, blas.NonUnit, n, a, lda)
	if !ok {
		return false
	}

	nbmin := 2
	if 1 < nb && nb < n {
		iws = max(n*nb, 1)
		if lwork < iws {
			nb = lwork / n
			nbmin = max(2, impl.Ilaenv(2, "DGETRI", " ", n, -1, -1, -1))
		}
	}
	ldwork := nb

	bi := blas64.Implementation()
	// Solve the equation inv(A)*L = inv(U) for inv(A).
	// TODO(btracey): Replace this with a more row-major oriented algorithm.
	if nb < nbmin || n <= nb {
		// Unblocked code.
		for j := n - 1; j >= 0; j-- {
			for i := j + 1; i < n; i++ {
				// Copy current column of L to work and replace with zeros.
				work[i] = a[i*lda+j]
				a[i*lda+j] = 0
			}
			// Compute current column of inv(A).
			if j < n-1 {
				bi.Dgemv(blas.NoTrans, n, n-j-1, -1, a[(j+1):], lda, work[(j+1):], 1, 1, a[j:], lda)
			}
		}
	} else {
		// Blocked code.
		nn := ((n - 1) / nb) * nb
		for j := nn; j >= 0; j -= nb {
			jb := min(nb, n-j)
			// Copy current block column of L to work and replace
			// with zeros.
			for jj := j; jj < j+jb; jj++ {
				for i := jj + 1; i < n; i++ {
					work[i*ldwork+(jj-j)] = a[i*lda+jj]
					a[i*lda+jj] = 0
				}
			}
			// Compute current block column of inv(A).
			if j+jb < n {
				bi.Dgemm(blas.NoTrans, blas.NoTrans, n, jb, n-j-jb, -1, a[(j+jb):], lda, work[(j+jb)*ldwork:], ldwork, 1, a[j:], lda)
			}
			bi.Dtrsm(blas.Right, blas.Lower, blas.NoTrans, blas.Unit, n, jb, 1, work[j*ldwork:], ldwork, a[j:], lda)
		}
	}
	// Apply column interchanges.
	for j := n - 2; j >= 0; j-- {
		jp := ipiv[j]
		if jp != j {
			bi.Dswap(n, a[j:], lda, a[jp:], lda)
		}
	}
	work[0] = float64(iws)
	return true
}
