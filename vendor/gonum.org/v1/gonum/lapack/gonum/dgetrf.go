// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
)

// Dgetrf computes the LU decomposition of the m×n matrix A.
// The LU decomposition is a factorization of A into
//  A = P * L * U
// where P is a permutation matrix, L is a unit lower triangular matrix, and
// U is a (usually) non-unit upper triangular matrix. On exit, L and U are stored
// in place into a.
//
// ipiv is a permutation vector. It indicates that row i of the matrix was
// changed with ipiv[i]. ipiv must have length at least min(m,n), and will panic
// otherwise. ipiv is zero-indexed.
//
// Dgetrf is the blocked version of the algorithm.
//
// Dgetrf returns whether the matrix A is singular. The LU decomposition will
// be computed regardless of the singularity of A, but division by zero
// will occur if the false is returned and the result is used to solve a
// system of equations.
func (impl Implementation) Dgetrf(m, n int, a []float64, lda int, ipiv []int) (ok bool) {
	mn := min(m, n)
	checkMatrix(m, n, a, lda)
	if len(ipiv) < mn {
		panic(badIpiv)
	}
	if m == 0 || n == 0 {
		return false
	}
	bi := blas64.Implementation()
	nb := impl.Ilaenv(1, "DGETRF", " ", m, n, -1, -1)
	if nb <= 1 || nb >= min(m, n) {
		// Use the unblocked algorithm.
		return impl.Dgetf2(m, n, a, lda, ipiv)
	}
	ok = true
	for j := 0; j < mn; j += nb {
		jb := min(mn-j, nb)
		blockOk := impl.Dgetf2(m-j, jb, a[j*lda+j:], lda, ipiv[j:])
		if !blockOk {
			ok = false
		}
		for i := j; i <= min(m-1, j+jb-1); i++ {
			ipiv[i] = j + ipiv[i]
		}
		impl.Dlaswp(j, a, lda, j, j+jb-1, ipiv[:j+jb], 1)
		if j+jb < n {
			impl.Dlaswp(n-j-jb, a[j+jb:], lda, j, j+jb-1, ipiv[:j+jb], 1)
			bi.Dtrsm(blas.Left, blas.Lower, blas.NoTrans, blas.Unit,
				jb, n-j-jb, 1,
				a[j*lda+j:], lda,
				a[j*lda+j+jb:], lda)
			if j+jb < m {
				bi.Dgemm(blas.NoTrans, blas.NoTrans, m-j-jb, n-j-jb, jb, -1,
					a[(j+jb)*lda+j:], lda,
					a[j*lda+j+jb:], lda,
					1, a[(j+jb)*lda+j+jb:], lda)
			}
		}
	}
	return ok
}
