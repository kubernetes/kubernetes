// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"math"

	"gonum.org/v1/gonum/blas/blas64"
)

// Dgetf2 computes the LU decomposition of the m×n matrix A.
// The LU decomposition is a factorization of a into
//  A = P * L * U
// where P is a permutation matrix, L is a unit lower triangular matrix, and
// U is a (usually) non-unit upper triangular matrix. On exit, L and U are stored
// in place into a.
//
// ipiv is a permutation vector. It indicates that row i of the matrix was
// changed with ipiv[i]. ipiv must have length at least min(m,n), and will panic
// otherwise. ipiv is zero-indexed.
//
// Dgetf2 returns whether the matrix A is singular. The LU decomposition will
// be computed regardless of the singularity of A, but division by zero
// will occur if the false is returned and the result is used to solve a
// system of equations.
//
// Dgetf2 is an internal routine. It is exported for testing purposes.
func (Implementation) Dgetf2(m, n int, a []float64, lda int, ipiv []int) (ok bool) {
	mn := min(m, n)
	checkMatrix(m, n, a, lda)
	if len(ipiv) < mn {
		panic(badIpiv)
	}
	if m == 0 || n == 0 {
		return true
	}
	bi := blas64.Implementation()
	sfmin := dlamchS
	ok = true
	for j := 0; j < mn; j++ {
		// Find a pivot and test for singularity.
		jp := j + bi.Idamax(m-j, a[j*lda+j:], lda)
		ipiv[j] = jp
		if a[jp*lda+j] == 0 {
			ok = false
		} else {
			// Swap the rows if necessary.
			if jp != j {
				bi.Dswap(n, a[j*lda:], 1, a[jp*lda:], 1)
			}
			if j < m-1 {
				aj := a[j*lda+j]
				if math.Abs(aj) >= sfmin {
					bi.Dscal(m-j-1, 1/aj, a[(j+1)*lda+j:], lda)
				} else {
					for i := 0; i < m-j-1; i++ {
						a[(j+1)*lda+j] = a[(j+1)*lda+j] / a[lda*j+j]
					}
				}
			}
		}
		if j < mn-1 {
			bi.Dger(m-j-1, n-j-1, -1, a[(j+1)*lda+j:], lda, a[j*lda+j+1:], 1, a[(j+1)*lda+j+1:], lda)
		}
	}
	return ok
}
