// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
)

// Dgetrs solves a system of equations using an LU factorization.
// The system of equations solved is
//  A * X = B if trans == blas.Trans
//  A^T * X = B if trans == blas.NoTrans
// A is a general n×n matrix with stride lda. B is a general matrix of size n×nrhs.
//
// On entry b contains the elements of the matrix B. On exit, b contains the
// elements of X, the solution to the system of equations.
//
// a and ipiv contain the LU factorization of A and the permutation indices as
// computed by Dgetrf. ipiv is zero-indexed.
func (impl Implementation) Dgetrs(trans blas.Transpose, n, nrhs int, a []float64, lda int, ipiv []int, b []float64, ldb int) {
	checkMatrix(n, n, a, lda)
	checkMatrix(n, nrhs, b, ldb)
	if len(ipiv) < n {
		panic(badIpiv)
	}
	if n == 0 || nrhs == 0 {
		return
	}
	if trans != blas.Trans && trans != blas.NoTrans {
		panic(badTrans)
	}
	bi := blas64.Implementation()
	if trans == blas.NoTrans {
		// Solve A * X = B.
		impl.Dlaswp(nrhs, b, ldb, 0, n-1, ipiv, 1)
		// Solve L * X = B, updating b.
		bi.Dtrsm(blas.Left, blas.Lower, blas.NoTrans, blas.Unit,
			n, nrhs, 1, a, lda, b, ldb)
		// Solve U * X = B, updating b.
		bi.Dtrsm(blas.Left, blas.Upper, blas.NoTrans, blas.NonUnit,
			n, nrhs, 1, a, lda, b, ldb)
		return
	}
	// Solve A^T * X = B.
	// Solve U^T * X = B, updating b.
	bi.Dtrsm(blas.Left, blas.Upper, blas.Trans, blas.NonUnit,
		n, nrhs, 1, a, lda, b, ldb)
	// Solve L^T * X = B, updating b.
	bi.Dtrsm(blas.Left, blas.Lower, blas.Trans, blas.Unit,
		n, nrhs, 1, a, lda, b, ldb)
	impl.Dlaswp(nrhs, b, ldb, 0, n-1, ipiv, -1)
}
