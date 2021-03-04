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
//  A * X = B  if trans == blas.Trans
//  Aᵀ * X = B if trans == blas.NoTrans
// A is a general n×n matrix with stride lda. B is a general matrix of size n×nrhs.
//
// On entry b contains the elements of the matrix B. On exit, b contains the
// elements of X, the solution to the system of equations.
//
// a and ipiv contain the LU factorization of A and the permutation indices as
// computed by Dgetrf. ipiv is zero-indexed.
func (impl Implementation) Dgetrs(trans blas.Transpose, n, nrhs int, a []float64, lda int, ipiv []int, b []float64, ldb int) {
	switch {
	case trans != blas.NoTrans && trans != blas.Trans && trans != blas.ConjTrans:
		panic(badTrans)
	case n < 0:
		panic(nLT0)
	case nrhs < 0:
		panic(nrhsLT0)
	case lda < max(1, n):
		panic(badLdA)
	case ldb < max(1, nrhs):
		panic(badLdB)
	}

	// Quick return if possible.
	if n == 0 || nrhs == 0 {
		return
	}

	switch {
	case len(a) < (n-1)*lda+n:
		panic(shortA)
	case len(b) < (n-1)*ldb+nrhs:
		panic(shortB)
	case len(ipiv) != n:
		panic(badLenIpiv)
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
	// Solve Aᵀ * X = B.
	// Solve Uᵀ * X = B, updating b.
	bi.Dtrsm(blas.Left, blas.Upper, blas.Trans, blas.NonUnit,
		n, nrhs, 1, a, lda, b, ldb)
	// Solve Lᵀ * X = B, updating b.
	bi.Dtrsm(blas.Left, blas.Lower, blas.Trans, blas.Unit,
		n, nrhs, 1, a, lda, b, ldb)
	impl.Dlaswp(nrhs, b, ldb, 0, n-1, ipiv, -1)
}
