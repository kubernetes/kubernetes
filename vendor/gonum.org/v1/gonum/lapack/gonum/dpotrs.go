// Copyright ©2018 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
)

// Dpotrs solves a system of n linear equations A*X = B where A is an n×n
// symmetric positive definite matrix and B is an n×nrhs matrix. The matrix A is
// represented by its Cholesky factorization
//  A = Uᵀ*U  if uplo == blas.Upper
//  A = L*Lᵀ  if uplo == blas.Lower
// as computed by Dpotrf. On entry, B contains the right-hand side matrix B, on
// return it contains the solution matrix X.
func (Implementation) Dpotrs(uplo blas.Uplo, n, nrhs int, a []float64, lda int, b []float64, ldb int) {
	switch {
	case uplo != blas.Upper && uplo != blas.Lower:
		panic(badUplo)
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
	}

	bi := blas64.Implementation()

	if uplo == blas.Upper {
		// Solve Uᵀ * U * X = B where U is stored in the upper triangle of A.

		// Solve Uᵀ * X = B, overwriting B with X.
		bi.Dtrsm(blas.Left, blas.Upper, blas.Trans, blas.NonUnit, n, nrhs, 1, a, lda, b, ldb)
		// Solve U * X = B, overwriting B with X.
		bi.Dtrsm(blas.Left, blas.Upper, blas.NoTrans, blas.NonUnit, n, nrhs, 1, a, lda, b, ldb)
	} else {
		// Solve L * Lᵀ * X = B where L is stored in the lower triangle of A.

		// Solve L * X = B, overwriting B with X.
		bi.Dtrsm(blas.Left, blas.Lower, blas.NoTrans, blas.NonUnit, n, nrhs, 1, a, lda, b, ldb)
		// Solve Lᵀ * X = B, overwriting B with X.
		bi.Dtrsm(blas.Left, blas.Lower, blas.Trans, blas.NonUnit, n, nrhs, 1, a, lda, b, ldb)
	}
}
