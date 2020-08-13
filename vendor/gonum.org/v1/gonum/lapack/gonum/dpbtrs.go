// Copyright ©2019 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
)

// Dpbtrs solves a system of linear equations A*X = B with an n×n symmetric
// positive definite band matrix A using the Cholesky factorization
//  A = Uᵀ * U  if uplo == blas.Upper
//  A = L * Lᵀ  if uplo == blas.Lower
// computed by Dpbtrf. kd is the number of super- or sub-diagonals of A. See the
// documentation for Dpbtrf for a description of the band storage format of A.
//
// On entry, b contains the n×nrhs right hand side matrix B. On return, it is
// overwritten with the solution matrix X.
func (Implementation) Dpbtrs(uplo blas.Uplo, n, kd, nrhs int, ab []float64, ldab int, b []float64, ldb int) {
	switch {
	case uplo != blas.Upper && uplo != blas.Lower:
		panic(badUplo)
	case n < 0:
		panic(nLT0)
	case kd < 0:
		panic(kdLT0)
	case nrhs < 0:
		panic(nrhsLT0)
	case ldab < kd+1:
		panic(badLdA)
	case ldb < max(1, nrhs):
		panic(badLdB)
	}

	// Quick return if possible.
	if n == 0 || nrhs == 0 {
		return
	}

	if len(ab) < (n-1)*ldab+kd+1 {
		panic(shortAB)
	}
	if len(b) < (n-1)*ldb+nrhs {
		panic(shortB)
	}

	bi := blas64.Implementation()
	if uplo == blas.Upper {
		// Solve A*X = B where A = Uᵀ*U.
		for j := 0; j < nrhs; j++ {
			// Solve Uᵀ*Y = B, overwriting B with Y.
			bi.Dtbsv(blas.Upper, blas.Trans, blas.NonUnit, n, kd, ab, ldab, b[j:], ldb)
			// Solve U*X = Y, overwriting Y with X.
			bi.Dtbsv(blas.Upper, blas.NoTrans, blas.NonUnit, n, kd, ab, ldab, b[j:], ldb)
		}
	} else {
		// Solve A*X = B where A = L*Lᵀ.
		for j := 0; j < nrhs; j++ {
			// Solve L*Y = B, overwriting B with Y.
			bi.Dtbsv(blas.Lower, blas.NoTrans, blas.NonUnit, n, kd, ab, ldab, b[j:], ldb)
			// Solve Lᵀ*X = Y, overwriting Y with X.
			bi.Dtbsv(blas.Lower, blas.Trans, blas.NonUnit, n, kd, ab, ldab, b[j:], ldb)
		}
	}
}
