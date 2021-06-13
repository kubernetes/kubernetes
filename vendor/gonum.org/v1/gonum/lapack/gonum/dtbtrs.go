// Copyright ©2020 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
)

// Dtbtrs solves a triangular system of the form
//  A * X = B   if trans == blas.NoTrans
//  Aᵀ * X = B  if trans == blas.Trans or blas.ConjTrans
// where A is an n×n triangular band matrix with kd super- or subdiagonals, and
// B is an n×nrhs matrix.
//
// Dtbtrs returns whether A is non-singular. If A is singular, no solution X is
// computed.
func (impl Implementation) Dtbtrs(uplo blas.Uplo, trans blas.Transpose, diag blas.Diag, n, kd, nrhs int, a []float64, lda int, b []float64, ldb int) (ok bool) {
	switch {
	case uplo != blas.Upper && uplo != blas.Lower:
		panic(badUplo)
	case trans != blas.NoTrans && trans != blas.Trans && trans != blas.ConjTrans:
		panic(badTrans)
	case diag != blas.NonUnit && diag != blas.Unit:
		panic(badDiag)
	case n < 0:
		panic(nLT0)
	case kd < 0:
		panic(kdLT0)
	case nrhs < 0:
		panic(nrhsLT0)
	case lda < kd+1:
		panic(badLdA)
	case ldb < max(1, nrhs):
		panic(badLdB)
	}

	// Quick return if possible.
	if n == 0 {
		return true
	}

	switch {
	case len(a) < (n-1)*lda+kd+1:
		panic(shortA)
	case len(b) < (n-1)*ldb+nrhs:
		panic(shortB)
	}

	// Check for singularity.
	if diag == blas.NonUnit {
		if uplo == blas.Upper {
			for i := 0; i < n; i++ {
				if a[i*lda] == 0 {
					return false
				}
			}
		} else {
			for i := 0; i < n; i++ {
				if a[i*lda+kd] == 0 {
					return false
				}
			}
		}
	}

	// Solve A * X = B  or Aᵀ * X = B.
	bi := blas64.Implementation()
	for j := 0; j < nrhs; j++ {
		bi.Dtbsv(uplo, trans, diag, n, kd, a, lda, b[j:], ldb)
	}
	return true
}
