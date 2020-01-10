// Copyright ©2019 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import "gonum.org/v1/gonum/blas"

// Dpotri computes the inverse of a real symmetric positive definite matrix A
// using its Cholesky factorization.
//
// On entry, a contains the triangular factor U or L from the Cholesky
// factorization A = Uᵀ*U or A = L*Lᵀ, as computed by Dpotrf.
// On return, a contains the upper or lower triangle of the (symmetric)
// inverse of A, overwriting the input factor U or L.
func (impl Implementation) Dpotri(uplo blas.Uplo, n int, a []float64, lda int) (ok bool) {
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
		return true
	}

	if len(a) < (n-1)*lda+n {
		panic(shortA)
	}

	// Invert the triangular Cholesky factor U or L.
	ok = impl.Dtrtri(uplo, blas.NonUnit, n, a, lda)
	if !ok {
		return false
	}

	// Form inv(U)*inv(U)ᵀ or inv(L)ᵀ*inv(L).
	impl.Dlauum(uplo, n, a, lda)
	return true
}
