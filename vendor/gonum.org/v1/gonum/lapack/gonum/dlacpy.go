// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import "gonum.org/v1/gonum/blas"

// Dlacpy copies the elements of A specified by uplo into B. Uplo can specify
// a triangular portion with blas.Upper or blas.Lower, or can specify all of the
// elemest with blas.All.
//
// Dlacpy is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dlacpy(uplo blas.Uplo, m, n int, a []float64, lda int, b []float64, ldb int) {
	switch {
	case uplo != blas.Upper && uplo != blas.Lower && uplo != blas.All:
		panic(badUplo)
	case m < 0:
		panic(mLT0)
	case n < 0:
		panic(nLT0)
	case lda < max(1, n):
		panic(badLdA)
	case ldb < max(1, n):
		panic(badLdB)
	}

	if m == 0 || n == 0 {
		return
	}

	switch {
	case len(a) < (m-1)*lda+n:
		panic(shortA)
	case len(b) < (m-1)*ldb+n:
		panic(shortB)
	}

	switch uplo {
	case blas.Upper:
		for i := 0; i < m; i++ {
			for j := i; j < n; j++ {
				b[i*ldb+j] = a[i*lda+j]
			}
		}
	case blas.Lower:
		for i := 0; i < m; i++ {
			for j := 0; j < min(i+1, n); j++ {
				b[i*ldb+j] = a[i*lda+j]
			}
		}
	case blas.All:
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				b[i*ldb+j] = a[i*lda+j]
			}
		}
	}
}
