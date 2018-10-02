// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import "gonum.org/v1/gonum/blas"

// Dlaset sets the off-diagonal elements of A to alpha, and the diagonal
// elements to beta. If uplo == blas.Upper, only the elements in the upper
// triangular part are set. If uplo == blas.Lower, only the elements in the
// lower triangular part are set. If uplo is otherwise, all of the elements of A
// are set.
//
// Dlaset is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dlaset(uplo blas.Uplo, m, n int, alpha, beta float64, a []float64, lda int) {
	checkMatrix(m, n, a, lda)
	if uplo == blas.Upper {
		for i := 0; i < m; i++ {
			for j := i + 1; j < n; j++ {
				a[i*lda+j] = alpha
			}
		}
	} else if uplo == blas.Lower {
		for i := 0; i < m; i++ {
			for j := 0; j < min(i+1, n); j++ {
				a[i*lda+j] = alpha
			}
		}
	} else {
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				a[i*lda+j] = alpha
			}
		}
	}
	for i := 0; i < min(m, n); i++ {
		a[i*lda+i] = beta
	}
}
