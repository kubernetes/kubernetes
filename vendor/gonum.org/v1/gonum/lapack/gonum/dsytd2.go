// Copyright ©2016 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
)

// Dsytd2 reduces a symmetric n×n matrix A to symmetric tridiagonal form T by
// an orthogonal similarity transformation
//  Q^T * A * Q = T
// On entry, the matrix is contained in the specified triangle of a. On exit,
// if uplo == blas.Upper, the diagonal and first super-diagonal of a are
// overwritten with the elements of T. The elements above the first super-diagonal
// are overwritten with the elementary reflectors that are used with
// the elements written to tau in order to construct Q. If uplo == blas.Lower,
// the elements are written in the lower triangular region.
//
// d must have length at least n. e and tau must have length at least n-1. Dsytd2
// will panic if these sizes are not met.
//
// Q is represented as a product of elementary reflectors.
// If uplo == blas.Upper
//  Q = H_{n-2} * ... * H_1 * H_0
// and if uplo == blas.Lower
//  Q = H_0 * H_1 * ... * H_{n-2}
// where
//  H_i = I - tau * v * v^T
// where tau is stored in tau[i], and v is stored in a.
//
// If uplo == blas.Upper, v[0:i-1] is stored in A[0:i-1,i+1], v[i] = 1, and
// v[i+1:] = 0. The elements of a are
//  [ d   e  v2  v3  v4]
//  [     d   e  v3  v4]
//  [         d   e  v4]
//  [             d   e]
//  [                 d]
// If uplo == blas.Lower, v[0:i+1] = 0, v[i+1] = 1, and v[i+2:] is stored in
// A[i+2:n,i].
// The elements of a are
//  [ d                ]
//  [ e   d            ]
//  [v1   e   d        ]
//  [v1  v2   e   d    ]
//  [v1  v2  v3   e   d]
//
// Dsytd2 is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dsytd2(uplo blas.Uplo, n int, a []float64, lda int, d, e, tau []float64) {
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
		return
	}

	switch {
	case len(a) < (n-1)*lda+n:
		panic(shortA)
	case len(d) < n:
		panic(shortD)
	case len(e) < n-1:
		panic(shortE)
	case len(tau) < n-1:
		panic(shortTau)
	}

	bi := blas64.Implementation()

	if uplo == blas.Upper {
		// Reduce the upper triangle of A.
		for i := n - 2; i >= 0; i-- {
			// Generate elementary reflector H_i = I - tau * v * v^T to
			// annihilate A[i:i-1, i+1].
			var taui float64
			a[i*lda+i+1], taui = impl.Dlarfg(i+1, a[i*lda+i+1], a[i+1:], lda)
			e[i] = a[i*lda+i+1]
			if taui != 0 {
				// Apply H_i from both sides to A[0:i,0:i].
				a[i*lda+i+1] = 1

				// Compute x := tau * A * v storing x in tau[0:i].
				bi.Dsymv(uplo, i+1, taui, a, lda, a[i+1:], lda, 0, tau, 1)

				// Compute w := x - 1/2 * tau * (x^T * v) * v.
				alpha := -0.5 * taui * bi.Ddot(i+1, tau, 1, a[i+1:], lda)
				bi.Daxpy(i+1, alpha, a[i+1:], lda, tau, 1)

				// Apply the transformation as a rank-2 update
				// A = A - v * w^T - w * v^T.
				bi.Dsyr2(uplo, i+1, -1, a[i+1:], lda, tau, 1, a, lda)
				a[i*lda+i+1] = e[i]
			}
			d[i+1] = a[(i+1)*lda+i+1]
			tau[i] = taui
		}
		d[0] = a[0]
		return
	}
	// Reduce the lower triangle of A.
	for i := 0; i < n-1; i++ {
		// Generate elementary reflector H_i = I - tau * v * v^T to
		// annihilate A[i+2:n, i].
		var taui float64
		a[(i+1)*lda+i], taui = impl.Dlarfg(n-i-1, a[(i+1)*lda+i], a[min(i+2, n-1)*lda+i:], lda)
		e[i] = a[(i+1)*lda+i]
		if taui != 0 {
			// Apply H_i from both sides to A[i+1:n, i+1:n].
			a[(i+1)*lda+i] = 1

			// Compute x := tau * A * v, storing y in tau[i:n-1].
			bi.Dsymv(uplo, n-i-1, taui, a[(i+1)*lda+i+1:], lda, a[(i+1)*lda+i:], lda, 0, tau[i:], 1)

			// Compute w := x - 1/2 * tau * (x^T * v) * v.
			alpha := -0.5 * taui * bi.Ddot(n-i-1, tau[i:], 1, a[(i+1)*lda+i:], lda)
			bi.Daxpy(n-i-1, alpha, a[(i+1)*lda+i:], lda, tau[i:], 1)

			// Apply the transformation as a rank-2 update
			// A = A - v * w^T - w * v^T.
			bi.Dsyr2(uplo, n-i-1, -1, a[(i+1)*lda+i:], lda, tau[i:], 1, a[(i+1)*lda+i+1:], lda)
			a[(i+1)*lda+i] = e[i]
		}
		d[i] = a[i*lda+i]
		tau[i] = taui
	}
	d[n-1] = a[(n-1)*lda+n-1]
}
