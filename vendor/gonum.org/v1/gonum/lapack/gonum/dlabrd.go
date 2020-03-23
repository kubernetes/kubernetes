// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
)

// Dlabrd reduces the first NB rows and columns of a real general m×n matrix
// A to upper or lower bidiagonal form by an orthogonal transformation
//  Q**T * A * P
// If m >= n, A is reduced to upper bidiagonal form and upon exit the elements
// on and below the diagonal in the first nb columns represent the elementary
// reflectors, and the elements above the diagonal in the first nb rows represent
// the matrix P. If m < n, A is reduced to lower bidiagonal form and the elements
// P is instead stored above the diagonal.
//
// The reduction to bidiagonal form is stored in d and e, where d are the diagonal
// elements, and e are the off-diagonal elements.
//
// The matrices Q and P are products of elementary reflectors
//  Q = H_0 * H_1 * ... * H_{nb-1}
//  P = G_0 * G_1 * ... * G_{nb-1}
// where
//  H_i = I - tauQ[i] * v_i * v_iᵀ
//  G_i = I - tauP[i] * u_i * u_iᵀ
//
// As an example, on exit the entries of A when m = 6, n = 5, and nb = 2
//  [ 1   1  u1  u1  u1]
//  [v1   1   1  u2  u2]
//  [v1  v2   a   a   a]
//  [v1  v2   a   a   a]
//  [v1  v2   a   a   a]
//  [v1  v2   a   a   a]
// and when m = 5, n = 6, and nb = 2
//  [ 1  u1  u1  u1  u1  u1]
//  [ 1   1  u2  u2  u2  u2]
//  [v1   1   a   a   a   a]
//  [v1  v2   a   a   a   a]
//  [v1  v2   a   a   a   a]
//
// Dlabrd also returns the matrices X and Y which are used with U and V to
// apply the transformation to the unreduced part of the matrix
//  A := A - V*Yᵀ - X*Uᵀ
// and returns the matrices X and Y which are needed to apply the
// transformation to the unreduced part of A.
//
// X is an m×nb matrix, Y is an n×nb matrix. d, e, taup, and tauq must all have
// length at least nb. Dlabrd will panic if these size constraints are violated.
//
// Dlabrd is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dlabrd(m, n, nb int, a []float64, lda int, d, e, tauQ, tauP, x []float64, ldx int, y []float64, ldy int) {
	switch {
	case m < 0:
		panic(mLT0)
	case n < 0:
		panic(nLT0)
	case nb < 0:
		panic(nbLT0)
	case nb > n:
		panic(nbGTN)
	case nb > m:
		panic(nbGTM)
	case lda < max(1, n):
		panic(badLdA)
	case ldx < max(1, nb):
		panic(badLdX)
	case ldy < max(1, nb):
		panic(badLdY)
	}

	if m == 0 || n == 0 || nb == 0 {
		return
	}

	switch {
	case len(a) < (m-1)*lda+n:
		panic(shortA)
	case len(d) < nb:
		panic(shortD)
	case len(e) < nb:
		panic(shortE)
	case len(tauQ) < nb:
		panic(shortTauQ)
	case len(tauP) < nb:
		panic(shortTauP)
	case len(x) < (m-1)*ldx+nb:
		panic(shortX)
	case len(y) < (n-1)*ldy+nb:
		panic(shortY)
	}

	bi := blas64.Implementation()

	if m >= n {
		// Reduce to upper bidiagonal form.
		for i := 0; i < nb; i++ {
			bi.Dgemv(blas.NoTrans, m-i, i, -1, a[i*lda:], lda, y[i*ldy:], 1, 1, a[i*lda+i:], lda)
			bi.Dgemv(blas.NoTrans, m-i, i, -1, x[i*ldx:], ldx, a[i:], lda, 1, a[i*lda+i:], lda)

			a[i*lda+i], tauQ[i] = impl.Dlarfg(m-i, a[i*lda+i], a[min(i+1, m-1)*lda+i:], lda)
			d[i] = a[i*lda+i]
			if i < n-1 {
				// Compute Y[i+1:n, i].
				a[i*lda+i] = 1
				bi.Dgemv(blas.Trans, m-i, n-i-1, 1, a[i*lda+i+1:], lda, a[i*lda+i:], lda, 0, y[(i+1)*ldy+i:], ldy)
				bi.Dgemv(blas.Trans, m-i, i, 1, a[i*lda:], lda, a[i*lda+i:], lda, 0, y[i:], ldy)
				bi.Dgemv(blas.NoTrans, n-i-1, i, -1, y[(i+1)*ldy:], ldy, y[i:], ldy, 1, y[(i+1)*ldy+i:], ldy)
				bi.Dgemv(blas.Trans, m-i, i, 1, x[i*ldx:], ldx, a[i*lda+i:], lda, 0, y[i:], ldy)
				bi.Dgemv(blas.Trans, i, n-i-1, -1, a[i+1:], lda, y[i:], ldy, 1, y[(i+1)*ldy+i:], ldy)
				bi.Dscal(n-i-1, tauQ[i], y[(i+1)*ldy+i:], ldy)

				// Update A[i, i+1:n].
				bi.Dgemv(blas.NoTrans, n-i-1, i+1, -1, y[(i+1)*ldy:], ldy, a[i*lda:], 1, 1, a[i*lda+i+1:], 1)
				bi.Dgemv(blas.Trans, i, n-i-1, -1, a[i+1:], lda, x[i*ldx:], 1, 1, a[i*lda+i+1:], 1)

				// Generate reflection P[i] to annihilate A[i, i+2:n].
				a[i*lda+i+1], tauP[i] = impl.Dlarfg(n-i-1, a[i*lda+i+1], a[i*lda+min(i+2, n-1):], 1)
				e[i] = a[i*lda+i+1]
				a[i*lda+i+1] = 1

				// Compute X[i+1:m, i].
				bi.Dgemv(blas.NoTrans, m-i-1, n-i-1, 1, a[(i+1)*lda+i+1:], lda, a[i*lda+i+1:], 1, 0, x[(i+1)*ldx+i:], ldx)
				bi.Dgemv(blas.Trans, n-i-1, i+1, 1, y[(i+1)*ldy:], ldy, a[i*lda+i+1:], 1, 0, x[i:], ldx)
				bi.Dgemv(blas.NoTrans, m-i-1, i+1, -1, a[(i+1)*lda:], lda, x[i:], ldx, 1, x[(i+1)*ldx+i:], ldx)
				bi.Dgemv(blas.NoTrans, i, n-i-1, 1, a[i+1:], lda, a[i*lda+i+1:], 1, 0, x[i:], ldx)
				bi.Dgemv(blas.NoTrans, m-i-1, i, -1, x[(i+1)*ldx:], ldx, x[i:], ldx, 1, x[(i+1)*ldx+i:], ldx)
				bi.Dscal(m-i-1, tauP[i], x[(i+1)*ldx+i:], ldx)
			}
		}
		return
	}
	// Reduce to lower bidiagonal form.
	for i := 0; i < nb; i++ {
		// Update A[i,i:n]
		bi.Dgemv(blas.NoTrans, n-i, i, -1, y[i*ldy:], ldy, a[i*lda:], 1, 1, a[i*lda+i:], 1)
		bi.Dgemv(blas.Trans, i, n-i, -1, a[i:], lda, x[i*ldx:], 1, 1, a[i*lda+i:], 1)

		// Generate reflection P[i] to annihilate A[i, i+1:n]
		a[i*lda+i], tauP[i] = impl.Dlarfg(n-i, a[i*lda+i], a[i*lda+min(i+1, n-1):], 1)
		d[i] = a[i*lda+i]
		if i < m-1 {
			a[i*lda+i] = 1
			// Compute X[i+1:m, i].
			bi.Dgemv(blas.NoTrans, m-i-1, n-i, 1, a[(i+1)*lda+i:], lda, a[i*lda+i:], 1, 0, x[(i+1)*ldx+i:], ldx)
			bi.Dgemv(blas.Trans, n-i, i, 1, y[i*ldy:], ldy, a[i*lda+i:], 1, 0, x[i:], ldx)
			bi.Dgemv(blas.NoTrans, m-i-1, i, -1, a[(i+1)*lda:], lda, x[i:], ldx, 1, x[(i+1)*ldx+i:], ldx)
			bi.Dgemv(blas.NoTrans, i, n-i, 1, a[i:], lda, a[i*lda+i:], 1, 0, x[i:], ldx)
			bi.Dgemv(blas.NoTrans, m-i-1, i, -1, x[(i+1)*ldx:], ldx, x[i:], ldx, 1, x[(i+1)*ldx+i:], ldx)
			bi.Dscal(m-i-1, tauP[i], x[(i+1)*ldx+i:], ldx)

			// Update A[i+1:m, i].
			bi.Dgemv(blas.NoTrans, m-i-1, i, -1, a[(i+1)*lda:], lda, y[i*ldy:], 1, 1, a[(i+1)*lda+i:], lda)
			bi.Dgemv(blas.NoTrans, m-i-1, i+1, -1, x[(i+1)*ldx:], ldx, a[i:], lda, 1, a[(i+1)*lda+i:], lda)

			// Generate reflection Q[i] to annihilate A[i+2:m, i].
			a[(i+1)*lda+i], tauQ[i] = impl.Dlarfg(m-i-1, a[(i+1)*lda+i], a[min(i+2, m-1)*lda+i:], lda)
			e[i] = a[(i+1)*lda+i]
			a[(i+1)*lda+i] = 1

			// Compute Y[i+1:n, i].
			bi.Dgemv(blas.Trans, m-i-1, n-i-1, 1, a[(i+1)*lda+i+1:], lda, a[(i+1)*lda+i:], lda, 0, y[(i+1)*ldy+i:], ldy)
			bi.Dgemv(blas.Trans, m-i-1, i, 1, a[(i+1)*lda:], lda, a[(i+1)*lda+i:], lda, 0, y[i:], ldy)
			bi.Dgemv(blas.NoTrans, n-i-1, i, -1, y[(i+1)*ldy:], ldy, y[i:], ldy, 1, y[(i+1)*ldy+i:], ldy)
			bi.Dgemv(blas.Trans, m-i-1, i+1, 1, x[(i+1)*ldx:], ldx, a[(i+1)*lda+i:], lda, 0, y[i:], ldy)
			bi.Dgemv(blas.Trans, i+1, n-i-1, -1, a[i+1:], lda, y[i:], ldy, 1, y[(i+1)*ldy+i:], ldy)
			bi.Dscal(n-i-1, tauQ[i], y[(i+1)*ldy+i:], ldy)
		}
	}
}
