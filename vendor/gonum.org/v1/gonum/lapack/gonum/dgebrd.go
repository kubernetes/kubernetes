// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
)

// Dgebrd reduces a general m×n matrix A to upper or lower bidiagonal form B by
// an orthogonal transformation:
//  Q^T * A * P = B.
// The diagonal elements of B are stored in d and the off-diagonal elements are stored
// in e. These are additionally stored along the diagonal of A and the off-diagonal
// of A. If m >= n B is an upper-bidiagonal matrix, and if m < n B is a
// lower-bidiagonal matrix.
//
// The remaining elements of A store the data needed to construct Q and P.
// The matrices Q and P are products of elementary reflectors
//  if m >= n, Q = H_0 * H_1 * ... * H_{n-1},
//             P = G_0 * G_1 * ... * G_{n-2},
//  if m < n,  Q = H_0 * H_1 * ... * H_{m-2},
//             P = G_0 * G_1 * ... * G_{m-1},
// where
//  H_i = I - tauQ[i] * v_i * v_i^T,
//  G_i = I - tauP[i] * u_i * u_i^T.
//
// As an example, on exit the entries of A when m = 6, and n = 5
//  [ d   e  u1  u1  u1]
//  [v1   d   e  u2  u2]
//  [v1  v2   d   e  u3]
//  [v1  v2  v3   d   e]
//  [v1  v2  v3  v4   d]
//  [v1  v2  v3  v4  v5]
// and when m = 5, n = 6
//  [ d  u1  u1  u1  u1  u1]
//  [ e   d  u2  u2  u2  u2]
//  [v1   e   d  u3  u3  u3]
//  [v1  v2   e   d  u4  u4]
//  [v1  v2  v3   e   d  u5]
//
// d, tauQ, and tauP must all have length at least min(m,n), and e must have
// length min(m,n) - 1, unless lwork is -1 when there is no check except for
// work which must have a length of at least one.
//
// work is temporary storage, and lwork specifies the usable memory length.
// At minimum, lwork >= max(1,m,n) or be -1 and this function will panic otherwise.
// Dgebrd is blocked decomposition, but the block size is limited
// by the temporary space available. If lwork == -1, instead of performing Dgebrd,
// the optimal work length will be stored into work[0].
//
// Dgebrd is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dgebrd(m, n int, a []float64, lda int, d, e, tauQ, tauP, work []float64, lwork int) {
	checkMatrix(m, n, a, lda)
	// Calculate optimal work.
	nb := impl.Ilaenv(1, "DGEBRD", " ", m, n, -1, -1)
	var lworkOpt int
	if lwork == -1 {
		if len(work) < 1 {
			panic(badWork)
		}
		lworkOpt = ((m + n) * nb)
		work[0] = float64(max(1, lworkOpt))
		return
	}
	minmn := min(m, n)
	if len(d) < minmn {
		panic(badD)
	}
	if len(e) < minmn-1 {
		panic(badE)
	}
	if len(tauQ) < minmn {
		panic(badTauQ)
	}
	if len(tauP) < minmn {
		panic(badTauP)
	}
	ws := max(m, n)
	if lwork < max(1, ws) {
		panic(badWork)
	}
	if len(work) < lwork {
		panic(badWork)
	}
	var nx int
	if nb > 1 && nb < minmn {
		nx = max(nb, impl.Ilaenv(3, "DGEBRD", " ", m, n, -1, -1))
		if nx < minmn {
			ws = (m + n) * nb
			if lwork < ws {
				nbmin := impl.Ilaenv(2, "DGEBRD", " ", m, n, -1, -1)
				if lwork >= (m+n)*nbmin {
					nb = lwork / (m + n)
				} else {
					nb = minmn
					nx = minmn
				}
			}
		}
	} else {
		nx = minmn
	}
	bi := blas64.Implementation()
	ldworkx := nb
	ldworky := nb
	var i int
	// Netlib lapack has minmn - nx, but this makes the last nx rows (which by
	// default is large) be unblocked. As written here, the blocking is more
	// consistent.
	for i = 0; i < minmn-nb; i += nb {
		// Reduce rows and columns i:i+nb to bidiagonal form and return
		// the matrices X and Y which are needed to update the unreduced
		// part of the matrix.
		// X is stored in the first m rows of work, y in the next rows.
		x := work[:m*ldworkx]
		y := work[m*ldworkx:]
		impl.Dlabrd(m-i, n-i, nb, a[i*lda+i:], lda,
			d[i:], e[i:], tauQ[i:], tauP[i:],
			x, ldworkx, y, ldworky)

		// Update the trailing submatrix A[i+nb:m,i+nb:n], using an update
		// of the form  A := A - V*Y**T - X*U**T
		bi.Dgemm(blas.NoTrans, blas.Trans, m-i-nb, n-i-nb, nb,
			-1, a[(i+nb)*lda+i:], lda, y[nb*ldworky:], ldworky,
			1, a[(i+nb)*lda+i+nb:], lda)

		bi.Dgemm(blas.NoTrans, blas.NoTrans, m-i-nb, n-i-nb, nb,
			-1, x[nb*ldworkx:], ldworkx, a[i*lda+i+nb:], lda,
			1, a[(i+nb)*lda+i+nb:], lda)

		// Copy diagonal and off-diagonal elements of B back into A.
		if m >= n {
			for j := i; j < i+nb; j++ {
				a[j*lda+j] = d[j]
				a[j*lda+j+1] = e[j]
			}
		} else {
			for j := i; j < i+nb; j++ {
				a[j*lda+j] = d[j]
				a[(j+1)*lda+j] = e[j]
			}
		}
	}
	// Use unblocked code to reduce the remainder of the matrix.
	impl.Dgebd2(m-i, n-i, a[i*lda+i:], lda, d[i:], e[i:], tauQ[i:], tauP[i:], work)
	work[0] = float64(lworkOpt)
}
