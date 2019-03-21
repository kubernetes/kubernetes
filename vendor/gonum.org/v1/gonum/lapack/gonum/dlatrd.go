// Copyright ©2016 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
)

// Dlatrd reduces nb rows and columns of a real n×n symmetric matrix A to symmetric
// tridiagonal form. It computes the orthonormal similarity transformation
//  Q^T * A * Q
// and returns the matrices V and W to apply to the unreduced part of A. If
// uplo == blas.Upper, the upper triangle is supplied and the last nb rows are
// reduced. If uplo == blas.Lower, the lower triangle is supplied and the first
// nb rows are reduced.
//
// a contains the symmetric matrix on entry with active triangular half specified
// by uplo. On exit, the nb columns have been reduced to tridiagonal form. The
// diagonal contains the diagonal of the reduced matrix, the off-diagonal is
// set to 1, and the remaining elements contain the data to construct Q.
//
// If uplo == blas.Upper, with n = 5 and nb = 2 on exit a is
//  [ a   a   a  v4  v5]
//  [     a   a  v4  v5]
//  [         a   1  v5]
//  [             d   1]
//  [                 d]
//
// If uplo == blas.Lower, with n = 5 and nb = 2, on exit a is
//  [ d                ]
//  [ 1   d            ]
//  [v1   1   a        ]
//  [v1  v2   a   a    ]
//  [v1  v2   a   a   a]
//
// e contains the superdiagonal elements of the reduced matrix. If uplo == blas.Upper,
// e[n-nb:n-1] contains the last nb columns of the reduced matrix, while if
// uplo == blas.Lower, e[:nb] contains the first nb columns of the reduced matrix.
// e must have length at least n-1, and Dlatrd will panic otherwise.
//
// tau contains the scalar factors of the elementary reflectors needed to construct Q.
// The reflectors are stored in tau[n-nb:n-1] if uplo == blas.Upper, and in
// tau[:nb] if uplo == blas.Lower. tau must have length n-1, and Dlatrd will panic
// otherwise.
//
// w is an n×nb matrix. On exit it contains the data to update the unreduced part
// of A.
//
// The matrix Q is represented as a product of elementary reflectors. Each reflector
// H has the form
//  I - tau * v * v^T
// If uplo == blas.Upper,
//  Q = H_{n-1} * H_{n-2} * ... * H_{n-nb}
// where v[:i-1] is stored in A[:i-1,i], v[i-1] = 1, and v[i:n] = 0.
//
// If uplo == blas.Lower,
//  Q = H_0 * H_1 * ... * H_{nb-1}
// where v[:i+1] = 0, v[i+1] = 1, and v[i+2:n] is stored in A[i+2:n,i].
//
// The vectors v form the n×nb matrix V which is used with W to apply a
// symmetric rank-2 update to the unreduced part of A
//  A = A - V * W^T - W * V^T
//
// Dlatrd is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dlatrd(uplo blas.Uplo, n, nb int, a []float64, lda int, e, tau, w []float64, ldw int) {
	checkMatrix(n, n, a, lda)
	checkMatrix(n, nb, w, ldw)
	if len(e) < n-1 {
		panic(badE)
	}
	if len(tau) < n-1 {
		panic(badTau)
	}
	if n <= 0 {
		return
	}
	bi := blas64.Implementation()
	if uplo == blas.Upper {
		for i := n - 1; i >= n-nb; i-- {
			iw := i - n + nb
			if i < n-1 {
				// Update A(0:i, i).
				bi.Dgemv(blas.NoTrans, i+1, n-i-1, -1, a[i+1:], lda,
					w[i*ldw+iw+1:], 1, 1, a[i:], lda)
				bi.Dgemv(blas.NoTrans, i+1, n-i-1, -1, w[iw+1:], ldw,
					a[i*lda+i+1:], 1, 1, a[i:], lda)
			}
			if i > 0 {
				// Generate elementary reflector H_i to annihilate A(0:i-2,i).
				e[i-1], tau[i-1] = impl.Dlarfg(i, a[(i-1)*lda+i], a[i:], lda)
				a[(i-1)*lda+i] = 1

				// Compute W(0:i-1, i).
				bi.Dsymv(blas.Upper, i, 1, a, lda, a[i:], lda, 0, w[iw:], ldw)
				if i < n-1 {
					bi.Dgemv(blas.Trans, i, n-i-1, 1, w[iw+1:], ldw,
						a[i:], lda, 0, w[(i+1)*ldw+iw:], ldw)
					bi.Dgemv(blas.NoTrans, i, n-i-1, -1, a[i+1:], lda,
						w[(i+1)*ldw+iw:], ldw, 1, w[iw:], ldw)
					bi.Dgemv(blas.Trans, i, n-i-1, 1, a[i+1:], lda,
						a[i:], lda, 0, w[(i+1)*ldw+iw:], ldw)
					bi.Dgemv(blas.NoTrans, i, n-i-1, -1, w[iw+1:], ldw,
						w[(i+1)*ldw+iw:], ldw, 1, w[iw:], ldw)
				}
				bi.Dscal(i, tau[i-1], w[iw:], ldw)
				alpha := -0.5 * tau[i-1] * bi.Ddot(i, w[iw:], ldw, a[i:], lda)
				bi.Daxpy(i, alpha, a[i:], lda, w[iw:], ldw)
			}
		}
	} else {
		// Reduce first nb columns of lower triangle.
		for i := 0; i < nb; i++ {
			// Update A(i:n, i)
			bi.Dgemv(blas.NoTrans, n-i, i, -1, a[i*lda:], lda,
				w[i*ldw:], 1, 1, a[i*lda+i:], lda)
			bi.Dgemv(blas.NoTrans, n-i, i, -1, w[i*ldw:], ldw,
				a[i*lda:], 1, 1, a[i*lda+i:], lda)
			if i < n-1 {
				// Generate elementary reflector H_i to annihilate A(i+2:n,i).
				e[i], tau[i] = impl.Dlarfg(n-i-1, a[(i+1)*lda+i], a[min(i+2, n-1)*lda+i:], lda)
				a[(i+1)*lda+i] = 1

				// Compute W(i+1:n,i).
				bi.Dsymv(blas.Lower, n-i-1, 1, a[(i+1)*lda+i+1:], lda,
					a[(i+1)*lda+i:], lda, 0, w[(i+1)*ldw+i:], ldw)
				bi.Dgemv(blas.Trans, n-i-1, i, 1, w[(i+1)*ldw:], ldw,
					a[(i+1)*lda+i:], lda, 0, w[i:], ldw)
				bi.Dgemv(blas.NoTrans, n-i-1, i, -1, a[(i+1)*lda:], lda,
					w[i:], ldw, 1, w[(i+1)*ldw+i:], ldw)
				bi.Dgemv(blas.Trans, n-i-1, i, 1, a[(i+1)*lda:], lda,
					a[(i+1)*lda+i:], lda, 0, w[i:], ldw)
				bi.Dgemv(blas.NoTrans, n-i-1, i, -1, w[(i+1)*ldw:], ldw,
					w[i:], ldw, 1, w[(i+1)*ldw+i:], ldw)
				bi.Dscal(n-i-1, tau[i], w[(i+1)*ldw+i:], ldw)
				alpha := -0.5 * tau[i] * bi.Ddot(n-i-1, w[(i+1)*ldw+i:], ldw,
					a[(i+1)*lda+i:], lda)
				bi.Daxpy(n-i-1, alpha, a[(i+1)*lda+i:], lda,
					w[(i+1)*ldw+i:], ldw)
			}
		}
	}
}
