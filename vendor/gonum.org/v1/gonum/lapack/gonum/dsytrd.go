// Copyright ©2016 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
)

// Dsytrd reduces a symmetric n×n matrix A to symmetric tridiagonal form by an
// orthogonal similarity transformation
//  Q^T * A * Q = T
// where Q is an orthonormal matrix and T is symmetric and tridiagonal.
//
// On entry, a contains the elements of the input matrix in the triangle specified
// by uplo. On exit, the diagonal and sub/super-diagonal are overwritten by the
// corresponding elements of the tridiagonal matrix T. The remaining elements in
// the triangle, along with the array tau, contain the data to construct Q as
// the product of elementary reflectors.
//
// If uplo == blas.Upper, Q is constructed with
//  Q = H_{n-2} * ... * H_1 * H_0
// where
//  H_i = I - tau_i * v * v^T
// v is constructed as v[i+1:n] = 0, v[i] = 1, v[0:i-1] is stored in A[0:i-1, i+1].
// The elements of A are
//  [ d   e  v1  v2  v3]
//  [     d   e  v2  v3]
//  [         d   e  v3]
//  [             d   e]
//  [                 e]
//
// If uplo == blas.Lower, Q is constructed with
//  Q = H_0 * H_1 * ... * H_{n-2}
// where
//  H_i = I - tau_i * v * v^T
// v is constructed as v[0:i+1] = 0, v[i+1] = 1, v[i+2:n] is stored in A[i+2:n, i].
// The elements of A are
//  [ d                ]
//  [ e   d            ]
//  [v0   e   d        ]
//  [v0  v1   e   d    ]
//  [v0  v1  v2   e   d]
//
// d must have length n, and e and tau must have length n-1. Dsytrd will panic if
// these conditions are not met.
//
// work is temporary storage, and lwork specifies the usable memory length. At minimum,
// lwork >= 1, and Dsytrd will panic otherwise. The amount of blocking is
// limited by the usable length.
// If lwork == -1, instead of computing Dsytrd the optimal work length is stored
// into work[0].
//
// Dsytrd is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dsytrd(uplo blas.Uplo, n int, a []float64, lda int, d, e, tau, work []float64, lwork int) {
	checkMatrix(n, n, a, lda)
	if len(d) < n {
		panic(badD)
	}
	if len(e) < n-1 {
		panic(badE)
	}
	if len(tau) < n-1 {
		panic(badTau)
	}
	if len(work) < lwork {
		panic(shortWork)
	}
	if lwork != -1 && lwork < 1 {
		panic(badWork)
	}

	var upper bool
	var opts string
	switch uplo {
	case blas.Upper:
		upper = true
		opts = "U"
	case blas.Lower:
		opts = "L"
	default:
		panic(badUplo)
	}

	if n == 0 {
		work[0] = 1
		return
	}

	nb := impl.Ilaenv(1, "DSYTRD", opts, n, -1, -1, -1)
	lworkopt := n * nb
	if lwork == -1 {
		work[0] = float64(lworkopt)
		return
	}

	nx := n
	bi := blas64.Implementation()
	var ldwork int
	if 1 < nb && nb < n {
		// Determine when to cross over from blocked to unblocked code. The last
		// block is always handled by unblocked code.
		opts := "L"
		if upper {
			opts = "U"
		}
		nx = max(nb, impl.Ilaenv(3, "DSYTRD", opts, n, -1, -1, -1))
		if nx < n {
			// Determine if workspace is large enough for blocked code.
			ldwork = nb
			iws := n * ldwork
			if lwork < iws {
				// Not enough workspace to use optimal nb: determine the minimum
				// value of nb and reduce nb or force use of unblocked code by
				// setting nx = n.
				nb = max(lwork/n, 1)
				nbmin := impl.Ilaenv(2, "DSYTRD", opts, n, -1, -1, -1)
				if nb < nbmin {
					nx = n
				}
			}
		} else {
			nx = n
		}
	} else {
		nb = 1
	}
	ldwork = nb

	if upper {
		// Reduce the upper triangle of A. Columns 0:kk are handled by the
		// unblocked method.
		var i int
		kk := n - ((n-nx+nb-1)/nb)*nb
		for i = n - nb; i >= kk; i -= nb {
			// Reduce columns i:i+nb to tridiagonal form and form the matrix W
			// which is needed to update the unreduced part of the matrix.
			impl.Dlatrd(uplo, i+nb, nb, a, lda, e, tau, work, ldwork)

			// Update the unreduced submatrix A[0:i-1,0:i-1], using an update
			// of the form A = A - V*W^T - W*V^T.
			bi.Dsyr2k(uplo, blas.NoTrans, i, nb, -1, a[i:], lda, work, ldwork, 1, a, lda)

			// Copy superdiagonal elements back into A, and diagonal elements into D.
			for j := i; j < i+nb; j++ {
				a[(j-1)*lda+j] = e[j-1]
				d[j] = a[j*lda+j]
			}
		}
		// Use unblocked code to reduce the last or only block
		// check that i == kk.
		impl.Dsytd2(uplo, kk, a, lda, d, e, tau)
	} else {
		var i int
		// Reduce the lower triangle of A.
		for i = 0; i < n-nx; i += nb {
			// Reduce columns 0:i+nb to tridiagonal form and form the matrix W
			// which is needed to update the unreduced part of the matrix.
			impl.Dlatrd(uplo, n-i, nb, a[i*lda+i:], lda, e[i:], tau[i:], work, ldwork)

			// Update the unreduced submatrix A[i+ib:n, i+ib:n], using an update
			// of the form A = A + V*W^T - W*V^T.
			bi.Dsyr2k(uplo, blas.NoTrans, n-i-nb, nb, -1, a[(i+nb)*lda+i:], lda,
				work[nb*ldwork:], ldwork, 1, a[(i+nb)*lda+i+nb:], lda)

			// Copy subdiagonal elements back into A, and diagonal elements into D.
			for j := i; j < i+nb; j++ {
				a[(j+1)*lda+j] = e[j]
				d[j] = a[j*lda+j]
			}
		}
		// Use unblocked code to reduce the last or only block.
		impl.Dsytd2(uplo, n-i, a[i*lda+i:], lda, d[i:], e[i:], tau[i:])
	}
	work[0] = float64(lworkopt)
}
