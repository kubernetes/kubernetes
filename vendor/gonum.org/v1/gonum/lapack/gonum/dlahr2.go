// Copyright ©2016 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
)

// Dlahr2 reduces the first nb columns of a real general n×(n-k+1) matrix A so
// that elements below the k-th subdiagonal are zero. The reduction is performed
// by an orthogonal similarity transformation Q^T * A * Q. Dlahr2 returns the
// matrices V and T which determine Q as a block reflector I - V*T*V^T, and
// also the matrix Y = A * V * T.
//
// The matrix Q is represented as a product of nb elementary reflectors
//  Q = H_0 * H_1 * ... * H_{nb-1}.
// Each H_i has the form
//  H_i = I - tau[i] * v * v^T,
// where v is a real vector with v[0:i+k-1] = 0 and v[i+k-1] = 1. v[i+k:n] is
// stored on exit in A[i+k+1:n,i].
//
// The elements of the vectors v together form the (n-k+1)×nb matrix
// V which is needed, with T and Y, to apply the transformation to the
// unreduced part of the matrix, using an update of the form
//  A = (I - V*T*V^T) * (A - Y*V^T).
//
// On entry, a contains the n×(n-k+1) general matrix A. On return, the elements
// on and above the k-th subdiagonal in the first nb columns are overwritten
// with the corresponding elements of the reduced matrix; the elements below the
// k-th subdiagonal, with the slice tau, represent the matrix Q as a product of
// elementary reflectors. The other columns of A are unchanged.
//
// The contents of A on exit are illustrated by the following example
// with n = 7, k = 3 and nb = 2:
//  [ a   a   a   a   a ]
//  [ a   a   a   a   a ]
//  [ a   a   a   a   a ]
//  [ h   h   a   a   a ]
//  [ v0  h   a   a   a ]
//  [ v0  v1  a   a   a ]
//  [ v0  v1  a   a   a ]
// where a denotes an element of the original matrix A, h denotes a
// modified element of the upper Hessenberg matrix H, and vi denotes an
// element of the vector defining H_i.
//
// k is the offset for the reduction. Elements below the k-th subdiagonal in the
// first nb columns are reduced to zero.
//
// nb is the number of columns to be reduced.
//
// On entry, a represents the n×(n-k+1) matrix A. On return, the elements on and
// above the k-th subdiagonal in the first nb columns are overwritten with the
// corresponding elements of the reduced matrix. The elements below the k-th
// subdiagonal, with the slice tau, represent the matrix Q as a product of
// elementary reflectors. The other columns of A are unchanged.
//
// tau will contain the scalar factors of the elementary reflectors. It must
// have length at least nb.
//
// t and ldt represent the nb×nb upper triangular matrix T, and y and ldy
// represent the n×nb matrix Y.
//
// Dlahr2 is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dlahr2(n, k, nb int, a []float64, lda int, tau, t []float64, ldt int, y []float64, ldy int) {
	checkMatrix(n, n-k+1, a, lda)
	if len(tau) < nb {
		panic(badTau)
	}
	checkMatrix(nb, nb, t, ldt)
	checkMatrix(n, nb, y, ldy)

	// Quick return if possible.
	if n <= 1 {
		return
	}

	bi := blas64.Implementation()
	var ei float64
	for i := 0; i < nb; i++ {
		if i > 0 {
			// Update A[k:n,i].

			// Update i-th column of A - Y * V^T.
			bi.Dgemv(blas.NoTrans, n-k, i,
				-1, y[k*ldy:], ldy,
				a[(k+i-1)*lda:], 1,
				1, a[k*lda+i:], lda)

			// Apply I - V * T^T * V^T to this column (call it b)
			// from the left, using the last column of T as
			// workspace.
			// Let V = [ V1 ]   and   b = [ b1 ]   (first i rows)
			//         [ V2 ]             [ b2 ]
			// where V1 is unit lower triangular.
			//
			// w := V1^T * b1.
			bi.Dcopy(i, a[k*lda+i:], lda, t[nb-1:], ldt)
			bi.Dtrmv(blas.Lower, blas.Trans, blas.Unit, i,
				a[k*lda:], lda, t[nb-1:], ldt)

			// w := w + V2^T * b2.
			bi.Dgemv(blas.Trans, n-k-i, i,
				1, a[(k+i)*lda:], lda,
				a[(k+i)*lda+i:], lda,
				1, t[nb-1:], ldt)

			// w := T^T * w.
			bi.Dtrmv(blas.Upper, blas.Trans, blas.NonUnit, i,
				t, ldt, t[nb-1:], ldt)

			// b2 := b2 - V2*w.
			bi.Dgemv(blas.NoTrans, n-k-i, i,
				-1, a[(k+i)*lda:], lda,
				t[nb-1:], ldt,
				1, a[(k+i)*lda+i:], lda)

			// b1 := b1 - V1*w.
			bi.Dtrmv(blas.Lower, blas.NoTrans, blas.Unit, i,
				a[k*lda:], lda, t[nb-1:], ldt)
			bi.Daxpy(i, -1, t[nb-1:], ldt, a[k*lda+i:], lda)

			a[(k+i-1)*lda+i-1] = ei
		}

		// Generate the elementary reflector H_i to annihilate
		// A[k+i+1:n,i].
		ei, tau[i] = impl.Dlarfg(n-k-i, a[(k+i)*lda+i], a[min(k+i+1, n-1)*lda+i:], lda)
		a[(k+i)*lda+i] = 1

		// Compute Y[k:n,i].
		bi.Dgemv(blas.NoTrans, n-k, n-k-i,
			1, a[k*lda+i+1:], lda,
			a[(k+i)*lda+i:], lda,
			0, y[k*ldy+i:], ldy)
		bi.Dgemv(blas.Trans, n-k-i, i,
			1, a[(k+i)*lda:], lda,
			a[(k+i)*lda+i:], lda,
			0, t[i:], ldt)
		bi.Dgemv(blas.NoTrans, n-k, i,
			-1, y[k*ldy:], ldy,
			t[i:], ldt,
			1, y[k*ldy+i:], ldy)
		bi.Dscal(n-k, tau[i], y[k*ldy+i:], ldy)

		// Compute T[0:i,i].
		bi.Dscal(i, -tau[i], t[i:], ldt)
		bi.Dtrmv(blas.Upper, blas.NoTrans, blas.NonUnit, i,
			t, ldt, t[i:], ldt)

		t[i*ldt+i] = tau[i]
	}
	a[(k+nb-1)*lda+nb-1] = ei

	// Compute Y[0:k,0:nb].
	impl.Dlacpy(blas.All, k, nb, a[1:], lda, y, ldy)
	bi.Dtrmm(blas.Right, blas.Lower, blas.NoTrans, blas.Unit, k, nb,
		1, a[k*lda:], lda, y, ldy)
	if n > k+nb {
		bi.Dgemm(blas.NoTrans, blas.NoTrans, k, nb, n-k-nb,
			1, a[1+nb:], lda,
			a[(k+nb)*lda:], lda,
			1, y, ldy)
	}
	bi.Dtrmm(blas.Right, blas.Upper, blas.NoTrans, blas.NonUnit, k, nb,
		1, t, ldt, y, ldy)
}
