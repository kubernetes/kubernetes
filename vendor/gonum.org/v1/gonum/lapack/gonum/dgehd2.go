// Copyright ©2016 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import "gonum.org/v1/gonum/blas"

// Dgehd2 reduces a block of a general n×n matrix A to upper Hessenberg form H
// by an orthogonal similarity transformation Qᵀ * A * Q = H.
//
// The matrix Q is represented as a product of (ihi-ilo) elementary
// reflectors
//  Q = H_{ilo} H_{ilo+1} ... H_{ihi-1}.
// Each H_i has the form
//  H_i = I - tau[i] * v * vᵀ
// where v is a real vector with v[0:i+1] = 0, v[i+1] = 1 and v[ihi+1:n] = 0.
// v[i+2:ihi+1] is stored on exit in A[i+2:ihi+1,i].
//
// On entry, a contains the n×n general matrix to be reduced. On return, the
// upper triangle and the first subdiagonal of A are overwritten with the upper
// Hessenberg matrix H, and the elements below the first subdiagonal, with the
// slice tau, represent the orthogonal matrix Q as a product of elementary
// reflectors.
//
// The contents of A are illustrated by the following example, with n = 7, ilo =
// 1 and ihi = 5.
// On entry,
//  [ a   a   a   a   a   a   a ]
//  [     a   a   a   a   a   a ]
//  [     a   a   a   a   a   a ]
//  [     a   a   a   a   a   a ]
//  [     a   a   a   a   a   a ]
//  [     a   a   a   a   a   a ]
//  [                         a ]
// on return,
//  [ a   a   h   h   h   h   a ]
//  [     a   h   h   h   h   a ]
//  [     h   h   h   h   h   h ]
//  [     v1  h   h   h   h   h ]
//  [     v1  v2  h   h   h   h ]
//  [     v1  v2  v3  h   h   h ]
//  [                         a ]
// where a denotes an element of the original matrix A, h denotes a
// modified element of the upper Hessenberg matrix H, and vi denotes an
// element of the vector defining H_i.
//
// ilo and ihi determine the block of A that will be reduced to upper Hessenberg
// form. It must hold that 0 <= ilo <= ihi <= max(0, n-1), otherwise Dgehd2 will
// panic.
//
// On return, tau will contain the scalar factors of the elementary reflectors.
// It must have length equal to n-1, otherwise Dgehd2 will panic.
//
// work must have length at least n, otherwise Dgehd2 will panic.
//
// Dgehd2 is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dgehd2(n, ilo, ihi int, a []float64, lda int, tau, work []float64) {
	switch {
	case n < 0:
		panic(nLT0)
	case ilo < 0 || max(0, n-1) < ilo:
		panic(badIlo)
	case ihi < min(ilo, n-1) || n <= ihi:
		panic(badIhi)
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
	case len(tau) != n-1:
		panic(badLenTau)
	case len(work) < n:
		panic(shortWork)
	}

	for i := ilo; i < ihi; i++ {
		// Compute elementary reflector H_i to annihilate A[i+2:ihi+1,i].
		var aii float64
		aii, tau[i] = impl.Dlarfg(ihi-i, a[(i+1)*lda+i], a[min(i+2, n-1)*lda+i:], lda)
		a[(i+1)*lda+i] = 1

		// Apply H_i to A[0:ihi+1,i+1:ihi+1] from the right.
		impl.Dlarf(blas.Right, ihi+1, ihi-i, a[(i+1)*lda+i:], lda, tau[i], a[i+1:], lda, work)

		// Apply H_i to A[i+1:ihi+1,i+1:n] from the left.
		impl.Dlarf(blas.Left, ihi-i, n-i-1, a[(i+1)*lda+i:], lda, tau[i], a[(i+1)*lda+i+1:], lda, work)
		a[(i+1)*lda+i] = aii
	}
}
