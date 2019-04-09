// Copyright ©2016 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import "gonum.org/v1/gonum/blas"

// Dormhr multiplies an m×n general matrix C with an nq×nq orthogonal matrix Q
//  Q * C,    if side == blas.Left and trans == blas.NoTrans,
//  Q^T * C,  if side == blas.Left and trans == blas.Trans,
//  C * Q,    if side == blas.Right and trans == blas.NoTrans,
//  C * Q^T,  if side == blas.Right and trans == blas.Trans,
// where nq == m if side == blas.Left and nq == n if side == blas.Right.
//
// Q is defined implicitly as the product of ihi-ilo elementary reflectors, as
// returned by Dgehrd:
//  Q = H_{ilo} H_{ilo+1} ... H_{ihi-1}.
// Q is equal to the identity matrix except in the submatrix
// Q[ilo+1:ihi+1,ilo+1:ihi+1].
//
// ilo and ihi must have the same values as in the previous call of Dgehrd. It
// must hold that
//  0 <= ilo <= ihi < m,   if m > 0 and side == blas.Left,
//  ilo = 0 and ihi = -1,  if m = 0 and side == blas.Left,
//  0 <= ilo <= ihi < n,   if n > 0 and side == blas.Right,
//  ilo = 0 and ihi = -1,  if n = 0 and side == blas.Right.
//
// a and lda represent an m×m matrix if side == blas.Left and an n×n matrix if
// side == blas.Right. The matrix contains vectors which define the elementary
// reflectors, as returned by Dgehrd.
//
// tau contains the scalar factors of the elementary reflectors, as returned by
// Dgehrd. tau must have length m-1 if side == blas.Left and n-1 if side ==
// blas.Right.
//
// c and ldc represent the m×n matrix C. On return, c is overwritten by the
// product with Q.
//
// work must have length at least max(1,lwork), and lwork must be at least
// max(1,n), if side == blas.Left, and max(1,m), if side == blas.Right. For
// optimum performance lwork should be at least n*nb if side == blas.Left and
// m*nb if side == blas.Right, where nb is the optimal block size. On return,
// work[0] will contain the optimal value of lwork.
//
// If lwork == -1, instead of performing Dormhr, only the optimal value of lwork
// will be stored in work[0].
//
// If any requirement on input sizes is not met, Dormhr will panic.
//
// Dormhr is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dormhr(side blas.Side, trans blas.Transpose, m, n, ilo, ihi int, a []float64, lda int, tau, c []float64, ldc int, work []float64, lwork int) {
	nq := n // The order of Q.
	nw := m // The minimum length of work.
	if side == blas.Left {
		nq = m
		nw = n
	}
	switch {
	case side != blas.Left && side != blas.Right:
		panic(badSide)
	case trans != blas.NoTrans && trans != blas.Trans:
		panic(badTrans)
	case m < 0:
		panic(mLT0)
	case n < 0:
		panic(nLT0)
	case ilo < 0 || max(1, nq) <= ilo:
		panic(badIlo)
	case ihi < min(ilo, nq-1) || nq <= ihi:
		panic(badIhi)
	case lda < max(1, nq):
		panic(badLdA)
	case lwork < max(1, nw) && lwork != -1:
		panic(badLWork)
	case len(work) < max(1, lwork):
		panic(shortWork)
	}

	// Quick return if possible.
	if m == 0 || n == 0 {
		work[0] = 1
		return
	}

	nh := ihi - ilo
	var nb int
	if side == blas.Left {
		opts := "LN"
		if trans == blas.Trans {
			opts = "LT"
		}
		nb = impl.Ilaenv(1, "DORMQR", opts, nh, n, nh, -1)
	} else {
		opts := "RN"
		if trans == blas.Trans {
			opts = "RT"
		}
		nb = impl.Ilaenv(1, "DORMQR", opts, m, nh, nh, -1)
	}
	lwkopt := max(1, nw) * nb
	if lwork == -1 {
		work[0] = float64(lwkopt)
		return
	}

	if nh == 0 {
		work[0] = 1
		return
	}

	switch {
	case len(a) < (nq-1)*lda+nq:
		panic(shortA)
	case len(c) < (m-1)*ldc+n:
		panic(shortC)
	case len(tau) != nq-1:
		panic(badLenTau)
	}

	if side == blas.Left {
		impl.Dormqr(side, trans, nh, n, nh, a[(ilo+1)*lda+ilo:], lda,
			tau[ilo:ihi], c[(ilo+1)*ldc:], ldc, work, lwork)
	} else {
		impl.Dormqr(side, trans, m, nh, nh, a[(ilo+1)*lda+ilo:], lda,
			tau[ilo:ihi], c[ilo+1:], ldc, work, lwork)
	}
	work[0] = float64(lwkopt)
}
