// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/lapack"
)

// Dormbr applies a multiplicative update to the matrix C based on a
// decomposition computed by Dgebrd.
//
// Dormbr overwrites the m×n matrix C with
//  Q * C   if vect == lapack.ApplyQ, side == blas.Left, and trans == blas.NoTrans
//  C * Q   if vect == lapack.ApplyQ, side == blas.Right, and trans == blas.NoTrans
//  Q^T * C if vect == lapack.ApplyQ, side == blas.Left, and trans == blas.Trans
//  C * Q^T if vect == lapack.ApplyQ, side == blas.Right, and trans == blas.Trans
//
//  P * C   if vect == lapack.ApplyP, side == blas.Left, and trans == blas.NoTrans
//  C * P   if vect == lapack.ApplyP, side == blas.Right, and trans == blas.NoTrans
//  P^T * C if vect == lapack.ApplyP, side == blas.Left, and trans == blas.Trans
//  C * P^T if vect == lapack.ApplyP, side == blas.Right, and trans == blas.Trans
// where P and Q are the orthogonal matrices determined by Dgebrd when reducing
// a matrix A to bidiagonal form: A = Q * B * P^T. See Dgebrd for the
// definitions of Q and P.
//
// If vect == lapack.ApplyQ, A is assumed to have been an nq×k matrix, while if
// vect == lapack.ApplyP, A is assumed to have been a k×nq matrix. nq = m if
// side == blas.Left, while nq = n if side == blas.Right.
//
// tau must have length min(nq,k), and Dormbr will panic otherwise. tau contains
// the elementary reflectors to construct Q or P depending on the value of
// vect.
//
// work must have length at least max(1,lwork), and lwork must be either -1 or
// at least max(1,n) if side == blas.Left, and at least max(1,m) if side ==
// blas.Right. For optimum performance lwork should be at least n*nb if side ==
// blas.Left, and at least m*nb if side == blas.Right, where nb is the optimal
// block size. On return, work[0] will contain the optimal value of lwork.
//
// If lwork == -1, the function only calculates the optimal value of lwork and
// returns it in work[0].
//
// Dormbr is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dormbr(vect lapack.ApplyOrtho, side blas.Side, trans blas.Transpose, m, n, k int, a []float64, lda int, tau, c []float64, ldc int, work []float64, lwork int) {
	nq := n
	nw := m
	if side == blas.Left {
		nq = m
		nw = n
	}
	applyQ := vect == lapack.ApplyQ
	switch {
	case !applyQ && vect != lapack.ApplyP:
		panic(badApplyOrtho)
	case side != blas.Left && side != blas.Right:
		panic(badSide)
	case trans != blas.NoTrans && trans != blas.Trans:
		panic(badTrans)
	case m < 0:
		panic(mLT0)
	case n < 0:
		panic(nLT0)
	case k < 0:
		panic(kLT0)
	case applyQ && lda < max(1, min(nq, k)):
		panic(badLdA)
	case !applyQ && lda < max(1, nq):
		panic(badLdA)
	case ldc < max(1, n):
		panic(badLdC)
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

	// The current implementation does not use opts, but a future change may
	// use these options so construct them.
	var opts string
	if side == blas.Left {
		opts = "L"
	} else {
		opts = "R"
	}
	if trans == blas.Trans {
		opts += "T"
	} else {
		opts += "N"
	}
	var nb int
	if applyQ {
		if side == blas.Left {
			nb = impl.Ilaenv(1, "DORMQR", opts, m-1, n, m-1, -1)
		} else {
			nb = impl.Ilaenv(1, "DORMQR", opts, m, n-1, n-1, -1)
		}
	} else {
		if side == blas.Left {
			nb = impl.Ilaenv(1, "DORMLQ", opts, m-1, n, m-1, -1)
		} else {
			nb = impl.Ilaenv(1, "DORMLQ", opts, m, n-1, n-1, -1)
		}
	}
	lworkopt := max(1, nw) * nb
	if lwork == -1 {
		work[0] = float64(lworkopt)
		return
	}

	minnqk := min(nq, k)
	switch {
	case applyQ && len(a) < (nq-1)*lda+minnqk:
		panic(shortA)
	case !applyQ && len(a) < (minnqk-1)*lda+nq:
		panic(shortA)
	case len(tau) < minnqk:
		panic(shortTau)
	case len(c) < (m-1)*ldc+n:
		panic(shortC)
	}

	if applyQ {
		// Change the operation to get Q depending on the size of the initial
		// matrix to Dgebrd. The size matters due to the storage location of
		// the off-diagonal elements.
		if nq >= k {
			impl.Dormqr(side, trans, m, n, k, a, lda, tau[:k], c, ldc, work, lwork)
		} else if nq > 1 {
			mi := m
			ni := n - 1
			i1 := 0
			i2 := 1
			if side == blas.Left {
				mi = m - 1
				ni = n
				i1 = 1
				i2 = 0
			}
			impl.Dormqr(side, trans, mi, ni, nq-1, a[1*lda:], lda, tau[:nq-1], c[i1*ldc+i2:], ldc, work, lwork)
		}
		work[0] = float64(lworkopt)
		return
	}

	transt := blas.Trans
	if trans == blas.Trans {
		transt = blas.NoTrans
	}

	// Change the operation to get P depending on the size of the initial
	// matrix to Dgebrd. The size matters due to the storage location of
	// the off-diagonal elements.
	if nq > k {
		impl.Dormlq(side, transt, m, n, k, a, lda, tau, c, ldc, work, lwork)
	} else if nq > 1 {
		mi := m
		ni := n - 1
		i1 := 0
		i2 := 1
		if side == blas.Left {
			mi = m - 1
			ni = n
			i1 = 1
			i2 = 0
		}
		impl.Dormlq(side, transt, mi, ni, nq-1, a[1:], lda, tau, c[i1*ldc+i2:], ldc, work, lwork)
	}
	work[0] = float64(lworkopt)
}
