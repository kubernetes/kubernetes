// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/lapack"
)

// Dormlq multiplies the matrix C by the orthogonal matrix Q defined by the
// slices a and tau. A and tau are as returned from Dgelqf.
//  C = Q * C    if side == blas.Left and trans == blas.NoTrans
//  C = Q^T * C  if side == blas.Left and trans == blas.Trans
//  C = C * Q    if side == blas.Right and trans == blas.NoTrans
//  C = C * Q^T  if side == blas.Right and trans == blas.Trans
// If side == blas.Left, A is a matrix of side k×m, and if side == blas.Right
// A is of size k×n. This uses a blocked algorithm.
//
// work is temporary storage, and lwork specifies the usable memory length.
// At minimum, lwork >= m if side == blas.Left and lwork >= n if side == blas.Right,
// and this function will panic otherwise.
// Dormlq uses a block algorithm, but the block size is limited
// by the temporary space available. If lwork == -1, instead of performing Dormlq,
// the optimal work length will be stored into work[0].
//
// tau contains the Householder scales and must have length at least k, and
// this function will panic otherwise.
func (impl Implementation) Dormlq(side blas.Side, trans blas.Transpose, m, n, k int, a []float64, lda int, tau, c []float64, ldc int, work []float64, lwork int) {
	left := side == blas.Left
	nw := m
	if left {
		nw = n
	}
	switch {
	case !left && side != blas.Right:
		panic(badSide)
	case trans != blas.Trans && trans != blas.NoTrans:
		panic(badTrans)
	case m < 0:
		panic(mLT0)
	case n < 0:
		panic(nLT0)
	case k < 0:
		panic(kLT0)
	case left && k > m:
		panic(kGTM)
	case !left && k > n:
		panic(kGTN)
	case left && lda < max(1, m):
		panic(badLdA)
	case !left && lda < max(1, n):
		panic(badLdA)
	case lwork < max(1, nw) && lwork != -1:
		panic(badLWork)
	case len(work) < max(1, lwork):
		panic(shortWork)
	}

	// Quick return if possible.
	if m == 0 || n == 0 || k == 0 {
		work[0] = 1
		return
	}

	const (
		nbmax = 64
		ldt   = nbmax
		tsize = nbmax * ldt
	)
	opts := string(side) + string(trans)
	nb := min(nbmax, impl.Ilaenv(1, "DORMLQ", opts, m, n, k, -1))
	lworkopt := max(1, nw)*nb + tsize
	if lwork == -1 {
		work[0] = float64(lworkopt)
		return
	}

	switch {
	case left && len(a) < (k-1)*lda+m:
		panic(shortA)
	case !left && len(a) < (k-1)*lda+n:
		panic(shortA)
	case len(tau) < k:
		panic(shortTau)
	case len(c) < (m-1)*ldc+n:
		panic(shortC)
	}

	nbmin := 2
	if 1 < nb && nb < k {
		iws := nw*nb + tsize
		if lwork < iws {
			nb = (lwork - tsize) / nw
			nbmin = max(2, impl.Ilaenv(2, "DORMLQ", opts, m, n, k, -1))
		}
	}
	if nb < nbmin || k <= nb {
		// Call unblocked code.
		impl.Dorml2(side, trans, m, n, k, a, lda, tau, c, ldc, work)
		work[0] = float64(lworkopt)
		return
	}

	t := work[:tsize]
	wrk := work[tsize:]
	ldwrk := nb

	notrans := trans == blas.NoTrans
	transt := blas.NoTrans
	if notrans {
		transt = blas.Trans
	}

	switch {
	case left && notrans:
		for i := 0; i < k; i += nb {
			ib := min(nb, k-i)
			impl.Dlarft(lapack.Forward, lapack.RowWise, m-i, ib,
				a[i*lda+i:], lda,
				tau[i:],
				t, ldt)
			impl.Dlarfb(side, transt, lapack.Forward, lapack.RowWise, m-i, n, ib,
				a[i*lda+i:], lda,
				t, ldt,
				c[i*ldc:], ldc,
				wrk, ldwrk)
		}

	case left && !notrans:
		for i := ((k - 1) / nb) * nb; i >= 0; i -= nb {
			ib := min(nb, k-i)
			impl.Dlarft(lapack.Forward, lapack.RowWise, m-i, ib,
				a[i*lda+i:], lda,
				tau[i:],
				t, ldt)
			impl.Dlarfb(side, transt, lapack.Forward, lapack.RowWise, m-i, n, ib,
				a[i*lda+i:], lda,
				t, ldt,
				c[i*ldc:], ldc,
				wrk, ldwrk)
		}

	case !left && notrans:
		for i := ((k - 1) / nb) * nb; i >= 0; i -= nb {
			ib := min(nb, k-i)
			impl.Dlarft(lapack.Forward, lapack.RowWise, n-i, ib,
				a[i*lda+i:], lda,
				tau[i:],
				t, ldt)
			impl.Dlarfb(side, transt, lapack.Forward, lapack.RowWise, m, n-i, ib,
				a[i*lda+i:], lda,
				t, ldt,
				c[i:], ldc,
				wrk, ldwrk)
		}

	case !left && !notrans:
		for i := 0; i < k; i += nb {
			ib := min(nb, k-i)
			impl.Dlarft(lapack.Forward, lapack.RowWise, n-i, ib,
				a[i*lda+i:], lda,
				tau[i:],
				t, ldt)
			impl.Dlarfb(side, transt, lapack.Forward, lapack.RowWise, m, n-i, ib,
				a[i*lda+i:], lda,
				t, ldt,
				c[i:], ldc,
				wrk, ldwrk)
		}
	}
	work[0] = float64(lworkopt)
}
