// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/lapack"
)

// Dgelqf computes the LQ factorization of the m×n matrix A using a blocked
// algorithm. See the documentation for Dgelq2 for a description of the
// parameters at entry and exit.
//
// work is temporary storage, and lwork specifies the usable memory length.
// At minimum, lwork >= m, and this function will panic otherwise.
// Dgelqf is a blocked LQ factorization, but the block size is limited
// by the temporary space available. If lwork == -1, instead of performing Dgelqf,
// the optimal work length will be stored into work[0].
//
// tau must have length at least min(m,n), and this function will panic otherwise.
func (impl Implementation) Dgelqf(m, n int, a []float64, lda int, tau, work []float64, lwork int) {
	switch {
	case m < 0:
		panic(mLT0)
	case n < 0:
		panic(nLT0)
	case lda < max(1, n):
		panic(badLdA)
	case lwork < max(1, m) && lwork != -1:
		panic(badLWork)
	case len(work) < max(1, lwork):
		panic(shortWork)
	}

	k := min(m, n)
	if k == 0 {
		work[0] = 1
		return
	}

	nb := impl.Ilaenv(1, "DGELQF", " ", m, n, -1, -1)
	if lwork == -1 {
		work[0] = float64(m * nb)
		return
	}

	if len(a) < (m-1)*lda+n {
		panic(shortA)
	}
	if len(tau) < k {
		panic(shortTau)
	}

	// Find the optimal blocking size based on the size of available memory
	// and optimal machine parameters.
	nbmin := 2
	var nx int
	iws := m
	if 1 < nb && nb < k {
		nx = max(0, impl.Ilaenv(3, "DGELQF", " ", m, n, -1, -1))
		if nx < k {
			iws = m * nb
			if lwork < iws {
				nb = lwork / m
				nbmin = max(2, impl.Ilaenv(2, "DGELQF", " ", m, n, -1, -1))
			}
		}
	}
	ldwork := nb
	// Computed blocked LQ factorization.
	var i int
	if nbmin <= nb && nb < k && nx < k {
		for i = 0; i < k-nx; i += nb {
			ib := min(k-i, nb)
			impl.Dgelq2(ib, n-i, a[i*lda+i:], lda, tau[i:], work)
			if i+ib < m {
				impl.Dlarft(lapack.Forward, lapack.RowWise, n-i, ib,
					a[i*lda+i:], lda,
					tau[i:],
					work, ldwork)
				impl.Dlarfb(blas.Right, blas.NoTrans, lapack.Forward, lapack.RowWise,
					m-i-ib, n-i, ib,
					a[i*lda+i:], lda,
					work, ldwork,
					a[(i+ib)*lda+i:], lda,
					work[ib*ldwork:], ldwork)
			}
		}
	}
	// Perform unblocked LQ factorization on the remainder.
	if i < k {
		impl.Dgelq2(m-i, n-i, a[i*lda+i:], lda, tau[i:], work)
	}
	work[0] = float64(iws)
}
