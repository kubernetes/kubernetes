// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/lapack"
)

// Dorglq generates an m×n matrix Q with orthonormal columns defined by the
// product of elementary reflectors as computed by Dgelqf.
//  Q = H_0 * H_1 * ... * H_{k-1}
// Dorglq is the blocked version of Dorgl2 that makes greater use of level-3 BLAS
// routines.
//
// len(tau) >= k, 0 <= k <= m, and 0 <= m <= n.
//
// work is temporary storage, and lwork specifies the usable memory length. At minimum,
// lwork >= m, and the amount of blocking is limited by the usable length.
// If lwork == -1, instead of computing Dorglq the optimal work length is stored
// into work[0].
//
// Dorglq will panic if the conditions on input values are not met.
//
// Dorglq is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dorglq(m, n, k int, a []float64, lda int, tau, work []float64, lwork int) {
	switch {
	case m < 0:
		panic(mLT0)
	case n < m:
		panic(nLTM)
	case k < 0:
		panic(kLT0)
	case k > m:
		panic(kGTM)
	case lda < max(1, n):
		panic(badLdA)
	case lwork < max(1, m) && lwork != -1:
		panic(badLWork)
	case len(work) < max(1, lwork):
		panic(shortWork)
	}

	if m == 0 {
		work[0] = 1
		return
	}

	nb := impl.Ilaenv(1, "DORGLQ", " ", m, n, k, -1)
	if lwork == -1 {
		work[0] = float64(m * nb)
		return
	}

	switch {
	case len(a) < (m-1)*lda+n:
		panic(shortA)
	case len(tau) < k:
		panic(shortTau)
	}

	nbmin := 2 // Minimum block size
	var nx int // Crossover size from blocked to unbloked code
	iws := m   // Length of work needed
	var ldwork int
	if 1 < nb && nb < k {
		nx = max(0, impl.Ilaenv(3, "DORGLQ", " ", m, n, k, -1))
		if nx < k {
			ldwork = nb
			iws = m * ldwork
			if lwork < iws {
				nb = lwork / m
				ldwork = nb
				nbmin = max(2, impl.Ilaenv(2, "DORGLQ", " ", m, n, k, -1))
			}
		}
	}

	var ki, kk int
	if nbmin <= nb && nb < k && nx < k {
		// The first kk rows are handled by the blocked method.
		ki = ((k - nx - 1) / nb) * nb
		kk = min(k, ki+nb)
		for i := kk; i < m; i++ {
			for j := 0; j < kk; j++ {
				a[i*lda+j] = 0
			}
		}
	}
	if kk < m {
		// Perform the operation on colums kk to the end.
		impl.Dorgl2(m-kk, n-kk, k-kk, a[kk*lda+kk:], lda, tau[kk:], work)
	}
	if kk > 0 {
		// Perform the operation on column-blocks
		for i := ki; i >= 0; i -= nb {
			ib := min(nb, k-i)
			if i+ib < m {
				impl.Dlarft(lapack.Forward, lapack.RowWise,
					n-i, ib,
					a[i*lda+i:], lda,
					tau[i:],
					work, ldwork)

				impl.Dlarfb(blas.Right, blas.Trans, lapack.Forward, lapack.RowWise,
					m-i-ib, n-i, ib,
					a[i*lda+i:], lda,
					work, ldwork,
					a[(i+ib)*lda+i:], lda,
					work[ib*ldwork:], ldwork)
			}
			impl.Dorgl2(ib, n-i, ib, a[i*lda+i:], lda, tau[i:], work)
			for l := i; l < i+ib; l++ {
				for j := 0; j < i; j++ {
					a[l*lda+j] = 0
				}
			}
		}
	}
	work[0] = float64(iws)
}
