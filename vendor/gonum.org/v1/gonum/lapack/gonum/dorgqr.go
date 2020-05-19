// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/lapack"
)

// Dorgqr generates an m×n matrix Q with orthonormal columns defined by the
// product of elementary reflectors
//  Q = H_0 * H_1 * ... * H_{k-1}
// as computed by Dgeqrf.
// Dorgqr is the blocked version of Dorg2r that makes greater use of level-3 BLAS
// routines.
//
// The length of tau must be at least k, and the length of work must be at least n.
// It also must be that 0 <= k <= n and 0 <= n <= m.
//
// work is temporary storage, and lwork specifies the usable memory length. At
// minimum, lwork >= n, and the amount of blocking is limited by the usable
// length. If lwork == -1, instead of computing Dorgqr the optimal work length
// is stored into work[0].
//
// Dorgqr will panic if the conditions on input values are not met.
//
// Dorgqr is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dorgqr(m, n, k int, a []float64, lda int, tau, work []float64, lwork int) {
	switch {
	case m < 0:
		panic(mLT0)
	case n < 0:
		panic(nLT0)
	case n > m:
		panic(nGTM)
	case k < 0:
		panic(kLT0)
	case k > n:
		panic(kGTN)
	case lda < max(1, n) && lwork != -1:
		// Normally, we follow the reference and require the leading
		// dimension to be always valid, even in case of workspace
		// queries. However, if a caller provided a placeholder value
		// for lda (and a) when doing a workspace query that didn't
		// fulfill the condition here, it would cause a panic. This is
		// exactly what Dgesvd does.
		panic(badLdA)
	case lwork < max(1, n) && lwork != -1:
		panic(badLWork)
	case len(work) < max(1, lwork):
		panic(shortWork)
	}

	if n == 0 {
		work[0] = 1
		return
	}

	nb := impl.Ilaenv(1, "DORGQR", " ", m, n, k, -1)
	// work is treated as an n×nb matrix
	if lwork == -1 {
		work[0] = float64(n * nb)
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
	iws := n   // Length of work needed
	var ldwork int
	if 1 < nb && nb < k {
		nx = max(0, impl.Ilaenv(3, "DORGQR", " ", m, n, k, -1))
		if nx < k {
			ldwork = nb
			iws = n * ldwork
			if lwork < iws {
				nb = lwork / n
				ldwork = nb
				nbmin = max(2, impl.Ilaenv(2, "DORGQR", " ", m, n, k, -1))
			}
		}
	}
	var ki, kk int
	if nbmin <= nb && nb < k && nx < k {
		// The first kk columns are handled by the blocked method.
		ki = ((k - nx - 1) / nb) * nb
		kk = min(k, ki+nb)
		for i := 0; i < kk; i++ {
			for j := kk; j < n; j++ {
				a[i*lda+j] = 0
			}
		}
	}
	if kk < n {
		// Perform the operation on colums kk to the end.
		impl.Dorg2r(m-kk, n-kk, k-kk, a[kk*lda+kk:], lda, tau[kk:], work)
	}
	if kk > 0 {
		// Perform the operation on column-blocks.
		for i := ki; i >= 0; i -= nb {
			ib := min(nb, k-i)
			if i+ib < n {
				impl.Dlarft(lapack.Forward, lapack.ColumnWise,
					m-i, ib,
					a[i*lda+i:], lda,
					tau[i:],
					work, ldwork)

				impl.Dlarfb(blas.Left, blas.NoTrans, lapack.Forward, lapack.ColumnWise,
					m-i, n-i-ib, ib,
					a[i*lda+i:], lda,
					work, ldwork,
					a[i*lda+i+ib:], lda,
					work[ib*ldwork:], ldwork)
			}
			impl.Dorg2r(m-i, ib, ib, a[i*lda+i:], lda, tau[i:], work)
			// Set rows 0:i-1 of current block to zero.
			for j := i; j < i+ib; j++ {
				for l := 0; l < i; l++ {
					a[l*lda+j] = 0
				}
			}
		}
	}
	work[0] = float64(iws)
}
