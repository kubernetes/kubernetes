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
	nb := impl.Ilaenv(1, "DORGQR", " ", m, n, k, -1)
	// work is treated as an n×nb matrix
	if lwork == -1 {
		work[0] = float64(max(1, n) * nb)
		return
	}
	checkMatrix(m, n, a, lda)
	if k < 0 {
		panic(kLT0)
	}
	if k > n {
		panic(kGTN)
	}
	if n > m {
		panic(mLTN)
	}
	if len(tau) < k {
		panic(badTau)
	}
	if len(work) < lwork {
		panic(shortWork)
	}
	if lwork < n {
		panic(badWork)
	}
	if n == 0 {
		return
	}
	nbmin := 2 // Minimum number of blocks
	var nx int // Minimum number of rows
	iws := n   // Length of work needed
	var ldwork int
	if nb > 1 && nb < k {
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
	if nb >= nbmin && nb < k && nx < k {
		// The first kk columns are handled by the blocked method.
		// Note: lapack has nx here, but this means the last nx rows are handled
		// serially which could be quite different than nb.
		ki = ((k - nb - 1) / nb) * nb
		kk = min(k, ki+nb)
		for j := kk; j < n; j++ {
			for i := 0; i < kk; i++ {
				a[i*lda+j] = 0
			}
		}
	}
	if kk < n {
		// Perform the operation on colums kk to the end.
		impl.Dorg2r(m-kk, n-kk, k-kk, a[kk*lda+kk:], lda, tau[kk:], work)
	}
	if kk == 0 {
		return
	}
	// Perform the operation on column-blocks
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
		// Set rows 0:i-1 of current block to zero
		for j := i; j < i+ib; j++ {
			for l := 0; l < i; l++ {
				a[l*lda+j] = 0
			}
		}
	}
}
