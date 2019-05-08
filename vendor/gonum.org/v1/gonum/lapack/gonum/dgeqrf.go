// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/lapack"
)

// Dgeqrf computes the QR factorization of the m×n matrix A using a blocked
// algorithm. See the documentation for Dgeqr2 for a description of the
// parameters at entry and exit.
//
// work is temporary storage, and lwork specifies the usable memory length.
// The length of work must be at least max(1, lwork) and lwork must be -1
// or at least n, otherwise this function will panic.
// Dgeqrf is a blocked QR factorization, but the block size is limited
// by the temporary space available. If lwork == -1, instead of performing Dgeqrf,
// the optimal work length will be stored into work[0].
//
// tau must have length at least min(m,n), and this function will panic otherwise.
func (impl Implementation) Dgeqrf(m, n int, a []float64, lda int, tau, work []float64, lwork int) {
	switch {
	case m < 0:
		panic(mLT0)
	case n < 0:
		panic(nLT0)
	case lda < max(1, n):
		panic(badLdA)
	case lwork < max(1, n) && lwork != -1:
		panic(badLWork)
	case len(work) < max(1, lwork):
		panic(shortWork)
	}

	// Quick return if possible.
	k := min(m, n)
	if k == 0 {
		work[0] = 1
		return
	}

	// nb is the optimal blocksize, i.e. the number of columns transformed at a time.
	nb := impl.Ilaenv(1, "DGEQRF", " ", m, n, -1, -1)
	if lwork == -1 {
		work[0] = float64(n * nb)
		return
	}

	if len(a) < (m-1)*lda+n {
		panic(shortA)
	}
	if len(tau) < k {
		panic(shortTau)
	}

	nbmin := 2 // Minimal block size.
	var nx int // Use unblocked (unless changed in the next for loop)
	iws := n
	// Only consider blocked if the suggested block size is > 1 and the
	// number of rows or columns is sufficiently large.
	if 1 < nb && nb < k {
		// nx is the block size at which the code switches from blocked
		// to unblocked.
		nx = max(0, impl.Ilaenv(3, "DGEQRF", " ", m, n, -1, -1))
		if k > nx {
			iws = n * nb
			if lwork < iws {
				// Not enough workspace to use the optimal block
				// size. Get the minimum block size instead.
				nb = lwork / n
				nbmin = max(2, impl.Ilaenv(2, "DGEQRF", " ", m, n, -1, -1))
			}
		}
	}

	// Compute QR using a blocked algorithm.
	var i int
	if nbmin <= nb && nb < k && nx < k {
		ldwork := nb
		for i = 0; i < k-nx; i += nb {
			ib := min(k-i, nb)
			// Compute the QR factorization of the current block.
			impl.Dgeqr2(m-i, ib, a[i*lda+i:], lda, tau[i:], work)
			if i+ib < n {
				// Form the triangular factor of the block reflector and apply H^T
				// In Dlarft, work becomes the T matrix.
				impl.Dlarft(lapack.Forward, lapack.ColumnWise, m-i, ib,
					a[i*lda+i:], lda,
					tau[i:],
					work, ldwork)
				impl.Dlarfb(blas.Left, blas.Trans, lapack.Forward, lapack.ColumnWise,
					m-i, n-i-ib, ib,
					a[i*lda+i:], lda,
					work, ldwork,
					a[i*lda+i+ib:], lda,
					work[ib*ldwork:], ldwork)
			}
		}
	}
	// Call unblocked code on the remaining columns.
	if i < k {
		impl.Dgeqr2(m-i, n-i, a[i*lda+i:], lda, tau[i:], work)
	}
	work[0] = float64(iws)
}
