// Copyright ©2016 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/lapack"
)

// Dorgql generates the m×n matrix Q with orthonormal columns defined as the
// last n columns of a product of k elementary reflectors of order m
//  Q = H_{k-1} * ... * H_1 * H_0.
//
// It must hold that
//  0 <= k <= n <= m,
// and Dorgql will panic otherwise.
//
// On entry, the (n-k+i)-th column of A must contain the vector which defines
// the elementary reflector H_i, for i=0,...,k-1, and tau[i] must contain its
// scalar factor. On return, a contains the m×n matrix Q.
//
// tau must have length at least k, and Dorgql will panic otherwise.
//
// work must have length at least max(1,lwork), and lwork must be at least
// max(1,n), otherwise Dorgql will panic. For optimum performance lwork must
// be a sufficiently large multiple of n.
//
// If lwork == -1, instead of computing Dorgql the optimal work length is stored
// into work[0].
//
// Dorgql is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dorgql(m, n, k int, a []float64, lda int, tau, work []float64, lwork int) {
	switch {
	case n < 0:
		panic(nLT0)
	case m < n:
		panic(mLTN)
	case k < 0:
		panic(kLT0)
	case k > n:
		panic(kGTN)
	case lwork < max(1, n) && lwork != -1:
		panic(badWork)
	case len(work) < lwork:
		panic(shortWork)
	}
	if lwork != -1 {
		checkMatrix(m, n, a, lda)
		if len(tau) < k {
			panic(badTau)
		}
	}

	if n == 0 {
		work[0] = 1
		return
	}

	nb := impl.Ilaenv(1, "DORGQL", " ", m, n, k, -1)
	if lwork == -1 {
		work[0] = float64(n * nb)
		return
	}

	nbmin := 2
	var nx, ldwork int
	iws := n
	if nb > 1 && nb < k {
		// Determine when to cross over from blocked to unblocked code.
		nx = max(0, impl.Ilaenv(3, "DORGQL", " ", m, n, k, -1))
		if nx < k {
			// Determine if workspace is large enough for blocked code.
			iws = n * nb
			if lwork < iws {
				// Not enough workspace to use optimal nb: reduce nb and determine
				// the minimum value of nb.
				nb = lwork / n
				nbmin = max(2, impl.Ilaenv(2, "DORGQL", " ", m, n, k, -1))
			}
			ldwork = nb
		}
	}

	var kk int
	if nb >= nbmin && nb < k && nx < k {
		// Use blocked code after the first block. The last kk columns are handled
		// by the block method.
		kk = min(k, ((k-nx+nb-1)/nb)*nb)

		// Set A(m-kk:m, 0:n-kk) to zero.
		for i := m - kk; i < m; i++ {
			for j := 0; j < n-kk; j++ {
				a[i*lda+j] = 0
			}
		}
	}

	// Use unblocked code for the first or only block.
	impl.Dorg2l(m-kk, n-kk, k-kk, a, lda, tau, work)
	if kk > 0 {
		// Use blocked code.
		for i := k - kk; i < k; i += nb {
			ib := min(nb, k-i)
			if n-k+i > 0 {
				// Form the triangular factor of the block reflector
				// H = H_{i+ib-1} * ... * H_{i+1} * H_i.
				impl.Dlarft(lapack.Backward, lapack.ColumnWise, m-k+i+ib, ib,
					a[n-k+i:], lda, tau[i:], work, ldwork)

				// Apply H to A[0:m-k+i+ib, 0:n-k+i] from the left.
				impl.Dlarfb(blas.Left, blas.NoTrans, lapack.Backward, lapack.ColumnWise,
					m-k+i+ib, n-k+i, ib, a[n-k+i:], lda, work, ldwork,
					a, lda, work[ib*ldwork:], ldwork)
			}

			// Apply H to rows 0:m-k+i+ib of current block.
			impl.Dorg2l(m-k+i+ib, ib, ib, a[n-k+i:], lda, tau[i:], work)

			// Set rows m-k+i+ib:m of current block to zero.
			for j := n - k + i; j < n-k+i+ib; j++ {
				for l := m - k + i + ib; l < m; l++ {
					a[l*lda+j] = 0
				}
			}
		}
	}
	work[0] = float64(iws)
}
