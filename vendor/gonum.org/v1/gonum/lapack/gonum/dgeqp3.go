// Copyright ©2017 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
)

// Dgeqp3 computes a QR factorization with column pivoting of the
// m×n matrix A: A*P = Q*R using Level 3 BLAS.
//
// The matrix Q is represented as a product of elementary reflectors
//  Q = H_0 H_1 . . . H_{k-1}, where k = min(m,n).
// Each H_i has the form
//  H_i = I - tau * v * v^T
// where tau and v are real vectors with v[0:i-1] = 0 and v[i] = 1;
// v[i:m] is stored on exit in A[i:m, i], and tau in tau[i].
//
// jpvt specifies a column pivot to be applied to A. If
// jpvt[j] is at least zero, the jth column of A is permuted
// to the front of A*P (a leading column), if jpvt[j] is -1
// the jth column of A is a free column. If jpvt[j] < -1, Dgeqp3
// will panic. On return, jpvt holds the permutation that was
// applied; the jth column of A*P was the jpvt[j] column of A.
// jpvt must have length n or Dgeqp3 will panic.
//
// tau holds the scalar factors of the elementary reflectors.
// It must have length min(m, n), otherwise Dgeqp3 will panic.
//
// work must have length at least max(1,lwork), and lwork must be at least
// 3*n+1, otherwise Dgeqp3 will panic. For optimal performance lwork must
// be at least 2*n+(n+1)*nb, where nb is the optimal blocksize. On return,
// work[0] will contain the optimal value of lwork.
//
// If lwork == -1, instead of performing Dgeqp3, only the optimal value of lwork
// will be stored in work[0].
//
// Dgeqp3 is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dgeqp3(m, n int, a []float64, lda int, jpvt []int, tau, work []float64, lwork int) {
	const (
		inb    = 1
		inbmin = 2
		ixover = 3
	)
	checkMatrix(m, n, a, lda)

	if len(jpvt) != n {
		panic(badIpiv)
	}
	for _, v := range jpvt {
		if v < -1 || n <= v {
			panic("lapack: jpvt element out of range")
		}
	}
	minmn := min(m, n)
	if len(work) < max(1, lwork) {
		panic(badWork)
	}

	var iws, lwkopt, nb int
	if minmn == 0 {
		iws = 1
		lwkopt = 1
	} else {
		iws = 3*n + 1
		nb = impl.Ilaenv(inb, "DGEQRF", " ", m, n, -1, -1)
		lwkopt = 2*n + (n+1)*nb
	}
	work[0] = float64(lwkopt)

	if lwork == -1 {
		return
	}

	if len(tau) < minmn {
		panic(badTau)
	}

	bi := blas64.Implementation()

	// Move initial columns up front.
	var nfxd int
	for j := 0; j < n; j++ {
		if jpvt[j] == -1 {
			jpvt[j] = j
			continue
		}
		if j != nfxd {
			bi.Dswap(m, a[j:], lda, a[nfxd:], lda)
			jpvt[j], jpvt[nfxd] = jpvt[nfxd], j
		} else {
			jpvt[j] = j
		}
		nfxd++
	}

	// Factorize nfxd columns.
	//
	// Compute the QR factorization of nfxd columns and update remaining columns.
	if nfxd > 0 {
		na := min(m, nfxd)
		impl.Dgeqrf(m, na, a, lda, tau, work, lwork)
		iws = max(iws, int(work[0]))
		if na < n {
			impl.Dormqr(blas.Left, blas.Trans, m, n-na, na, a, lda, tau[:na], a[na:], lda,
				work, lwork)
			iws = max(iws, int(work[0]))
		}
	}

	if nfxd >= minmn {
		work[0] = float64(iws)
		return
	}

	// Factorize free columns.
	sm := m - nfxd
	sn := n - nfxd
	sminmn := minmn - nfxd

	// Determine the block size.
	nb = impl.Ilaenv(inb, "DGEQRF", " ", sm, sn, -1, -1)
	nbmin := 2
	nx := 0

	if 1 < nb && nb < sminmn {
		// Determine when to cross over from blocked to unblocked code.
		nx = max(0, impl.Ilaenv(ixover, "DGEQRF", " ", sm, sn, -1, -1))

		if nx < sminmn {
			// Determine if workspace is large enough for blocked code.
			minws := 2*sn + (sn+1)*nb
			iws = max(iws, minws)
			if lwork < minws {
				// Not enough workspace to use optimal nb. Reduce
				// nb and determine the minimum value of nb.
				nb = (lwork - 2*sn) / (sn + 1)
				nbmin = max(2, impl.Ilaenv(inbmin, "DGEQRF", " ", sm, sn, -1, -1))
			}
		}
	}

	// Initialize partial column norms.
	// The first n elements of work store the exact column norms.
	for j := nfxd; j < n; j++ {
		work[j] = bi.Dnrm2(sm, a[nfxd*lda+j:], lda)
		work[n+j] = work[j]
	}
	j := nfxd
	if nbmin <= nb && nb < sminmn && nx < sminmn {
		// Use blocked code initially.

		// Compute factorization.
		var fjb int
		for topbmn := minmn - nx; j < topbmn; j += fjb {
			jb := min(nb, topbmn-j)

			// Factorize jb columns among columns j:n.
			fjb = impl.Dlaqps(m, n-j, j, jb, a[j:], lda, jpvt[j:], tau[j:],
				work[j:n], work[j+n:2*n], work[2*n:2*n+jb], work[2*n+jb:], jb)
		}
	}

	// Use unblocked code to factor the last or only block.
	if j < minmn {
		impl.Dlaqp2(m, n-j, j, a[j:], lda, jpvt[j:], tau[j:],
			work[j:n], work[j+n:2*n], work[2*n:])
	}

	work[0] = float64(iws)
}
