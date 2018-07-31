// Copyright ©2017 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import "gonum.org/v1/gonum/blas"

// Dgerq2 computes an RQ factorization of the m×n matrix A,
//  A = R * Q.
// On exit, if m <= n, the upper triangle of the subarray
// A[0:m, n-m:n] contains the m×m upper triangular matrix R.
// If m >= n, the elements on and above the (m-n)-th subdiagonal
// contain the m×n upper trapezoidal matrix R.
// The remaining elements, with tau, represent the
// orthogonal matrix Q as a product of min(m,n) elementary
// reflectors.
//
// The matrix Q is represented as a product of elementary reflectors
//  Q = H_0 H_1 . . . H_{min(m,n)-1}.
// Each H(i) has the form
//  H_i = I - tau_i * v * v^T
// where v is a vector with v[0:n-k+i-1] stored in A[m-k+i, 0:n-k+i-1],
// v[n-k+i:n] = 0 and v[n-k+i] = 1.
//
// tau must have length min(m,n) and work must have length m, otherwise
// Dgerq2 will panic.
//
// Dgerq2 is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dgerq2(m, n int, a []float64, lda int, tau, work []float64) {
	checkMatrix(m, n, a, lda)
	k := min(m, n)
	if len(tau) < k {
		panic(badTau)
	}
	if len(work) < m {
		panic(badWork)
	}

	for i := k - 1; i >= 0; i-- {
		// Generate elementary reflector H[i] to annihilate
		// A[m-k+i, 0:n-k+i-1].
		mki := m - k + i
		nki := n - k + i
		var aii float64
		aii, tau[i] = impl.Dlarfg(nki+1, a[mki*lda+nki], a[mki*lda:], 1)

		// Apply H[i] to A[0:m-k+i-1, 0:n-k+i] from the right.
		a[mki*lda+nki] = 1
		impl.Dlarf(blas.Right, mki, nki+1, a[mki*lda:], 1, tau[i], a, lda, work)
		a[mki*lda+nki] = aii
	}
}
