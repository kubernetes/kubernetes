// Copyright ©2016 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import "gonum.org/v1/gonum/blas"

// Dgeql2 computes the QL factorization of the m×n matrix A. That is, Dgeql2
// computes Q and L such that
//  A = Q * L
// where Q is an m×m orthonormal matrix and L is a lower trapezoidal matrix.
//
// Q is represented as a product of elementary reflectors,
//  Q = H_{k-1} * ... * H_1 * H_0
// where k = min(m,n) and each H_i has the form
//  H_i = I - tau[i] * v_i * v_i^T
// Vector v_i has v[m-k+i+1:m] = 0, v[m-k+i] = 1, and v[:m-k+i+1] is stored on
// exit in A[0:m-k+i-1, n-k+i].
//
// tau must have length at least min(m,n), and Dgeql2 will panic otherwise.
//
// work is temporary memory storage and must have length at least n.
//
// Dgeql2 is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dgeql2(m, n int, a []float64, lda int, tau, work []float64) {
	checkMatrix(m, n, a, lda)
	if len(tau) < min(m, n) {
		panic(badTau)
	}
	if len(work) < n {
		panic(badWork)
	}
	k := min(m, n)
	var aii float64
	for i := k - 1; i >= 0; i-- {
		// Generate elementary reflector H_i to annihilate A[0:m-k+i-1, n-k+i].
		aii, tau[i] = impl.Dlarfg(m-k+i+1, a[(m-k+i)*lda+n-k+i], a[n-k+i:], lda)

		// Apply H_i to A[0:m-k+i, 0:n-k+i-1] from the left.
		a[(m-k+i)*lda+n-k+i] = 1
		impl.Dlarf(blas.Left, m-k+i+1, n-k+i, a[n-k+i:], lda, tau[i], a, lda, work)
		a[(m-k+i)*lda+n-k+i] = aii
	}
}
