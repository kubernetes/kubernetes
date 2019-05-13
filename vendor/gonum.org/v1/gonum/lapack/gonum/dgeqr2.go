// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import "gonum.org/v1/gonum/blas"

// Dgeqr2 computes a QR factorization of the m×n matrix A.
//
// In a QR factorization, Q is an m×m orthonormal matrix, and R is an
// upper triangular m×n matrix.
//
// A is modified to contain the information to construct Q and R.
// The upper triangle of a contains the matrix R. The lower triangular elements
// (not including the diagonal) contain the elementary reflectors. tau is modified
// to contain the reflector scales. tau must have length at least min(m,n), and
// this function will panic otherwise.
//
// The ith elementary reflector can be explicitly constructed by first extracting
// the
//  v[j] = 0           j < i
//  v[j] = 1           j == i
//  v[j] = a[j*lda+i]  j > i
// and computing H_i = I - tau[i] * v * v^T.
//
// The orthonormal matrix Q can be constructed from a product of these elementary
// reflectors, Q = H_0 * H_1 * ... * H_{k-1}, where k = min(m,n).
//
// work is temporary storage of length at least n and this function will panic otherwise.
//
// Dgeqr2 is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dgeqr2(m, n int, a []float64, lda int, tau, work []float64) {
	// TODO(btracey): This is oriented such that columns of a are eliminated.
	// This likely could be re-arranged to take better advantage of row-major
	// storage.

	switch {
	case m < 0:
		panic(mLT0)
	case n < 0:
		panic(nLT0)
	case lda < max(1, n):
		panic(badLdA)
	case len(work) < n:
		panic(shortWork)
	}

	// Quick return if possible.
	k := min(m, n)
	if k == 0 {
		return
	}

	switch {
	case len(a) < (m-1)*lda+n:
		panic(shortA)
	case len(tau) < k:
		panic(shortTau)
	}

	for i := 0; i < k; i++ {
		// Generate elementary reflector H_i.
		a[i*lda+i], tau[i] = impl.Dlarfg(m-i, a[i*lda+i], a[min((i+1), m-1)*lda+i:], lda)
		if i < n-1 {
			aii := a[i*lda+i]
			a[i*lda+i] = 1
			impl.Dlarf(blas.Left, m-i, n-i-1,
				a[i*lda+i:], lda,
				tau[i],
				a[i*lda+i+1:], lda,
				work)
			a[i*lda+i] = aii
		}
	}
}
