// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import "gonum.org/v1/gonum/blas"

// Dgelq2 computes the LQ factorization of the m×n matrix A.
//
// In an LQ factorization, L is a lower triangular m×n matrix, and Q is an n×n
// orthonormal matrix.
//
// a is modified to contain the information to construct L and Q.
// The lower triangle of a contains the matrix L. The upper triangular elements
// (not including the diagonal) contain the elementary reflectors. tau is modified
// to contain the reflector scales. tau must have length of at least k = min(m,n)
// and this function will panic otherwise.
//
// See Dgeqr2 for a description of the elementary reflectors and orthonormal
// matrix Q. Q is constructed as a product of these elementary reflectors,
// Q = H_{k-1} * ... * H_1 * H_0.
//
// work is temporary storage of length at least m and this function will panic otherwise.
//
// Dgelq2 is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dgelq2(m, n int, a []float64, lda int, tau, work []float64) {
	checkMatrix(m, n, a, lda)
	k := min(m, n)
	if len(tau) < k {
		panic(badTau)
	}
	if len(work) < m {
		panic(badWork)
	}
	for i := 0; i < k; i++ {
		a[i*lda+i], tau[i] = impl.Dlarfg(n-i, a[i*lda+i], a[i*lda+min(i+1, n-1):], 1)
		if i < m-1 {
			aii := a[i*lda+i]
			a[i*lda+i] = 1
			impl.Dlarf(blas.Right, m-i-1, n-i,
				a[i*lda+i:], 1,
				tau[i],
				a[(i+1)*lda+i:], lda,
				work)
			a[i*lda+i] = aii
		}
	}
}
