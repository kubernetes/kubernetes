// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import "gonum.org/v1/gonum/blas"

// Dorml2 multiplies a general matrix C by an orthogonal matrix from an LQ factorization
// determined by Dgelqf.
//  C = Q * C   if side == blas.Left and trans == blas.NoTrans
//  C = Qᵀ * C  if side == blas.Left and trans == blas.Trans
//  C = C * Q   if side == blas.Right and trans == blas.NoTrans
//  C = C * Qᵀ  if side == blas.Right and trans == blas.Trans
// If side == blas.Left, a is a matrix of side k×m, and if side == blas.Right
// a is of size k×n.
//
// tau contains the Householder factors and is of length at least k and this function will
// panic otherwise.
//
// work is temporary storage of length at least n if side == blas.Left
// and at least m if side == blas.Right and this function will panic otherwise.
//
// Dorml2 is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dorml2(side blas.Side, trans blas.Transpose, m, n, k int, a []float64, lda int, tau, c []float64, ldc int, work []float64) {
	left := side == blas.Left
	switch {
	case !left && side != blas.Right:
		panic(badSide)
	case trans != blas.Trans && trans != blas.NoTrans:
		panic(badTrans)
	case m < 0:
		panic(mLT0)
	case n < 0:
		panic(nLT0)
	case k < 0:
		panic(kLT0)
	case left && k > m:
		panic(kGTM)
	case !left && k > n:
		panic(kGTN)
	case left && lda < max(1, m):
		panic(badLdA)
	case !left && lda < max(1, n):
		panic(badLdA)
	}

	// Quick return if possible.
	if m == 0 || n == 0 || k == 0 {
		return
	}

	switch {
	case left && len(a) < (k-1)*lda+m:
		panic(shortA)
	case !left && len(a) < (k-1)*lda+n:
		panic(shortA)
	case len(tau) < k:
		panic(shortTau)
	case len(c) < (m-1)*ldc+n:
		panic(shortC)
	case left && len(work) < n:
		panic(shortWork)
	case !left && len(work) < m:
		panic(shortWork)
	}

	notrans := trans == blas.NoTrans
	switch {
	case left && notrans:
		for i := 0; i < k; i++ {
			aii := a[i*lda+i]
			a[i*lda+i] = 1
			impl.Dlarf(side, m-i, n, a[i*lda+i:], 1, tau[i], c[i*ldc:], ldc, work)
			a[i*lda+i] = aii
		}

	case left && !notrans:
		for i := k - 1; i >= 0; i-- {
			aii := a[i*lda+i]
			a[i*lda+i] = 1
			impl.Dlarf(side, m-i, n, a[i*lda+i:], 1, tau[i], c[i*ldc:], ldc, work)
			a[i*lda+i] = aii
		}

	case !left && notrans:
		for i := k - 1; i >= 0; i-- {
			aii := a[i*lda+i]
			a[i*lda+i] = 1
			impl.Dlarf(side, m, n-i, a[i*lda+i:], 1, tau[i], c[i:], ldc, work)
			a[i*lda+i] = aii
		}

	case !left && !notrans:
		for i := 0; i < k; i++ {
			aii := a[i*lda+i]
			a[i*lda+i] = 1
			impl.Dlarf(side, m, n-i, a[i*lda+i:], 1, tau[i], c[i:], ldc, work)
			a[i*lda+i] = aii
		}
	}
}
