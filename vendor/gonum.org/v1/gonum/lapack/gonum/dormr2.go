// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import "gonum.org/v1/gonum/blas"

// Dormr2 multiplies a general matrix C by an orthogonal matrix from a RQ factorization
// determined by Dgerqf.
//  C = Q * C   if side == blas.Left and trans == blas.NoTrans
//  C = Qᵀ * C  if side == blas.Left and trans == blas.Trans
//  C = C * Q   if side == blas.Right and trans == blas.NoTrans
//  C = C * Qᵀ  if side == blas.Right and trans == blas.Trans
// If side == blas.Left, a is a matrix of size k×m, and if side == blas.Right
// a is of size k×n.
//
// tau contains the Householder factors and is of length at least k and this function
// will panic otherwise.
//
// work is temporary storage of length at least n if side == blas.Left
// and at least m if side == blas.Right and this function will panic otherwise.
//
// Dormr2 is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dormr2(side blas.Side, trans blas.Transpose, m, n, k int, a []float64, lda int, tau, c []float64, ldc int, work []float64) {
	left := side == blas.Left
	nq := n
	nw := m
	if left {
		nq = m
		nw = n
	}
	switch {
	case !left && side != blas.Right:
		panic(badSide)
	case trans != blas.NoTrans && trans != blas.Trans:
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
	case lda < max(1, nq):
		panic(badLdA)
	case ldc < max(1, n):
		panic(badLdC)
	}

	// Quick return if possible.
	if m == 0 || n == 0 || k == 0 {
		return
	}

	switch {
	case len(a) < (k-1)*lda+nq:
		panic(shortA)
	case len(tau) < k:
		panic(shortTau)
	case len(c) < (m-1)*ldc+n:
		panic(shortC)
	case len(work) < nw:
		panic(shortWork)
	}

	if left {
		if trans == blas.NoTrans {
			for i := k - 1; i >= 0; i-- {
				aii := a[i*lda+(m-k+i)]
				a[i*lda+(m-k+i)] = 1
				impl.Dlarf(side, m-k+i+1, n, a[i*lda:], 1, tau[i], c, ldc, work)
				a[i*lda+(m-k+i)] = aii
			}
			return
		}
		for i := 0; i < k; i++ {
			aii := a[i*lda+(m-k+i)]
			a[i*lda+(m-k+i)] = 1
			impl.Dlarf(side, m-k+i+1, n, a[i*lda:], 1, tau[i], c, ldc, work)
			a[i*lda+(m-k+i)] = aii
		}
		return
	}
	if trans == blas.NoTrans {
		for i := 0; i < k; i++ {
			aii := a[i*lda+(n-k+i)]
			a[i*lda+(n-k+i)] = 1
			impl.Dlarf(side, m, n-k+i+1, a[i*lda:], 1, tau[i], c, ldc, work)
			a[i*lda+(n-k+i)] = aii
		}
		return
	}
	for i := k - 1; i >= 0; i-- {
		aii := a[i*lda+(n-k+i)]
		a[i*lda+(n-k+i)] = 1
		impl.Dlarf(side, m, n-k+i+1, a[i*lda:], 1, tau[i], c, ldc, work)
		a[i*lda+(n-k+i)] = aii
	}
}
