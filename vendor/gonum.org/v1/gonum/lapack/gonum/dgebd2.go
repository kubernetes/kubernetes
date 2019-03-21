// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import "gonum.org/v1/gonum/blas"

// Dgebd2 reduces an m×n matrix A to upper or lower bidiagonal form by an orthogonal
// transformation.
//  Q^T * A * P = B
// if m >= n, B is upper diagonal, otherwise B is lower bidiagonal.
// d is the diagonal, len = min(m,n)
// e is the off-diagonal len = min(m,n)-1
//
// Dgebd2 is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dgebd2(m, n int, a []float64, lda int, d, e, tauQ, tauP, work []float64) {
	checkMatrix(m, n, a, lda)
	if len(d) < min(m, n) {
		panic(badD)
	}
	if len(e) < min(m, n)-1 {
		panic(badE)
	}
	if len(tauQ) < min(m, n) {
		panic(badTauQ)
	}
	if len(tauP) < min(m, n) {
		panic(badTauP)
	}
	if len(work) < max(m, n) {
		panic(badWork)
	}
	if m >= n {
		for i := 0; i < n; i++ {
			a[i*lda+i], tauQ[i] = impl.Dlarfg(m-i, a[i*lda+i], a[min(i+1, m-1)*lda+i:], lda)
			d[i] = a[i*lda+i]
			a[i*lda+i] = 1
			// Apply H_i to A[i:m, i+1:n] from the left.
			if i < n-1 {
				impl.Dlarf(blas.Left, m-i, n-i-1, a[i*lda+i:], lda, tauQ[i], a[i*lda+i+1:], lda, work)
			}
			a[i*lda+i] = d[i]
			if i < n-1 {
				a[i*lda+i+1], tauP[i] = impl.Dlarfg(n-i-1, a[i*lda+i+1], a[i*lda+min(i+2, n-1):], 1)
				e[i] = a[i*lda+i+1]
				a[i*lda+i+1] = 1
				impl.Dlarf(blas.Right, m-i-1, n-i-1, a[i*lda+i+1:], 1, tauP[i], a[(i+1)*lda+i+1:], lda, work)
				a[i*lda+i+1] = e[i]
			} else {
				tauP[i] = 0
			}
		}
		return
	}
	for i := 0; i < m; i++ {
		a[i*lda+i], tauP[i] = impl.Dlarfg(n-i, a[i*lda+i], a[i*lda+min(i+1, n-1):], 1)
		d[i] = a[i*lda+i]
		a[i*lda+i] = 1
		if i < m-1 {
			impl.Dlarf(blas.Right, m-i-1, n-i, a[i*lda+i:], 1, tauP[i], a[(i+1)*lda+i:], lda, work)
		}
		a[i*lda+i] = d[i]
		if i < m-1 {
			a[(i+1)*lda+i], tauQ[i] = impl.Dlarfg(m-i-1, a[(i+1)*lda+i], a[min(i+2, m-1)*lda+i:], lda)
			e[i] = a[(i+1)*lda+i]
			a[(i+1)*lda+i] = 1
			impl.Dlarf(blas.Left, m-i-1, n-i-1, a[(i+1)*lda+i:], lda, tauQ[i], a[(i+1)*lda+i+1:], lda, work)
			a[(i+1)*lda+i] = e[i]
		} else {
			tauQ[i] = 0
		}
	}
}
