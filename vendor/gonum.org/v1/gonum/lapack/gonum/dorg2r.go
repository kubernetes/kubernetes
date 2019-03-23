// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
)

// Dorg2r generates an m×n matrix Q with orthonormal columns defined by the
// product of elementary reflectors as computed by Dgeqrf.
//  Q = H_0 * H_1 * ... * H_{k-1}
// len(tau) >= k, 0 <= k <= n, 0 <= n <= m, len(work) >= n.
// Dorg2r will panic if these conditions are not met.
//
// Dorg2r is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dorg2r(m, n, k int, a []float64, lda int, tau []float64, work []float64) {
	checkMatrix(m, n, a, lda)
	if len(tau) < k {
		panic(badTau)
	}
	if len(work) < n {
		panic(badWork)
	}
	if k > n {
		panic(kGTN)
	}
	if n > m {
		panic(mLTN)
	}
	if len(work) < n {
		panic(badWork)
	}
	if n == 0 {
		return
	}
	bi := blas64.Implementation()
	// Initialize columns k+1:n to columns of the unit matrix.
	for l := 0; l < m; l++ {
		for j := k; j < n; j++ {
			a[l*lda+j] = 0
		}
	}
	for j := k; j < n; j++ {
		a[j*lda+j] = 1
	}
	for i := k - 1; i >= 0; i-- {
		for i := range work {
			work[i] = 0
		}
		if i < n-1 {
			a[i*lda+i] = 1
			impl.Dlarf(blas.Left, m-i, n-i-1, a[i*lda+i:], lda, tau[i], a[i*lda+i+1:], lda, work)
		}
		if i < m-1 {
			bi.Dscal(m-i-1, -tau[i], a[(i+1)*lda+i:], lda)
		}
		a[i*lda+i] = 1 - tau[i]
		for l := 0; l < i; l++ {
			a[l*lda+i] = 0
		}
	}
}
