// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
)

// Dorgl2 generates an m×n matrix Q with orthonormal rows defined by the
// first m rows product of elementary reflectors as computed by Dgelqf.
//  Q = H_0 * H_1 * ... * H_{k-1}
// len(tau) >= k, 0 <= k <= m, 0 <= m <= n, len(work) >= m.
// Dorgl2 will panic if these conditions are not met.
//
// Dorgl2 is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dorgl2(m, n, k int, a []float64, lda int, tau, work []float64) {
	switch {
	case m < 0:
		panic(mLT0)
	case n < m:
		panic(nLTM)
	case k < 0:
		panic(kLT0)
	case k > m:
		panic(kGTM)
	case lda < max(1, m):
		panic(badLdA)
	}

	if m == 0 {
		return
	}

	switch {
	case len(a) < (m-1)*lda+n:
		panic(shortA)
	case len(tau) < k:
		panic(shortTau)
	case len(work) < m:
		panic(shortWork)
	}

	bi := blas64.Implementation()

	if k < m {
		for i := k; i < m; i++ {
			for j := 0; j < n; j++ {
				a[i*lda+j] = 0
			}
		}
		for j := k; j < m; j++ {
			a[j*lda+j] = 1
		}
	}
	for i := k - 1; i >= 0; i-- {
		if i < n-1 {
			if i < m-1 {
				a[i*lda+i] = 1
				impl.Dlarf(blas.Right, m-i-1, n-i, a[i*lda+i:], 1, tau[i], a[(i+1)*lda+i:], lda, work)
			}
			bi.Dscal(n-i-1, -tau[i], a[i*lda+i+1:], 1)
		}
		a[i*lda+i] = 1 - tau[i]
		for l := 0; l < i; l++ {
			a[i*lda+l] = 0
		}
	}
}
