// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import "gonum.org/v1/gonum/blas/blas64"

// Dlaswp swaps the rows k1 to k2 of a rectangular matrix A according to the
// indices in ipiv so that row k is swapped with ipiv[k].
//
// n is the number of columns of A and incX is the increment for ipiv. If incX
// is 1, the swaps are applied from k1 to k2. If incX is -1, the swaps are
// applied in reverse order from k2 to k1. For other values of incX Dlaswp will
// panic. ipiv must have length k2+1, otherwise Dlaswp will panic.
//
// The indices k1, k2, and the elements of ipiv are zero-based.
//
// Dlaswp is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dlaswp(n int, a []float64, lda int, k1, k2 int, ipiv []int, incX int) {
	switch {
	case n < 0:
		panic(nLT0)
	case k2 < 0:
		panic(badK2)
	case k1 < 0 || k2 < k1:
		panic(badK1)
	case lda < max(1, n):
		panic(badLdA)
	case len(a) < (k2-1)*lda+n:
		panic(shortA)
	case len(ipiv) != k2+1:
		panic(badLenIpiv)
	case incX != 1 && incX != -1:
		panic(absIncNotOne)
	}

	if n == 0 {
		return
	}

	bi := blas64.Implementation()
	if incX == 1 {
		for k := k1; k <= k2; k++ {
			bi.Dswap(n, a[k*lda:], 1, a[ipiv[k]*lda:], 1)
		}
		return
	}
	for k := k2; k >= k1; k-- {
		bi.Dswap(n, a[k*lda:], 1, a[ipiv[k]*lda:], 1)
	}
}
