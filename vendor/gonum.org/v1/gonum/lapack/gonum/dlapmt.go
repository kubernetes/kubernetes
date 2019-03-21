// Copyright ©2017 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import "gonum.org/v1/gonum/blas/blas64"

// Dlapmt rearranges the columns of the m×n matrix X as specified by the
// permutation k_0, k_1, ..., k_n-1 of the integers 0, ..., n-1.
//
// If forward is true a forward permutation is performed:
//
//  X[0:m, k[j]] is moved to X[0:m, j] for j = 0, 1, ..., n-1.
//
// otherwise a backward permutation is performed:
//
//  X[0:m, j] is moved to X[0:m, k[j]] for j = 0, 1, ..., n-1.
//
// k must have length n, otherwise Dlapmt will panic. k is zero-indexed.
func (impl Implementation) Dlapmt(forward bool, m, n int, x []float64, ldx int, k []int) {
	checkMatrix(m, n, x, ldx)
	if len(k) != n {
		panic(badKperm)
	}

	if n <= 1 {
		return
	}

	for i, v := range k {
		v++
		k[i] = -v
	}

	bi := blas64.Implementation()

	if forward {
		for j, v := range k {
			if v >= 0 {
				continue
			}
			k[j] = -v
			i := -v - 1
			for k[i] < 0 {
				bi.Dswap(m, x[j:], ldx, x[i:], ldx)

				k[i] = -k[i]
				j = i
				i = k[i] - 1
			}
		}
	} else {
		for i, v := range k {
			if v >= 0 {
				continue
			}
			k[i] = -v
			j := -v - 1
			for j != i {
				bi.Dswap(m, x[j:], ldx, x[i:], ldx)

				k[j] = -k[j]
				j = k[j] - 1
			}
		}
	}

	for i := range k {
		k[i]--
	}
}
