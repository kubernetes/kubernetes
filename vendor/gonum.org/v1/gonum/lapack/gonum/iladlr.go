// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

// Iladlr scans a matrix for its last non-zero row. Returns -1 if the matrix
// is all zeros.
//
// Iladlr is an internal routine. It is exported for testing purposes.
func (Implementation) Iladlr(m, n int, a []float64, lda int) int {
	switch {
	case m < 0:
		panic(mLT0)
	case n < 0:
		panic(nLT0)
	case lda < max(1, n):
		panic(badLdA)
	}

	if n == 0 || m == 0 {
		return -1
	}

	if len(a) < (m-1)*lda+n {
		panic(shortA)
	}

	// Check the common case where the corner is non-zero
	if a[(m-1)*lda] != 0 || a[(m-1)*lda+n-1] != 0 {
		return m - 1
	}
	for i := m - 1; i >= 0; i-- {
		for j := 0; j < n; j++ {
			if a[i*lda+j] != 0 {
				return i
			}
		}
	}
	return -1
}
