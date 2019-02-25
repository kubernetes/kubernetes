// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

// Iladlc scans a matrix for its last non-zero column. Returns -1 if the matrix
// is all zeros.
//
// Iladlc is an internal routine. It is exported for testing purposes.
func (Implementation) Iladlc(m, n int, a []float64, lda int) int {
	if n == 0 || m == 0 {
		return n - 1
	}
	checkMatrix(m, n, a, lda)

	// Test common case where corner is non-zero.
	if a[n-1] != 0 || a[(m-1)*lda+(n-1)] != 0 {
		return n - 1
	}

	// Scan each row tracking the highest column seen.
	highest := -1
	for i := 0; i < m; i++ {
		for j := n - 1; j >= 0; j-- {
			if a[i*lda+j] != 0 {
				highest = max(highest, j)
				break
			}
		}
	}
	return highest
}
