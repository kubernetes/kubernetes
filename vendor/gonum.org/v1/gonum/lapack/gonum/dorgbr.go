// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import "gonum.org/v1/gonum/lapack"

// Dorgbr generates one of the matrices Q or P^T computed by Dgebrd
// computed from the decomposition Dgebrd. See Dgebd2 for the description of
// Q and P^T.
//
// If vect == lapack.ApplyQ, then a is assumed to have been an m×k matrix and
// Q is of order m. If m >= k, then Dorgbr returns the first n columns of Q
// where m >= n >= k. If m < k, then Dorgbr returns Q as an m×m matrix.
//
// If vect == lapack.ApplyP, then A is assumed to have been a k×n matrix, and
// P^T is of order n. If k < n, then Dorgbr returns the first m rows of P^T,
// where n >= m >= k. If k >= n, then Dorgbr returns P^T as an n×n matrix.
//
// Dorgbr is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dorgbr(vect lapack.DecompUpdate, m, n, k int, a []float64, lda int, tau, work []float64, lwork int) {
	mn := min(m, n)
	wantq := vect == lapack.ApplyQ
	if wantq {
		if m < n || n < min(m, k) || m < min(m, k) {
			panic(badDims)
		}
	} else {
		if n < m || m < min(n, k) || n < min(n, k) {
			panic(badDims)
		}
	}
	if wantq {
		if m >= k {
			checkMatrix(m, k, a, lda)
		} else {
			checkMatrix(m, m, a, lda)
		}
	} else {
		if n >= k {
			checkMatrix(k, n, a, lda)
		} else {
			checkMatrix(n, n, a, lda)
		}
	}
	work[0] = 1
	if wantq {
		if m >= k {
			impl.Dorgqr(m, n, k, a, lda, tau, work, -1)
		} else if m > 1 {
			impl.Dorgqr(m-1, m-1, m-1, a[lda+1:], lda, tau, work, -1)
		}
	} else {
		if k < n {
			impl.Dorglq(m, n, k, a, lda, tau, work, -1)
		} else if n > 1 {
			impl.Dorglq(n-1, n-1, n-1, a[lda+1:], lda, tau, work, -1)
		}
	}
	lworkopt := int(work[0])
	lworkopt = max(lworkopt, mn)
	if lwork == -1 {
		work[0] = float64(lworkopt)
		return
	}
	if len(work) < lwork {
		panic(badWork)
	}
	if lwork < mn {
		panic(badWork)
	}
	if m == 0 || n == 0 {
		work[0] = 1
		return
	}
	if wantq {
		// Form Q, determined by a call to Dgebrd to reduce an m×k matrix.
		if m >= k {
			impl.Dorgqr(m, n, k, a, lda, tau, work, lwork)
		} else {
			// Shift the vectors which define the elementary reflectors one
			// column to the right, and set the first row and column of Q to
			// those of the unit matrix.
			for j := m - 1; j >= 1; j-- {
				a[j] = 0
				for i := j + 1; i < m; i++ {
					a[i*lda+j] = a[i*lda+j-1]
				}
			}
			a[0] = 1
			for i := 1; i < m; i++ {
				a[i*lda] = 0
			}
			if m > 1 {
				// Form Q[1:m-1, 1:m-1]
				impl.Dorgqr(m-1, m-1, m-1, a[lda+1:], lda, tau, work, lwork)
			}
		}
	} else {
		// Form P^T, determined by a call to Dgebrd to reduce a k×n matrix.
		if k < n {
			impl.Dorglq(m, n, k, a, lda, tau, work, lwork)
		} else {
			// Shift the vectors which define the elementary reflectors one
			// row downward, and set the first row and column of P^T to
			// those of the unit matrix.
			a[0] = 1
			for i := 1; i < n; i++ {
				a[i*lda] = 0
			}
			for j := 1; j < n; j++ {
				for i := j - 1; i >= 1; i-- {
					a[i*lda+j] = a[(i-1)*lda+j]
				}
				a[j] = 0
			}
			if n > 1 {
				impl.Dorglq(n-1, n-1, n-1, a[lda+1:], lda, tau, work, lwork)
			}
		}
	}
	work[0] = float64(lworkopt)
}
