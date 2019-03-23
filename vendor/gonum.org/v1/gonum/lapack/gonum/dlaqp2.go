// Copyright ©2017 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"math"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
)

// Dlaqp2 computes a QR factorization with column pivoting of the block A[offset:m, 0:n]
// of the m×n matrix A. The block A[0:offset, 0:n] is accordingly pivoted, but not factorized.
//
// On exit, the upper triangle of block A[offset:m, 0:n] is the triangular factor obtained.
// The elements in block A[offset:m, 0:n] below the diagonal, together with tau, represent
// the orthogonal matrix Q as a product of elementary reflectors.
//
// offset is number of rows of the matrix A that must be pivoted but not factorized.
// offset must not be negative otherwise Dlaqp2 will panic.
//
// On exit, jpvt holds the permutation that was applied; the jth column of A*P was the
// jpvt[j] column of A. jpvt must have length n, otherwise Dlaqp2 will panic.
//
// On exit tau holds the scalar factors of the elementary reflectors. It must have length
// at least min(m-offset, n) otherwise Dlaqp2 will panic.
//
// vn1 and vn2 hold the partial and complete column norms respectively. They must have length n,
// otherwise Dlaqp2 will panic.
//
// work must have length n, otherwise Dlaqp2 will panic.
//
// Dlaqp2 is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dlaqp2(m, n, offset int, a []float64, lda int, jpvt []int, tau, vn1, vn2, work []float64) {
	checkMatrix(m, n, a, lda)
	if len(jpvt) != n {
		panic(badIpiv)
	}
	mn := min(m-offset, n)
	if len(tau) < mn {
		panic(badTau)
	}
	if len(vn1) < n {
		panic(badVn1)
	}
	if len(vn2) < n {
		panic(badVn2)
	}
	if len(work) < n {
		panic(badWork)
	}

	tol3z := math.Sqrt(dlamchE)

	bi := blas64.Implementation()

	// Compute factorization.
	for i := 0; i < mn; i++ {
		offpi := offset + i

		// Determine ith pivot column and swap if necessary.
		p := i + bi.Idamax(n-i, vn1[i:], 1)
		if p != i {
			bi.Dswap(m, a[p:], lda, a[i:], lda)
			jpvt[p], jpvt[i] = jpvt[i], jpvt[p]
			vn1[p] = vn1[i]
			vn2[p] = vn2[i]
		}

		// Generate elementary reflector H_i.
		if offpi < m-1 {
			a[offpi*lda+i], tau[i] = impl.Dlarfg(m-offpi, a[offpi*lda+i], a[(offpi+1)*lda+i:], lda)
		} else {
			tau[i] = 0
		}

		if i < n-1 {
			// Apply H_i^T to A[offset+i:m, i:n] from the left.
			aii := a[offpi*lda+i]
			a[offpi*lda+i] = 1
			impl.Dlarf(blas.Left, m-offpi, n-i-1, a[offpi*lda+i:], lda, tau[i], a[offpi*lda+i+1:], lda, work)
			a[offpi*lda+i] = aii
		}

		// Update partial column norms.
		for j := i + 1; j < n; j++ {
			if vn1[j] == 0 {
				continue
			}

			// The following marked lines follow from the
			// analysis in Lapack Working Note 176.
			r := math.Abs(a[offpi*lda+j]) / vn1[j] // *
			temp := math.Max(0, 1-r*r)             // *
			r = vn1[j] / vn2[j]                    // *
			temp2 := temp * r * r                  // *
			if temp2 < tol3z {
				var v float64
				if offpi < m-1 {
					v = bi.Dnrm2(m-offpi-1, a[(offpi+1)*lda+j:], lda)
				}
				vn1[j] = v
				vn2[j] = v
			} else {
				vn1[j] *= math.Sqrt(temp) // *
			}
		}
	}
}
