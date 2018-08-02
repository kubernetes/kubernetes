// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"math"

	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/gonum/lapack"
)

// Dlasq1 computes the singular values of an n×n bidiagonal matrix with diagonal
// d and off-diagonal e. On exit, d contains the singular values in decreasing
// order, and e is overwritten. d must have length at least n, e must have
// length at least n-1, and the input work must have length at least 4*n. Dlasq1
// will panic if these conditions are not met.
//
// Dlasq1 is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dlasq1(n int, d, e, work []float64) (info int) {
	// TODO(btracey): replace info with an error.
	if n < 0 {
		panic(nLT0)
	}
	if len(work) < 4*n {
		panic(badWork)
	}
	if len(d) < n {
		panic("lapack: length of d less than n")
	}
	if len(e) < n-1 {
		panic("lapack: length of e less than n-1")
	}
	if n == 0 {
		return info
	}
	if n == 1 {
		d[0] = math.Abs(d[0])
		return info
	}
	if n == 2 {
		d[1], d[0] = impl.Dlas2(d[0], e[0], d[1])
		return info
	}
	// Estimate the largest singular value.
	var sigmx float64
	for i := 0; i < n-1; i++ {
		d[i] = math.Abs(d[i])
		sigmx = math.Max(sigmx, math.Abs(e[i]))
	}
	d[n-1] = math.Abs(d[n-1])
	// Early return if sigmx is zero (matrix is already diagonal).
	if sigmx == 0 {
		impl.Dlasrt(lapack.SortDecreasing, n, d)
		return info
	}

	for i := 0; i < n; i++ {
		sigmx = math.Max(sigmx, d[i])
	}

	// Copy D and E into WORK (in the Z format) and scale (squaring the
	// input data makes scaling by a power of the radix pointless).

	eps := dlamchP
	safmin := dlamchS
	scale := math.Sqrt(eps / safmin)
	bi := blas64.Implementation()
	bi.Dcopy(n, d, 1, work, 2)
	bi.Dcopy(n-1, e, 1, work[1:], 2)
	impl.Dlascl(lapack.General, 0, 0, sigmx, scale, 2*n-1, 1, work, 1)

	// Compute the q's and e's.
	for i := 0; i < 2*n-1; i++ {
		work[i] *= work[i]
	}
	work[2*n-1] = 0

	info = impl.Dlasq2(n, work)
	if info == 0 {
		for i := 0; i < n; i++ {
			d[i] = math.Sqrt(work[i])
		}
		impl.Dlascl(lapack.General, 0, 0, scale, sigmx, n, 1, d, 1)
	} else if info == 2 {
		// Maximum number of iterations exceeded. Move data from work
		// into D and E so the calling subroutine can try to finish.
		for i := 0; i < n; i++ {
			d[i] = math.Sqrt(work[2*i])
			e[i] = math.Sqrt(work[2*i+1])
		}
		impl.Dlascl(lapack.General, 0, 0, scale, sigmx, n, 1, d, 1)
		impl.Dlascl(lapack.General, 0, 0, scale, sigmx, n, 1, e, 1)
	}
	return info
}
