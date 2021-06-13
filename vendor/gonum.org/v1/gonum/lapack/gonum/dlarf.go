// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
)

// Dlarf applies an elementary reflector H to an m×n matrix C:
//  C = H * C  if side == blas.Left
//  C = C * H  if side == blas.Right
// H is represented in the form
//  H = I - tau * v * vᵀ
// where tau is a scalar and v is a vector.
//
// work must have length at least m if side == blas.Left and
// at least n if side == blas.Right.
//
// Dlarf is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dlarf(side blas.Side, m, n int, v []float64, incv int, tau float64, c []float64, ldc int, work []float64) {
	switch {
	case side != blas.Left && side != blas.Right:
		panic(badSide)
	case m < 0:
		panic(mLT0)
	case n < 0:
		panic(nLT0)
	case incv == 0:
		panic(zeroIncV)
	case ldc < max(1, n):
		panic(badLdC)
	}

	if m == 0 || n == 0 {
		return
	}

	applyleft := side == blas.Left
	lenV := n
	if applyleft {
		lenV = m
	}

	switch {
	case len(v) < 1+(lenV-1)*abs(incv):
		panic(shortV)
	case len(c) < (m-1)*ldc+n:
		panic(shortC)
	case (applyleft && len(work) < n) || (!applyleft && len(work) < m):
		panic(shortWork)
	}

	lastv := -1 // last non-zero element of v
	lastc := -1 // last non-zero row/column of C
	if tau != 0 {
		if applyleft {
			lastv = m - 1
		} else {
			lastv = n - 1
		}
		var i int
		if incv > 0 {
			i = lastv * incv
		}
		// Look for the last non-zero row in v.
		for lastv >= 0 && v[i] == 0 {
			lastv--
			i -= incv
		}
		if applyleft {
			// Scan for the last non-zero column in C[0:lastv, :]
			lastc = impl.Iladlc(lastv+1, n, c, ldc)
		} else {
			// Scan for the last non-zero row in C[:, 0:lastv]
			lastc = impl.Iladlr(m, lastv+1, c, ldc)
		}
	}
	if lastv == -1 || lastc == -1 {
		return
	}
	bi := blas64.Implementation()
	if applyleft {
		// Form H * C
		// w[0:lastc+1] = c[1:lastv+1, 1:lastc+1]ᵀ * v[1:lastv+1,1]
		bi.Dgemv(blas.Trans, lastv+1, lastc+1, 1, c, ldc, v, incv, 0, work, 1)
		// c[0: lastv, 0: lastc] = c[...] - w[0:lastv, 1] * v[1:lastc, 1]ᵀ
		bi.Dger(lastv+1, lastc+1, -tau, v, incv, work, 1, c, ldc)
	} else {
		// Form C * H
		// w[0:lastc+1,1] := c[0:lastc+1,0:lastv+1] * v[0:lastv+1,1]
		bi.Dgemv(blas.NoTrans, lastc+1, lastv+1, 1, c, ldc, v, incv, 0, work, 1)
		// c[0:lastc+1,0:lastv+1] = c[...] - w[0:lastc+1,0] * v[0:lastv+1,0]ᵀ
		bi.Dger(lastc+1, lastv+1, -tau, work, 1, v, incv, c, ldc)
	}
}
