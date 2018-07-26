// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
)

// Dlarf applies an elementary reflector to a general rectangular matrix c.
// This computes
//  c = h * c if side == Left
//  c = c * h if side == right
// where
//  h = 1 - tau * v * v^T
// and c is an m * n matrix.
//
// work is temporary storage of length at least m if side == Left and at least
// n if side == Right. This function will panic if this length requirement is not met.
//
// Dlarf is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dlarf(side blas.Side, m, n int, v []float64, incv int, tau float64, c []float64, ldc int, work []float64) {
	applyleft := side == blas.Left
	if (applyleft && len(work) < n) || (!applyleft && len(work) < m) {
		panic(badWork)
	}
	checkMatrix(m, n, c, ldc)

	// v has length m if applyleft and n otherwise.
	lenV := n
	if applyleft {
		lenV = m
	}

	checkVector(lenV, v, incv)

	lastv := 0 // last non-zero element of v
	lastc := 0 // last non-zero row/column of c
	if tau != 0 {
		var i int
		if applyleft {
			lastv = m - 1
		} else {
			lastv = n - 1
		}
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
	// Sometimes 1-indexing is nicer ...
	bi := blas64.Implementation()
	if applyleft {
		// Form H * C
		// w[0:lastc+1] = c[1:lastv+1, 1:lastc+1]^T * v[1:lastv+1,1]
		bi.Dgemv(blas.Trans, lastv+1, lastc+1, 1, c, ldc, v, incv, 0, work, 1)
		// c[0: lastv, 0: lastc] = c[...] - w[0:lastv, 1] * v[1:lastc, 1]^T
		bi.Dger(lastv+1, lastc+1, -tau, v, incv, work, 1, c, ldc)
		return
	}
	// Form C*H
	// w[0:lastc+1,1] := c[0:lastc+1,0:lastv+1] * v[0:lastv+1,1]
	bi.Dgemv(blas.NoTrans, lastc+1, lastv+1, 1, c, ldc, v, incv, 0, work, 1)
	// c[0:lastc+1,0:lastv+1] = c[...] - w[0:lastc+1,0] * v[0:lastv+1,0]^T
	bi.Dger(lastc+1, lastv+1, -tau, work, 1, v, incv, c, ldc)
}
