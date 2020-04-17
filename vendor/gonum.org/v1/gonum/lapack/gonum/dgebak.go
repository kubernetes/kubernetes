// Copyright ©2016 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/gonum/lapack"
)

// Dgebak updates an n×m matrix V as
//  V = P D V       if side == lapack.EVRight,
//  V = P D^{-1} V  if side == lapack.EVLeft,
// where P and D are n×n permutation and scaling matrices, respectively,
// implicitly represented by job, scale, ilo and ihi as returned by Dgebal.
//
// Typically, columns of the matrix V contain the right or left (determined by
// side) eigenvectors of the balanced matrix output by Dgebal, and Dgebak forms
// the eigenvectors of the original matrix.
//
// Dgebak is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dgebak(job lapack.BalanceJob, side lapack.EVSide, n, ilo, ihi int, scale []float64, m int, v []float64, ldv int) {
	switch {
	case job != lapack.BalanceNone && job != lapack.Permute && job != lapack.Scale && job != lapack.PermuteScale:
		panic(badBalanceJob)
	case side != lapack.EVLeft && side != lapack.EVRight:
		panic(badEVSide)
	case n < 0:
		panic(nLT0)
	case ilo < 0 || max(0, n-1) < ilo:
		panic(badIlo)
	case ihi < min(ilo, n-1) || n <= ihi:
		panic(badIhi)
	case m < 0:
		panic(mLT0)
	case ldv < max(1, m):
		panic(badLdV)
	}

	// Quick return if possible.
	if n == 0 || m == 0 {
		return
	}

	if len(scale) < n {
		panic(shortScale)
	}
	if len(v) < (n-1)*ldv+m {
		panic(shortV)
	}

	// Quick return if possible.
	if job == lapack.BalanceNone {
		return
	}

	bi := blas64.Implementation()
	if ilo != ihi && job != lapack.Permute {
		// Backward balance.
		if side == lapack.EVRight {
			for i := ilo; i <= ihi; i++ {
				bi.Dscal(m, scale[i], v[i*ldv:], 1)
			}
		} else {
			for i := ilo; i <= ihi; i++ {
				bi.Dscal(m, 1/scale[i], v[i*ldv:], 1)
			}
		}
	}
	if job == lapack.Scale {
		return
	}
	// Backward permutation.
	for i := ilo - 1; i >= 0; i-- {
		k := int(scale[i])
		if k == i {
			continue
		}
		bi.Dswap(m, v[i*ldv:], 1, v[k*ldv:], 1)
	}
	for i := ihi + 1; i < n; i++ {
		k := int(scale[i])
		if k == i {
			continue
		}
		bi.Dswap(m, v[i*ldv:], 1, v[k*ldv:], 1)
	}
}
