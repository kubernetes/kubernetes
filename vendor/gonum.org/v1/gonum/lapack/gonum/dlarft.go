// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/gonum/lapack"
)

// Dlarft forms the triangular factor T of a block reflector H, storing the answer
// in t.
//  H = I - V * T * Vᵀ  if store == lapack.ColumnWise
//  H = I - Vᵀ * T * V  if store == lapack.RowWise
// H is defined by a product of the elementary reflectors where
//  H = H_0 * H_1 * ... * H_{k-1}  if direct == lapack.Forward
//  H = H_{k-1} * ... * H_1 * H_0  if direct == lapack.Backward
//
// t is a k×k triangular matrix. t is upper triangular if direct = lapack.Forward
// and lower triangular otherwise. This function will panic if t is not of
// sufficient size.
//
// store describes the storage of the elementary reflectors in v. See
// Dlarfb for a description of layout.
//
// tau contains the scalar factors of the elementary reflectors H_i.
//
// Dlarft is an internal routine. It is exported for testing purposes.
func (Implementation) Dlarft(direct lapack.Direct, store lapack.StoreV, n, k int, v []float64, ldv int, tau []float64, t []float64, ldt int) {
	mv, nv := n, k
	if store == lapack.RowWise {
		mv, nv = k, n
	}
	switch {
	case direct != lapack.Forward && direct != lapack.Backward:
		panic(badDirect)
	case store != lapack.RowWise && store != lapack.ColumnWise:
		panic(badStoreV)
	case n < 0:
		panic(nLT0)
	case k < 1:
		panic(kLT1)
	case ldv < max(1, nv):
		panic(badLdV)
	case len(tau) < k:
		panic(shortTau)
	case ldt < max(1, k):
		panic(shortT)
	}

	if n == 0 {
		return
	}

	switch {
	case len(v) < (mv-1)*ldv+nv:
		panic(shortV)
	case len(t) < (k-1)*ldt+k:
		panic(shortT)
	}

	bi := blas64.Implementation()

	// TODO(btracey): There are a number of minor obvious loop optimizations here.
	// TODO(btracey): It may be possible to rearrange some of the code so that
	// index of 1 is more common in the Dgemv.
	if direct == lapack.Forward {
		prevlastv := n - 1
		for i := 0; i < k; i++ {
			prevlastv = max(i, prevlastv)
			if tau[i] == 0 {
				for j := 0; j <= i; j++ {
					t[j*ldt+i] = 0
				}
				continue
			}
			var lastv int
			if store == lapack.ColumnWise {
				// skip trailing zeros
				for lastv = n - 1; lastv >= i+1; lastv-- {
					if v[lastv*ldv+i] != 0 {
						break
					}
				}
				for j := 0; j < i; j++ {
					t[j*ldt+i] = -tau[i] * v[i*ldv+j]
				}
				j := min(lastv, prevlastv)
				bi.Dgemv(blas.Trans, j-i, i,
					-tau[i], v[(i+1)*ldv:], ldv, v[(i+1)*ldv+i:], ldv,
					1, t[i:], ldt)
			} else {
				for lastv = n - 1; lastv >= i+1; lastv-- {
					if v[i*ldv+lastv] != 0 {
						break
					}
				}
				for j := 0; j < i; j++ {
					t[j*ldt+i] = -tau[i] * v[j*ldv+i]
				}
				j := min(lastv, prevlastv)
				bi.Dgemv(blas.NoTrans, i, j-i,
					-tau[i], v[i+1:], ldv, v[i*ldv+i+1:], 1,
					1, t[i:], ldt)
			}
			bi.Dtrmv(blas.Upper, blas.NoTrans, blas.NonUnit, i, t, ldt, t[i:], ldt)
			t[i*ldt+i] = tau[i]
			if i > 1 {
				prevlastv = max(prevlastv, lastv)
			} else {
				prevlastv = lastv
			}
		}
		return
	}
	prevlastv := 0
	for i := k - 1; i >= 0; i-- {
		if tau[i] == 0 {
			for j := i; j < k; j++ {
				t[j*ldt+i] = 0
			}
			continue
		}
		var lastv int
		if i < k-1 {
			if store == lapack.ColumnWise {
				for lastv = 0; lastv < i; lastv++ {
					if v[lastv*ldv+i] != 0 {
						break
					}
				}
				for j := i + 1; j < k; j++ {
					t[j*ldt+i] = -tau[i] * v[(n-k+i)*ldv+j]
				}
				j := max(lastv, prevlastv)
				bi.Dgemv(blas.Trans, n-k+i-j, k-i-1,
					-tau[i], v[j*ldv+i+1:], ldv, v[j*ldv+i:], ldv,
					1, t[(i+1)*ldt+i:], ldt)
			} else {
				for lastv = 0; lastv < i; lastv++ {
					if v[i*ldv+lastv] != 0 {
						break
					}
				}
				for j := i + 1; j < k; j++ {
					t[j*ldt+i] = -tau[i] * v[j*ldv+n-k+i]
				}
				j := max(lastv, prevlastv)
				bi.Dgemv(blas.NoTrans, k-i-1, n-k+i-j,
					-tau[i], v[(i+1)*ldv+j:], ldv, v[i*ldv+j:], 1,
					1, t[(i+1)*ldt+i:], ldt)
			}
			bi.Dtrmv(blas.Lower, blas.NoTrans, blas.NonUnit, k-i-1,
				t[(i+1)*ldt+i+1:], ldt,
				t[(i+1)*ldt+i:], ldt)
			if i > 0 {
				prevlastv = min(prevlastv, lastv)
			} else {
				prevlastv = lastv
			}
		}
		t[i*ldt+i] = tau[i]
	}
}
