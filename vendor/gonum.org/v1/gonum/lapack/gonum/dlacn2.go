// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"math"

	"gonum.org/v1/gonum/blas/blas64"
)

// Dlacn2 estimates the 1-norm of an n×n matrix A using sequential updates with
// matrix-vector products provided externally.
//
// Dlacn2 is called sequentially and it returns the value of est and kase to be
// used on the next call.
// On the initial call, kase must be 0.
// In between calls, x must be overwritten by
//  A * X    if kase was returned as 1,
//  A^T * X  if kase was returned as 2,
// and all other parameters must not be changed.
// On the final return, kase is returned as 0, v contains A*W where W is a
// vector, and est = norm(V)/norm(W) is a lower bound for 1-norm of A.
//
// v, x, and isgn must all have length n and n must be at least 1, otherwise
// Dlacn2 will panic. isave is used for temporary storage.
//
// Dlacn2 is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dlacn2(n int, v, x []float64, isgn []int, est float64, kase int, isave *[3]int) (float64, int) {
	switch {
	case n < 1:
		panic(nLT1)
	case len(v) < n:
		panic(shortV)
	case len(x) < n:
		panic(shortX)
	case len(isgn) < n:
		panic(shortIsgn)
	case isave[0] < 0 || 5 < isave[0]:
		panic(badIsave)
	case isave[0] == 0 && kase != 0:
		panic(badIsave)
	}

	const itmax = 5
	bi := blas64.Implementation()

	if kase == 0 {
		for i := 0; i < n; i++ {
			x[i] = 1 / float64(n)
		}
		kase = 1
		isave[0] = 1
		return est, kase
	}
	switch isave[0] {
	case 1:
		if n == 1 {
			v[0] = x[0]
			est = math.Abs(v[0])
			kase = 0
			return est, kase
		}
		est = bi.Dasum(n, x, 1)
		for i := 0; i < n; i++ {
			x[i] = math.Copysign(1, x[i])
			isgn[i] = int(x[i])
		}
		kase = 2
		isave[0] = 2
		return est, kase
	case 2:
		isave[1] = bi.Idamax(n, x, 1)
		isave[2] = 2
		for i := 0; i < n; i++ {
			x[i] = 0
		}
		x[isave[1]] = 1
		kase = 1
		isave[0] = 3
		return est, kase
	case 3:
		bi.Dcopy(n, x, 1, v, 1)
		estold := est
		est = bi.Dasum(n, v, 1)
		sameSigns := true
		for i := 0; i < n; i++ {
			if int(math.Copysign(1, x[i])) != isgn[i] {
				sameSigns = false
				break
			}
		}
		if !sameSigns && est > estold {
			for i := 0; i < n; i++ {
				x[i] = math.Copysign(1, x[i])
				isgn[i] = int(x[i])
			}
			kase = 2
			isave[0] = 4
			return est, kase
		}
	case 4:
		jlast := isave[1]
		isave[1] = bi.Idamax(n, x, 1)
		if x[jlast] != math.Abs(x[isave[1]]) && isave[2] < itmax {
			isave[2] += 1
			for i := 0; i < n; i++ {
				x[i] = 0
			}
			x[isave[1]] = 1
			kase = 1
			isave[0] = 3
			return est, kase
		}
	case 5:
		tmp := 2 * (bi.Dasum(n, x, 1)) / float64(3*n)
		if tmp > est {
			bi.Dcopy(n, x, 1, v, 1)
			est = tmp
		}
		kase = 0
		return est, kase
	}
	// Iteration complete. Final stage
	altsgn := 1.0
	for i := 0; i < n; i++ {
		x[i] = altsgn * (1 + float64(i)/float64(n-1))
		altsgn *= -1
	}
	kase = 1
	isave[0] = 5
	return est, kase
}
