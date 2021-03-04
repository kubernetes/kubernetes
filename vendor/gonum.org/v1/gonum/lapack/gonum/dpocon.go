// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"math"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
)

// Dpocon estimates the reciprocal of the condition number of a positive-definite
// matrix A given the Cholesky decomposition of A. The condition number computed
// is based on the 1-norm and the ∞-norm.
//
// anorm is the 1-norm and the ∞-norm of the original matrix A.
//
// work is a temporary data slice of length at least 3*n and Dpocon will panic otherwise.
//
// iwork is a temporary data slice of length at least n and Dpocon will panic otherwise.
func (impl Implementation) Dpocon(uplo blas.Uplo, n int, a []float64, lda int, anorm float64, work []float64, iwork []int) float64 {
	switch {
	case uplo != blas.Upper && uplo != blas.Lower:
		panic(badUplo)
	case n < 0:
		panic(nLT0)
	case lda < max(1, n):
		panic(badLdA)
	case anorm < 0:
		panic(negANorm)
	}

	// Quick return if possible.
	if n == 0 {
		return 1
	}

	switch {
	case len(a) < (n-1)*lda+n:
		panic(shortA)
	case len(work) < 3*n:
		panic(shortWork)
	case len(iwork) < n:
		panic(shortIWork)
	}

	if anorm == 0 {
		return 0
	}

	bi := blas64.Implementation()

	var (
		smlnum = dlamchS
		rcond  float64
		sl, su float64
		normin bool
		ainvnm float64
		kase   int
		isave  [3]int
	)
	for {
		ainvnm, kase = impl.Dlacn2(n, work[n:], work, iwork, ainvnm, kase, &isave)
		if kase == 0 {
			if ainvnm != 0 {
				rcond = (1 / ainvnm) / anorm
			}
			return rcond
		}
		if uplo == blas.Upper {
			sl = impl.Dlatrs(blas.Upper, blas.Trans, blas.NonUnit, normin, n, a, lda, work, work[2*n:])
			normin = true
			su = impl.Dlatrs(blas.Upper, blas.NoTrans, blas.NonUnit, normin, n, a, lda, work, work[2*n:])
		} else {
			sl = impl.Dlatrs(blas.Lower, blas.NoTrans, blas.NonUnit, normin, n, a, lda, work, work[2*n:])
			normin = true
			su = impl.Dlatrs(blas.Lower, blas.Trans, blas.NonUnit, normin, n, a, lda, work, work[2*n:])
		}
		scale := sl * su
		if scale != 1 {
			ix := bi.Idamax(n, work, 1)
			if scale == 0 || scale < math.Abs(work[ix])*smlnum {
				return rcond
			}
			impl.Drscl(n, scale, work, 1)
		}
	}
}
