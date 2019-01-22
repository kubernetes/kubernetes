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
	checkMatrix(n, n, a, lda)
	if uplo != blas.Upper && uplo != blas.Lower {
		panic(badUplo)
	}
	if len(work) < 3*n {
		panic(badWork)
	}
	if len(iwork) < n {
		panic(badWork)
	}
	var rcond float64
	if n == 0 {
		return 1
	}
	if anorm == 0 {
		return rcond
	}

	bi := blas64.Implementation()
	var ainvnm float64
	smlnum := dlamchS
	upper := uplo == blas.Upper
	var kase int
	var normin bool
	isave := new([3]int)
	var sl, su float64
	for {
		ainvnm, kase = impl.Dlacn2(n, work[n:], work, iwork, ainvnm, kase, isave)
		if kase == 0 {
			if ainvnm != 0 {
				rcond = (1 / ainvnm) / anorm
			}
			return rcond
		}
		if upper {
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
