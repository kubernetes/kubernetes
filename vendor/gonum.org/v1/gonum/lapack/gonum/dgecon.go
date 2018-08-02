// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"math"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/gonum/lapack"
)

// Dgecon estimates the reciprocal of the condition number of the n×n matrix A
// given the LU decomposition of the matrix. The condition number computed may
// be based on the 1-norm or the ∞-norm.
//
// The slice a contains the result of the LU decomposition of A as computed by Dgetrf.
//
// anorm is the corresponding 1-norm or ∞-norm of the original matrix A.
//
// work is a temporary data slice of length at least 4*n and Dgecon will panic otherwise.
//
// iwork is a temporary data slice of length at least n and Dgecon will panic otherwise.
func (impl Implementation) Dgecon(norm lapack.MatrixNorm, n int, a []float64, lda int, anorm float64, work []float64, iwork []int) float64 {
	checkMatrix(n, n, a, lda)
	if norm != lapack.MaxColumnSum && norm != lapack.MaxRowSum {
		panic(badNorm)
	}
	if len(work) < 4*n {
		panic(badWork)
	}
	if len(iwork) < n {
		panic(badWork)
	}

	if n == 0 {
		return 1
	} else if anorm == 0 {
		return 0
	}

	bi := blas64.Implementation()
	var rcond, ainvnm float64
	var kase int
	var normin bool
	isave := new([3]int)
	onenrm := norm == lapack.MaxColumnSum
	smlnum := dlamchS
	kase1 := 2
	if onenrm {
		kase1 = 1
	}
	for {
		ainvnm, kase = impl.Dlacn2(n, work[n:], work, iwork, ainvnm, kase, isave)
		if kase == 0 {
			if ainvnm != 0 {
				rcond = (1 / ainvnm) / anorm
			}
			return rcond
		}
		var sl, su float64
		if kase == kase1 {
			sl = impl.Dlatrs(blas.Lower, blas.NoTrans, blas.Unit, normin, n, a, lda, work, work[2*n:])
			su = impl.Dlatrs(blas.Upper, blas.NoTrans, blas.NonUnit, normin, n, a, lda, work, work[3*n:])
		} else {
			su = impl.Dlatrs(blas.Upper, blas.Trans, blas.NonUnit, normin, n, a, lda, work, work[3*n:])
			sl = impl.Dlatrs(blas.Lower, blas.Trans, blas.Unit, normin, n, a, lda, work, work[2*n:])
		}
		scale := sl * su
		normin = true
		if scale != 1 {
			ix := bi.Idamax(n, work, 1)
			if scale == 0 || scale < math.Abs(work[ix])*smlnum {
				return rcond
			}
			impl.Drscl(n, scale, work, 1)
		}
	}
}
