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

// Dtrcon estimates the reciprocal of the condition number of a triangular matrix A.
// The condition number computed may be based on the 1-norm or the ∞-norm.
//
// work is a temporary data slice of length at least 3*n and Dtrcon will panic otherwise.
//
// iwork is a temporary data slice of length at least n and Dtrcon will panic otherwise.
func (impl Implementation) Dtrcon(norm lapack.MatrixNorm, uplo blas.Uplo, diag blas.Diag, n int, a []float64, lda int, work []float64, iwork []int) float64 {
	if norm != lapack.MaxColumnSum && norm != lapack.MaxRowSum {
		panic(badNorm)
	}
	if uplo != blas.Upper && uplo != blas.Lower {
		panic(badUplo)
	}
	if diag != blas.NonUnit && diag != blas.Unit {
		panic(badDiag)
	}
	if len(work) < 3*n {
		panic(badWork)
	}
	if len(iwork) < n {
		panic(badWork)
	}
	if n == 0 {
		return 1
	}
	bi := blas64.Implementation()

	var rcond float64
	smlnum := dlamchS * float64(n)

	anorm := impl.Dlantr(norm, uplo, diag, n, n, a, lda, work)

	if anorm <= 0 {
		return rcond
	}
	var ainvnm float64
	var normin bool
	kase1 := 2
	if norm == lapack.MaxColumnSum {
		kase1 = 1
	}
	var kase int
	isave := new([3]int)
	var scale float64
	for {
		ainvnm, kase = impl.Dlacn2(n, work[n:], work, iwork, ainvnm, kase, isave)
		if kase == 0 {
			if ainvnm != 0 {
				rcond = (1 / anorm) / ainvnm
			}
			return rcond
		}
		if kase == kase1 {
			scale = impl.Dlatrs(uplo, blas.NoTrans, diag, normin, n, a, lda, work, work[2*n:])
		} else {
			scale = impl.Dlatrs(uplo, blas.Trans, diag, normin, n, a, lda, work, work[2*n:])
		}
		normin = true
		if scale != 1 {
			ix := bi.Idamax(n, work, 1)
			xnorm := math.Abs(work[ix])
			if scale == 0 || scale < xnorm*smlnum {
				return rcond
			}
			impl.Drscl(n, scale, work, 1)
		}
	}
}
