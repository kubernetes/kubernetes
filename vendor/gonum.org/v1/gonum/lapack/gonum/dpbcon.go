// Copyright ©2019 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"math"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
)

// Dpbcon returns an estimate of the reciprocal of the condition number (in the
// 1-norm) of an n×n symmetric positive definite band matrix using the Cholesky
// factorization
//  A = Uᵀ*U  if uplo == blas.Upper
//  A = L*Lᵀ  if uplo == blas.Lower
// computed by Dpbtrf. The estimate is obtained for norm(inv(A)), and the
// reciprocal of the condition number is computed as
//  rcond = 1 / (anorm * norm(inv(A))).
//
// The length of work must be at least 3*n and the length of iwork must be at
// least n.
func (impl Implementation) Dpbcon(uplo blas.Uplo, n, kd int, ab []float64, ldab int, anorm float64, work []float64, iwork []int) (rcond float64) {
	switch {
	case uplo != blas.Upper && uplo != blas.Lower:
		panic(badUplo)
	case n < 0:
		panic(nLT0)
	case kd < 0:
		panic(kdLT0)
	case ldab < kd+1:
		panic(badLdA)
	case anorm < 0:
		panic(badNorm)
	}

	// Quick return if possible.
	if n == 0 {
		return 1
	}

	switch {
	case len(ab) < (n-1)*ldab+kd+1:
		panic(shortAB)
	case len(work) < 3*n:
		panic(shortWork)
	case len(iwork) < n:
		panic(shortIWork)
	}

	// Quick return if possible.
	if anorm == 0 {
		return 0
	}

	const smlnum = dlamchS

	var (
		ainvnm float64
		kase   int
		isave  [3]int
		normin bool

		// Denote work slices.
		x     = work[:n]
		v     = work[n : 2*n]
		cnorm = work[2*n : 3*n]
	)
	// Estimate the 1-norm of the inverse.
	bi := blas64.Implementation()
	for {
		ainvnm, kase = impl.Dlacn2(n, v, x, iwork, ainvnm, kase, &isave)
		if kase == 0 {
			break
		}
		var op1, op2 blas.Transpose
		if uplo == blas.Upper {
			// Multiply x by inv(Uᵀ),
			op1 = blas.Trans
			// then by inv(Uᵀ).
			op2 = blas.NoTrans
		} else {
			// Multiply x by inv(L),
			op1 = blas.NoTrans
			// then by inv(Lᵀ).
			op2 = blas.Trans
		}
		scaleL := impl.Dlatbs(uplo, op1, blas.NonUnit, normin, n, kd, ab, ldab, x, cnorm)
		normin = true
		scaleU := impl.Dlatbs(uplo, op2, blas.NonUnit, normin, n, kd, ab, ldab, x, cnorm)
		// Multiply x by 1/scale if doing so will not cause overflow.
		scale := scaleL * scaleU
		if scale != 1 {
			ix := bi.Idamax(n, x, 1)
			if scale < math.Abs(x[ix])*smlnum || scale == 0 {
				return 0
			}
			impl.Drscl(n, scale, x, 1)
		}
	}
	if ainvnm == 0 {
		return 0
	}
	// Return the estimate of the reciprocal condition number.
	return (1 / ainvnm) / anorm
}
