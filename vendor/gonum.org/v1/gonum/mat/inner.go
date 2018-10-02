// Copyright ©2014 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat

import (
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/gonum/internal/asm/f64"
)

// Inner computes the generalized inner product
//   x^T A y
// between column vectors x and y with matrix A. This is only a true inner product if
// A is symmetric positive definite, though the operation works for any matrix A.
//
// Inner panics if x.Len != m or y.Len != n when A is an m x n matrix.
func Inner(x Vector, a Matrix, y Vector) float64 {
	m, n := a.Dims()
	if x.Len() != m {
		panic(ErrShape)
	}
	if y.Len() != n {
		panic(ErrShape)
	}
	if m == 0 || n == 0 {
		return 0
	}

	var sum float64

	switch a := a.(type) {
	case RawSymmetricer:
		amat := a.RawSymmetric()
		if amat.Uplo != blas.Upper {
			// Panic as a string not a mat.Error.
			panic(badSymTriangle)
		}
		var xmat, ymat blas64.Vector
		if xrv, ok := x.(RawVectorer); ok {
			xmat = xrv.RawVector()
		} else {
			break
		}
		if yrv, ok := y.(RawVectorer); ok {
			ymat = yrv.RawVector()
		} else {
			break
		}
		for i := 0; i < x.Len(); i++ {
			xi := x.AtVec(i)
			if xi != 0 {
				if ymat.Inc == 1 {
					sum += xi * f64.DotUnitary(
						amat.Data[i*amat.Stride+i:i*amat.Stride+n],
						ymat.Data[i:],
					)
				} else {
					sum += xi * f64.DotInc(
						amat.Data[i*amat.Stride+i:i*amat.Stride+n],
						ymat.Data[i*ymat.Inc:], uintptr(n-i),
						1, uintptr(ymat.Inc),
						0, 0,
					)
				}
			}
			yi := y.AtVec(i)
			if i != n-1 && yi != 0 {
				if xmat.Inc == 1 {
					sum += yi * f64.DotUnitary(
						amat.Data[i*amat.Stride+i+1:i*amat.Stride+n],
						xmat.Data[i+1:],
					)
				} else {
					sum += yi * f64.DotInc(
						amat.Data[i*amat.Stride+i+1:i*amat.Stride+n],
						xmat.Data[(i+1)*xmat.Inc:], uintptr(n-i-1),
						1, uintptr(xmat.Inc),
						0, 0,
					)
				}
			}
		}
		return sum
	case RawMatrixer:
		amat := a.RawMatrix()
		var ymat blas64.Vector
		if yrv, ok := y.(RawVectorer); ok {
			ymat = yrv.RawVector()
		} else {
			break
		}
		for i := 0; i < x.Len(); i++ {
			xi := x.AtVec(i)
			if xi != 0 {
				if ymat.Inc == 1 {
					sum += xi * f64.DotUnitary(
						amat.Data[i*amat.Stride:i*amat.Stride+n],
						ymat.Data,
					)
				} else {
					sum += xi * f64.DotInc(
						amat.Data[i*amat.Stride:i*amat.Stride+n],
						ymat.Data, uintptr(n),
						1, uintptr(ymat.Inc),
						0, 0,
					)
				}
			}
		}
		return sum
	}
	for i := 0; i < x.Len(); i++ {
		xi := x.AtVec(i)
		for j := 0; j < y.Len(); j++ {
			sum += xi * a.At(i, j) * y.AtVec(j)
		}
	}
	return sum
}
