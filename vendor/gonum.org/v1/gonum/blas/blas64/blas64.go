// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package blas64

import (
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/gonum"
)

var blas64 blas.Float64 = gonum.Implementation{}

// Use sets the BLAS float64 implementation to be used by subsequent BLAS calls.
// The default implementation is
// gonum.org/v1/gonum/blas/gonum.Implementation.
func Use(b blas.Float64) {
	blas64 = b
}

// Implementation returns the current BLAS float64 implementation.
//
// Implementation allows direct calls to the current the BLAS float64 implementation
// giving finer control of parameters.
func Implementation() blas.Float64 {
	return blas64
}

// Vector represents a vector with an associated element increment.
type Vector struct {
	N    int
	Data []float64
	Inc  int
}

// General represents a matrix using the conventional storage scheme.
type General struct {
	Rows, Cols int
	Data       []float64
	Stride     int
}

// Band represents a band matrix using the band storage scheme.
type Band struct {
	Rows, Cols int
	KL, KU     int
	Data       []float64
	Stride     int
}

// Triangular represents a triangular matrix using the conventional storage scheme.
type Triangular struct {
	Uplo   blas.Uplo
	Diag   blas.Diag
	N      int
	Data   []float64
	Stride int
}

// TriangularBand represents a triangular matrix using the band storage scheme.
type TriangularBand struct {
	Uplo   blas.Uplo
	Diag   blas.Diag
	N, K   int
	Data   []float64
	Stride int
}

// TriangularPacked represents a triangular matrix using the packed storage scheme.
type TriangularPacked struct {
	Uplo blas.Uplo
	Diag blas.Diag
	N    int
	Data []float64
}

// Symmetric represents a symmetric matrix using the conventional storage scheme.
type Symmetric struct {
	Uplo   blas.Uplo
	N      int
	Data   []float64
	Stride int
}

// SymmetricBand represents a symmetric matrix using the band storage scheme.
type SymmetricBand struct {
	Uplo   blas.Uplo
	N, K   int
	Data   []float64
	Stride int
}

// SymmetricPacked represents a symmetric matrix using the packed storage scheme.
type SymmetricPacked struct {
	Uplo blas.Uplo
	N    int
	Data []float64
}

// Level 1

const (
	negInc    = "blas64: negative vector increment"
	badLength = "blas64: vector length mismatch"
)

// Dot computes the dot product of the two vectors:
//  \sum_i x[i]*y[i].
func Dot(x, y Vector) float64 {
	if x.N != y.N {
		panic(badLength)
	}
	return blas64.Ddot(x.N, x.Data, x.Inc, y.Data, y.Inc)
}

// Nrm2 computes the Euclidean norm of the vector x:
//  sqrt(\sum_i x[i]*x[i]).
//
// Nrm2 will panic if the vector increment is negative.
func Nrm2(x Vector) float64 {
	if x.Inc < 0 {
		panic(negInc)
	}
	return blas64.Dnrm2(x.N, x.Data, x.Inc)
}

// Asum computes the sum of the absolute values of the elements of x:
//  \sum_i |x[i]|.
//
// Asum will panic if the vector increment is negative.
func Asum(x Vector) float64 {
	if x.Inc < 0 {
		panic(negInc)
	}
	return blas64.Dasum(x.N, x.Data, x.Inc)
}

// Iamax returns the index of an element of x with the largest absolute value.
// If there are multiple such indices the earliest is returned.
// Iamax returns -1 if n == 0.
//
// Iamax will panic if the vector increment is negative.
func Iamax(x Vector) int {
	if x.Inc < 0 {
		panic(negInc)
	}
	return blas64.Idamax(x.N, x.Data, x.Inc)
}

// Swap exchanges the elements of the two vectors:
//  x[i], y[i] = y[i], x[i] for all i.
func Swap(x, y Vector) {
	if x.N != y.N {
		panic(badLength)
	}
	blas64.Dswap(x.N, x.Data, x.Inc, y.Data, y.Inc)
}

// Copy copies the elements of x into the elements of y:
//  y[i] = x[i] for all i.
// Copy requires that the lengths of x and y match and will panic otherwise.
func Copy(x, y Vector) {
	if x.N != y.N {
		panic(badLength)
	}
	blas64.Dcopy(x.N, x.Data, x.Inc, y.Data, y.Inc)
}

// Axpy adds x scaled by alpha to y:
//  y[i] += alpha*x[i] for all i.
func Axpy(alpha float64, x, y Vector) {
	if x.N != y.N {
		panic(badLength)
	}
	blas64.Daxpy(x.N, alpha, x.Data, x.Inc, y.Data, y.Inc)
}

// Rotg computes the parameters of a Givens plane rotation so that
//  ⎡ c s⎤   ⎡a⎤   ⎡r⎤
//  ⎣-s c⎦ * ⎣b⎦ = ⎣0⎦
// where a and b are the Cartesian coordinates of a given point.
// c, s, and r are defined as
//  r = ±Sqrt(a^2 + b^2),
//  c = a/r, the cosine of the rotation angle,
//  s = a/r, the sine of the rotation angle,
// and z is defined such that
//  if |a| > |b|,        z = s,
//  otherwise if c != 0, z = 1/c,
//  otherwise            z = 1.
func Rotg(a, b float64) (c, s, r, z float64) {
	return blas64.Drotg(a, b)
}

// Rotmg computes the modified Givens rotation. See
// http://www.netlib.org/lapack/explore-html/df/deb/drotmg_8f.html
// for more details.
func Rotmg(d1, d2, b1, b2 float64) (p blas.DrotmParams, rd1, rd2, rb1 float64) {
	return blas64.Drotmg(d1, d2, b1, b2)
}

// Rot applies a plane transformation to n points represented by the vectors x
// and y:
//  x[i] =  c*x[i] + s*y[i],
//  y[i] = -s*x[i] + c*y[i], for all i.
func Rot(x, y Vector, c, s float64) {
	if x.N != y.N {
		panic(badLength)
	}
	blas64.Drot(x.N, x.Data, x.Inc, y.Data, y.Inc, c, s)
}

// Rotm applies the modified Givens rotation to n points represented by the
// vectors x and y.
func Rotm(x, y Vector, p blas.DrotmParams) {
	if x.N != y.N {
		panic(badLength)
	}
	blas64.Drotm(x.N, x.Data, x.Inc, y.Data, y.Inc, p)
}

// Scal scales the vector x by alpha:
//  x[i] *= alpha for all i.
//
// Scal will panic if the vector increment is negative.
func Scal(alpha float64, x Vector) {
	if x.Inc < 0 {
		panic(negInc)
	}
	blas64.Dscal(x.N, alpha, x.Data, x.Inc)
}

// Level 2

// Gemv computes
//  y = alpha * A * x + beta * y,   if t == blas.NoTrans,
//  y = alpha * A^T * x + beta * y, if t == blas.Trans or blas.ConjTrans,
// where A is an m×n dense matrix, x and y are vectors, and alpha and beta are scalars.
func Gemv(t blas.Transpose, alpha float64, a General, x Vector, beta float64, y Vector) {
	blas64.Dgemv(t, a.Rows, a.Cols, alpha, a.Data, a.Stride, x.Data, x.Inc, beta, y.Data, y.Inc)
}

// Gbmv computes
//  y = alpha * A * x + beta * y,   if t == blas.NoTrans,
//  y = alpha * A^T * x + beta * y, if t == blas.Trans or blas.ConjTrans,
// where A is an m×n band matrix, x and y are vectors, and alpha and beta are scalars.
func Gbmv(t blas.Transpose, alpha float64, a Band, x Vector, beta float64, y Vector) {
	blas64.Dgbmv(t, a.Rows, a.Cols, a.KL, a.KU, alpha, a.Data, a.Stride, x.Data, x.Inc, beta, y.Data, y.Inc)
}

// Trmv computes
//  x = A * x,   if t == blas.NoTrans,
//  x = A^T * x, if t == blas.Trans or blas.ConjTrans,
// where A is an n×n triangular matrix, and x is a vector.
func Trmv(t blas.Transpose, a Triangular, x Vector) {
	blas64.Dtrmv(a.Uplo, t, a.Diag, a.N, a.Data, a.Stride, x.Data, x.Inc)
}

// Tbmv computes
//  x = A * x,   if t == blas.NoTrans,
//  x = A^T * x, if t == blas.Trans or blas.ConjTrans,
// where A is an n×n triangular band matrix, and x is a vector.
func Tbmv(t blas.Transpose, a TriangularBand, x Vector) {
	blas64.Dtbmv(a.Uplo, t, a.Diag, a.N, a.K, a.Data, a.Stride, x.Data, x.Inc)
}

// Tpmv computes
//  x = A * x,   if t == blas.NoTrans,
//  x = A^T * x, if t == blas.Trans or blas.ConjTrans,
// where A is an n×n triangular matrix in packed format, and x is a vector.
func Tpmv(t blas.Transpose, a TriangularPacked, x Vector) {
	blas64.Dtpmv(a.Uplo, t, a.Diag, a.N, a.Data, x.Data, x.Inc)
}

// Trsv solves
//  A * x = b,   if t == blas.NoTrans,
//  A^T * x = b, if t == blas.Trans or blas.ConjTrans,
// where A is an n×n triangular matrix, and x and b are vectors.
//
// At entry to the function, x contains the values of b, and the result is
// stored in-place into x.
//
// No test for singularity or near-singularity is included in this
// routine. Such tests must be performed before calling this routine.
func Trsv(t blas.Transpose, a Triangular, x Vector) {
	blas64.Dtrsv(a.Uplo, t, a.Diag, a.N, a.Data, a.Stride, x.Data, x.Inc)
}

// Tbsv solves
//  A * x = b,   if t == blas.NoTrans,
//  A^T * x = b, if t == blas.Trans or blas.ConjTrans,
// where A is an n×n triangular band matrix, and x and b are vectors.
//
// At entry to the function, x contains the values of b, and the result is
// stored in place into x.
//
// No test for singularity or near-singularity is included in this
// routine. Such tests must be performed before calling this routine.
func Tbsv(t blas.Transpose, a TriangularBand, x Vector) {
	blas64.Dtbsv(a.Uplo, t, a.Diag, a.N, a.K, a.Data, a.Stride, x.Data, x.Inc)
}

// Tpsv solves
//  A * x = b,   if t == blas.NoTrans,
//  A^T * x = b, if t == blas.Trans or blas.ConjTrans,
// where A is an n×n triangular matrix in packed format, and x and b are
// vectors.
//
// At entry to the function, x contains the values of b, and the result is
// stored in place into x.
//
// No test for singularity or near-singularity is included in this
// routine. Such tests must be performed before calling this routine.
func Tpsv(t blas.Transpose, a TriangularPacked, x Vector) {
	blas64.Dtpsv(a.Uplo, t, a.Diag, a.N, a.Data, x.Data, x.Inc)
}

// Symv computes
//    y = alpha * A * x + beta * y,
// where A is an n×n symmetric matrix, x and y are vectors, and alpha and
// beta are scalars.
func Symv(alpha float64, a Symmetric, x Vector, beta float64, y Vector) {
	blas64.Dsymv(a.Uplo, a.N, alpha, a.Data, a.Stride, x.Data, x.Inc, beta, y.Data, y.Inc)
}

// Sbmv performs
//  y = alpha * A * x + beta * y,
// where A is an n×n symmetric band matrix, x and y are vectors, and alpha
// and beta are scalars.
func Sbmv(alpha float64, a SymmetricBand, x Vector, beta float64, y Vector) {
	blas64.Dsbmv(a.Uplo, a.N, a.K, alpha, a.Data, a.Stride, x.Data, x.Inc, beta, y.Data, y.Inc)
}

// Spmv performs
//    y = alpha * A * x + beta * y,
// where A is an n×n symmetric matrix in packed format, x and y are vectors,
// and alpha and beta are scalars.
func Spmv(alpha float64, a SymmetricPacked, x Vector, beta float64, y Vector) {
	blas64.Dspmv(a.Uplo, a.N, alpha, a.Data, x.Data, x.Inc, beta, y.Data, y.Inc)
}

// Ger performs a rank-1 update
//  A += alpha * x * y^T,
// where A is an m×n dense matrix, x and y are vectors, and alpha is a scalar.
func Ger(alpha float64, x, y Vector, a General) {
	blas64.Dger(a.Rows, a.Cols, alpha, x.Data, x.Inc, y.Data, y.Inc, a.Data, a.Stride)
}

// Syr performs a rank-1 update
//  A += alpha * x * x^T,
// where A is an n×n symmetric matrix, x is a vector, and alpha is a scalar.
func Syr(alpha float64, x Vector, a Symmetric) {
	blas64.Dsyr(a.Uplo, a.N, alpha, x.Data, x.Inc, a.Data, a.Stride)
}

// Spr performs the rank-1 update
//  A += alpha * x * x^T,
// where A is an n×n symmetric matrix in packed format, x is a vector, and
// alpha is a scalar.
func Spr(alpha float64, x Vector, a SymmetricPacked) {
	blas64.Dspr(a.Uplo, a.N, alpha, x.Data, x.Inc, a.Data)
}

// Syr2 performs a rank-2 update
//  A += alpha * x * y^T + alpha * y * x^T,
// where A is a symmetric n×n matrix, x and y are vectors, and alpha is a scalar.
func Syr2(alpha float64, x, y Vector, a Symmetric) {
	blas64.Dsyr2(a.Uplo, a.N, alpha, x.Data, x.Inc, y.Data, y.Inc, a.Data, a.Stride)
}

// Spr2 performs a rank-2 update
//  A += alpha * x * y^T + alpha * y * x^T,
// where A is an n×n symmetric matrix in packed format, x and y are vectors,
// and alpha is a scalar.
func Spr2(alpha float64, x, y Vector, a SymmetricPacked) {
	blas64.Dspr2(a.Uplo, a.N, alpha, x.Data, x.Inc, y.Data, y.Inc, a.Data)
}

// Level 3

// Gemm computes
//  C = alpha * A * B + beta * C,
// where A, B, and C are dense matrices, and alpha and beta are scalars.
// tA and tB specify whether A or B are transposed.
func Gemm(tA, tB blas.Transpose, alpha float64, a, b General, beta float64, c General) {
	var m, n, k int
	if tA == blas.NoTrans {
		m, k = a.Rows, a.Cols
	} else {
		m, k = a.Cols, a.Rows
	}
	if tB == blas.NoTrans {
		n = b.Cols
	} else {
		n = b.Rows
	}
	blas64.Dgemm(tA, tB, m, n, k, alpha, a.Data, a.Stride, b.Data, b.Stride, beta, c.Data, c.Stride)
}

// Symm performs
//  C = alpha * A * B + beta * C, if s == blas.Left,
//  C = alpha * B * A + beta * C, if s == blas.Right,
// where A is an n×n or m×m symmetric matrix, B and C are m×n matrices, and
// alpha is a scalar.
func Symm(s blas.Side, alpha float64, a Symmetric, b General, beta float64, c General) {
	var m, n int
	if s == blas.Left {
		m, n = a.N, b.Cols
	} else {
		m, n = b.Rows, a.N
	}
	blas64.Dsymm(s, a.Uplo, m, n, alpha, a.Data, a.Stride, b.Data, b.Stride, beta, c.Data, c.Stride)
}

// Syrk performs a symmetric rank-k update
//  C = alpha * A * A^T + beta * C, if t == blas.NoTrans,
//  C = alpha * A^T * A + beta * C, if t == blas.Trans or blas.ConjTrans,
// where C is an n×n symmetric matrix, A is an n×k matrix if t == blas.NoTrans and
// a k×n matrix otherwise, and alpha and beta are scalars.
func Syrk(t blas.Transpose, alpha float64, a General, beta float64, c Symmetric) {
	var n, k int
	if t == blas.NoTrans {
		n, k = a.Rows, a.Cols
	} else {
		n, k = a.Cols, a.Rows
	}
	blas64.Dsyrk(c.Uplo, t, n, k, alpha, a.Data, a.Stride, beta, c.Data, c.Stride)
}

// Syr2k performs a symmetric rank-2k update
//  C = alpha * A * B^T + alpha * B * A^T + beta * C, if t == blas.NoTrans,
//  C = alpha * A^T * B + alpha * B^T * A + beta * C, if t == blas.Trans or blas.ConjTrans,
// where C is an n×n symmetric matrix, A and B are n×k matrices if t == NoTrans
// and k×n matrices otherwise, and alpha and beta are scalars.
func Syr2k(t blas.Transpose, alpha float64, a, b General, beta float64, c Symmetric) {
	var n, k int
	if t == blas.NoTrans {
		n, k = a.Rows, a.Cols
	} else {
		n, k = a.Cols, a.Rows
	}
	blas64.Dsyr2k(c.Uplo, t, n, k, alpha, a.Data, a.Stride, b.Data, b.Stride, beta, c.Data, c.Stride)
}

// Trmm performs
//  B = alpha * A * B,   if tA == blas.NoTrans and s == blas.Left,
//  B = alpha * A^T * B, if tA == blas.Trans or blas.ConjTrans, and s == blas.Left,
//  B = alpha * B * A,   if tA == blas.NoTrans and s == blas.Right,
//  B = alpha * B * A^T, if tA == blas.Trans or blas.ConjTrans, and s == blas.Right,
// where A is an n×n or m×m triangular matrix, B is an m×n matrix, and alpha is
// a scalar.
func Trmm(s blas.Side, tA blas.Transpose, alpha float64, a Triangular, b General) {
	blas64.Dtrmm(s, a.Uplo, tA, a.Diag, b.Rows, b.Cols, alpha, a.Data, a.Stride, b.Data, b.Stride)
}

// Trsm solves
//  A * X = alpha * B,   if tA == blas.NoTrans and s == blas.Left,
//  A^T * X = alpha * B, if tA == blas.Trans or blas.ConjTrans, and s == blas.Left,
//  X * A = alpha * B,   if tA == blas.NoTrans and s == blas.Right,
//  X * A^T = alpha * B, if tA == blas.Trans or blas.ConjTrans, and s == blas.Right,
// where A is an n×n or m×m triangular matrix, X and B are m×n matrices, and
// alpha is a scalar.
//
// At entry to the function, X contains the values of B, and the result is
// stored in-place into X.
//
// No check is made that A is invertible.
func Trsm(s blas.Side, tA blas.Transpose, alpha float64, a Triangular, b General) {
	blas64.Dtrsm(s, a.Uplo, tA, a.Diag, b.Rows, b.Cols, alpha, a.Data, a.Stride, b.Data, b.Stride)
}
