// Copyright ©2017 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import "gonum.org/v1/gonum/blas"

var (
	_ blas.Complex64  = Implementation{}
	_ blas.Complex128 = Implementation{}
)

// TODO(btracey): Replace this as complex routines are added, and instead
// automatically generate the complex64 routines from the complex128 ones.

var noComplex = "native: implementation does not implement this routine, see the cgo wrapper in gonum.org/v1/netlib/blas"

// Level 1 complex64 routines.

func (Implementation) Cdotu(n int, x []complex64, incX int, y []complex64, incY int) (dotu complex64) {
	panic(noComplex)
}
func (Implementation) Cdotc(n int, x []complex64, incX int, y []complex64, incY int) (dotc complex64) {
	panic(noComplex)
}
func (Implementation) Scnrm2(n int, x []complex64, incX int) float32 {
	panic(noComplex)
}
func (Implementation) Scasum(n int, x []complex64, incX int) float32 {
	panic(noComplex)
}
func (Implementation) Icamax(n int, x []complex64, incX int) int {
	panic(noComplex)
}
func (Implementation) Cswap(n int, x []complex64, incX int, y []complex64, incY int) {
	panic(noComplex)
}
func (Implementation) Ccopy(n int, x []complex64, incX int, y []complex64, incY int) {
	panic(noComplex)
}
func (Implementation) Caxpy(n int, alpha complex64, x []complex64, incX int, y []complex64, incY int) {
	panic(noComplex)
}
func (Implementation) Cscal(n int, alpha complex64, x []complex64, incX int) {
	panic(noComplex)
}
func (Implementation) Csscal(n int, alpha float32, x []complex64, incX int) {
	panic(noComplex)
}

// Level 2 complex64 routines.

func (Implementation) Cgemv(tA blas.Transpose, m, n int, alpha complex64, a []complex64, lda int, x []complex64, incX int, beta complex64, y []complex64, incY int) {
	panic(noComplex)
}
func (Implementation) Cgbmv(tA blas.Transpose, m, n, kL, kU int, alpha complex64, a []complex64, lda int, x []complex64, incX int, beta complex64, y []complex64, incY int) {
	panic(noComplex)
}
func (Implementation) Ctrmv(ul blas.Uplo, tA blas.Transpose, d blas.Diag, n int, a []complex64, lda int, x []complex64, incX int) {
	panic(noComplex)
}
func (Implementation) Ctbmv(ul blas.Uplo, tA blas.Transpose, d blas.Diag, n, k int, a []complex64, lda int, x []complex64, incX int) {
	panic(noComplex)
}
func (Implementation) Ctpmv(ul blas.Uplo, tA blas.Transpose, d blas.Diag, n int, ap []complex64, x []complex64, incX int) {
	panic(noComplex)
}
func (Implementation) Ctrsv(ul blas.Uplo, tA blas.Transpose, d blas.Diag, n int, a []complex64, lda int, x []complex64, incX int) {
	panic(noComplex)
}
func (Implementation) Ctbsv(ul blas.Uplo, tA blas.Transpose, d blas.Diag, n, k int, a []complex64, lda int, x []complex64, incX int) {
	panic(noComplex)
}
func (Implementation) Ctpsv(ul blas.Uplo, tA blas.Transpose, d blas.Diag, n int, ap []complex64, x []complex64, incX int) {
	panic(noComplex)
}
func (Implementation) Chemv(ul blas.Uplo, n int, alpha complex64, a []complex64, lda int, x []complex64, incX int, beta complex64, y []complex64, incY int) {
	panic(noComplex)
}
func (Implementation) Chbmv(ul blas.Uplo, n, k int, alpha complex64, a []complex64, lda int, x []complex64, incX int, beta complex64, y []complex64, incY int) {
	panic(noComplex)
}
func (Implementation) Chpmv(ul blas.Uplo, n int, alpha complex64, ap []complex64, x []complex64, incX int, beta complex64, y []complex64, incY int) {
	panic(noComplex)
}
func (Implementation) Cgeru(m, n int, alpha complex64, x []complex64, incX int, y []complex64, incY int, a []complex64, lda int) {
	panic(noComplex)
}
func (Implementation) Cgerc(m, n int, alpha complex64, x []complex64, incX int, y []complex64, incY int, a []complex64, lda int) {
	panic(noComplex)
}
func (Implementation) Cher(ul blas.Uplo, n int, alpha float32, x []complex64, incX int, a []complex64, lda int) {
	panic(noComplex)
}
func (Implementation) Chpr(ul blas.Uplo, n int, alpha float32, x []complex64, incX int, a []complex64) {
	panic(noComplex)
}
func (Implementation) Cher2(ul blas.Uplo, n int, alpha complex64, x []complex64, incX int, y []complex64, incY int, a []complex64, lda int) {
	panic(noComplex)
}
func (Implementation) Chpr2(ul blas.Uplo, n int, alpha complex64, x []complex64, incX int, y []complex64, incY int, ap []complex64) {
	panic(noComplex)
}

// Level 3 complex64 routines.

func (Implementation) Cgemm(tA, tB blas.Transpose, m, n, k int, alpha complex64, a []complex64, lda int, b []complex64, ldb int, beta complex64, c []complex64, ldc int) {
	panic(noComplex)
}
func (Implementation) Csymm(s blas.Side, ul blas.Uplo, m, n int, alpha complex64, a []complex64, lda int, b []complex64, ldb int, beta complex64, c []complex64, ldc int) {
	panic(noComplex)
}
func (Implementation) Csyrk(ul blas.Uplo, t blas.Transpose, n, k int, alpha complex64, a []complex64, lda int, beta complex64, c []complex64, ldc int) {
	panic(noComplex)
}
func (Implementation) Csyr2k(ul blas.Uplo, t blas.Transpose, n, k int, alpha complex64, a []complex64, lda int, b []complex64, ldb int, beta complex64, c []complex64, ldc int) {
	panic(noComplex)
}
func (Implementation) Ctrmm(s blas.Side, ul blas.Uplo, tA blas.Transpose, d blas.Diag, m, n int, alpha complex64, a []complex64, lda int, b []complex64, ldb int) {
	panic(noComplex)
}
func (Implementation) Ctrsm(s blas.Side, ul blas.Uplo, tA blas.Transpose, d blas.Diag, m, n int, alpha complex64, a []complex64, lda int, b []complex64, ldb int) {
	panic(noComplex)
}
func (Implementation) Chemm(s blas.Side, ul blas.Uplo, m, n int, alpha complex64, a []complex64, lda int, b []complex64, ldb int, beta complex64, c []complex64, ldc int) {
	panic(noComplex)
}
func (Implementation) Cherk(ul blas.Uplo, t blas.Transpose, n, k int, alpha float32, a []complex64, lda int, beta float32, c []complex64, ldc int) {
	panic(noComplex)
}
func (Implementation) Cher2k(ul blas.Uplo, t blas.Transpose, n, k int, alpha complex64, a []complex64, lda int, b []complex64, ldb int, beta float32, c []complex64, ldc int) {
	panic(noComplex)
}

// Level 3 complex128 routines.

func (Implementation) Zgemm(tA, tB blas.Transpose, m, n, k int, alpha complex128, a []complex128, lda int, b []complex128, ldb int, beta complex128, c []complex128, ldc int) {
	panic(noComplex)
}
func (Implementation) Zsymm(s blas.Side, ul blas.Uplo, m, n int, alpha complex128, a []complex128, lda int, b []complex128, ldb int, beta complex128, c []complex128, ldc int) {
	panic(noComplex)
}
func (Implementation) Zsyrk(ul blas.Uplo, t blas.Transpose, n, k int, alpha complex128, a []complex128, lda int, beta complex128, c []complex128, ldc int) {
	panic(noComplex)
}
func (Implementation) Zsyr2k(ul blas.Uplo, t blas.Transpose, n, k int, alpha complex128, a []complex128, lda int, b []complex128, ldb int, beta complex128, c []complex128, ldc int) {
	panic(noComplex)
}
func (Implementation) Ztrmm(s blas.Side, ul blas.Uplo, tA blas.Transpose, d blas.Diag, m, n int, alpha complex128, a []complex128, lda int, b []complex128, ldb int) {
	panic(noComplex)
}
func (Implementation) Ztrsm(s blas.Side, ul blas.Uplo, tA blas.Transpose, d blas.Diag, m, n int, alpha complex128, a []complex128, lda int, b []complex128, ldb int) {
	panic(noComplex)
}
func (Implementation) Zhemm(s blas.Side, ul blas.Uplo, m, n int, alpha complex128, a []complex128, lda int, b []complex128, ldb int, beta complex128, c []complex128, ldc int) {
	panic(noComplex)
}
func (Implementation) Zherk(ul blas.Uplo, t blas.Transpose, n, k int, alpha float64, a []complex128, lda int, beta float64, c []complex128, ldc int) {
	panic(noComplex)
}
func (Implementation) Zher2k(ul blas.Uplo, t blas.Transpose, n, k int, alpha complex128, a []complex128, lda int, b []complex128, ldb int, beta float64, c []complex128, ldc int) {
	panic(noComplex)
}
