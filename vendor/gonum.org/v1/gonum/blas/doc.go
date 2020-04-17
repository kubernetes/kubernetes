// Copyright ©2017 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package blas provides interfaces for the BLAS linear algebra standard.

All methods must perform appropriate parameter checking and panic if
provided parameters that do not conform to the requirements specified
by the BLAS standard.

Quick Reference Guide to the BLAS from http://www.netlib.org/lapack/lug/node145.html

This version is modified to remove the "order" option. All matrix operations are
on row-order matrices.

Level 1 BLAS

	        dim scalar vector   vector   scalars              5-element prefixes
	                                                          struct

	_rotg (                                      a, b )                S, D
	_rotmg(                              d1, d2, a, b )                S, D
	_rot  ( n,         x, incX, y, incY,               c, s )          S, D
	_rotm ( n,         x, incX, y, incY,                      param )  S, D
	_swap ( n,         x, incX, y, incY )                              S, D, C, Z
	_scal ( n,  alpha, x, incX )                                       S, D, C, Z, Cs, Zd
	_copy ( n,         x, incX, y, incY )                              S, D, C, Z
	_axpy ( n,  alpha, x, incX, y, incY )                              S, D, C, Z
	_dot  ( n,         x, incX, y, incY )                              S, D, Ds
	_dotu ( n,         x, incX, y, incY )                              C, Z
	_dotc ( n,         x, incX, y, incY )                              C, Z
	__dot ( n,  alpha, x, incX, y, incY )                              Sds
	_nrm2 ( n,         x, incX )                                       S, D, Sc, Dz
	_asum ( n,         x, incX )                                       S, D, Sc, Dz
	I_amax( n,         x, incX )                                       s, d, c, z

Level 2 BLAS

	        options                   dim   b-width scalar matrix  vector   scalar vector   prefixes

	_gemv (        trans,      m, n,         alpha, a, lda, x, incX, beta,  y, incY ) S, D, C, Z
	_gbmv (        trans,      m, n, kL, kU, alpha, a, lda, x, incX, beta,  y, incY ) S, D, C, Z
	_hemv ( uplo,                 n,         alpha, a, lda, x, incX, beta,  y, incY ) C, Z
	_hbmv ( uplo,                 n, k,      alpha, a, lda, x, incX, beta,  y, incY ) C, Z
	_hpmv ( uplo,                 n,         alpha, ap,     x, incX, beta,  y, incY ) C, Z
	_symv ( uplo,                 n,         alpha, a, lda, x, incX, beta,  y, incY ) S, D
	_sbmv ( uplo,                 n, k,      alpha, a, lda, x, incX, beta,  y, incY ) S, D
	_spmv ( uplo,                 n,         alpha, ap,     x, incX, beta,  y, incY ) S, D
	_trmv ( uplo, trans, diag,    n,                a, lda, x, incX )                 S, D, C, Z
	_tbmv ( uplo, trans, diag,    n, k,             a, lda, x, incX )                 S, D, C, Z
	_tpmv ( uplo, trans, diag,    n,                ap,     x, incX )                 S, D, C, Z
	_trsv ( uplo, trans, diag,    n,                a, lda, x, incX )                 S, D, C, Z
	_tbsv ( uplo, trans, diag,    n, k,             a, lda, x, incX )                 S, D, C, Z
	_tpsv ( uplo, trans, diag,    n,                ap,     x, incX )                 S, D, C, Z

	        options                   dim   scalar vector   vector   matrix  prefixes

	_ger  (                    m, n, alpha, x, incX, y, incY, a, lda ) S, D
	_geru (                    m, n, alpha, x, incX, y, incY, a, lda ) C, Z
	_gerc (                    m, n, alpha, x, incX, y, incY, a, lda ) C, Z
	_her  ( uplo,                 n, alpha, x, incX,          a, lda ) C, Z
	_hpr  ( uplo,                 n, alpha, x, incX,          ap )     C, Z
	_her2 ( uplo,                 n, alpha, x, incX, y, incY, a, lda ) C, Z
	_hpr2 ( uplo,                 n, alpha, x, incX, y, incY, ap )     C, Z
	_syr  ( uplo,                 n, alpha, x, incX,          a, lda ) S, D
	_spr  ( uplo,                 n, alpha, x, incX,          ap )     S, D
	_syr2 ( uplo,                 n, alpha, x, incX, y, incY, a, lda ) S, D
	_spr2 ( uplo,                 n, alpha, x, incX, y, incY, ap )     S, D

Level 3 BLAS

	        options                                 dim      scalar matrix  matrix  scalar matrix  prefixes

	_gemm (             transA, transB,      m, n, k, alpha, a, lda, b, ldb, beta,  c, ldc ) S, D, C, Z
	_symm ( side, uplo,                      m, n,    alpha, a, lda, b, ldb, beta,  c, ldc ) S, D, C, Z
	_hemm ( side, uplo,                      m, n,    alpha, a, lda, b, ldb, beta,  c, ldc ) C, Z
	_syrk (       uplo, trans,                  n, k, alpha, a, lda,         beta,  c, ldc ) S, D, C, Z
	_herk (       uplo, trans,                  n, k, alpha, a, lda,         beta,  c, ldc ) C, Z
	_syr2k(       uplo, trans,                  n, k, alpha, a, lda, b, ldb, beta,  c, ldc ) S, D, C, Z
	_her2k(       uplo, trans,                  n, k, alpha, a, lda, b, ldb, beta,  c, ldc ) C, Z
	_trmm ( side, uplo, transA,        diag, m, n,    alpha, a, lda, b, ldb )                S, D, C, Z
	_trsm ( side, uplo, transA,        diag, m, n,    alpha, a, lda, b, ldb )                S, D, C, Z

Meaning of prefixes

	S - float32	C - complex64
	D - float64	Z - complex128

Matrix types

	GE - GEneral 		GB - General Band
	SY - SYmmetric 		SB - Symmetric Band 	SP - Symmetric Packed
	HE - HErmitian 		HB - Hermitian Band 	HP - Hermitian Packed
	TR - TRiangular 	TB - Triangular Band 	TP - Triangular Packed

Options

	trans 	= NoTrans, Trans, ConjTrans
	uplo 	= Upper, Lower
	diag 	= Nonunit, Unit
	side 	= Left, Right (A or op(A) on the left, or A or op(A) on the right)

For real matrices, Trans and ConjTrans have the same meaning.
For Hermitian matrices, trans = Trans is not allowed.
For complex symmetric matrices, trans = ConjTrans is not allowed.
*/
package blas // import "gonum.org/v1/gonum/blas"
