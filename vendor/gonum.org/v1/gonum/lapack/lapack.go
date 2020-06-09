// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lapack

import "gonum.org/v1/gonum/blas"

// Complex128 defines the public complex128 LAPACK API supported by gonum/lapack.
type Complex128 interface{}

// Float64 defines the public float64 LAPACK API supported by gonum/lapack.
type Float64 interface {
	Dgecon(norm MatrixNorm, n int, a []float64, lda int, anorm float64, work []float64, iwork []int) float64
	Dgeev(jobvl LeftEVJob, jobvr RightEVJob, n int, a []float64, lda int, wr, wi []float64, vl []float64, ldvl int, vr []float64, ldvr int, work []float64, lwork int) (first int)
	Dgels(trans blas.Transpose, m, n, nrhs int, a []float64, lda int, b []float64, ldb int, work []float64, lwork int) bool
	Dgelqf(m, n int, a []float64, lda int, tau, work []float64, lwork int)
	Dgeqrf(m, n int, a []float64, lda int, tau, work []float64, lwork int)
	Dgesvd(jobU, jobVT SVDJob, m, n int, a []float64, lda int, s, u []float64, ldu int, vt []float64, ldvt int, work []float64, lwork int) (ok bool)
	Dgetrf(m, n int, a []float64, lda int, ipiv []int) (ok bool)
	Dgetri(n int, a []float64, lda int, ipiv []int, work []float64, lwork int) (ok bool)
	Dgetrs(trans blas.Transpose, n, nrhs int, a []float64, lda int, ipiv []int, b []float64, ldb int)
	Dggsvd3(jobU, jobV, jobQ GSVDJob, m, n, p int, a []float64, lda int, b []float64, ldb int, alpha, beta, u []float64, ldu int, v []float64, ldv int, q []float64, ldq int, work []float64, lwork int, iwork []int) (k, l int, ok bool)
	Dlantr(norm MatrixNorm, uplo blas.Uplo, diag blas.Diag, m, n int, a []float64, lda int, work []float64) float64
	Dlange(norm MatrixNorm, m, n int, a []float64, lda int, work []float64) float64
	Dlansy(norm MatrixNorm, uplo blas.Uplo, n int, a []float64, lda int, work []float64) float64
	Dlapmt(forward bool, m, n int, x []float64, ldx int, k []int)
	Dormqr(side blas.Side, trans blas.Transpose, m, n, k int, a []float64, lda int, tau, c []float64, ldc int, work []float64, lwork int)
	Dormlq(side blas.Side, trans blas.Transpose, m, n, k int, a []float64, lda int, tau, c []float64, ldc int, work []float64, lwork int)
	Dpocon(uplo blas.Uplo, n int, a []float64, lda int, anorm float64, work []float64, iwork []int) float64
	Dpotrf(ul blas.Uplo, n int, a []float64, lda int) (ok bool)
	Dpotri(ul blas.Uplo, n int, a []float64, lda int) (ok bool)
	Dpotrs(ul blas.Uplo, n, nrhs int, a []float64, lda int, b []float64, ldb int)
	Dsyev(jobz EVJob, uplo blas.Uplo, n int, a []float64, lda int, w, work []float64, lwork int) (ok bool)
	Dtrcon(norm MatrixNorm, uplo blas.Uplo, diag blas.Diag, n int, a []float64, lda int, work []float64, iwork []int) float64
	Dtrtri(uplo blas.Uplo, diag blas.Diag, n int, a []float64, lda int) (ok bool)
	Dtrtrs(uplo blas.Uplo, trans blas.Transpose, diag blas.Diag, n, nrhs int, a []float64, lda int, b []float64, ldb int) (ok bool)
}

// Direct specifies the direction of the multiplication for the Householder matrix.
type Direct byte

const (
	Forward  Direct = 'F' // Reflectors are right-multiplied, H_0 * H_1 * ... * H_{k-1}.
	Backward Direct = 'B' // Reflectors are left-multiplied, H_{k-1} * ... * H_1 * H_0.
)

// Sort is the sorting order.
type Sort byte

const (
	SortIncreasing Sort = 'I'
	SortDecreasing Sort = 'D'
)

// StoreV indicates the storage direction of elementary reflectors.
type StoreV byte

const (
	ColumnWise StoreV = 'C' // Reflector stored in a column of the matrix.
	RowWise    StoreV = 'R' // Reflector stored in a row of the matrix.
)

// MatrixNorm represents the kind of matrix norm to compute.
type MatrixNorm byte

const (
	MaxAbs       MatrixNorm = 'M' // max(abs(A(i,j)))
	MaxColumnSum MatrixNorm = 'O' // Maximum absolute column sum (one norm)
	MaxRowSum    MatrixNorm = 'I' // Maximum absolute row sum (infinity norm)
	Frobenius    MatrixNorm = 'F' // Frobenius norm (sqrt of sum of squares)
)

// MatrixType represents the kind of matrix represented in the data.
type MatrixType byte

const (
	General  MatrixType = 'G' // A general dense matrix.
	UpperTri MatrixType = 'U' // An upper triangular matrix.
	LowerTri MatrixType = 'L' // A lower triangular matrix.
)

// Pivot specifies the pivot type for plane rotations.
type Pivot byte

const (
	Variable Pivot = 'V'
	Top      Pivot = 'T'
	Bottom   Pivot = 'B'
)

// ApplyOrtho specifies which orthogonal matrix is applied in Dormbr.
type ApplyOrtho byte

const (
	ApplyP ApplyOrtho = 'P' // Apply P or Pᵀ.
	ApplyQ ApplyOrtho = 'Q' // Apply Q or Qᵀ.
)

// GenOrtho specifies which orthogonal matrix is generated in Dorgbr.
type GenOrtho byte

const (
	GeneratePT GenOrtho = 'P' // Generate Pᵀ.
	GenerateQ  GenOrtho = 'Q' // Generate Q.
)

// SVDJob specifies the singular vector computation type for SVD.
type SVDJob byte

const (
	SVDAll       SVDJob = 'A' // Compute all columns of the orthogonal matrix U or V.
	SVDStore     SVDJob = 'S' // Compute the singular vectors and store them in the orthogonal matrix U or V.
	SVDOverwrite SVDJob = 'O' // Compute the singular vectors and overwrite them on the input matrix A.
	SVDNone      SVDJob = 'N' // Do not compute singular vectors.
)

// GSVDJob specifies the singular vector computation type for Generalized SVD.
type GSVDJob byte

const (
	GSVDU    GSVDJob = 'U' // Compute orthogonal matrix U.
	GSVDV    GSVDJob = 'V' // Compute orthogonal matrix V.
	GSVDQ    GSVDJob = 'Q' // Compute orthogonal matrix Q.
	GSVDUnit GSVDJob = 'I' // Use unit-initialized matrix.
	GSVDNone GSVDJob = 'N' // Do not compute orthogonal matrix.
)

// EVComp specifies how eigenvectors are computed in Dsteqr.
type EVComp byte

const (
	EVOrig     EVComp = 'V' // Compute eigenvectors of the original symmetric matrix.
	EVTridiag  EVComp = 'I' // Compute eigenvectors of the tridiagonal matrix.
	EVCompNone EVComp = 'N' // Do not compute eigenvectors.
)

// EVJob specifies whether eigenvectors are computed in Dsyev.
type EVJob byte

const (
	EVCompute EVJob = 'V' // Compute eigenvectors.
	EVNone    EVJob = 'N' // Do not compute eigenvectors.
)

// LeftEVJob specifies whether left eigenvectors are computed in Dgeev.
type LeftEVJob byte

const (
	LeftEVCompute LeftEVJob = 'V' // Compute left eigenvectors.
	LeftEVNone    LeftEVJob = 'N' // Do not compute left eigenvectors.
)

// RightEVJob specifies whether right eigenvectors are computed in Dgeev.
type RightEVJob byte

const (
	RightEVCompute RightEVJob = 'V' // Compute right eigenvectors.
	RightEVNone    RightEVJob = 'N' // Do not compute right eigenvectors.
)

// BalanceJob specifies matrix balancing operation.
type BalanceJob byte

const (
	Permute      BalanceJob = 'P'
	Scale        BalanceJob = 'S'
	PermuteScale BalanceJob = 'B'
	BalanceNone  BalanceJob = 'N'
)

// SchurJob specifies whether the Schur form is computed in Dhseqr.
type SchurJob byte

const (
	EigenvaluesOnly     SchurJob = 'E'
	EigenvaluesAndSchur SchurJob = 'S'
)

// SchurComp specifies whether and how the Schur vectors are computed in Dhseqr.
type SchurComp byte

const (
	SchurOrig SchurComp = 'V' // Compute Schur vectors of the original matrix.
	SchurHess SchurComp = 'I' // Compute Schur vectors of the upper Hessenberg matrix.
	SchurNone SchurComp = 'N' // Do not compute Schur vectors.
)

// UpdateSchurComp specifies whether the matrix of Schur vectors is updated in Dtrexc.
type UpdateSchurComp byte

const (
	UpdateSchur     UpdateSchurComp = 'V' // Update the matrix of Schur vectors.
	UpdateSchurNone UpdateSchurComp = 'N' // Do not update the matrix of Schur vectors.
)

// EVSide specifies what eigenvectors are computed in Dtrevc3.
type EVSide byte

const (
	EVRight EVSide = 'R' // Compute only right eigenvectors.
	EVLeft  EVSide = 'L' // Compute only left eigenvectors.
	EVBoth  EVSide = 'B' // Compute both right and left eigenvectors.
)

// EVHowMany specifies which eigenvectors are computed in Dtrevc3 and how.
type EVHowMany byte

const (
	EVAll      EVHowMany = 'A' // Compute all right and/or left eigenvectors.
	EVAllMulQ  EVHowMany = 'B' // Compute all right and/or left eigenvectors multiplied by an input matrix.
	EVSelected EVHowMany = 'S' // Compute selected right and/or left eigenvectors.
)
