// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lapack64

import (
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/gonum/lapack"
	"gonum.org/v1/gonum/lapack/gonum"
)

var lapack64 lapack.Float64 = gonum.Implementation{}

// Use sets the LAPACK float64 implementation to be used by subsequent BLAS calls.
// The default implementation is native.Implementation.
func Use(l lapack.Float64) {
	lapack64 = l
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// Potrf computes the Cholesky factorization of a.
// The factorization has the form
//  A = Uᵀ * U  if a.Uplo == blas.Upper, or
//  A = L * Lᵀ  if a.Uplo == blas.Lower,
// where U is an upper triangular matrix and L is lower triangular.
// The triangular matrix is returned in t, and the underlying data between
// a and t is shared. The returned bool indicates whether a is positive
// definite and the factorization could be finished.
func Potrf(a blas64.Symmetric) (t blas64.Triangular, ok bool) {
	ok = lapack64.Dpotrf(a.Uplo, a.N, a.Data, max(1, a.Stride))
	t.Uplo = a.Uplo
	t.N = a.N
	t.Data = a.Data
	t.Stride = a.Stride
	t.Diag = blas.NonUnit
	return
}

// Potri computes the inverse of a real symmetric positive definite matrix A
// using its Cholesky factorization.
//
// On entry, t contains the triangular factor U or L from the Cholesky
// factorization A = Uᵀ*U or A = L*Lᵀ, as computed by Potrf.
//
// On return, the upper or lower triangle of the (symmetric) inverse of A is
// stored in t, overwriting the input factor U or L, and also returned in a. The
// underlying data between a and t is shared.
//
// The returned bool indicates whether the inverse was computed successfully.
func Potri(t blas64.Triangular) (a blas64.Symmetric, ok bool) {
	ok = lapack64.Dpotri(t.Uplo, t.N, t.Data, max(1, t.Stride))
	a.Uplo = t.Uplo
	a.N = t.N
	a.Data = t.Data
	a.Stride = t.Stride
	return
}

// Potrs solves a system of n linear equations A*X = B where A is an n×n
// symmetric positive definite matrix and B is an n×nrhs matrix, using the
// Cholesky factorization A = Uᵀ*U or A = L*Lᵀ. t contains the corresponding
// triangular factor as returned by Potrf. On entry, B contains the right-hand
// side matrix B, on return it contains the solution matrix X.
func Potrs(t blas64.Triangular, b blas64.General) {
	lapack64.Dpotrs(t.Uplo, t.N, b.Cols, t.Data, max(1, t.Stride), b.Data, max(1, b.Stride))
}

// Gecon estimates the reciprocal of the condition number of the n×n matrix A
// given the LU decomposition of the matrix. The condition number computed may
// be based on the 1-norm or the ∞-norm.
//
// a contains the result of the LU decomposition of A as computed by Getrf.
//
// anorm is the corresponding 1-norm or ∞-norm of the original matrix A.
//
// work is a temporary data slice of length at least 4*n and Gecon will panic otherwise.
//
// iwork is a temporary data slice of length at least n and Gecon will panic otherwise.
func Gecon(norm lapack.MatrixNorm, a blas64.General, anorm float64, work []float64, iwork []int) float64 {
	return lapack64.Dgecon(norm, a.Cols, a.Data, max(1, a.Stride), anorm, work, iwork)
}

// Gels finds a minimum-norm solution based on the matrices A and B using the
// QR or LQ factorization. Gels returns false if the matrix
// A is singular, and true if this solution was successfully found.
//
// The minimization problem solved depends on the input parameters.
//
//  1. If m >= n and trans == blas.NoTrans, Gels finds X such that || A*X - B||_2
//     is minimized.
//  2. If m < n and trans == blas.NoTrans, Gels finds the minimum norm solution of
//     A * X = B.
//  3. If m >= n and trans == blas.Trans, Gels finds the minimum norm solution of
//     Aᵀ * X = B.
//  4. If m < n and trans == blas.Trans, Gels finds X such that || A*X - B||_2
//     is minimized.
// Note that the least-squares solutions (cases 1 and 3) perform the minimization
// per column of B. This is not the same as finding the minimum-norm matrix.
//
// The matrix A is a general matrix of size m×n and is modified during this call.
// The input matrix B is of size max(m,n)×nrhs, and serves two purposes. On entry,
// the elements of b specify the input matrix B. B has size m×nrhs if
// trans == blas.NoTrans, and n×nrhs if trans == blas.Trans. On exit, the
// leading submatrix of b contains the solution vectors X. If trans == blas.NoTrans,
// this submatrix is of size n×nrhs, and of size m×nrhs otherwise.
//
// Work is temporary storage, and lwork specifies the usable memory length.
// At minimum, lwork >= max(m,n) + max(m,n,nrhs), and this function will panic
// otherwise. A longer work will enable blocked algorithms to be called.
// In the special case that lwork == -1, work[0] will be set to the optimal working
// length.
func Gels(trans blas.Transpose, a blas64.General, b blas64.General, work []float64, lwork int) bool {
	return lapack64.Dgels(trans, a.Rows, a.Cols, b.Cols, a.Data, max(1, a.Stride), b.Data, max(1, b.Stride), work, lwork)
}

// Geqrf computes the QR factorization of the m×n matrix A using a blocked
// algorithm. A is modified to contain the information to construct Q and R.
// The upper triangle of a contains the matrix R. The lower triangular elements
// (not including the diagonal) contain the elementary reflectors. tau is modified
// to contain the reflector scales. tau must have length at least min(m,n), and
// this function will panic otherwise.
//
// The ith elementary reflector can be explicitly constructed by first extracting
// the
//  v[j] = 0           j < i
//  v[j] = 1           j == i
//  v[j] = a[j*lda+i]  j > i
// and computing H_i = I - tau[i] * v * vᵀ.
//
// The orthonormal matrix Q can be constucted from a product of these elementary
// reflectors, Q = H_0 * H_1 * ... * H_{k-1}, where k = min(m,n).
//
// Work is temporary storage, and lwork specifies the usable memory length.
// At minimum, lwork >= m and this function will panic otherwise.
// Geqrf is a blocked QR factorization, but the block size is limited
// by the temporary space available. If lwork == -1, instead of performing Geqrf,
// the optimal work length will be stored into work[0].
func Geqrf(a blas64.General, tau, work []float64, lwork int) {
	lapack64.Dgeqrf(a.Rows, a.Cols, a.Data, max(1, a.Stride), tau, work, lwork)
}

// Gelqf computes the LQ factorization of the m×n matrix A using a blocked
// algorithm. A is modified to contain the information to construct L and Q. The
// lower triangle of a contains the matrix L. The elements above the diagonal
// and the slice tau represent the matrix Q. tau is modified to contain the
// reflector scales. tau must have length at least min(m,n), and this function
// will panic otherwise.
//
// See Geqrf for a description of the elementary reflectors and orthonormal
// matrix Q. Q is constructed as a product of these elementary reflectors,
// Q = H_{k-1} * ... * H_1 * H_0.
//
// Work is temporary storage, and lwork specifies the usable memory length.
// At minimum, lwork >= m and this function will panic otherwise.
// Gelqf is a blocked LQ factorization, but the block size is limited
// by the temporary space available. If lwork == -1, instead of performing Gelqf,
// the optimal work length will be stored into work[0].
func Gelqf(a blas64.General, tau, work []float64, lwork int) {
	lapack64.Dgelqf(a.Rows, a.Cols, a.Data, max(1, a.Stride), tau, work, lwork)
}

// Gesvd computes the singular value decomposition of the input matrix A.
//
// The singular value decomposition is
//  A = U * Sigma * Vᵀ
// where Sigma is an m×n diagonal matrix containing the singular values of A,
// U is an m×m orthogonal matrix and V is an n×n orthogonal matrix. The first
// min(m,n) columns of U and V are the left and right singular vectors of A
// respectively.
//
// jobU and jobVT are options for computing the singular vectors. The behavior
// is as follows
//  jobU == lapack.SVDAll       All m columns of U are returned in u
//  jobU == lapack.SVDStore     The first min(m,n) columns are returned in u
//  jobU == lapack.SVDOverwrite The first min(m,n) columns of U are written into a
//  jobU == lapack.SVDNone      The columns of U are not computed.
// The behavior is the same for jobVT and the rows of Vᵀ. At most one of jobU
// and jobVT can equal lapack.SVDOverwrite, and Gesvd will panic otherwise.
//
// On entry, a contains the data for the m×n matrix A. During the call to Gesvd
// the data is overwritten. On exit, A contains the appropriate singular vectors
// if either job is lapack.SVDOverwrite.
//
// s is a slice of length at least min(m,n) and on exit contains the singular
// values in decreasing order.
//
// u contains the left singular vectors on exit, stored columnwise. If
// jobU == lapack.SVDAll, u is of size m×m. If jobU == lapack.SVDStore u is
// of size m×min(m,n). If jobU == lapack.SVDOverwrite or lapack.SVDNone, u is
// not used.
//
// vt contains the left singular vectors on exit, stored rowwise. If
// jobV == lapack.SVDAll, vt is of size n×m. If jobVT == lapack.SVDStore vt is
// of size min(m,n)×n. If jobVT == lapack.SVDOverwrite or lapack.SVDNone, vt is
// not used.
//
// work is a slice for storing temporary memory, and lwork is the usable size of
// the slice. lwork must be at least max(5*min(m,n), 3*min(m,n)+max(m,n)).
// If lwork == -1, instead of performing Gesvd, the optimal work length will be
// stored into work[0]. Gesvd will panic if the working memory has insufficient
// storage.
//
// Gesvd returns whether the decomposition successfully completed.
func Gesvd(jobU, jobVT lapack.SVDJob, a, u, vt blas64.General, s, work []float64, lwork int) (ok bool) {
	return lapack64.Dgesvd(jobU, jobVT, a.Rows, a.Cols, a.Data, max(1, a.Stride), s, u.Data, max(1, u.Stride), vt.Data, max(1, vt.Stride), work, lwork)
}

// Getrf computes the LU decomposition of the m×n matrix A.
// The LU decomposition is a factorization of A into
//  A = P * L * U
// where P is a permutation matrix, L is a unit lower triangular matrix, and
// U is a (usually) non-unit upper triangular matrix. On exit, L and U are stored
// in place into a.
//
// ipiv is a permutation vector. It indicates that row i of the matrix was
// changed with ipiv[i]. ipiv must have length at least min(m,n), and will panic
// otherwise. ipiv is zero-indexed.
//
// Getrf is the blocked version of the algorithm.
//
// Getrf returns whether the matrix A is singular. The LU decomposition will
// be computed regardless of the singularity of A, but division by zero
// will occur if the false is returned and the result is used to solve a
// system of equations.
func Getrf(a blas64.General, ipiv []int) bool {
	return lapack64.Dgetrf(a.Rows, a.Cols, a.Data, max(1, a.Stride), ipiv)
}

// Getri computes the inverse of the matrix A using the LU factorization computed
// by Getrf. On entry, a contains the PLU decomposition of A as computed by
// Getrf and on exit contains the reciprocal of the original matrix.
//
// Getri will not perform the inversion if the matrix is singular, and returns
// a boolean indicating whether the inversion was successful.
//
// Work is temporary storage, and lwork specifies the usable memory length.
// At minimum, lwork >= n and this function will panic otherwise.
// Getri is a blocked inversion, but the block size is limited
// by the temporary space available. If lwork == -1, instead of performing Getri,
// the optimal work length will be stored into work[0].
func Getri(a blas64.General, ipiv []int, work []float64, lwork int) (ok bool) {
	return lapack64.Dgetri(a.Cols, a.Data, max(1, a.Stride), ipiv, work, lwork)
}

// Getrs solves a system of equations using an LU factorization.
// The system of equations solved is
//  A * X = B   if trans == blas.Trans
//  Aᵀ * X = B  if trans == blas.NoTrans
// A is a general n×n matrix with stride lda. B is a general matrix of size n×nrhs.
//
// On entry b contains the elements of the matrix B. On exit, b contains the
// elements of X, the solution to the system of equations.
//
// a and ipiv contain the LU factorization of A and the permutation indices as
// computed by Getrf. ipiv is zero-indexed.
func Getrs(trans blas.Transpose, a blas64.General, b blas64.General, ipiv []int) {
	lapack64.Dgetrs(trans, a.Cols, b.Cols, a.Data, max(1, a.Stride), ipiv, b.Data, max(1, b.Stride))
}

// Ggsvd3 computes the generalized singular value decomposition (GSVD)
// of an m×n matrix A and p×n matrix B:
//  Uᵀ*A*Q = D1*[ 0 R ]
//
//  Vᵀ*B*Q = D2*[ 0 R ]
// where U, V and Q are orthogonal matrices.
//
// Ggsvd3 returns k and l, the dimensions of the sub-blocks. k+l
// is the effective numerical rank of the (m+p)×n matrix [ Aᵀ Bᵀ ]ᵀ.
// R is a (k+l)×(k+l) nonsingular upper triangular matrix, D1 and
// D2 are m×(k+l) and p×(k+l) diagonal matrices and of the following
// structures, respectively:
//
// If m-k-l >= 0,
//
//                    k  l
//       D1 =     k [ I  0 ]
//                l [ 0  C ]
//            m-k-l [ 0  0 ]
//
//                  k  l
//       D2 = l   [ 0  S ]
//            p-l [ 0  0 ]
//
//               n-k-l  k    l
//  [ 0 R ] = k [  0   R11  R12 ] k
//            l [  0    0   R22 ] l
//
// where
//
//  C = diag( alpha_k, ... , alpha_{k+l} ),
//  S = diag( beta_k,  ... , beta_{k+l} ),
//  C^2 + S^2 = I.
//
// R is stored in
//  A[0:k+l, n-k-l:n]
// on exit.
//
// If m-k-l < 0,
//
//                 k m-k k+l-m
//      D1 =   k [ I  0    0  ]
//           m-k [ 0  C    0  ]
//
//                   k m-k k+l-m
//      D2 =   m-k [ 0  S    0  ]
//           k+l-m [ 0  0    I  ]
//             p-l [ 0  0    0  ]
//
//                 n-k-l  k   m-k  k+l-m
//  [ 0 R ] =    k [ 0    R11  R12  R13 ]
//             m-k [ 0     0   R22  R23 ]
//           k+l-m [ 0     0    0   R33 ]
//
// where
//  C = diag( alpha_k, ... , alpha_m ),
//  S = diag( beta_k,  ... , beta_m ),
//  C^2 + S^2 = I.
//
//  R = [ R11 R12 R13 ] is stored in A[1:m, n-k-l+1:n]
//      [  0  R22 R23 ]
// and R33 is stored in
//  B[m-k:l, n+m-k-l:n] on exit.
//
// Ggsvd3 computes C, S, R, and optionally the orthogonal transformation
// matrices U, V and Q.
//
// jobU, jobV and jobQ are options for computing the orthogonal matrices. The behavior
// is as follows
//  jobU == lapack.GSVDU        Compute orthogonal matrix U
//  jobU == lapack.GSVDNone     Do not compute orthogonal matrix.
// The behavior is the same for jobV and jobQ with the exception that instead of
// lapack.GSVDU these accept lapack.GSVDV and lapack.GSVDQ respectively.
// The matrices U, V and Q must be m×m, p×p and n×n respectively unless the
// relevant job parameter is lapack.GSVDNone.
//
// alpha and beta must have length n or Ggsvd3 will panic. On exit, alpha and
// beta contain the generalized singular value pairs of A and B
//   alpha[0:k] = 1,
//   beta[0:k]  = 0,
// if m-k-l >= 0,
//   alpha[k:k+l] = diag(C),
//   beta[k:k+l]  = diag(S),
// if m-k-l < 0,
//   alpha[k:m]= C, alpha[m:k+l]= 0
//   beta[k:m] = S, beta[m:k+l] = 1.
// if k+l < n,
//   alpha[k+l:n] = 0 and
//   beta[k+l:n]  = 0.
//
// On exit, iwork contains the permutation required to sort alpha descending.
//
// iwork must have length n, work must have length at least max(1, lwork), and
// lwork must be -1 or greater than n, otherwise Ggsvd3 will panic. If
// lwork is -1, work[0] holds the optimal lwork on return, but Ggsvd3 does
// not perform the GSVD.
func Ggsvd3(jobU, jobV, jobQ lapack.GSVDJob, a, b blas64.General, alpha, beta []float64, u, v, q blas64.General, work []float64, lwork int, iwork []int) (k, l int, ok bool) {
	return lapack64.Dggsvd3(jobU, jobV, jobQ, a.Rows, a.Cols, b.Rows, a.Data, max(1, a.Stride), b.Data, max(1, b.Stride), alpha, beta, u.Data, max(1, u.Stride), v.Data, max(1, v.Stride), q.Data, max(1, q.Stride), work, lwork, iwork)
}

// Lange computes the matrix norm of the general m×n matrix A. The input norm
// specifies the norm computed.
//  lapack.MaxAbs: the maximum absolute value of an element.
//  lapack.MaxColumnSum: the maximum column sum of the absolute values of the entries.
//  lapack.MaxRowSum: the maximum row sum of the absolute values of the entries.
//  lapack.Frobenius: the square root of the sum of the squares of the entries.
// If norm == lapack.MaxColumnSum, work must be of length n, and this function will panic otherwise.
// There are no restrictions on work for the other matrix norms.
func Lange(norm lapack.MatrixNorm, a blas64.General, work []float64) float64 {
	return lapack64.Dlange(norm, a.Rows, a.Cols, a.Data, max(1, a.Stride), work)
}

// Lansy computes the specified norm of an n×n symmetric matrix. If
// norm == lapack.MaxColumnSum or norm == lapackMaxRowSum work must have length
// at least n and this function will panic otherwise.
// There are no restrictions on work for the other matrix norms.
func Lansy(norm lapack.MatrixNorm, a blas64.Symmetric, work []float64) float64 {
	return lapack64.Dlansy(norm, a.Uplo, a.N, a.Data, max(1, a.Stride), work)
}

// Lantr computes the specified norm of an m×n trapezoidal matrix A. If
// norm == lapack.MaxColumnSum work must have length at least n and this function
// will panic otherwise. There are no restrictions on work for the other matrix norms.
func Lantr(norm lapack.MatrixNorm, a blas64.Triangular, work []float64) float64 {
	return lapack64.Dlantr(norm, a.Uplo, a.Diag, a.N, a.N, a.Data, max(1, a.Stride), work)
}

// Lapmt rearranges the columns of the m×n matrix X as specified by the
// permutation k_0, k_1, ..., k_{n-1} of the integers 0, ..., n-1.
//
// If forward is true a forward permutation is performed:
//
//  X[0:m, k[j]] is moved to X[0:m, j] for j = 0, 1, ..., n-1.
//
// otherwise a backward permutation is performed:
//
//  X[0:m, j] is moved to X[0:m, k[j]] for j = 0, 1, ..., n-1.
//
// k must have length n, otherwise Lapmt will panic. k is zero-indexed.
func Lapmt(forward bool, x blas64.General, k []int) {
	lapack64.Dlapmt(forward, x.Rows, x.Cols, x.Data, max(1, x.Stride), k)
}

// Ormlq multiplies the matrix C by the othogonal matrix Q defined by
// A and tau. A and tau are as returned from Gelqf.
//  C = Q * C   if side == blas.Left and trans == blas.NoTrans
//  C = Qᵀ * C  if side == blas.Left and trans == blas.Trans
//  C = C * Q   if side == blas.Right and trans == blas.NoTrans
//  C = C * Qᵀ  if side == blas.Right and trans == blas.Trans
// If side == blas.Left, A is a matrix of side k×m, and if side == blas.Right
// A is of size k×n. This uses a blocked algorithm.
//
// Work is temporary storage, and lwork specifies the usable memory length.
// At minimum, lwork >= m if side == blas.Left and lwork >= n if side == blas.Right,
// and this function will panic otherwise.
// Ormlq uses a block algorithm, but the block size is limited
// by the temporary space available. If lwork == -1, instead of performing Ormlq,
// the optimal work length will be stored into work[0].
//
// Tau contains the Householder scales and must have length at least k, and
// this function will panic otherwise.
func Ormlq(side blas.Side, trans blas.Transpose, a blas64.General, tau []float64, c blas64.General, work []float64, lwork int) {
	lapack64.Dormlq(side, trans, c.Rows, c.Cols, a.Rows, a.Data, max(1, a.Stride), tau, c.Data, max(1, c.Stride), work, lwork)
}

// Ormqr multiplies an m×n matrix C by an orthogonal matrix Q as
//  C = Q * C   if side == blas.Left  and trans == blas.NoTrans,
//  C = Qᵀ * C  if side == blas.Left  and trans == blas.Trans,
//  C = C * Q   if side == blas.Right and trans == blas.NoTrans,
//  C = C * Qᵀ  if side == blas.Right and trans == blas.Trans,
// where Q is defined as the product of k elementary reflectors
//  Q = H_0 * H_1 * ... * H_{k-1}.
//
// If side == blas.Left, A is an m×k matrix and 0 <= k <= m.
// If side == blas.Right, A is an n×k matrix and 0 <= k <= n.
// The ith column of A contains the vector which defines the elementary
// reflector H_i and tau[i] contains its scalar factor. tau must have length k
// and Ormqr will panic otherwise. Geqrf returns A and tau in the required
// form.
//
// work must have length at least max(1,lwork), and lwork must be at least n if
// side == blas.Left and at least m if side == blas.Right, otherwise Ormqr will
// panic.
//
// work is temporary storage, and lwork specifies the usable memory length. At
// minimum, lwork >= m if side == blas.Left and lwork >= n if side ==
// blas.Right, and this function will panic otherwise. Larger values of lwork
// will generally give better performance. On return, work[0] will contain the
// optimal value of lwork.
//
// If lwork is -1, instead of performing Ormqr, the optimal workspace size will
// be stored into work[0].
func Ormqr(side blas.Side, trans blas.Transpose, a blas64.General, tau []float64, c blas64.General, work []float64, lwork int) {
	lapack64.Dormqr(side, trans, c.Rows, c.Cols, a.Cols, a.Data, max(1, a.Stride), tau, c.Data, max(1, c.Stride), work, lwork)
}

// Pocon estimates the reciprocal of the condition number of a positive-definite
// matrix A given the Cholesky decmposition of A. The condition number computed
// is based on the 1-norm and the ∞-norm.
//
// anorm is the 1-norm and the ∞-norm of the original matrix A.
//
// work is a temporary data slice of length at least 3*n and Pocon will panic otherwise.
//
// iwork is a temporary data slice of length at least n and Pocon will panic otherwise.
func Pocon(a blas64.Symmetric, anorm float64, work []float64, iwork []int) float64 {
	return lapack64.Dpocon(a.Uplo, a.N, a.Data, max(1, a.Stride), anorm, work, iwork)
}

// Syev computes all eigenvalues and, optionally, the eigenvectors of a real
// symmetric matrix A.
//
// w contains the eigenvalues in ascending order upon return. w must have length
// at least n, and Syev will panic otherwise.
//
// On entry, a contains the elements of the symmetric matrix A in the triangular
// portion specified by uplo. If jobz == lapack.EVCompute, a contains the
// orthonormal eigenvectors of A on exit, otherwise jobz must be lapack.EVNone
// and on exit the specified triangular region is overwritten.
//
// Work is temporary storage, and lwork specifies the usable memory length. At minimum,
// lwork >= 3*n-1, and Syev will panic otherwise. The amount of blocking is
// limited by the usable length. If lwork == -1, instead of computing Syev the
// optimal work length is stored into work[0].
func Syev(jobz lapack.EVJob, a blas64.Symmetric, w, work []float64, lwork int) (ok bool) {
	return lapack64.Dsyev(jobz, a.Uplo, a.N, a.Data, max(1, a.Stride), w, work, lwork)
}

// Trcon estimates the reciprocal of the condition number of a triangular matrix A.
// The condition number computed may be based on the 1-norm or the ∞-norm.
//
// work is a temporary data slice of length at least 3*n and Trcon will panic otherwise.
//
// iwork is a temporary data slice of length at least n and Trcon will panic otherwise.
func Trcon(norm lapack.MatrixNorm, a blas64.Triangular, work []float64, iwork []int) float64 {
	return lapack64.Dtrcon(norm, a.Uplo, a.Diag, a.N, a.Data, max(1, a.Stride), work, iwork)
}

// Trtri computes the inverse of a triangular matrix, storing the result in place
// into a.
//
// Trtri will not perform the inversion if the matrix is singular, and returns
// a boolean indicating whether the inversion was successful.
func Trtri(a blas64.Triangular) (ok bool) {
	return lapack64.Dtrtri(a.Uplo, a.Diag, a.N, a.Data, max(1, a.Stride))
}

// Trtrs solves a triangular system of the form A * X = B or Aᵀ * X = B. Trtrs
// returns whether the solve completed successfully. If A is singular, no solve is performed.
func Trtrs(trans blas.Transpose, a blas64.Triangular, b blas64.General) (ok bool) {
	return lapack64.Dtrtrs(a.Uplo, trans, a.Diag, a.N, b.Cols, a.Data, max(1, a.Stride), b.Data, max(1, b.Stride))
}

// Geev computes the eigenvalues and, optionally, the left and/or right
// eigenvectors for an n×n real nonsymmetric matrix A.
//
// The right eigenvector v_j of A corresponding to an eigenvalue λ_j
// is defined by
//  A v_j = λ_j v_j,
// and the left eigenvector u_j corresponding to an eigenvalue λ_j is defined by
//  u_jᴴ A = λ_j u_jᴴ,
// where u_jᴴ is the conjugate transpose of u_j.
//
// On return, A will be overwritten and the left and right eigenvectors will be
// stored, respectively, in the columns of the n×n matrices VL and VR in the
// same order as their eigenvalues. If the j-th eigenvalue is real, then
//  u_j = VL[:,j],
//  v_j = VR[:,j],
// and if it is not real, then j and j+1 form a complex conjugate pair and the
// eigenvectors can be recovered as
//  u_j     = VL[:,j] + i*VL[:,j+1],
//  u_{j+1} = VL[:,j] - i*VL[:,j+1],
//  v_j     = VR[:,j] + i*VR[:,j+1],
//  v_{j+1} = VR[:,j] - i*VR[:,j+1],
// where i is the imaginary unit. The computed eigenvectors are normalized to
// have Euclidean norm equal to 1 and largest component real.
//
// Left eigenvectors will be computed only if jobvl == lapack.LeftEVCompute,
// otherwise jobvl must be lapack.LeftEVNone.
// Right eigenvectors will be computed only if jobvr == lapack.RightEVCompute,
// otherwise jobvr must be lapack.RightEVNone.
// For other values of jobvl and jobvr Geev will panic.
//
// On return, wr and wi will contain the real and imaginary parts, respectively,
// of the computed eigenvalues. Complex conjugate pairs of eigenvalues appear
// consecutively with the eigenvalue having the positive imaginary part first.
// wr and wi must have length n, and Geev will panic otherwise.
//
// work must have length at least lwork and lwork must be at least max(1,4*n) if
// the left or right eigenvectors are computed, and at least max(1,3*n) if no
// eigenvectors are computed. For good performance, lwork must generally be
// larger. On return, optimal value of lwork will be stored in work[0].
//
// If lwork == -1, instead of performing Geev, the function only calculates the
// optimal vaule of lwork and stores it into work[0].
//
// On return, first will be the index of the first valid eigenvalue.
// If first == 0, all eigenvalues and eigenvectors have been computed.
// If first is positive, Geev failed to compute all the eigenvalues, no
// eigenvectors have been computed and wr[first:] and wi[first:] contain those
// eigenvalues which have converged.
func Geev(jobvl lapack.LeftEVJob, jobvr lapack.RightEVJob, a blas64.General, wr, wi []float64, vl, vr blas64.General, work []float64, lwork int) (first int) {
	n := a.Rows
	if a.Cols != n {
		panic("lapack64: matrix not square")
	}
	if jobvl == lapack.LeftEVCompute && (vl.Rows != n || vl.Cols != n) {
		panic("lapack64: bad size of VL")
	}
	if jobvr == lapack.RightEVCompute && (vr.Rows != n || vr.Cols != n) {
		panic("lapack64: bad size of VR")
	}
	return lapack64.Dgeev(jobvl, jobvr, n, a.Data, max(1, a.Stride), wr, wi, vl.Data, max(1, vl.Stride), vr.Data, max(1, vr.Stride), work, lwork)
}
