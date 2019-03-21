// Copyright ©2017 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"math"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/gonum/lapack"
)

// Dtgsja computes the generalized singular value decomposition (GSVD)
// of two real upper triangular or trapezoidal matrices A and B.
//
// A and B have the following forms, which may be obtained by the
// preprocessing subroutine Dggsvp from a general m×n matrix A and p×n
// matrix B:
//
//            n-k-l  k    l
//  A =    k [  0   A12  A13 ] if m-k-l >= 0;
//         l [  0    0   A23 ]
//     m-k-l [  0    0    0  ]
//
//            n-k-l  k    l
//  A =    k [  0   A12  A13 ] if m-k-l < 0;
//       m-k [  0    0   A23 ]
//
//            n-k-l  k    l
//  B =    l [  0    0   B13 ]
//       p-l [  0    0    0  ]
//
// where the k×k matrix A12 and l×l matrix B13 are non-singular
// upper triangular. A23 is l×l upper triangular if m-k-l >= 0,
// otherwise A23 is (m-k)×l upper trapezoidal.
//
// On exit,
//
//  U^T*A*Q = D1*[ 0 R ], V^T*B*Q = D2*[ 0 R ],
//
// where U, V and Q are orthogonal matrices.
// R is a non-singular upper triangular matrix, and D1 and D2 are
// diagonal matrices, which are of the following structures:
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
//  R = [ R11 R12 R13 ] is stored in A[0:m, n-k-l:n]
//      [  0  R22 R23 ]
// and R33 is stored in
//  B[m-k:l, n+m-k-l:n] on exit.
//
// The computation of the orthogonal transformation matrices U, V or Q
// is optional. These matrices may either be formed explicitly, or they
// may be post-multiplied into input matrices U1, V1, or Q1.
//
// Dtgsja essentially uses a variant of Kogbetliantz algorithm to reduce
// min(l,m-k)×l triangular or trapezoidal matrix A23 and l×l
// matrix B13 to the form:
//
//  U1^T*A13*Q1 = C1*R1; V1^T*B13*Q1 = S1*R1,
//
// where U1, V1 and Q1 are orthogonal matrices. C1 and S1 are diagonal
// matrices satisfying
//
//  C1^2 + S1^2 = I,
//
// and R1 is an l×l non-singular upper triangular matrix.
//
// jobU, jobV and jobQ are options for computing the orthogonal matrices. The behavior
// is as follows
//  jobU == lapack.GSVDU        Compute orthogonal matrix U
//  jobU == lapack.GSVDUnit     Use unit-initialized matrix
//  jobU == lapack.GSVDNone     Do not compute orthogonal matrix.
// The behavior is the same for jobV and jobQ with the exception that instead of
// lapack.GSVDU these accept lapack.GSVDV and lapack.GSVDQ respectively.
// The matrices U, V and Q must be m×m, p×p and n×n respectively unless the
// relevant job parameter is lapack.GSVDNone.
//
// k and l specify the sub-blocks in the input matrices A and B:
//  A23 = A[k:min(k+l,m), n-l:n) and B13 = B[0:l, n-l:n]
// of A and B, whose GSVD is going to be computed by Dtgsja.
//
// tola and tolb are the convergence criteria for the Jacobi-Kogbetliantz
// iteration procedure. Generally, they are the same as used in the preprocessing
// step, for example,
//  tola = max(m, n)*norm(A)*eps,
//  tolb = max(p, n)*norm(B)*eps,
// where eps is the machine epsilon.
//
// work must have length at least 2*n, otherwise Dtgsja will panic.
//
// alpha and beta must have length n or Dtgsja will panic. On exit, alpha and
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
// On exit, A[n-k:n, 0:min(k+l,m)] contains the triangular matrix R or part of R
// and if necessary, B[m-k:l, n+m-k-l:n] contains a part of R.
//
// Dtgsja returns whether the routine converged and the number of iteration cycles
// that were run.
//
// Dtgsja is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dtgsja(jobU, jobV, jobQ lapack.GSVDJob, m, p, n, k, l int, a []float64, lda int, b []float64, ldb int, tola, tolb float64, alpha, beta, u []float64, ldu int, v []float64, ldv int, q []float64, ldq int, work []float64) (cycles int, ok bool) {
	const maxit = 40

	checkMatrix(m, n, a, lda)
	checkMatrix(p, n, b, ldb)

	if len(alpha) != n {
		panic(badAlpha)
	}
	if len(beta) != n {
		panic(badBeta)
	}

	initu := jobU == lapack.GSVDUnit
	wantu := initu || jobU == lapack.GSVDU
	if !initu && !wantu && jobU != lapack.GSVDNone {
		panic(badGSVDJob + "U")
	}
	if jobU != lapack.GSVDNone {
		checkMatrix(m, m, u, ldu)
	}

	initv := jobV == lapack.GSVDUnit
	wantv := initv || jobV == lapack.GSVDV
	if !initv && !wantv && jobV != lapack.GSVDNone {
		panic(badGSVDJob + "V")
	}
	if jobV != lapack.GSVDNone {
		checkMatrix(p, p, v, ldv)
	}

	initq := jobQ == lapack.GSVDUnit
	wantq := initq || jobQ == lapack.GSVDQ
	if !initq && !wantq && jobQ != lapack.GSVDNone {
		panic(badGSVDJob + "Q")
	}
	if jobQ != lapack.GSVDNone {
		checkMatrix(n, n, q, ldq)
	}

	if len(work) < 2*n {
		panic(badWork)
	}

	// Initialize U, V and Q, if necessary
	if initu {
		impl.Dlaset(blas.All, m, m, 0, 1, u, ldu)
	}
	if initv {
		impl.Dlaset(blas.All, p, p, 0, 1, v, ldv)
	}
	if initq {
		impl.Dlaset(blas.All, n, n, 0, 1, q, ldq)
	}

	bi := blas64.Implementation()
	minTol := math.Min(tola, tolb)

	// Loop until convergence.
	upper := false
	for cycles = 1; cycles <= maxit; cycles++ {
		upper = !upper

		for i := 0; i < l-1; i++ {
			for j := i + 1; j < l; j++ {
				var a1, a2, a3 float64
				if k+i < m {
					a1 = a[(k+i)*lda+n-l+i]
				}
				if k+j < m {
					a3 = a[(k+j)*lda+n-l+j]
				}

				b1 := b[i*ldb+n-l+i]
				b3 := b[j*ldb+n-l+j]

				var b2 float64
				if upper {
					if k+i < m {
						a2 = a[(k+i)*lda+n-l+j]
					}
					b2 = b[i*ldb+n-l+j]
				} else {
					if k+j < m {
						a2 = a[(k+j)*lda+n-l+i]
					}
					b2 = b[j*ldb+n-l+i]
				}

				csu, snu, csv, snv, csq, snq := impl.Dlags2(upper, a1, a2, a3, b1, b2, b3)

				// Update (k+i)-th and (k+j)-th rows of matrix A: U^T*A.
				if k+j < m {
					bi.Drot(l, a[(k+j)*lda+n-l:], 1, a[(k+i)*lda+n-l:], 1, csu, snu)
				}

				// Update i-th and j-th rows of matrix B: V^T*B.
				bi.Drot(l, b[j*ldb+n-l:], 1, b[i*ldb+n-l:], 1, csv, snv)

				// Update (n-l+i)-th and (n-l+j)-th columns of matrices
				// A and B: A*Q and B*Q.
				bi.Drot(min(k+l, m), a[n-l+j:], lda, a[n-l+i:], lda, csq, snq)
				bi.Drot(l, b[n-l+j:], ldb, b[n-l+i:], ldb, csq, snq)

				if upper {
					if k+i < m {
						a[(k+i)*lda+n-l+j] = 0
					}
					b[i*ldb+n-l+j] = 0
				} else {
					if k+j < m {
						a[(k+j)*lda+n-l+i] = 0
					}
					b[j*ldb+n-l+i] = 0
				}

				// Update orthogonal matrices U, V, Q, if desired.
				if wantu && k+j < m {
					bi.Drot(m, u[k+j:], ldu, u[k+i:], ldu, csu, snu)
				}
				if wantv {
					bi.Drot(p, v[j:], ldv, v[i:], ldv, csv, snv)
				}
				if wantq {
					bi.Drot(n, q[n-l+j:], ldq, q[n-l+i:], ldq, csq, snq)
				}
			}
		}

		if !upper {
			// The matrices A13 and B13 were lower triangular at the start
			// of the cycle, and are now upper triangular.
			//
			// Convergence test: test the parallelism of the corresponding
			// rows of A and B.
			var error float64
			for i := 0; i < min(l, m-k); i++ {
				bi.Dcopy(l-i, a[(k+i)*lda+n-l+i:], 1, work, 1)
				bi.Dcopy(l-i, b[i*ldb+n-l+i:], 1, work[l:], 1)
				ssmin := impl.Dlapll(l-i, work, 1, work[l:], 1)
				error = math.Max(error, ssmin)
			}
			if math.Abs(error) <= minTol {
				// The algorithm has converged.
				// Compute the generalized singular value pairs (alpha, beta)
				// and set the triangular matrix R to array A.
				for i := 0; i < k; i++ {
					alpha[i] = 1
					beta[i] = 0
				}

				for i := 0; i < min(l, m-k); i++ {
					a1 := a[(k+i)*lda+n-l+i]
					b1 := b[i*ldb+n-l+i]

					if a1 != 0 {
						gamma := b1 / a1

						// Change sign if necessary.
						if gamma < 0 {
							bi.Dscal(l-i, -1, b[i*ldb+n-l+i:], 1)
							if wantv {
								bi.Dscal(p, -1, v[i:], ldv)
							}
						}
						beta[k+i], alpha[k+i], _ = impl.Dlartg(math.Abs(gamma), 1)

						if alpha[k+i] >= beta[k+i] {
							bi.Dscal(l-i, 1/alpha[k+i], a[(k+i)*lda+n-l+i:], 1)
						} else {
							bi.Dscal(l-i, 1/beta[k+i], b[i*ldb+n-l+i:], 1)
							bi.Dcopy(l-i, b[i*ldb+n-l+i:], 1, a[(k+i)*lda+n-l+i:], 1)
						}
					} else {
						alpha[k+i] = 0
						beta[k+i] = 1
						bi.Dcopy(l-i, b[i*ldb+n-l+i:], 1, a[(k+i)*lda+n-l+i:], 1)
					}
				}

				for i := m; i < k+l; i++ {
					alpha[i] = 0
					beta[i] = 1
				}
				if k+l < n {
					for i := k + l; i < n; i++ {
						alpha[i] = 0
						beta[i] = 0
					}
				}

				return cycles, true
			}
		}
	}

	// The algorithm has not converged after maxit cycles.
	return cycles, false
}
