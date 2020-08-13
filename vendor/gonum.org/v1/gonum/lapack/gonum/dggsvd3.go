// Copyright ©2017 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"math"

	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/gonum/lapack"
)

// Dggsvd3 computes the generalized singular value decomposition (GSVD)
// of an m×n matrix A and p×n matrix B:
//  Uᵀ*A*Q = D1*[ 0 R ]
//
//  Vᵀ*B*Q = D2*[ 0 R ]
// where U, V and Q are orthogonal matrices.
//
// Dggsvd3 returns k and l, the dimensions of the sub-blocks. k+l
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
// Dggsvd3 computes C, S, R, and optionally the orthogonal transformation
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
// alpha and beta must have length n or Dggsvd3 will panic. On exit, alpha and
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
// lwork must be -1 or greater than n, otherwise Dggsvd3 will panic. If
// lwork is -1, work[0] holds the optimal lwork on return, but Dggsvd3 does
// not perform the GSVD.
func (impl Implementation) Dggsvd3(jobU, jobV, jobQ lapack.GSVDJob, m, n, p int, a []float64, lda int, b []float64, ldb int, alpha, beta, u []float64, ldu int, v []float64, ldv int, q []float64, ldq int, work []float64, lwork int, iwork []int) (k, l int, ok bool) {
	wantu := jobU == lapack.GSVDU
	wantv := jobV == lapack.GSVDV
	wantq := jobQ == lapack.GSVDQ
	switch {
	case !wantu && jobU != lapack.GSVDNone:
		panic(badGSVDJob + "U")
	case !wantv && jobV != lapack.GSVDNone:
		panic(badGSVDJob + "V")
	case !wantq && jobQ != lapack.GSVDNone:
		panic(badGSVDJob + "Q")
	case m < 0:
		panic(mLT0)
	case n < 0:
		panic(nLT0)
	case p < 0:
		panic(pLT0)
	case lda < max(1, n):
		panic(badLdA)
	case ldb < max(1, n):
		panic(badLdB)
	case ldu < 1, wantu && ldu < m:
		panic(badLdU)
	case ldv < 1, wantv && ldv < p:
		panic(badLdV)
	case ldq < 1, wantq && ldq < n:
		panic(badLdQ)
	case len(iwork) < n:
		panic(shortWork)
	case lwork < 1 && lwork != -1:
		panic(badLWork)
	case len(work) < max(1, lwork):
		panic(shortWork)
	}

	// Determine optimal work length.
	impl.Dggsvp3(jobU, jobV, jobQ,
		m, p, n,
		a, lda,
		b, ldb,
		0, 0,
		u, ldu,
		v, ldv,
		q, ldq,
		iwork,
		work, work, -1)
	lwkopt := n + int(work[0])
	lwkopt = max(lwkopt, 2*n)
	lwkopt = max(lwkopt, 1)
	work[0] = float64(lwkopt)
	if lwork == -1 {
		return 0, 0, true
	}

	switch {
	case len(a) < (m-1)*lda+n:
		panic(shortA)
	case len(b) < (p-1)*ldb+n:
		panic(shortB)
	case wantu && len(u) < (m-1)*ldu+m:
		panic(shortU)
	case wantv && len(v) < (p-1)*ldv+p:
		panic(shortV)
	case wantq && len(q) < (n-1)*ldq+n:
		panic(shortQ)
	case len(alpha) != n:
		panic(badLenAlpha)
	case len(beta) != n:
		panic(badLenBeta)
	}

	// Compute the Frobenius norm of matrices A and B.
	anorm := impl.Dlange(lapack.Frobenius, m, n, a, lda, nil)
	bnorm := impl.Dlange(lapack.Frobenius, p, n, b, ldb, nil)

	// Get machine precision and set up threshold for determining
	// the effective numerical rank of the matrices A and B.
	tola := float64(max(m, n)) * math.Max(anorm, dlamchS) * dlamchP
	tolb := float64(max(p, n)) * math.Max(bnorm, dlamchS) * dlamchP

	// Preprocessing.
	k, l = impl.Dggsvp3(jobU, jobV, jobQ,
		m, p, n,
		a, lda,
		b, ldb,
		tola, tolb,
		u, ldu,
		v, ldv,
		q, ldq,
		iwork,
		work[:n], work[n:], lwork-n)

	// Compute the GSVD of two upper "triangular" matrices.
	_, ok = impl.Dtgsja(jobU, jobV, jobQ,
		m, p, n,
		k, l,
		a, lda,
		b, ldb,
		tola, tolb,
		alpha, beta,
		u, ldu,
		v, ldv,
		q, ldq,
		work)

	// Sort the singular values and store the pivot indices in iwork
	// Copy alpha to work, then sort alpha in work.
	bi := blas64.Implementation()
	bi.Dcopy(n, alpha, 1, work[:n], 1)
	ibnd := min(l, m-k)
	for i := 0; i < ibnd; i++ {
		// Scan for largest alpha_{k+i}.
		isub := i
		smax := work[k+i]
		for j := i + 1; j < ibnd; j++ {
			if v := work[k+j]; v > smax {
				isub = j
				smax = v
			}
		}
		if isub != i {
			work[k+isub] = work[k+i]
			work[k+i] = smax
			iwork[k+i] = k + isub
		} else {
			iwork[k+i] = k + i
		}
	}

	work[0] = float64(lwkopt)

	return k, l, ok
}
