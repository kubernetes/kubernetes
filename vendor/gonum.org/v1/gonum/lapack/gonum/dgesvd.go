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

const noSVDO = "dgesvd: not coded for overwrite"

// Dgesvd computes the singular value decomposition of the input matrix A.
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
// and jobVT can equal lapack.SVDOverwrite, and Dgesvd will panic otherwise.
//
// On entry, a contains the data for the m×n matrix A. During the call to Dgesvd
// the data is overwritten. On exit, A contains the appropriate singular vectors
// if either job is lapack.SVDOverwrite.
//
// s is a slice of length at least min(m,n) and on exit contains the singular
// values in decreasing order.
//
// u contains the left singular vectors on exit, stored column-wise. If
// jobU == lapack.SVDAll, u is of size m×m. If jobU == lapack.SVDStore u is
// of size m×min(m,n). If jobU == lapack.SVDOverwrite or lapack.SVDNone, u is
// not used.
//
// vt contains the left singular vectors on exit, stored row-wise. If
// jobV == lapack.SVDAll, vt is of size n×n. If jobVT == lapack.SVDStore vt is
// of size min(m,n)×n. If jobVT == lapack.SVDOverwrite or lapack.SVDNone, vt is
// not used.
//
// work is a slice for storing temporary memory, and lwork is the usable size of
// the slice. lwork must be at least max(5*min(m,n), 3*min(m,n)+max(m,n)).
// If lwork == -1, instead of performing Dgesvd, the optimal work length will be
// stored into work[0]. Dgesvd will panic if the working memory has insufficient
// storage.
//
// Dgesvd returns whether the decomposition successfully completed.
func (impl Implementation) Dgesvd(jobU, jobVT lapack.SVDJob, m, n int, a []float64, lda int, s, u []float64, ldu int, vt []float64, ldvt int, work []float64, lwork int) (ok bool) {
	if jobU == lapack.SVDOverwrite || jobVT == lapack.SVDOverwrite {
		panic(noSVDO)
	}

	wantua := jobU == lapack.SVDAll
	wantus := jobU == lapack.SVDStore
	wantuas := wantua || wantus
	wantuo := jobU == lapack.SVDOverwrite
	wantun := jobU == lapack.SVDNone
	if !(wantua || wantus || wantuo || wantun) {
		panic(badSVDJob)
	}

	wantva := jobVT == lapack.SVDAll
	wantvs := jobVT == lapack.SVDStore
	wantvas := wantva || wantvs
	wantvo := jobVT == lapack.SVDOverwrite
	wantvn := jobVT == lapack.SVDNone
	if !(wantva || wantvs || wantvo || wantvn) {
		panic(badSVDJob)
	}

	if wantuo && wantvo {
		panic(bothSVDOver)
	}

	minmn := min(m, n)
	minwork := 1
	if minmn > 0 {
		minwork = max(3*minmn+max(m, n), 5*minmn)
	}
	switch {
	case m < 0:
		panic(mLT0)
	case n < 0:
		panic(nLT0)
	case lda < max(1, n):
		panic(badLdA)
	case ldu < 1, wantua && ldu < m, wantus && ldu < minmn:
		panic(badLdU)
	case ldvt < 1 || (wantvas && ldvt < n):
		panic(badLdVT)
	case lwork < minwork && lwork != -1:
		panic(badLWork)
	case len(work) < max(1, lwork):
		panic(shortWork)
	}

	// Quick return if possible.
	if minmn == 0 {
		work[0] = 1
		return true
	}

	// Compute optimal workspace size for subroutines.
	opts := string(jobU) + string(jobVT)
	mnthr := impl.Ilaenv(6, "DGESVD", opts, m, n, 0, 0)
	maxwrk := 1
	var wrkbl, bdspac int
	if m >= n {
		bdspac = 5 * n
		impl.Dgeqrf(m, n, a, lda, nil, work, -1)
		lwork_dgeqrf := int(work[0])

		impl.Dorgqr(m, n, n, a, lda, nil, work, -1)
		lwork_dorgqr_n := int(work[0])
		impl.Dorgqr(m, m, n, a, lda, nil, work, -1)
		lwork_dorgqr_m := int(work[0])

		impl.Dgebrd(n, n, a, lda, s, nil, nil, nil, work, -1)
		lwork_dgebrd := int(work[0])

		impl.Dorgbr(lapack.GeneratePT, n, n, n, a, lda, nil, work, -1)
		lwork_dorgbr_p := int(work[0])

		impl.Dorgbr(lapack.GenerateQ, n, n, n, a, lda, nil, work, -1)
		lwork_dorgbr_q := int(work[0])

		if m >= mnthr {
			if wantun {
				// Path 1 (m much larger than n, jobU == None)
				maxwrk = n + lwork_dgeqrf
				maxwrk = max(maxwrk, 3*n+lwork_dgebrd)
				if wantvo || wantvas {
					maxwrk = max(maxwrk, 3*n+lwork_dorgbr_p)
				}
				maxwrk = max(maxwrk, bdspac)
			} else if wantuo && wantvn {
				// Path 2 (m much larger than n, jobU == Overwrite, jobVT == None)
				wrkbl = n + lwork_dgeqrf
				wrkbl = max(wrkbl, n+lwork_dorgqr_n)
				wrkbl = max(wrkbl, 3*n+lwork_dgebrd)
				wrkbl = max(wrkbl, 3*n+lwork_dorgbr_q)
				wrkbl = max(wrkbl, bdspac)
				maxwrk = max(n*n+wrkbl, n*n+m*n+n)
			} else if wantuo && wantvas {
				// Path 3 (m much larger than n, jobU == Overwrite, jobVT == Store or All)
				wrkbl = n + lwork_dgeqrf
				wrkbl = max(wrkbl, n+lwork_dorgqr_n)
				wrkbl = max(wrkbl, 3*n+lwork_dgebrd)
				wrkbl = max(wrkbl, 3*n+lwork_dorgbr_q)
				wrkbl = max(wrkbl, 3*n+lwork_dorgbr_p)
				wrkbl = max(wrkbl, bdspac)
				maxwrk = max(n*n+wrkbl, n*n+m*n+n)
			} else if wantus && wantvn {
				// Path 4 (m much larger than n, jobU == Store, jobVT == None)
				wrkbl = n + lwork_dgeqrf
				wrkbl = max(wrkbl, n+lwork_dorgqr_n)
				wrkbl = max(wrkbl, 3*n+lwork_dgebrd)
				wrkbl = max(wrkbl, 3*n+lwork_dorgbr_q)
				wrkbl = max(wrkbl, bdspac)
				maxwrk = n*n + wrkbl
			} else if wantus && wantvo {
				// Path 5 (m much larger than n, jobU == Store, jobVT == Overwrite)
				wrkbl = n + lwork_dgeqrf
				wrkbl = max(wrkbl, n+lwork_dorgqr_n)
				wrkbl = max(wrkbl, 3*n+lwork_dgebrd)
				wrkbl = max(wrkbl, 3*n+lwork_dorgbr_q)
				wrkbl = max(wrkbl, 3*n+lwork_dorgbr_p)
				wrkbl = max(wrkbl, bdspac)
				maxwrk = 2*n*n + wrkbl
			} else if wantus && wantvas {
				// Path 6 (m much larger than n, jobU == Store, jobVT == Store or All)
				wrkbl = n + lwork_dgeqrf
				wrkbl = max(wrkbl, n+lwork_dorgqr_n)
				wrkbl = max(wrkbl, 3*n+lwork_dgebrd)
				wrkbl = max(wrkbl, 3*n+lwork_dorgbr_q)
				wrkbl = max(wrkbl, 3*n+lwork_dorgbr_p)
				wrkbl = max(wrkbl, bdspac)
				maxwrk = n*n + wrkbl
			} else if wantua && wantvn {
				// Path 7 (m much larger than n, jobU == All, jobVT == None)
				wrkbl = n + lwork_dgeqrf
				wrkbl = max(wrkbl, n+lwork_dorgqr_m)
				wrkbl = max(wrkbl, 3*n+lwork_dgebrd)
				wrkbl = max(wrkbl, 3*n+lwork_dorgbr_q)
				wrkbl = max(wrkbl, bdspac)
				maxwrk = n*n + wrkbl
			} else if wantua && wantvo {
				// Path 8 (m much larger than n, jobU == All, jobVT == Overwrite)
				wrkbl = n + lwork_dgeqrf
				wrkbl = max(wrkbl, n+lwork_dorgqr_m)
				wrkbl = max(wrkbl, 3*n+lwork_dgebrd)
				wrkbl = max(wrkbl, 3*n+lwork_dorgbr_q)
				wrkbl = max(wrkbl, 3*n+lwork_dorgbr_p)
				wrkbl = max(wrkbl, bdspac)
				maxwrk = 2*n*n + wrkbl
			} else if wantua && wantvas {
				// Path 9 (m much larger than n, jobU == All, jobVT == Store or All)
				wrkbl = n + lwork_dgeqrf
				wrkbl = max(wrkbl, n+lwork_dorgqr_m)
				wrkbl = max(wrkbl, 3*n+lwork_dgebrd)
				wrkbl = max(wrkbl, 3*n+lwork_dorgbr_q)
				wrkbl = max(wrkbl, 3*n+lwork_dorgbr_p)
				wrkbl = max(wrkbl, bdspac)
				maxwrk = n*n + wrkbl
			}
		} else {
			// Path 10 (m at least n, but not much larger)
			impl.Dgebrd(m, n, a, lda, s, nil, nil, nil, work, -1)
			lwork_dgebrd := int(work[0])
			maxwrk = 3*n + lwork_dgebrd
			if wantus || wantuo {
				impl.Dorgbr(lapack.GenerateQ, m, n, n, a, lda, nil, work, -1)
				lwork_dorgbr_q = int(work[0])
				maxwrk = max(maxwrk, 3*n+lwork_dorgbr_q)
			}
			if wantua {
				impl.Dorgbr(lapack.GenerateQ, m, m, n, a, lda, nil, work, -1)
				lwork_dorgbr_q := int(work[0])
				maxwrk = max(maxwrk, 3*n+lwork_dorgbr_q)
			}
			if !wantvn {
				maxwrk = max(maxwrk, 3*n+lwork_dorgbr_p)
			}
			maxwrk = max(maxwrk, bdspac)
		}
	} else {
		bdspac = 5 * m

		impl.Dgelqf(m, n, a, lda, nil, work, -1)
		lwork_dgelqf := int(work[0])

		impl.Dorglq(n, n, m, nil, n, nil, work, -1)
		lwork_dorglq_n := int(work[0])
		impl.Dorglq(m, n, m, a, lda, nil, work, -1)
		lwork_dorglq_m := int(work[0])

		impl.Dgebrd(m, m, a, lda, s, nil, nil, nil, work, -1)
		lwork_dgebrd := int(work[0])

		impl.Dorgbr(lapack.GeneratePT, m, m, m, a, n, nil, work, -1)
		lwork_dorgbr_p := int(work[0])

		impl.Dorgbr(lapack.GenerateQ, m, m, m, a, n, nil, work, -1)
		lwork_dorgbr_q := int(work[0])

		if n >= mnthr {
			if wantvn {
				// Path 1t (n much larger than m, jobVT == None)
				maxwrk = m + lwork_dgelqf
				maxwrk = max(maxwrk, 3*m+lwork_dgebrd)
				if wantuo || wantuas {
					maxwrk = max(maxwrk, 3*m+lwork_dorgbr_q)
				}
				maxwrk = max(maxwrk, bdspac)
			} else if wantvo && wantun {
				// Path 2t (n much larger than m, jobU == None, jobVT == Overwrite)
				wrkbl = m + lwork_dgelqf
				wrkbl = max(wrkbl, m+lwork_dorglq_m)
				wrkbl = max(wrkbl, 3*m+lwork_dgebrd)
				wrkbl = max(wrkbl, 3*m+lwork_dorgbr_p)
				wrkbl = max(wrkbl, bdspac)
				maxwrk = max(m*m+wrkbl, m*m+m*n+m)
			} else if wantvo && wantuas {
				// Path 3t (n much larger than m, jobU == Store or All, jobVT == Overwrite)
				wrkbl = m + lwork_dgelqf
				wrkbl = max(wrkbl, m+lwork_dorglq_m)
				wrkbl = max(wrkbl, 3*m+lwork_dgebrd)
				wrkbl = max(wrkbl, 3*m+lwork_dorgbr_p)
				wrkbl = max(wrkbl, 3*m+lwork_dorgbr_q)
				wrkbl = max(wrkbl, bdspac)
				maxwrk = max(m*m+wrkbl, m*m+m*n+m)
			} else if wantvs && wantun {
				// Path 4t (n much larger than m, jobU == None, jobVT == Store)
				wrkbl = m + lwork_dgelqf
				wrkbl = max(wrkbl, m+lwork_dorglq_m)
				wrkbl = max(wrkbl, 3*m+lwork_dgebrd)
				wrkbl = max(wrkbl, 3*m+lwork_dorgbr_p)
				wrkbl = max(wrkbl, bdspac)
				maxwrk = m*m + wrkbl
			} else if wantvs && wantuo {
				// Path 5t (n much larger than m, jobU == Overwrite, jobVT == Store)
				wrkbl = m + lwork_dgelqf
				wrkbl = max(wrkbl, m+lwork_dorglq_m)
				wrkbl = max(wrkbl, 3*m+lwork_dgebrd)
				wrkbl = max(wrkbl, 3*m+lwork_dorgbr_p)
				wrkbl = max(wrkbl, 3*m+lwork_dorgbr_q)
				wrkbl = max(wrkbl, bdspac)
				maxwrk = 2*m*m + wrkbl
			} else if wantvs && wantuas {
				// Path 6t (n much larger than m, jobU == Store or All, jobVT == Store)
				wrkbl = m + lwork_dgelqf
				wrkbl = max(wrkbl, m+lwork_dorglq_m)
				wrkbl = max(wrkbl, 3*m+lwork_dgebrd)
				wrkbl = max(wrkbl, 3*m+lwork_dorgbr_p)
				wrkbl = max(wrkbl, 3*m+lwork_dorgbr_q)
				wrkbl = max(wrkbl, bdspac)
				maxwrk = m*m + wrkbl
			} else if wantva && wantun {
				// Path 7t (n much larger than m, jobU== None, jobVT == All)
				wrkbl = m + lwork_dgelqf
				wrkbl = max(wrkbl, m+lwork_dorglq_n)
				wrkbl = max(wrkbl, 3*m+lwork_dgebrd)
				wrkbl = max(wrkbl, 3*m+lwork_dorgbr_p)
				wrkbl = max(wrkbl, bdspac)
				maxwrk = m*m + wrkbl
			} else if wantva && wantuo {
				// Path 8t (n much larger than m, jobU == Overwrite, jobVT == All)
				wrkbl = m + lwork_dgelqf
				wrkbl = max(wrkbl, m+lwork_dorglq_n)
				wrkbl = max(wrkbl, 3*m+lwork_dgebrd)
				wrkbl = max(wrkbl, 3*m+lwork_dorgbr_p)
				wrkbl = max(wrkbl, 3*m+lwork_dorgbr_q)
				wrkbl = max(wrkbl, bdspac)
				maxwrk = 2*m*m + wrkbl
			} else if wantva && wantuas {
				// Path 9t (n much larger than m, jobU == Store or All, jobVT == All)
				wrkbl = m + lwork_dgelqf
				wrkbl = max(wrkbl, m+lwork_dorglq_n)
				wrkbl = max(wrkbl, 3*m+lwork_dgebrd)
				wrkbl = max(wrkbl, 3*m+lwork_dorgbr_p)
				wrkbl = max(wrkbl, 3*m+lwork_dorgbr_q)
				wrkbl = max(wrkbl, bdspac)
				maxwrk = m*m + wrkbl
			}
		} else {
			// Path 10t (n greater than m, but not much larger)
			impl.Dgebrd(m, n, a, lda, s, nil, nil, nil, work, -1)
			lwork_dgebrd = int(work[0])
			maxwrk = 3*m + lwork_dgebrd
			if wantvs || wantvo {
				impl.Dorgbr(lapack.GeneratePT, m, n, m, a, n, nil, work, -1)
				lwork_dorgbr_p = int(work[0])
				maxwrk = max(maxwrk, 3*m+lwork_dorgbr_p)
			}
			if wantva {
				impl.Dorgbr(lapack.GeneratePT, n, n, m, a, n, nil, work, -1)
				lwork_dorgbr_p = int(work[0])
				maxwrk = max(maxwrk, 3*m+lwork_dorgbr_p)
			}
			if !wantun {
				maxwrk = max(maxwrk, 3*m+lwork_dorgbr_q)
			}
			maxwrk = max(maxwrk, bdspac)
		}
	}

	maxwrk = max(maxwrk, minwork)
	if lwork == -1 {
		work[0] = float64(maxwrk)
		return true
	}

	if len(a) < (m-1)*lda+n {
		panic(shortA)
	}
	if len(s) < minmn {
		panic(shortS)
	}
	if (len(u) < (m-1)*ldu+m && wantua) || (len(u) < (m-1)*ldu+minmn && wantus) {
		panic(shortU)
	}
	if (len(vt) < (n-1)*ldvt+n && wantva) || (len(vt) < (minmn-1)*ldvt+n && wantvs) {
		panic(shortVT)
	}

	// Perform decomposition.
	eps := dlamchE
	smlnum := math.Sqrt(dlamchS) / eps
	bignum := 1 / smlnum

	// Scale A if max element outside range [smlnum, bignum].
	anrm := impl.Dlange(lapack.MaxAbs, m, n, a, lda, nil)
	var iscl bool
	if anrm > 0 && anrm < smlnum {
		iscl = true
		impl.Dlascl(lapack.General, 0, 0, anrm, smlnum, m, n, a, lda)
	} else if anrm > bignum {
		iscl = true
		impl.Dlascl(lapack.General, 0, 0, anrm, bignum, m, n, a, lda)
	}

	bi := blas64.Implementation()
	var ie int
	if m >= n {
		// If A has sufficiently more rows than columns, use the QR decomposition.
		if m >= mnthr {
			// m >> n
			if wantun {
				// Path 1.
				itau := 0
				iwork := itau + n

				// Compute A = Q * R.
				impl.Dgeqrf(m, n, a, lda, work[itau:], work[iwork:], lwork-iwork)

				// Zero out below R.
				impl.Dlaset(blas.Lower, n-1, n-1, 0, 0, a[lda:], lda)
				ie = 0
				itauq := ie + n
				itaup := itauq + n
				iwork = itaup + n
				// Bidiagonalize R in A.
				impl.Dgebrd(n, n, a, lda, s, work[ie:], work[itauq:],
					work[itaup:], work[iwork:], lwork-iwork)
				ncvt := 0
				if wantvo || wantvas {
					impl.Dorgbr(lapack.GeneratePT, n, n, n, a, lda, work[itaup:],
						work[iwork:], lwork-iwork)
					ncvt = n
				}
				iwork = ie + n

				// Perform bidiagonal QR iteration computing right singular vectors
				// of A in A if desired.
				ok = impl.Dbdsqr(blas.Upper, n, ncvt, 0, 0, s, work[ie:],
					a, lda, work, 1, work, 1, work[iwork:])

				// If right singular vectors desired in VT, copy them there.
				if wantvas {
					impl.Dlacpy(blas.All, n, n, a, lda, vt, ldvt)
				}
			} else if wantuo && wantvn {
				// Path 2
				panic(noSVDO)
			} else if wantuo && wantvas {
				// Path 3
				panic(noSVDO)
			} else if wantus {
				if wantvn {
					// Path 4
					if lwork >= n*n+max(4*n, bdspac) {
						// Sufficient workspace for a fast algorithm.
						ir := 0
						var ldworkr int
						if lwork >= wrkbl+lda*n {
							ldworkr = lda
						} else {
							ldworkr = n
						}
						itau := ir + ldworkr*n
						iwork := itau + n
						// Compute A = Q * R.
						impl.Dgeqrf(m, n, a, lda, work[itau:], work[iwork:], lwork-iwork)

						// Copy R to work[ir:], zeroing out below it.
						impl.Dlacpy(blas.Upper, n, n, a, lda, work[ir:], ldworkr)
						impl.Dlaset(blas.Lower, n-1, n-1, 0, 0, work[ir+ldworkr:], ldworkr)

						// Generate Q in A.
						impl.Dorgqr(m, n, n, a, lda, work[itau:], work[iwork:], lwork-iwork)
						ie := itau
						itauq := ie + n
						itaup := itauq + n
						iwork = itaup + n

						// Bidiagonalize R in work[ir:].
						impl.Dgebrd(n, n, work[ir:], ldworkr, s, work[ie:],
							work[itauq:], work[itaup:], work[iwork:], lwork-iwork)

						// Generate left vectors bidiagonalizing R in work[ir:].
						impl.Dorgbr(lapack.GenerateQ, n, n, n, work[ir:], ldworkr,
							work[itauq:], work[iwork:], lwork-iwork)
						iwork = ie + n

						// Perform bidiagonal QR iteration, compuing left singular
						// vectors of R in work[ir:].
						ok = impl.Dbdsqr(blas.Upper, n, 0, n, 0, s, work[ie:], work, 1,
							work[ir:], ldworkr, work, 1, work[iwork:])

						// Multiply Q in A by left singular vectors of R in
						// work[ir:], storing result in U.
						bi.Dgemm(blas.NoTrans, blas.NoTrans, m, n, n, 1, a, lda,
							work[ir:], ldworkr, 0, u, ldu)
					} else {
						// Insufficient workspace for a fast algorithm.
						itau := 0
						iwork := itau + n

						// Compute A = Q*R, copying result to U.
						impl.Dgeqrf(m, n, a, lda, work[itau:], work[iwork:], lwork-iwork)
						impl.Dlacpy(blas.Lower, m, n, a, lda, u, ldu)

						// Generate Q in U.
						impl.Dorgqr(m, n, n, u, ldu, work[itau:], work[iwork:], lwork-iwork)
						ie := itau
						itauq := ie + n
						itaup := itauq + n
						iwork = itaup + n

						// Zero out below R in A.
						impl.Dlaset(blas.Lower, n-1, n-1, 0, 0, a[lda:], lda)

						// Bidiagonalize R in A.
						impl.Dgebrd(n, n, a, lda, s, work[ie:],
							work[itauq:], work[itaup:], work[iwork:], lwork-iwork)

						// Multiply Q in U by left vectors bidiagonalizing R.
						impl.Dormbr(lapack.ApplyQ, blas.Right, blas.NoTrans, m, n, n,
							a, lda, work[itauq:], u, ldu, work[iwork:], lwork-iwork)
						iwork = ie + n

						// Perform bidiagonal QR iteration, computing left
						// singular vectors of A in U.
						ok = impl.Dbdsqr(blas.Upper, n, 0, m, 0, s, work[ie:], work, 1,
							u, ldu, work, 1, work[iwork:])
					}
				} else if wantvo {
					// Path 5
					panic(noSVDO)
				} else if wantvas {
					// Path 6
					if lwork >= n*n+max(4*n, bdspac) {
						// Sufficient workspace for a fast algorithm.
						iu := 0
						var ldworku int
						if lwork >= wrkbl+lda*n {
							ldworku = lda
						} else {
							ldworku = n
						}
						itau := iu + ldworku*n
						iwork := itau + n

						// Compute A = Q * R.
						impl.Dgeqrf(m, n, a, lda, work[itau:], work[iwork:], lwork-iwork)
						// Copy R to work[iu:], zeroing out below it.
						impl.Dlacpy(blas.Upper, n, n, a, lda, work[iu:], ldworku)
						impl.Dlaset(blas.Lower, n-1, n-1, 0, 0, work[iu+ldworku:], ldworku)

						// Generate Q in A.
						impl.Dorgqr(m, n, n, a, lda, work[itau:], work[iwork:], lwork-iwork)

						ie := itau
						itauq := ie + n
						itaup := itauq + n
						iwork = itaup + n

						// Bidiagonalize R in work[iu:], copying result to VT.
						impl.Dgebrd(n, n, work[iu:], ldworku, s, work[ie:],
							work[itauq:], work[itaup:], work[iwork:], lwork-iwork)
						impl.Dlacpy(blas.Upper, n, n, work[iu:], ldworku, vt, ldvt)

						// Generate left bidiagonalizing vectors in work[iu:].
						impl.Dorgbr(lapack.GenerateQ, n, n, n, work[iu:], ldworku,
							work[itauq:], work[iwork:], lwork-iwork)

						// Generate right bidiagonalizing vectors in VT.
						impl.Dorgbr(lapack.GeneratePT, n, n, n, vt, ldvt,
							work[itaup:], work[iwork:], lwork-iwork)
						iwork = ie + n

						// Perform bidiagonal QR iteration, computing left singular
						// vectors of R in work[iu:], and computing right singular
						// vectors of R in VT.
						ok = impl.Dbdsqr(blas.Upper, n, n, n, 0, s, work[ie:],
							vt, ldvt, work[iu:], ldworku, work, 1, work[iwork:])

						// Multiply Q in A by left singular vectors of R in
						// work[iu:], storing result in U.
						bi.Dgemm(blas.NoTrans, blas.NoTrans, m, n, n, 1, a, lda,
							work[iu:], ldworku, 0, u, ldu)
					} else {
						// Insufficient workspace for a fast algorithm.
						itau := 0
						iwork := itau + n

						// Compute A = Q * R, copying result to U.
						impl.Dgeqrf(m, n, a, lda, work[itau:], work[iwork:], lwork-iwork)
						impl.Dlacpy(blas.Lower, m, n, a, lda, u, ldu)

						// Generate Q in U.
						impl.Dorgqr(m, n, n, u, ldu, work[itau:], work[iwork:], lwork-iwork)

						// Copy R to VT, zeroing out below it.
						impl.Dlacpy(blas.Upper, n, n, a, lda, vt, ldvt)
						impl.Dlaset(blas.Lower, n-1, n-1, 0, 0, vt[ldvt:], ldvt)

						ie := itau
						itauq := ie + n
						itaup := itauq + n
						iwork = itaup + n

						// Bidiagonalize R in VT.
						impl.Dgebrd(n, n, vt, ldvt, s, work[ie:],
							work[itauq:], work[itaup:], work[iwork:], lwork-iwork)

						// Multiply Q in U by left bidiagonalizing vectors in VT.
						impl.Dormbr(lapack.ApplyQ, blas.Right, blas.NoTrans, m, n, n,
							vt, ldvt, work[itauq:], u, ldu, work[iwork:], lwork-iwork)

						// Generate right bidiagonalizing vectors in VT.
						impl.Dorgbr(lapack.GeneratePT, n, n, n, vt, ldvt,
							work[itaup:], work[iwork:], lwork-iwork)
						iwork = ie + n

						// Perform bidiagonal QR iteration, computing left singular
						// vectors of A in U and computing right singular vectors
						// of A in VT.
						ok = impl.Dbdsqr(blas.Upper, n, n, m, 0, s, work[ie:],
							vt, ldvt, u, ldu, work, 1, work[iwork:])
					}
				}
			} else if wantua {
				if wantvn {
					// Path 7
					if lwork >= n*n+max(max(n+m, 4*n), bdspac) {
						// Sufficient workspace for a fast algorithm.
						ir := 0
						var ldworkr int
						if lwork >= wrkbl+lda*n {
							ldworkr = lda
						} else {
							ldworkr = n
						}
						itau := ir + ldworkr*n
						iwork := itau + n

						// Compute A = Q*R, copying result to U.
						impl.Dgeqrf(m, n, a, lda, work[itau:], work[iwork:], lwork-iwork)
						impl.Dlacpy(blas.Lower, m, n, a, lda, u, ldu)

						// Copy R to work[ir:], zeroing out below it.
						impl.Dlacpy(blas.Upper, n, n, a, lda, work[ir:], ldworkr)
						impl.Dlaset(blas.Lower, n-1, n-1, 0, 0, work[ir+ldworkr:], ldworkr)

						// Generate Q in U.
						impl.Dorgqr(m, m, n, u, ldu, work[itau:], work[iwork:], lwork-iwork)
						ie := itau
						itauq := ie + n
						itaup := itauq + n
						iwork = itaup + n

						// Bidiagonalize R in work[ir:].
						impl.Dgebrd(n, n, work[ir:], ldworkr, s, work[ie:],
							work[itauq:], work[itaup:], work[iwork:], lwork-iwork)

						// Generate left bidiagonalizing vectors in work[ir:].
						impl.Dorgbr(lapack.GenerateQ, n, n, n, work[ir:], ldworkr,
							work[itauq:], work[iwork:], lwork-iwork)
						iwork = ie + n

						// Perform bidiagonal QR iteration, computing left singular
						// vectors of R in work[ir:].
						ok = impl.Dbdsqr(blas.Upper, n, 0, n, 0, s, work[ie:], work, 1,
							work[ir:], ldworkr, work, 1, work[iwork:])

						// Multiply Q in U by left singular vectors of R in
						// work[ir:], storing result in A.
						bi.Dgemm(blas.NoTrans, blas.NoTrans, m, n, n, 1, u, ldu,
							work[ir:], ldworkr, 0, a, lda)

						// Copy left singular vectors of A from A to U.
						impl.Dlacpy(blas.All, m, n, a, lda, u, ldu)
					} else {
						// Insufficient workspace for a fast algorithm.
						itau := 0
						iwork := itau + n

						// Compute A = Q*R, copying result to U.
						impl.Dgeqrf(m, n, a, lda, work[itau:], work[iwork:], lwork-iwork)
						impl.Dlacpy(blas.Lower, m, n, a, lda, u, ldu)

						// Generate Q in U.
						impl.Dorgqr(m, m, n, u, ldu, work[itau:], work[iwork:], lwork-iwork)
						ie := itau
						itauq := ie + n
						itaup := itauq + n
						iwork = itaup + n

						// Zero out below R in A.
						impl.Dlaset(blas.Lower, n-1, n-1, 0, 0, a[lda:], lda)

						// Bidiagonalize R in A.
						impl.Dgebrd(n, n, a, lda, s, work[ie:],
							work[itauq:], work[itaup:], work[iwork:], lwork-iwork)

						// Multiply Q in U by left bidiagonalizing vectors in A.
						impl.Dormbr(lapack.ApplyQ, blas.Right, blas.NoTrans, m, n, n,
							a, lda, work[itauq:], u, ldu, work[iwork:], lwork-iwork)
						iwork = ie + n

						// Perform bidiagonal QR iteration, computing left
						// singular vectors of A in U.
						ok = impl.Dbdsqr(blas.Upper, n, 0, m, 0, s, work[ie:],
							work, 1, u, ldu, work, 1, work[iwork:])
					}
				} else if wantvo {
					// Path 8.
					panic(noSVDO)
				} else if wantvas {
					// Path 9.
					if lwork >= n*n+max(max(n+m, 4*n), bdspac) {
						// Sufficient workspace for a fast algorithm.
						iu := 0
						var ldworku int
						if lwork >= wrkbl+lda*n {
							ldworku = lda
						} else {
							ldworku = n
						}
						itau := iu + ldworku*n
						iwork := itau + n

						// Compute A = Q * R, copying result to U.
						impl.Dgeqrf(m, n, a, lda, work[itau:], work[iwork:], lwork-iwork)
						impl.Dlacpy(blas.Lower, m, n, a, lda, u, ldu)

						// Generate Q in U.
						impl.Dorgqr(m, m, n, u, ldu, work[itau:], work[iwork:], lwork-iwork)

						// Copy R to work[iu:], zeroing out below it.
						impl.Dlacpy(blas.Upper, n, n, a, lda, work[iu:], ldworku)
						impl.Dlaset(blas.Lower, n-1, n-1, 0, 0, work[iu+ldworku:], ldworku)

						ie = itau
						itauq := ie + n
						itaup := itauq + n
						iwork = itaup + n

						// Bidiagonalize R in work[iu:], copying result to VT.
						impl.Dgebrd(n, n, work[iu:], ldworku, s, work[ie:],
							work[itauq:], work[itaup:], work[iwork:], lwork-iwork)
						impl.Dlacpy(blas.Upper, n, n, work[iu:], ldworku, vt, ldvt)

						// Generate left bidiagonalizing vectors in work[iu:].
						impl.Dorgbr(lapack.GenerateQ, n, n, n, work[iu:], ldworku,
							work[itauq:], work[iwork:], lwork-iwork)

						// Generate right bidiagonalizing vectors in VT.
						impl.Dorgbr(lapack.GeneratePT, n, n, n, vt, ldvt,
							work[itaup:], work[iwork:], lwork-iwork)
						iwork = ie + n

						// Perform bidiagonal QR iteration, computing left singular
						// vectors of R in work[iu:] and computing right
						// singular vectors of R in VT.
						ok = impl.Dbdsqr(blas.Upper, n, n, n, 0, s, work[ie:],
							vt, ldvt, work[iu:], ldworku, work, 1, work[iwork:])

						// Multiply Q in U by left singular vectors of R in
						// work[iu:], storing result in A.
						bi.Dgemm(blas.NoTrans, blas.NoTrans, m, n, n, 1,
							u, ldu, work[iu:], ldworku, 0, a, lda)

						// Copy left singular vectors of A from A to U.
						impl.Dlacpy(blas.All, m, n, a, lda, u, ldu)

						/*
							// Bidiagonalize R in VT.
							impl.Dgebrd(n, n, vt, ldvt, s, work[ie:],
								work[itauq:], work[itaup:], work[iwork:], lwork-iwork)

							// Multiply Q in U by left bidiagonalizing vectors in VT.
							impl.Dormbr(lapack.ApplyQ, blas.Right, blas.NoTrans,
								m, n, n, vt, ldvt, work[itauq:], u, ldu, work[iwork:], lwork-iwork)

							// Generate right bidiagonalizing vectors in VT.
							impl.Dorgbr(lapack.GeneratePT, n, n, n, vt, ldvt,
								work[itaup:], work[iwork:], lwork-iwork)
							iwork = ie + n

							// Perform bidiagonal QR iteration, computing left singular
							// vectors of A in U and computing right singular vectors
							// of A in VT.
							ok = impl.Dbdsqr(blas.Upper, n, n, m, 0, s, work[ie:],
								vt, ldvt, u, ldu, work, 1, work[iwork:])
						*/
					} else {
						// Insufficient workspace for a fast algorithm.
						itau := 0
						iwork := itau + n

						// Compute A = Q*R, copying result to U.
						impl.Dgeqrf(m, n, a, lda, work[itau:], work[iwork:], lwork-iwork)
						impl.Dlacpy(blas.Lower, m, n, a, lda, u, ldu)

						// Generate Q in U.
						impl.Dorgqr(m, m, n, u, ldu, work[itau:], work[iwork:], lwork-iwork)

						// Copy R from A to VT, zeroing out below it.
						impl.Dlacpy(blas.Upper, n, n, a, lda, vt, ldvt)
						if n > 1 {
							impl.Dlaset(blas.Lower, n-1, n-1, 0, 0, vt[ldvt:], ldvt)
						}

						ie := itau
						itauq := ie + n
						itaup := itauq + n
						iwork = itaup + n

						// Bidiagonalize R in VT.
						impl.Dgebrd(n, n, vt, ldvt, s, work[ie:],
							work[itauq:], work[itaup:], work[iwork:], lwork-iwork)

						// Multiply Q in U by left bidiagonalizing vectors in VT.
						impl.Dormbr(lapack.ApplyQ, blas.Right, blas.NoTrans,
							m, n, n, vt, ldvt, work[itauq:], u, ldu, work[iwork:], lwork-iwork)

						// Generate right bidiagonizing vectors in VT.
						impl.Dorgbr(lapack.GeneratePT, n, n, n, vt, ldvt,
							work[itaup:], work[iwork:], lwork-iwork)
						iwork = ie + n

						// Perform bidiagonal QR iteration, computing left singular
						// vectors of A in U and computing right singular vectors
						// of A in VT.
						ok = impl.Dbdsqr(blas.Upper, n, n, m, 0, s, work[ie:],
							vt, ldvt, u, ldu, work, 1, work[iwork:])
					}
				}
			}
		} else {
			// Path 10.
			// M at least N, but not much larger.
			ie = 0
			itauq := ie + n
			itaup := itauq + n
			iwork := itaup + n

			// Bidiagonalize A.
			impl.Dgebrd(m, n, a, lda, s, work[ie:], work[itauq:],
				work[itaup:], work[iwork:], lwork-iwork)
			if wantuas {
				// Left singular vectors are desired in U. Copy result to U and
				// generate left biadiagonalizing vectors in U.
				impl.Dlacpy(blas.Lower, m, n, a, lda, u, ldu)
				var ncu int
				if wantus {
					ncu = n
				}
				if wantua {
					ncu = m
				}
				impl.Dorgbr(lapack.GenerateQ, m, ncu, n, u, ldu, work[itauq:], work[iwork:], lwork-iwork)
			}
			if wantvas {
				// Right singular vectors are desired in VT. Copy result to VT and
				// generate left biadiagonalizing vectors in VT.
				impl.Dlacpy(blas.Upper, n, n, a, lda, vt, ldvt)
				impl.Dorgbr(lapack.GeneratePT, n, n, n, vt, ldvt, work[itaup:], work[iwork:], lwork-iwork)
			}
			if wantuo {
				panic(noSVDO)
			}
			if wantvo {
				panic(noSVDO)
			}
			iwork = ie + n
			var nru, ncvt int
			if wantuas || wantuo {
				nru = m
			}
			if wantun {
				nru = 0
			}
			if wantvas || wantvo {
				ncvt = n
			}
			if wantvn {
				ncvt = 0
			}
			if !wantuo && !wantvo {
				// Perform bidiagonal QR iteration, if desired, computing left
				// singular vectors in U and right singular vectors in VT.
				ok = impl.Dbdsqr(blas.Upper, n, ncvt, nru, 0, s, work[ie:],
					vt, ldvt, u, ldu, work, 1, work[iwork:])
			} else {
				// There will be two branches when the implementation is complete.
				panic(noSVDO)
			}
		}
	} else {
		// A has more columns than rows. If A has sufficiently more columns than
		// rows, first reduce using the LQ decomposition.
		if n >= mnthr {
			// n >> m.
			if wantvn {
				// Path 1t.
				itau := 0
				iwork := itau + m

				// Compute A = L*Q.
				impl.Dgelqf(m, n, a, lda, work[itau:], work[iwork:], lwork-iwork)

				// Zero out above L.
				impl.Dlaset(blas.Upper, m-1, m-1, 0, 0, a[1:], lda)
				ie := 0
				itauq := ie + m
				itaup := itauq + m
				iwork = itaup + m

				// Bidiagonalize L in A.
				impl.Dgebrd(m, m, a, lda, s, work[ie:itauq],
					work[itauq:itaup], work[itaup:iwork], work[iwork:], lwork-iwork)
				if wantuo || wantuas {
					impl.Dorgbr(lapack.GenerateQ, m, m, m, a, lda,
						work[itauq:], work[iwork:], lwork-iwork)
				}
				iwork = ie + m
				nru := 0
				if wantuo || wantuas {
					nru = m
				}

				// Perform bidiagonal QR iteration, computing left singular vectors
				// of A in A if desired.
				ok = impl.Dbdsqr(blas.Upper, m, 0, nru, 0, s, work[ie:],
					work, 1, a, lda, work, 1, work[iwork:])

				// If left singular vectors desired in U, copy them there.
				if wantuas {
					impl.Dlacpy(blas.All, m, m, a, lda, u, ldu)
				}
			} else if wantvo && wantun {
				// Path 2t.
				panic(noSVDO)
			} else if wantvo && wantuas {
				// Path 3t.
				panic(noSVDO)
			} else if wantvs {
				if wantun {
					// Path 4t.
					if lwork >= m*m+max(4*m, bdspac) {
						// Sufficient workspace for a fast algorithm.
						ir := 0
						var ldworkr int
						if lwork >= wrkbl+lda*m {
							ldworkr = lda
						} else {
							ldworkr = m
						}
						itau := ir + ldworkr*m
						iwork := itau + m

						// Compute A = L*Q.
						impl.Dgelqf(m, n, a, lda, work[itau:], work[iwork:], lwork-iwork)

						// Copy L to work[ir:], zeroing out above it.
						impl.Dlacpy(blas.Lower, m, m, a, lda, work[ir:], ldworkr)
						impl.Dlaset(blas.Upper, m-1, m-1, 0, 0, work[ir+1:], ldworkr)

						// Generate Q in A.
						impl.Dorglq(m, n, m, a, lda, work[itau:], work[iwork:], lwork-iwork)
						ie := itau
						itauq := ie + m
						itaup := itauq + m
						iwork = itaup + m

						// Bidiagonalize L in work[ir:].
						impl.Dgebrd(m, m, work[ir:], ldworkr, s, work[ie:],
							work[itauq:], work[itaup:], work[iwork:], lwork-iwork)

						// Generate right vectors bidiagonalizing L in work[ir:].
						impl.Dorgbr(lapack.GeneratePT, m, m, m, work[ir:], ldworkr,
							work[itaup:], work[iwork:], lwork-iwork)
						iwork = ie + m

						// Perform bidiagonal QR iteration, computing right singular
						// vectors of L in work[ir:].
						ok = impl.Dbdsqr(blas.Upper, m, m, 0, 0, s, work[ie:],
							work[ir:], ldworkr, work, 1, work, 1, work[iwork:])

						// Multiply right singular vectors of L in work[ir:] by
						// Q in A, storing result in VT.
						bi.Dgemm(blas.NoTrans, blas.NoTrans, m, n, m, 1,
							work[ir:], ldworkr, a, lda, 0, vt, ldvt)
					} else {
						// Insufficient workspace for a fast algorithm.
						itau := 0
						iwork := itau + m

						// Compute A = L*Q.
						impl.Dgelqf(m, n, a, lda, work[itau:], work[iwork:], lwork-iwork)

						// Copy result to VT.
						impl.Dlacpy(blas.Upper, m, n, a, lda, vt, ldvt)

						// Generate Q in VT.
						impl.Dorglq(m, n, m, vt, ldvt, work[itau:], work[iwork:], lwork-iwork)
						ie := itau
						itauq := ie + m
						itaup := itauq + m
						iwork = itaup + m

						// Zero out above L in A.
						impl.Dlaset(blas.Upper, m-1, m-1, 0, 0, a[1:], lda)

						// Bidiagonalize L in A.
						impl.Dgebrd(m, m, a, lda, s, work[ie:],
							work[itauq:], work[itaup:], work[iwork:], lwork-iwork)

						// Multiply right vectors bidiagonalizing L by Q in VT.
						impl.Dormbr(lapack.ApplyP, blas.Left, blas.Trans, m, n, m,
							a, lda, work[itaup:], vt, ldvt, work[iwork:], lwork-iwork)
						iwork = ie + m

						// Perform bidiagonal QR iteration, computing right
						// singular vectors of A in VT.
						ok = impl.Dbdsqr(blas.Upper, m, n, 0, 0, s, work[ie:],
							vt, ldvt, work, 1, work, 1, work[iwork:])
					}
				} else if wantuo {
					// Path 5t.
					panic(noSVDO)
				} else if wantuas {
					// Path 6t.
					if lwork >= m*m+max(4*m, bdspac) {
						// Sufficient workspace for a fast algorithm.
						iu := 0
						var ldworku int
						if lwork >= wrkbl+lda*m {
							ldworku = lda
						} else {
							ldworku = m
						}
						itau := iu + ldworku*m
						iwork := itau + m

						// Compute A = L*Q.
						impl.Dgelqf(m, n, a, lda, work[itau:], work[iwork:], lwork-iwork)

						// Copy L to work[iu:], zeroing out above it.
						impl.Dlacpy(blas.Lower, m, m, a, lda, work[iu:], ldworku)
						impl.Dlaset(blas.Upper, m-1, m-1, 0, 0, work[iu+1:], ldworku)

						// Generate Q in A.
						impl.Dorglq(m, n, m, a, lda, work[itau:], work[iwork:], lwork-iwork)
						ie := itau
						itauq := ie + m
						itaup := itauq + m
						iwork = itaup + m

						// Bidiagonalize L in work[iu:], copying result to U.
						impl.Dgebrd(m, m, work[iu:], ldworku, s, work[ie:],
							work[itauq:], work[itaup:], work[iwork:], lwork-iwork)
						impl.Dlacpy(blas.Lower, m, m, work[iu:], ldworku, u, ldu)

						// Generate right bidiagionalizing vectors in work[iu:].
						impl.Dorgbr(lapack.GeneratePT, m, m, m, work[iu:], ldworku,
							work[itaup:], work[iwork:], lwork-iwork)

						// Generate left bidiagonalizing vectors in U.
						impl.Dorgbr(lapack.GenerateQ, m, m, m, u, ldu, work[itauq:], work[iwork:], lwork-iwork)
						iwork = ie + m

						// Perform bidiagonal QR iteration, computing left singular
						// vectors of L in U and computing right singular vectors of
						// L in work[iu:].
						ok = impl.Dbdsqr(blas.Upper, m, m, m, 0, s, work[ie:],
							work[iu:], ldworku, u, ldu, work, 1, work[iwork:])

						// Multiply right singular vectors of L in work[iu:] by
						// Q in A, storing result in VT.
						bi.Dgemm(blas.NoTrans, blas.NoTrans, m, n, m, 1,
							work[iu:], ldworku, a, lda, 0, vt, ldvt)
					} else {
						// Insufficient workspace for a fast algorithm.
						itau := 0
						iwork := itau + m

						// Compute A = L*Q, copying result to VT.
						impl.Dgelqf(m, n, a, lda, work[itau:], work[iwork:], lwork-iwork)
						impl.Dlacpy(blas.Upper, m, n, a, lda, vt, ldvt)

						// Generate Q in VT.
						impl.Dorglq(m, n, m, vt, ldvt, work[itau:], work[iwork:], lwork-iwork)

						// Copy L to U, zeroing out above it.
						impl.Dlacpy(blas.Lower, m, m, a, lda, u, ldu)
						impl.Dlaset(blas.Upper, m-1, m-1, 0, 0, u[1:], ldu)

						ie := itau
						itauq := ie + m
						itaup := itauq + m
						iwork = itaup + m

						// Bidiagonalize L in U.
						impl.Dgebrd(m, m, u, ldu, s, work[ie:],
							work[itauq:], work[itaup:], work[iwork:], lwork-iwork)

						// Multiply right bidiagonalizing vectors in U by Q in VT.
						impl.Dormbr(lapack.ApplyP, blas.Left, blas.Trans, m, n, m,
							u, ldu, work[itaup:], vt, ldvt, work[iwork:], lwork-iwork)

						// Generate left bidiagonalizing vectors in U.
						impl.Dorgbr(lapack.GenerateQ, m, m, m, u, ldu, work[itauq:], work[iwork:], lwork-iwork)
						iwork = ie + m

						// Perform bidiagonal QR iteration, computing left singular
						// vectors of A in U and computing right singular vectors
						// of A in VT.
						ok = impl.Dbdsqr(blas.Upper, m, n, m, 0, s, work[ie:], vt, ldvt,
							u, ldu, work, 1, work[iwork:])
					}
				}
			} else if wantva {
				if wantun {
					// Path 7t.
					if lwork >= m*m+max(max(n+m, 4*m), bdspac) {
						// Sufficient workspace for a fast algorithm.
						ir := 0
						var ldworkr int
						if lwork >= wrkbl+lda*m {
							ldworkr = lda
						} else {
							ldworkr = m
						}
						itau := ir + ldworkr*m
						iwork := itau + m

						// Compute A = L*Q, copying result to VT.
						impl.Dgelqf(m, n, a, lda, work[itau:], work[iwork:], lwork-iwork)
						impl.Dlacpy(blas.Upper, m, n, a, lda, vt, ldvt)

						// Copy L to work[ir:], zeroing out above it.
						impl.Dlacpy(blas.Lower, m, m, a, lda, work[ir:], ldworkr)
						impl.Dlaset(blas.Upper, m-1, m-1, 0, 0, work[ir+1:], ldworkr)

						// Generate Q in VT.
						impl.Dorglq(n, n, m, vt, ldvt, work[itau:], work[iwork:], lwork-iwork)

						ie := itau
						itauq := ie + m
						itaup := itauq + m
						iwork = itaup + m

						// Bidiagonalize L in work[ir:].
						impl.Dgebrd(m, m, work[ir:], ldworkr, s, work[ie:],
							work[itauq:], work[itaup:], work[iwork:], lwork-iwork)

						// Generate right bidiagonalizing vectors in work[ir:].
						impl.Dorgbr(lapack.GeneratePT, m, m, m, work[ir:], ldworkr,
							work[itaup:], work[iwork:], lwork-iwork)
						iwork = ie + m

						// Perform bidiagonal QR iteration, computing right
						// singular vectors of L in work[ir:].
						ok = impl.Dbdsqr(blas.Upper, m, m, 0, 0, s, work[ie:],
							work[ir:], ldworkr, work, 1, work, 1, work[iwork:])

						// Multiply right singular vectors of L in work[ir:] by
						// Q in VT, storing result in A.
						bi.Dgemm(blas.NoTrans, blas.NoTrans, m, n, m, 1,
							work[ir:], ldworkr, vt, ldvt, 0, a, lda)

						// Copy right singular vectors of A from A to VT.
						impl.Dlacpy(blas.All, m, n, a, lda, vt, ldvt)
					} else {
						// Insufficient workspace for a fast algorithm.
						itau := 0
						iwork := itau + m
						// Compute A = L * Q, copying result to VT.
						impl.Dgelqf(m, n, a, lda, work[itau:], work[iwork:], lwork-iwork)
						impl.Dlacpy(blas.Upper, m, n, a, lda, vt, ldvt)

						// Generate Q in VT.
						impl.Dorglq(n, n, m, vt, ldvt, work[itau:], work[iwork:], lwork-iwork)

						ie := itau
						itauq := ie + m
						itaup := itauq + m
						iwork = itaup + m

						// Zero out above L in A.
						impl.Dlaset(blas.Upper, m-1, m-1, 0, 0, a[1:], lda)

						// Bidiagonalize L in A.
						impl.Dgebrd(m, m, a, lda, s, work[ie:], work[itauq:],
							work[itaup:], work[iwork:], lwork-iwork)

						// Multiply right bidiagonalizing vectors in A by Q in VT.
						impl.Dormbr(lapack.ApplyP, blas.Left, blas.Trans, m, n, m,
							a, lda, work[itaup:], vt, ldvt, work[iwork:], lwork-iwork)
						iwork = ie + m

						// Perform bidiagonal QR iteration, computing right singular
						// vectors of A in VT.
						ok = impl.Dbdsqr(blas.Upper, m, n, 0, 0, s, work[ie:],
							vt, ldvt, work, 1, work, 1, work[iwork:])
					}
				} else if wantuo {
					panic(noSVDO)
				} else if wantuas {
					// Path 9t.
					if lwork >= m*m+max(max(m+n, 4*m), bdspac) {
						// Sufficient workspace for a fast algorithm.
						iu := 0

						var ldworku int
						if lwork >= wrkbl+lda*m {
							ldworku = lda
						} else {
							ldworku = m
						}
						itau := iu + ldworku*m
						iwork := itau + m

						// Generate A = L * Q copying result to VT.
						impl.Dgelqf(m, n, a, lda, work[itau:], work[iwork:], lwork-iwork)
						impl.Dlacpy(blas.Upper, m, n, a, lda, vt, ldvt)

						// Generate Q in VT.
						impl.Dorglq(n, n, m, vt, ldvt, work[itau:], work[iwork:], lwork-iwork)

						// Copy L to work[iu:], zeroing out above it.
						impl.Dlacpy(blas.Lower, m, m, a, lda, work[iu:], ldworku)
						impl.Dlaset(blas.Upper, m-1, m-1, 0, 0, work[iu+1:], ldworku)
						ie = itau
						itauq := ie + m
						itaup := itauq + m
						iwork = itaup + m

						// Bidiagonalize L in work[iu:], copying result to U.
						impl.Dgebrd(m, m, work[iu:], ldworku, s, work[ie:],
							work[itauq:], work[itaup:], work[iwork:], lwork-iwork)
						impl.Dlacpy(blas.Lower, m, m, work[iu:], ldworku, u, ldu)

						// Generate right bidiagonalizing vectors in work[iu:].
						impl.Dorgbr(lapack.GeneratePT, m, m, m, work[iu:], ldworku,
							work[itaup:], work[iwork:], lwork-iwork)

						// Generate left bidiagonalizing vectors in U.
						impl.Dorgbr(lapack.GenerateQ, m, m, m, u, ldu, work[itauq:], work[iwork:], lwork-iwork)
						iwork = ie + m

						// Perform bidiagonal QR iteration, computing left singular
						// vectors of L in U and computing right singular vectors
						// of L in work[iu:].
						ok = impl.Dbdsqr(blas.Upper, m, m, m, 0, s, work[ie:],
							work[iu:], ldworku, u, ldu, work, 1, work[iwork:])

						// Multiply right singular vectors of L in work[iu:]
						// Q in VT, storing result in A.
						bi.Dgemm(blas.NoTrans, blas.NoTrans, m, n, m, 1,
							work[iu:], ldworku, vt, ldvt, 0, a, lda)

						// Copy right singular vectors of A from A to VT.
						impl.Dlacpy(blas.All, m, n, a, lda, vt, ldvt)
					} else {
						// Insufficient workspace for a fast algorithm.
						itau := 0
						iwork := itau + m

						// Compute A = L * Q, copying result to VT.
						impl.Dgelqf(m, n, a, lda, work[itau:], work[iwork:], lwork-iwork)
						impl.Dlacpy(blas.Upper, m, n, a, lda, vt, ldvt)

						// Generate Q in VT.
						impl.Dorglq(n, n, m, vt, ldvt, work[itau:], work[iwork:], lwork-iwork)

						// Copy L to U, zeroing out above it.
						impl.Dlacpy(blas.Lower, m, m, a, lda, u, ldu)
						impl.Dlaset(blas.Upper, m-1, m-1, 0, 0, u[1:], ldu)

						ie = itau
						itauq := ie + m
						itaup := itauq + m
						iwork = itaup + m

						// Bidiagonalize L in U.
						impl.Dgebrd(m, m, u, ldu, s, work[ie:], work[itauq:],
							work[itaup:], work[iwork:], lwork-iwork)

						// Multiply right bidiagonalizing vectors in U by Q in VT.
						impl.Dormbr(lapack.ApplyP, blas.Left, blas.Trans, m, n, m,
							u, ldu, work[itaup:], vt, ldvt, work[iwork:], lwork-iwork)

						// Generate left bidiagonalizing vectors in U.
						impl.Dorgbr(lapack.GenerateQ, m, m, m, u, ldu, work[itauq:], work[iwork:], lwork-iwork)
						iwork = ie + m

						// Perform bidiagonal QR iteration, computing left singular
						// vectors of A in U and computing right singular vectors
						// of A in VT.
						ok = impl.Dbdsqr(blas.Upper, m, n, m, 0, s, work[ie:],
							vt, ldvt, u, ldu, work, 1, work[iwork:])
					}
				}
			}
		} else {
			// Path 10t.
			// N at least M, but not much larger.
			ie = 0
			itauq := ie + m
			itaup := itauq + m
			iwork := itaup + m

			// Bidiagonalize A.
			impl.Dgebrd(m, n, a, lda, s, work[ie:], work[itauq:], work[itaup:], work[iwork:], lwork-iwork)
			if wantuas {
				// If left singular vectors desired in U, copy result to U and
				// generate left bidiagonalizing vectors in U.
				impl.Dlacpy(blas.Lower, m, m, a, lda, u, ldu)
				impl.Dorgbr(lapack.GenerateQ, m, m, n, u, ldu, work[itauq:], work[iwork:], lwork-iwork)
			}
			if wantvas {
				// If right singular vectors desired in VT, copy result to VT
				// and generate right bidiagonalizing vectors in VT.
				impl.Dlacpy(blas.Upper, m, n, a, lda, vt, ldvt)
				var nrvt int
				if wantva {
					nrvt = n
				} else {
					nrvt = m
				}
				impl.Dorgbr(lapack.GeneratePT, nrvt, n, m, vt, ldvt, work[itaup:], work[iwork:], lwork-iwork)
			}
			if wantuo {
				panic(noSVDO)
			}
			if wantvo {
				panic(noSVDO)
			}
			iwork = ie + m
			var nru, ncvt int
			if wantuas || wantuo {
				nru = m
			}
			if wantvas || wantvo {
				ncvt = n
			}
			if !wantuo && !wantvo {
				// Perform bidiagonal QR iteration, if desired, computing left
				// singular vectors in U and computing right singular vectors in
				// VT.
				ok = impl.Dbdsqr(blas.Lower, m, ncvt, nru, 0, s, work[ie:],
					vt, ldvt, u, ldu, work, 1, work[iwork:])
			} else {
				// There will be two branches when the implementation is complete.
				panic(noSVDO)
			}
		}
	}
	if !ok {
		if ie > 1 {
			for i := 0; i < minmn-1; i++ {
				work[i+1] = work[i+ie]
			}
		}
		if ie < 1 {
			for i := minmn - 2; i >= 0; i-- {
				work[i+1] = work[i+ie]
			}
		}
	}
	// Undo scaling if necessary.
	if iscl {
		if anrm > bignum {
			impl.Dlascl(lapack.General, 0, 0, bignum, anrm, 1, minmn, s, minmn)
		}
		if !ok && anrm > bignum {
			impl.Dlascl(lapack.General, 0, 0, bignum, anrm, 1, minmn-1, work[1:], minmn)
		}
		if anrm < smlnum {
			impl.Dlascl(lapack.General, 0, 0, smlnum, anrm, 1, minmn, s, minmn)
		}
		if !ok && anrm < smlnum {
			impl.Dlascl(lapack.General, 0, 0, smlnum, anrm, 1, minmn-1, work[1:], minmn)
		}
	}
	work[0] = float64(maxwrk)
	return ok
}
