// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/lapack"
)

// Dgels finds a minimum-norm solution based on the matrices A and B using the
// QR or LQ factorization. Dgels returns false if the matrix
// A is singular, and true if this solution was successfully found.
//
// The minimization problem solved depends on the input parameters.
//
//  1. If m >= n and trans == blas.NoTrans, Dgels finds X such that || A*X - B||_2
//     is minimized.
//  2. If m < n and trans == blas.NoTrans, Dgels finds the minimum norm solution of
//     A * X = B.
//  3. If m >= n and trans == blas.Trans, Dgels finds the minimum norm solution of
//     A^T * X = B.
//  4. If m < n and trans == blas.Trans, Dgels finds X such that || A*X - B||_2
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
// work is temporary storage, and lwork specifies the usable memory length.
// At minimum, lwork >= max(m,n) + max(m,n,nrhs), and this function will panic
// otherwise. A longer work will enable blocked algorithms to be called.
// In the special case that lwork == -1, work[0] will be set to the optimal working
// length.
func (impl Implementation) Dgels(trans blas.Transpose, m, n, nrhs int, a []float64, lda int, b []float64, ldb int, work []float64, lwork int) bool {
	mn := min(m, n)
	minwrk := mn + max(mn, nrhs)
	switch {
	case trans != blas.NoTrans && trans != blas.Trans && trans != blas.ConjTrans:
		panic(badTrans)
	case m < 0:
		panic(mLT0)
	case n < 0:
		panic(nLT0)
	case nrhs < 0:
		panic(nrhsLT0)
	case lda < max(1, n):
		panic(badLdA)
	case ldb < max(1, nrhs):
		panic(badLdB)
	case lwork < max(1, minwrk) && lwork != -1:
		panic(badLWork)
	case len(work) < max(1, lwork):
		panic(shortWork)
	}

	// Quick return if possible.
	if mn == 0 || nrhs == 0 {
		impl.Dlaset(blas.All, max(m, n), nrhs, 0, 0, b, ldb)
		work[0] = 1
		return true
	}

	// Find optimal block size.
	var nb int
	if m >= n {
		nb = impl.Ilaenv(1, "DGEQRF", " ", m, n, -1, -1)
		if trans != blas.NoTrans {
			nb = max(nb, impl.Ilaenv(1, "DORMQR", "LN", m, nrhs, n, -1))
		} else {
			nb = max(nb, impl.Ilaenv(1, "DORMQR", "LT", m, nrhs, n, -1))
		}
	} else {
		nb = impl.Ilaenv(1, "DGELQF", " ", m, n, -1, -1)
		if trans != blas.NoTrans {
			nb = max(nb, impl.Ilaenv(1, "DORMLQ", "LT", n, nrhs, m, -1))
		} else {
			nb = max(nb, impl.Ilaenv(1, "DORMLQ", "LN", n, nrhs, m, -1))
		}
	}
	wsize := max(1, mn+max(mn, nrhs)*nb)
	work[0] = float64(wsize)

	if lwork == -1 {
		return true
	}

	switch {
	case len(a) < (m-1)*lda+n:
		panic(shortA)
	case len(b) < (max(m, n)-1)*ldb+nrhs:
		panic(shortB)
	}

	// Scale the input matrices if they contain extreme values.
	smlnum := dlamchS / dlamchP
	bignum := 1 / smlnum
	anrm := impl.Dlange(lapack.MaxAbs, m, n, a, lda, nil)
	var iascl int
	if anrm > 0 && anrm < smlnum {
		impl.Dlascl(lapack.General, 0, 0, anrm, smlnum, m, n, a, lda)
		iascl = 1
	} else if anrm > bignum {
		impl.Dlascl(lapack.General, 0, 0, anrm, bignum, m, n, a, lda)
	} else if anrm == 0 {
		// Matrix is all zeros.
		impl.Dlaset(blas.All, max(m, n), nrhs, 0, 0, b, ldb)
		return true
	}
	brow := m
	if trans != blas.NoTrans {
		brow = n
	}
	bnrm := impl.Dlange(lapack.MaxAbs, brow, nrhs, b, ldb, nil)
	ibscl := 0
	if bnrm > 0 && bnrm < smlnum {
		impl.Dlascl(lapack.General, 0, 0, bnrm, smlnum, brow, nrhs, b, ldb)
		ibscl = 1
	} else if bnrm > bignum {
		impl.Dlascl(lapack.General, 0, 0, bnrm, bignum, brow, nrhs, b, ldb)
		ibscl = 2
	}

	// Solve the minimization problem using a QR or an LQ decomposition.
	var scllen int
	if m >= n {
		impl.Dgeqrf(m, n, a, lda, work, work[mn:], lwork-mn)
		if trans == blas.NoTrans {
			impl.Dormqr(blas.Left, blas.Trans, m, nrhs, n,
				a, lda,
				work[:n],
				b, ldb,
				work[mn:], lwork-mn)
			ok := impl.Dtrtrs(blas.Upper, blas.NoTrans, blas.NonUnit, n, nrhs,
				a, lda,
				b, ldb)
			if !ok {
				return false
			}
			scllen = n
		} else {
			ok := impl.Dtrtrs(blas.Upper, blas.Trans, blas.NonUnit, n, nrhs,
				a, lda,
				b, ldb)
			if !ok {
				return false
			}
			for i := n; i < m; i++ {
				for j := 0; j < nrhs; j++ {
					b[i*ldb+j] = 0
				}
			}
			impl.Dormqr(blas.Left, blas.NoTrans, m, nrhs, n,
				a, lda,
				work[:n],
				b, ldb,
				work[mn:], lwork-mn)
			scllen = m
		}
	} else {
		impl.Dgelqf(m, n, a, lda, work, work[mn:], lwork-mn)
		if trans == blas.NoTrans {
			ok := impl.Dtrtrs(blas.Lower, blas.NoTrans, blas.NonUnit,
				m, nrhs,
				a, lda,
				b, ldb)
			if !ok {
				return false
			}
			for i := m; i < n; i++ {
				for j := 0; j < nrhs; j++ {
					b[i*ldb+j] = 0
				}
			}
			impl.Dormlq(blas.Left, blas.Trans, n, nrhs, m,
				a, lda,
				work,
				b, ldb,
				work[mn:], lwork-mn)
			scllen = n
		} else {
			impl.Dormlq(blas.Left, blas.NoTrans, n, nrhs, m,
				a, lda,
				work,
				b, ldb,
				work[mn:], lwork-mn)
			ok := impl.Dtrtrs(blas.Lower, blas.Trans, blas.NonUnit,
				m, nrhs,
				a, lda,
				b, ldb)
			if !ok {
				return false
			}
		}
	}

	// Adjust answer vector based on scaling.
	if iascl == 1 {
		impl.Dlascl(lapack.General, 0, 0, anrm, smlnum, scllen, nrhs, b, ldb)
	}
	if iascl == 2 {
		impl.Dlascl(lapack.General, 0, 0, anrm, bignum, scllen, nrhs, b, ldb)
	}
	if ibscl == 1 {
		impl.Dlascl(lapack.General, 0, 0, smlnum, bnrm, scllen, nrhs, b, ldb)
	}
	if ibscl == 2 {
		impl.Dlascl(lapack.General, 0, 0, bignum, bnrm, scllen, nrhs, b, ldb)
	}

	work[0] = float64(wsize)
	return true
}
