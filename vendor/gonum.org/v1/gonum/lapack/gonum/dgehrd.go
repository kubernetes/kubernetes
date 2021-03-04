// Copyright ©2016 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/gonum/lapack"
)

// Dgehrd reduces a block of a real n×n general matrix A to upper Hessenberg
// form H by an orthogonal similarity transformation Qᵀ * A * Q = H.
//
// The matrix Q is represented as a product of (ihi-ilo) elementary
// reflectors
//  Q = H_{ilo} H_{ilo+1} ... H_{ihi-1}.
// Each H_i has the form
//  H_i = I - tau[i] * v * vᵀ
// where v is a real vector with v[0:i+1] = 0, v[i+1] = 1 and v[ihi+1:n] = 0.
// v[i+2:ihi+1] is stored on exit in A[i+2:ihi+1,i].
//
// On entry, a contains the n×n general matrix to be reduced. On return, the
// upper triangle and the first subdiagonal of A will be overwritten with the
// upper Hessenberg matrix H, and the elements below the first subdiagonal, with
// the slice tau, represent the orthogonal matrix Q as a product of elementary
// reflectors.
//
// The contents of a are illustrated by the following example, with n = 7, ilo =
// 1 and ihi = 5.
// On entry,
//  [ a   a   a   a   a   a   a ]
//  [     a   a   a   a   a   a ]
//  [     a   a   a   a   a   a ]
//  [     a   a   a   a   a   a ]
//  [     a   a   a   a   a   a ]
//  [     a   a   a   a   a   a ]
//  [                         a ]
// on return,
//  [ a   a   h   h   h   h   a ]
//  [     a   h   h   h   h   a ]
//  [     h   h   h   h   h   h ]
//  [     v1  h   h   h   h   h ]
//  [     v1  v2  h   h   h   h ]
//  [     v1  v2  v3  h   h   h ]
//  [                         a ]
// where a denotes an element of the original matrix A, h denotes a
// modified element of the upper Hessenberg matrix H, and vi denotes an
// element of the vector defining H_i.
//
// ilo and ihi determine the block of A that will be reduced to upper Hessenberg
// form. It must hold that 0 <= ilo <= ihi < n if n > 0, and ilo == 0 and ihi ==
// -1 if n == 0, otherwise Dgehrd will panic.
//
// On return, tau will contain the scalar factors of the elementary reflectors.
// Elements tau[:ilo] and tau[ihi:] will be set to zero. tau must have length
// equal to n-1 if n > 0, otherwise Dgehrd will panic.
//
// work must have length at least lwork and lwork must be at least max(1,n),
// otherwise Dgehrd will panic. On return, work[0] contains the optimal value of
// lwork.
//
// If lwork == -1, instead of performing Dgehrd, only the optimal value of lwork
// will be stored in work[0].
//
// Dgehrd is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dgehrd(n, ilo, ihi int, a []float64, lda int, tau, work []float64, lwork int) {
	switch {
	case n < 0:
		panic(nLT0)
	case ilo < 0 || max(0, n-1) < ilo:
		panic(badIlo)
	case ihi < min(ilo, n-1) || n <= ihi:
		panic(badIhi)
	case lda < max(1, n):
		panic(badLdA)
	case lwork < max(1, n) && lwork != -1:
		panic(badLWork)
	case len(work) < lwork:
		panic(shortWork)
	}

	// Quick return if possible.
	if n == 0 {
		work[0] = 1
		return
	}

	const (
		nbmax = 64
		ldt   = nbmax + 1
		tsize = ldt * nbmax
	)
	// Compute the workspace requirements.
	nb := min(nbmax, impl.Ilaenv(1, "DGEHRD", " ", n, ilo, ihi, -1))
	lwkopt := n*nb + tsize
	if lwork == -1 {
		work[0] = float64(lwkopt)
		return
	}

	if len(a) < (n-1)*lda+n {
		panic(shortA)
	}
	if len(tau) != n-1 {
		panic(badLenTau)
	}

	// Set tau[:ilo] and tau[ihi:] to zero.
	for i := 0; i < ilo; i++ {
		tau[i] = 0
	}
	for i := ihi; i < n-1; i++ {
		tau[i] = 0
	}

	// Quick return if possible.
	nh := ihi - ilo + 1
	if nh <= 1 {
		work[0] = 1
		return
	}

	// Determine the block size.
	nbmin := 2
	var nx int
	if 1 < nb && nb < nh {
		// Determine when to cross over from blocked to unblocked code
		// (last block is always handled by unblocked code).
		nx = max(nb, impl.Ilaenv(3, "DGEHRD", " ", n, ilo, ihi, -1))
		if nx < nh {
			// Determine if workspace is large enough for blocked code.
			if lwork < n*nb+tsize {
				// Not enough workspace to use optimal nb:
				// determine the minimum value of nb, and reduce
				// nb or force use of unblocked code.
				nbmin = max(2, impl.Ilaenv(2, "DGEHRD", " ", n, ilo, ihi, -1))
				if lwork >= n*nbmin+tsize {
					nb = (lwork - tsize) / n
				} else {
					nb = 1
				}
			}
		}
	}
	ldwork := nb // work is used as an n×nb matrix.

	var i int
	if nb < nbmin || nh <= nb {
		// Use unblocked code below.
		i = ilo
	} else {
		// Use blocked code.
		bi := blas64.Implementation()
		iwt := n * nb // Size of the matrix Y and index where the matrix T starts in work.
		for i = ilo; i < ihi-nx; i += nb {
			ib := min(nb, ihi-i)

			// Reduce columns [i:i+ib] to Hessenberg form, returning the
			// matrices V and T of the block reflector H = I - V*T*Vᵀ
			// which performs the reduction, and also the matrix Y = A*V*T.
			impl.Dlahr2(ihi+1, i+1, ib, a[i:], lda, tau[i:], work[iwt:], ldt, work, ldwork)

			// Apply the block reflector H to A[:ihi+1,i+ib:ihi+1] from the
			// right, computing  A := A - Y * Vᵀ. V[i+ib,i+ib-1] must be set
			// to 1.
			ei := a[(i+ib)*lda+i+ib-1]
			a[(i+ib)*lda+i+ib-1] = 1
			bi.Dgemm(blas.NoTrans, blas.Trans, ihi+1, ihi-i-ib+1, ib,
				-1, work, ldwork,
				a[(i+ib)*lda+i:], lda,
				1, a[i+ib:], lda)
			a[(i+ib)*lda+i+ib-1] = ei

			// Apply the block reflector H to A[0:i+1,i+1:i+ib-1] from the
			// right.
			bi.Dtrmm(blas.Right, blas.Lower, blas.Trans, blas.Unit, i+1, ib-1,
				1, a[(i+1)*lda+i:], lda, work, ldwork)
			for j := 0; j <= ib-2; j++ {
				bi.Daxpy(i+1, -1, work[j:], ldwork, a[i+j+1:], lda)
			}

			// Apply the block reflector H to A[i+1:ihi+1,i+ib:n] from the
			// left.
			impl.Dlarfb(blas.Left, blas.Trans, lapack.Forward, lapack.ColumnWise,
				ihi-i, n-i-ib, ib,
				a[(i+1)*lda+i:], lda, work[iwt:], ldt, a[(i+1)*lda+i+ib:], lda, work, ldwork)
		}
	}
	// Use unblocked code to reduce the rest of the matrix.
	impl.Dgehd2(n, i, ihi, a, lda, tau, work)
	work[0] = float64(lwkopt)
}
