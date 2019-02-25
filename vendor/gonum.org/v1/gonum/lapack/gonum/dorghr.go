// Copyright ©2016 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

// Dorghr generates an n×n orthogonal matrix Q which is defined as the product
// of ihi-ilo elementary reflectors:
//  Q = H_{ilo} H_{ilo+1} ... H_{ihi-1}.
//
// a and lda represent an n×n matrix that contains the elementary reflectors, as
// returned by Dgehrd. On return, a is overwritten by the n×n orthogonal matrix
// Q. Q will be equal to the identity matrix except in the submatrix
// Q[ilo+1:ihi+1,ilo+1:ihi+1].
//
// ilo and ihi must have the same values as in the previous call of Dgehrd. It
// must hold that
//  0 <= ilo <= ihi < n,  if n > 0,
//  ilo = 0, ihi = -1,    if n == 0.
//
// tau contains the scalar factors of the elementary reflectors, as returned by
// Dgehrd. tau must have length n-1.
//
// work must have length at least max(1,lwork) and lwork must be at least
// ihi-ilo. For optimum performance lwork must be at least (ihi-ilo)*nb where nb
// is the optimal blocksize. On return, work[0] will contain the optimal value
// of lwork.
//
// If lwork == -1, instead of performing Dorghr, only the optimal value of lwork
// will be stored into work[0].
//
// If any requirement on input sizes is not met, Dorghr will panic.
//
// Dorghr is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dorghr(n, ilo, ihi int, a []float64, lda int, tau, work []float64, lwork int) {
	checkMatrix(n, n, a, lda)
	nh := ihi - ilo
	switch {
	case ilo < 0 || max(1, n) <= ilo:
		panic(badIlo)
	case ihi < min(ilo, n-1) || n <= ihi:
		panic(badIhi)
	case lwork < max(1, nh) && lwork != -1:
		panic(badWork)
	case len(work) < max(1, lwork):
		panic(shortWork)
	}

	lwkopt := max(1, nh) * impl.Ilaenv(1, "DORGQR", " ", nh, nh, nh, -1)
	if lwork == -1 {
		work[0] = float64(lwkopt)
		return
	}

	// Quick return if possible.
	if n == 0 {
		work[0] = 1
		return
	}

	// Shift the vectors which define the elementary reflectors one column
	// to the right.
	for i := ilo + 2; i < ihi+1; i++ {
		copy(a[i*lda+ilo+1:i*lda+i], a[i*lda+ilo:i*lda+i-1])
	}
	// Set the first ilo+1 and the last n-ihi-1 rows and columns to those of
	// the identity matrix.
	for i := 0; i < ilo+1; i++ {
		for j := 0; j < n; j++ {
			a[i*lda+j] = 0
		}
		a[i*lda+i] = 1
	}
	for i := ilo + 1; i < ihi+1; i++ {
		for j := 0; j <= ilo; j++ {
			a[i*lda+j] = 0
		}
		for j := i; j < n; j++ {
			a[i*lda+j] = 0
		}
	}
	for i := ihi + 1; i < n; i++ {
		for j := 0; j < n; j++ {
			a[i*lda+j] = 0
		}
		a[i*lda+i] = 1
	}
	if nh > 0 {
		// Generate Q[ilo+1:ihi+1,ilo+1:ihi+1].
		impl.Dorgqr(nh, nh, nh, a[(ilo+1)*lda+ilo+1:], lda, tau[ilo:ihi], work, lwork)
	}
	work[0] = float64(lwkopt)
}
