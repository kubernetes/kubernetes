// Copyright ©2016 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"math"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/gonum/lapack"
)

// Dsyev computes all eigenvalues and, optionally, the eigenvectors of a real
// symmetric matrix A.
//
// w contains the eigenvalues in ascending order upon return. w must have length
// at least n, and Dsyev will panic otherwise.
//
// On entry, a contains the elements of the symmetric matrix A in the triangular
// portion specified by uplo. If jobz == lapack.EVCompute, a contains the
// orthonormal eigenvectors of A on exit, otherwise jobz must be lapack.EVNone
// and on exit the specified triangular region is overwritten.
//
// work is temporary storage, and lwork specifies the usable memory length. At minimum,
// lwork >= 3*n-1, and Dsyev will panic otherwise. The amount of blocking is
// limited by the usable length. If lwork == -1, instead of computing Dsyev the
// optimal work length is stored into work[0].
func (impl Implementation) Dsyev(jobz lapack.EVJob, uplo blas.Uplo, n int, a []float64, lda int, w, work []float64, lwork int) (ok bool) {
	switch {
	case jobz != lapack.EVNone && jobz != lapack.EVCompute:
		panic(badEVJob)
	case uplo != blas.Upper && uplo != blas.Lower:
		panic(badUplo)
	case n < 0:
		panic(nLT0)
	case lda < max(1, n):
		panic(badLdA)
	case lwork < max(1, 3*n-1) && lwork != -1:
		panic(badLWork)
	case len(work) < max(1, lwork):
		panic(shortWork)
	}

	// Quick return if possible.
	if n == 0 {
		return true
	}

	var opts string
	if uplo == blas.Upper {
		opts = "U"
	} else {
		opts = "L"
	}
	nb := impl.Ilaenv(1, "DSYTRD", opts, n, -1, -1, -1)
	lworkopt := max(1, (nb+2)*n)
	if lwork == -1 {
		work[0] = float64(lworkopt)
		return
	}

	switch {
	case len(a) < (n-1)*lda+n:
		panic(shortA)
	case len(w) < n:
		panic(shortW)
	}

	if n == 1 {
		w[0] = a[0]
		work[0] = 2
		if jobz == lapack.EVCompute {
			a[0] = 1
		}
		return true
	}

	safmin := dlamchS
	eps := dlamchP
	smlnum := safmin / eps
	bignum := 1 / smlnum
	rmin := math.Sqrt(smlnum)
	rmax := math.Sqrt(bignum)

	// Scale matrix to allowable range, if necessary.
	anrm := impl.Dlansy(lapack.MaxAbs, uplo, n, a, lda, work)
	scaled := false
	var sigma float64
	if anrm > 0 && anrm < rmin {
		scaled = true
		sigma = rmin / anrm
	} else if anrm > rmax {
		scaled = true
		sigma = rmax / anrm
	}
	if scaled {
		kind := lapack.LowerTri
		if uplo == blas.Upper {
			kind = lapack.UpperTri
		}
		impl.Dlascl(kind, 0, 0, 1, sigma, n, n, a, lda)
	}
	var inde int
	indtau := inde + n
	indwork := indtau + n
	llwork := lwork - indwork
	impl.Dsytrd(uplo, n, a, lda, w, work[inde:], work[indtau:], work[indwork:], llwork)

	// For eigenvalues only, call Dsterf. For eigenvectors, first call Dorgtr
	// to generate the orthogonal matrix, then call Dsteqr.
	if jobz == lapack.EVNone {
		ok = impl.Dsterf(n, w, work[inde:])
	} else {
		impl.Dorgtr(uplo, n, a, lda, work[indtau:], work[indwork:], llwork)
		ok = impl.Dsteqr(lapack.EVComp(jobz), n, w, work[inde:], a, lda, work[indtau:])
	}
	if !ok {
		return false
	}

	// If the matrix was scaled, then rescale eigenvalues appropriately.
	if scaled {
		bi := blas64.Implementation()
		bi.Dscal(n, 1/sigma, w, 1)
	}
	work[0] = float64(lworkopt)
	return true
}
