// Copyright ©2016 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"math"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/lapack"
)

// Dhseqr computes the eigenvalues of an n×n Hessenberg matrix H and,
// optionally, the matrices T and Z from the Schur decomposition
//  H = Z T Zᵀ,
// where T is an n×n upper quasi-triangular matrix (the Schur form), and Z is
// the n×n orthogonal matrix of Schur vectors.
//
// Optionally Z may be postmultiplied into an input orthogonal matrix Q so that
// this routine can give the Schur factorization of a matrix A which has been
// reduced to the Hessenberg form H by the orthogonal matrix Q:
//  A = Q H Qᵀ = (QZ) T (QZ)ᵀ.
//
// If job == lapack.EigenvaluesOnly, only the eigenvalues will be computed.
// If job == lapack.EigenvaluesAndSchur, the eigenvalues and the Schur form T will
// be computed.
// For other values of job Dhseqr will panic.
//
// If compz == lapack.SchurNone, no Schur vectors will be computed and Z will not be
// referenced.
// If compz == lapack.SchurHess, on return Z will contain the matrix of Schur
// vectors of H.
// If compz == lapack.SchurOrig, on entry z is assumed to contain the orthogonal
// matrix Q that is the identity except for the submatrix
// Q[ilo:ihi+1,ilo:ihi+1]. On return z will be updated to the product Q*Z.
//
// ilo and ihi determine the block of H on which Dhseqr operates. It is assumed
// that H is already upper triangular in rows and columns [0:ilo] and [ihi+1:n],
// although it will be only checked that the block is isolated, that is,
//  ilo == 0   or H[ilo,ilo-1] == 0,
//  ihi == n-1 or H[ihi+1,ihi] == 0,
// and Dhseqr will panic otherwise. ilo and ihi are typically set by a previous
// call to Dgebal, otherwise they should be set to 0 and n-1, respectively. It
// must hold that
//  0 <= ilo <= ihi < n     if n > 0,
//  ilo == 0 and ihi == -1  if n == 0.
//
// wr and wi must have length n.
//
// work must have length at least lwork and lwork must be at least max(1,n)
// otherwise Dhseqr will panic. The minimum lwork delivers very good and
// sometimes optimal performance, although lwork as large as 11*n may be
// required. On return, work[0] will contain the optimal value of lwork.
//
// If lwork is -1, instead of performing Dhseqr, the function only estimates the
// optimal workspace size and stores it into work[0]. Neither h nor z are
// accessed.
//
// unconverged indicates whether Dhseqr computed all the eigenvalues.
//
// If unconverged == 0, all the eigenvalues have been computed and their real
// and imaginary parts will be stored on return in wr and wi, respectively. If
// two eigenvalues are computed as a complex conjugate pair, they are stored in
// consecutive elements of wr and wi, say the i-th and (i+1)th, with wi[i] > 0
// and wi[i+1] < 0.
//
// If unconverged == 0 and job == lapack.EigenvaluesAndSchur, on return H will
// contain the upper quasi-triangular matrix T from the Schur decomposition (the
// Schur form). 2×2 diagonal blocks (corresponding to complex conjugate pairs of
// eigenvalues) will be returned in standard form, with
//  H[i,i] == H[i+1,i+1],
// and
//  H[i+1,i]*H[i,i+1] < 0.
// The eigenvalues will be stored in wr and wi in the same order as on the
// diagonal of the Schur form returned in H, with
//  wr[i] = H[i,i],
// and, if H[i:i+2,i:i+2] is a 2×2 diagonal block,
//  wi[i]   = sqrt(-H[i+1,i]*H[i,i+1]),
//  wi[i+1] = -wi[i].
//
// If unconverged == 0 and job == lapack.EigenvaluesOnly, the contents of h
// on return is unspecified.
//
// If unconverged > 0, some eigenvalues have not converged, and the blocks
// [0:ilo] and [unconverged:n] of wr and wi will contain those eigenvalues which
// have been successfully computed. Failures are rare.
//
// If unconverged > 0 and job == lapack.EigenvaluesOnly, on return the
// remaining unconverged eigenvalues are the eigenvalues of the upper Hessenberg
// matrix H[ilo:unconverged,ilo:unconverged].
//
// If unconverged > 0 and job == lapack.EigenvaluesAndSchur, then on
// return
//  (initial H) U = U (final H),   (*)
// where U is an orthogonal matrix. The final H is upper Hessenberg and
// H[unconverged:ihi+1,unconverged:ihi+1] is upper quasi-triangular.
//
// If unconverged > 0 and compz == lapack.SchurOrig, then on return
//  (final Z) = (initial Z) U,
// where U is the orthogonal matrix in (*) regardless of the value of job.
//
// If unconverged > 0 and compz == lapack.SchurHess, then on return
//  (final Z) = U,
// where U is the orthogonal matrix in (*) regardless of the value of job.
//
// References:
//  [1] R. Byers. LAPACK 3.1 xHSEQR: Tuning and Implementation Notes on the
//      Small Bulge Multi-Shift QR Algorithm with Aggressive Early Deflation.
//      LAPACK Working Note 187 (2007)
//      URL: http://www.netlib.org/lapack/lawnspdf/lawn187.pdf
//  [2] K. Braman, R. Byers, R. Mathias. The Multishift QR Algorithm. Part I:
//      Maintaining Well-Focused Shifts and Level 3 Performance. SIAM J. Matrix
//      Anal. Appl. 23(4) (2002), pp. 929—947
//      URL: http://dx.doi.org/10.1137/S0895479801384573
//  [3] K. Braman, R. Byers, R. Mathias. The Multishift QR Algorithm. Part II:
//      Aggressive Early Deflation. SIAM J. Matrix Anal. Appl. 23(4) (2002), pp. 948—973
//      URL: http://dx.doi.org/10.1137/S0895479801384585
//
// Dhseqr is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dhseqr(job lapack.SchurJob, compz lapack.SchurComp, n, ilo, ihi int, h []float64, ldh int, wr, wi []float64, z []float64, ldz int, work []float64, lwork int) (unconverged int) {
	wantt := job == lapack.EigenvaluesAndSchur
	wantz := compz == lapack.SchurHess || compz == lapack.SchurOrig

	switch {
	case job != lapack.EigenvaluesOnly && job != lapack.EigenvaluesAndSchur:
		panic(badSchurJob)
	case compz != lapack.SchurNone && compz != lapack.SchurHess && compz != lapack.SchurOrig:
		panic(badSchurComp)
	case n < 0:
		panic(nLT0)
	case ilo < 0 || max(0, n-1) < ilo:
		panic(badIlo)
	case ihi < min(ilo, n-1) || n <= ihi:
		panic(badIhi)
	case ldh < max(1, n):
		panic(badLdH)
	case ldz < 1, wantz && ldz < n:
		panic(badLdZ)
	case lwork < max(1, n) && lwork != -1:
		panic(badLWork)
	case len(work) < max(1, lwork):
		panic(shortWork)
	}

	// Quick return if possible.
	if n == 0 {
		work[0] = 1
		return 0
	}

	// Quick return in case of a workspace query.
	if lwork == -1 {
		impl.Dlaqr04(wantt, wantz, n, ilo, ihi, h, ldh, wr, wi, ilo, ihi, z, ldz, work, -1, 1)
		work[0] = math.Max(float64(n), work[0])
		return 0
	}

	switch {
	case len(h) < (n-1)*ldh+n:
		panic(shortH)
	case wantz && len(z) < (n-1)*ldz+n:
		panic(shortZ)
	case len(wr) < n:
		panic(shortWr)
	case len(wi) < n:
		panic(shortWi)
	}

	const (
		// Matrices of order ntiny or smaller must be processed by
		// Dlahqr because of insufficient subdiagonal scratch space.
		// This is a hard limit.
		ntiny = 11

		// nl is the size of a local workspace to help small matrices
		// through a rare Dlahqr failure. nl > ntiny is required and
		// nl <= nmin = Ilaenv(ispec=12,...) is recommended (the default
		// value of nmin is 75). Using nl = 49 allows up to six
		// simultaneous shifts and a 16×16 deflation window.
		nl = 49
	)

	// Copy eigenvalues isolated by Dgebal.
	for i := 0; i < ilo; i++ {
		wr[i] = h[i*ldh+i]
		wi[i] = 0
	}
	for i := ihi + 1; i < n; i++ {
		wr[i] = h[i*ldh+i]
		wi[i] = 0
	}

	// Initialize Z to identity matrix if requested.
	if compz == lapack.SchurHess {
		impl.Dlaset(blas.All, n, n, 0, 1, z, ldz)
	}

	// Quick return if possible.
	if ilo == ihi {
		wr[ilo] = h[ilo*ldh+ilo]
		wi[ilo] = 0
		return 0
	}

	// Dlahqr/Dlaqr04 crossover point.
	nmin := impl.Ilaenv(12, "DHSEQR", string(job)+string(compz), n, ilo, ihi, lwork)
	nmin = max(ntiny, nmin)

	if n > nmin {
		// Dlaqr0 for big matrices.
		unconverged = impl.Dlaqr04(wantt, wantz, n, ilo, ihi, h, ldh, wr[:ihi+1], wi[:ihi+1],
			ilo, ihi, z, ldz, work, lwork, 1)
	} else {
		// Dlahqr for small matrices.
		unconverged = impl.Dlahqr(wantt, wantz, n, ilo, ihi, h, ldh, wr[:ihi+1], wi[:ihi+1],
			ilo, ihi, z, ldz)
		if unconverged > 0 {
			// A rare Dlahqr failure! Dlaqr04 sometimes succeeds
			// when Dlahqr fails.
			kbot := unconverged
			if n >= nl {
				// Larger matrices have enough subdiagonal
				// scratch space to call Dlaqr04 directly.
				unconverged = impl.Dlaqr04(wantt, wantz, n, ilo, kbot, h, ldh,
					wr[:ihi+1], wi[:ihi+1], ilo, ihi, z, ldz, work, lwork, 1)
			} else {
				// Tiny matrices don't have enough subdiagonal
				// scratch space to benefit from Dlaqr04. Hence,
				// tiny matrices must be copied into a larger
				// array before calling Dlaqr04.
				var hl [nl * nl]float64
				impl.Dlacpy(blas.All, n, n, h, ldh, hl[:], nl)
				impl.Dlaset(blas.All, nl, nl-n, 0, 0, hl[n:], nl)
				var workl [nl]float64
				unconverged = impl.Dlaqr04(wantt, wantz, nl, ilo, kbot, hl[:], nl,
					wr[:ihi+1], wi[:ihi+1], ilo, ihi, z, ldz, workl[:], nl, 1)
				work[0] = workl[0]
				if wantt || unconverged > 0 {
					impl.Dlacpy(blas.All, n, n, hl[:], nl, h, ldh)
				}
			}
		}
	}
	// Zero out under the first subdiagonal, if necessary.
	if (wantt || unconverged > 0) && n > 2 {
		impl.Dlaset(blas.Lower, n-2, n-2, 0, 0, h[2*ldh:], ldh)
	}

	work[0] = math.Max(float64(n), work[0])
	return unconverged
}
