// Copyright ©2016 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"math"

	"gonum.org/v1/gonum/blas"
)

// Dlaqr04 computes the eigenvalues of a block of an n×n upper Hessenberg matrix
// H, and optionally the matrices T and Z from the Schur decomposition
//  H = Z T Zᵀ
// where T is an upper quasi-triangular matrix (the Schur form), and Z is the
// orthogonal matrix of Schur vectors.
//
// wantt indicates whether the full Schur form T is required. If wantt is false,
// then only enough of H will be updated to preserve the eigenvalues.
//
// wantz indicates whether the n×n matrix of Schur vectors Z is required. If it
// is true, the orthogonal similarity transformation will be accumulated into
// Z[iloz:ihiz+1,ilo:ihi+1], otherwise Z will not be referenced.
//
// ilo and ihi determine the block of H on which Dlaqr04 operates. It must hold that
//  0 <= ilo <= ihi < n     if n > 0,
//  ilo == 0 and ihi == -1  if n == 0,
// and the block must be isolated, that is,
//  ilo == 0   or H[ilo,ilo-1] == 0,
//  ihi == n-1 or H[ihi+1,ihi] == 0,
// otherwise Dlaqr04 will panic.
//
// wr and wi must have length ihi+1.
//
// iloz and ihiz specify the rows of Z to which transformations will be applied
// if wantz is true. It must hold that
//  0 <= iloz <= ilo,  and  ihi <= ihiz < n,
// otherwise Dlaqr04 will panic.
//
// work must have length at least lwork and lwork must be
//  lwork >= 1  if n <= 11,
//  lwork >= n  if n > 11,
// otherwise Dlaqr04 will panic. lwork as large as 6*n may be required for
// optimal performance. On return, work[0] will contain the optimal value of
// lwork.
//
// If lwork is -1, instead of performing Dlaqr04, the function only estimates the
// optimal workspace size and stores it into work[0]. Neither h nor z are
// accessed.
//
// recur is the non-negative recursion depth. For recur > 0, Dlaqr04 behaves
// as DLAQR0, for recur == 0 it behaves as DLAQR4.
//
// unconverged indicates whether Dlaqr04 computed all the eigenvalues of H[ilo:ihi+1,ilo:ihi+1].
//
// If unconverged is zero and wantt is true, H will contain on return the upper
// quasi-triangular matrix T from the Schur decomposition. 2×2 diagonal blocks
// (corresponding to complex conjugate pairs of eigenvalues) will be returned in
// standard form, with H[i,i] == H[i+1,i+1] and H[i+1,i]*H[i,i+1] < 0.
//
// If unconverged is zero and if wantt is false, the contents of h on return is
// unspecified.
//
// If unconverged is zero, all the eigenvalues have been computed and their real
// and imaginary parts will be stored on return in wr[ilo:ihi+1] and
// wi[ilo:ihi+1], respectively. If two eigenvalues are computed as a complex
// conjugate pair, they are stored in consecutive elements of wr and wi, say the
// i-th and (i+1)th, with wi[i] > 0 and wi[i+1] < 0. If wantt is true, then the
// eigenvalues are stored in the same order as on the diagonal of the Schur form
// returned in H, with wr[i] = H[i,i] and, if H[i:i+2,i:i+2] is a 2×2 diagonal
// block, wi[i] = sqrt(-H[i+1,i]*H[i,i+1]) and wi[i+1] = -wi[i].
//
// If unconverged is positive, some eigenvalues have not converged, and
// wr[unconverged:ihi+1] and wi[unconverged:ihi+1] will contain those
// eigenvalues which have been successfully computed. Failures are rare.
//
// If unconverged is positive and wantt is true, then on return
//  (initial H)*U = U*(final H),   (*)
// where U is an orthogonal matrix. The final H is upper Hessenberg and
// H[unconverged:ihi+1,unconverged:ihi+1] is upper quasi-triangular.
//
// If unconverged is positive and wantt is false, on return the remaining
// unconverged eigenvalues are the eigenvalues of the upper Hessenberg matrix
// H[ilo:unconverged,ilo:unconverged].
//
// If unconverged is positive and wantz is true, then on return
//  (final Z) = (initial Z)*U,
// where U is the orthogonal matrix in (*) regardless of the value of wantt.
//
// References:
//  [1] K. Braman, R. Byers, R. Mathias. The Multishift QR Algorithm. Part I:
//      Maintaining Well-Focused Shifts and Level 3 Performance. SIAM J. Matrix
//      Anal. Appl. 23(4) (2002), pp. 929—947
//      URL: http://dx.doi.org/10.1137/S0895479801384573
//  [2] K. Braman, R. Byers, R. Mathias. The Multishift QR Algorithm. Part II:
//      Aggressive Early Deflation. SIAM J. Matrix Anal. Appl. 23(4) (2002), pp. 948—973
//      URL: http://dx.doi.org/10.1137/S0895479801384585
//
// Dlaqr04 is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dlaqr04(wantt, wantz bool, n, ilo, ihi int, h []float64, ldh int, wr, wi []float64, iloz, ihiz int, z []float64, ldz int, work []float64, lwork int, recur int) (unconverged int) {
	const (
		// Matrices of order ntiny or smaller must be processed by
		// Dlahqr because of insufficient subdiagonal scratch space.
		// This is a hard limit.
		ntiny = 11
		// Exceptional deflation windows: try to cure rare slow
		// convergence by varying the size of the deflation window after
		// kexnw iterations.
		kexnw = 5
		// Exceptional shifts: try to cure rare slow convergence with
		// ad-hoc exceptional shifts every kexsh iterations.
		kexsh = 6

		// See https://github.com/gonum/lapack/pull/151#discussion_r68162802
		// and the surrounding discussion for an explanation where these
		// constants come from.
		// TODO(vladimir-ch): Similar constants for exceptional shifts
		// are used also in dlahqr.go. The first constant is different
		// there, it is equal to 3. Why? And does it matter?
		wilk1 = 0.75
		wilk2 = -0.4375
	)

	switch {
	case n < 0:
		panic(nLT0)
	case ilo < 0 || max(0, n-1) < ilo:
		panic(badIlo)
	case ihi < min(ilo, n-1) || n <= ihi:
		panic(badIhi)
	case ldh < max(1, n):
		panic(badLdH)
	case wantz && (iloz < 0 || ilo < iloz):
		panic(badIloz)
	case wantz && (ihiz < ihi || n <= ihiz):
		panic(badIhiz)
	case ldz < 1, wantz && ldz < n:
		panic(badLdZ)
	case lwork < 1 && lwork != -1:
		panic(badLWork)
	// TODO(vladimir-ch): Enable if and when we figure out what the minimum
	// necessary lwork value is. Dlaqr04 says that the minimum is n which
	// clashes with Dlaqr23's opinion about optimal work when nw <= 2
	// (independent of n).
	// case lwork < n && n > ntiny && lwork != -1:
	// 	panic(badLWork)
	case len(work) < max(1, lwork):
		panic(shortWork)
	case recur < 0:
		panic(recurLT0)
	}

	// Quick return.
	if n == 0 {
		work[0] = 1
		return 0
	}

	if lwork != -1 {
		switch {
		case len(h) < (n-1)*ldh+n:
			panic(shortH)
		case len(wr) != ihi+1:
			panic(badLenWr)
		case len(wi) != ihi+1:
			panic(badLenWi)
		case wantz && len(z) < (n-1)*ldz+n:
			panic(shortZ)
		case ilo > 0 && h[ilo*ldh+ilo-1] != 0:
			panic(notIsolated)
		case ihi+1 < n && h[(ihi+1)*ldh+ihi] != 0:
			panic(notIsolated)
		}
	}

	if n <= ntiny {
		// Tiny matrices must use Dlahqr.
		if lwork == -1 {
			work[0] = 1
			return 0
		}
		return impl.Dlahqr(wantt, wantz, n, ilo, ihi, h, ldh, wr, wi, iloz, ihiz, z, ldz)
	}

	// Use small bulge multi-shift QR with aggressive early deflation on
	// larger-than-tiny matrices.
	var jbcmpz string
	if wantt {
		jbcmpz = "S"
	} else {
		jbcmpz = "E"
	}
	if wantz {
		jbcmpz += "V"
	} else {
		jbcmpz += "N"
	}

	var fname string
	if recur > 0 {
		fname = "DLAQR0"
	} else {
		fname = "DLAQR4"
	}
	// nwr is the recommended deflation window size. n is greater than 11,
	// so there is enough subdiagonal workspace for nwr >= 2 as required.
	// (In fact, there is enough subdiagonal space for nwr >= 3.)
	// TODO(vladimir-ch): If there is enough space for nwr >= 3, should we
	// use it?
	nwr := impl.Ilaenv(13, fname, jbcmpz, n, ilo, ihi, lwork)
	nwr = max(2, nwr)
	nwr = min(ihi-ilo+1, min((n-1)/3, nwr))

	// nsr is the recommended number of simultaneous shifts. n is greater
	// than 11, so there is enough subdiagonal workspace for nsr to be even
	// and greater than or equal to two as required.
	nsr := impl.Ilaenv(15, fname, jbcmpz, n, ilo, ihi, lwork)
	nsr = min(nsr, min((n+6)/9, ihi-ilo))
	nsr = max(2, nsr&^1)

	// Workspace query call to Dlaqr23.
	impl.Dlaqr23(wantt, wantz, n, ilo, ihi, nwr+1, h, ldh, iloz, ihiz, z, ldz,
		wr, wi, h, ldh, n, h, ldh, n, h, ldh, work, -1, recur)
	// Optimal workspace is max(Dlaqr5, Dlaqr23).
	lwkopt := max(3*nsr/2, int(work[0]))
	// Quick return in case of workspace query.
	if lwork == -1 {
		work[0] = float64(lwkopt)
		return 0
	}

	// Dlahqr/Dlaqr04 crossover point.
	nmin := impl.Ilaenv(12, fname, jbcmpz, n, ilo, ihi, lwork)
	nmin = max(ntiny, nmin)

	// Nibble determines when to skip a multi-shift QR sweep (Dlaqr5).
	nibble := impl.Ilaenv(14, fname, jbcmpz, n, ilo, ihi, lwork)
	nibble = max(0, nibble)

	// Computation mode of far-from-diagonal orthogonal updates in Dlaqr5.
	kacc22 := impl.Ilaenv(16, fname, jbcmpz, n, ilo, ihi, lwork)
	kacc22 = max(0, min(kacc22, 2))

	// nwmax is the largest possible deflation window for which there is
	// sufficient workspace.
	nwmax := min((n-1)/3, lwork/2)
	nw := nwmax // Start with maximum deflation window size.

	// nsmax is the largest number of simultaneous shifts for which there is
	// sufficient workspace.
	nsmax := min((n+6)/9, 2*lwork/3) &^ 1

	ndfl := 1 // Number of iterations since last deflation.
	ndec := 0 // Deflation window size decrement.

	// Main loop.
	var (
		itmax = max(30, 2*kexsh) * max(10, (ihi-ilo+1))
		it    = 0
	)
	for kbot := ihi; kbot >= ilo; {
		if it == itmax {
			unconverged = kbot + 1
			break
		}
		it++

		// Locate active block.
		ktop := ilo
		for k := kbot; k >= ilo+1; k-- {
			if h[k*ldh+k-1] == 0 {
				ktop = k
				break
			}
		}

		// Select deflation window size nw.
		//
		// Typical Case:
		//  If possible and advisable, nibble the entire active block.
		//  If not, use size min(nwr,nwmax) or min(nwr+1,nwmax)
		//  depending upon which has the smaller corresponding
		//  subdiagonal entry (a heuristic).
		//
		// Exceptional Case:
		//  If there have been no deflations in kexnw or more
		//  iterations, then vary the deflation window size. At first,
		//  because larger windows are, in general, more powerful than
		//  smaller ones, rapidly increase the window to the maximum
		//  possible. Then, gradually reduce the window size.
		nh := kbot - ktop + 1
		nwupbd := min(nh, nwmax)
		if ndfl < kexnw {
			nw = min(nwupbd, nwr)
		} else {
			nw = min(nwupbd, 2*nw)
		}
		if nw < nwmax {
			if nw >= nh-1 {
				nw = nh
			} else {
				kwtop := kbot - nw + 1
				if math.Abs(h[kwtop*ldh+kwtop-1]) > math.Abs(h[(kwtop-1)*ldh+kwtop-2]) {
					nw++
				}
			}
		}
		if ndfl < kexnw {
			ndec = -1
		} else if ndec >= 0 || nw >= nwupbd {
			ndec++
			if nw-ndec < 2 {
				ndec = 0
			}
			nw -= ndec
		}

		// Split workspace under the subdiagonal of H into:
		//  - an nw×nw work array V in the lower left-hand corner,
		//  - an nw×nhv horizontal work array along the bottom edge (nhv
		//    must be at least nw but more is better),
		//  - an nve×nw vertical work array along the left-hand-edge
		//    (nhv can be any positive integer but more is better).
		kv := n - nw
		kt := nw
		kwv := nw + 1
		nhv := n - kwv - kt
		// Aggressive early deflation.
		ls, ld := impl.Dlaqr23(wantt, wantz, n, ktop, kbot, nw,
			h, ldh, iloz, ihiz, z, ldz, wr[:kbot+1], wi[:kbot+1],
			h[kv*ldh:], ldh, nhv, h[kv*ldh+kt:], ldh, nhv, h[kwv*ldh:], ldh, work, lwork, recur)

		// Adjust kbot accounting for new deflations.
		kbot -= ld
		// ks points to the shifts.
		ks := kbot - ls + 1

		// Skip an expensive QR sweep if there is a (partly heuristic)
		// reason to expect that many eigenvalues will deflate without
		// it. Here, the QR sweep is skipped if many eigenvalues have
		// just been deflated or if the remaining active block is small.
		if ld > 0 && (100*ld > nw*nibble || kbot-ktop+1 <= min(nmin, nwmax)) {
			// ld is positive, note progress.
			ndfl = 1
			continue
		}

		// ns is the nominal number of simultaneous shifts. This may be
		// lowered (slightly) if Dlaqr23 did not provide that many
		// shifts.
		ns := min(min(nsmax, nsr), max(2, kbot-ktop)) &^ 1

		// If there have been no deflations in a multiple of kexsh
		// iterations, then try exceptional shifts. Otherwise use shifts
		// provided by Dlaqr23 above or from the eigenvalues of a
		// trailing principal submatrix.
		if ndfl%kexsh == 0 {
			ks = kbot - ns + 1
			for i := kbot; i > max(ks, ktop+1); i -= 2 {
				ss := math.Abs(h[i*ldh+i-1]) + math.Abs(h[(i-1)*ldh+i-2])
				aa := wilk1*ss + h[i*ldh+i]
				_, _, _, _, wr[i-1], wi[i-1], wr[i], wi[i], _, _ =
					impl.Dlanv2(aa, ss, wilk2*ss, aa)
			}
			if ks == ktop {
				wr[ks+1] = h[(ks+1)*ldh+ks+1]
				wi[ks+1] = 0
				wr[ks] = wr[ks+1]
				wi[ks] = wi[ks+1]
			}
		} else {
			// If we got ns/2 or fewer shifts, use Dlahqr or recur
			// into Dlaqr04 on a trailing principal submatrix to get
			// more. Since ns <= nsmax <=(n+6)/9, there is enough
			// space below the subdiagonal to fit an ns×ns scratch
			// array.
			if kbot-ks+1 <= ns/2 {
				ks = kbot - ns + 1
				kt = n - ns
				impl.Dlacpy(blas.All, ns, ns, h[ks*ldh+ks:], ldh, h[kt*ldh:], ldh)
				if ns > nmin && recur > 0 {
					ks += impl.Dlaqr04(false, false, ns, 1, ns-1, h[kt*ldh:], ldh,
						wr[ks:ks+ns], wi[ks:ks+ns], 0, 0, nil, 0, work, lwork, recur-1)
				} else {
					ks += impl.Dlahqr(false, false, ns, 0, ns-1, h[kt*ldh:], ldh,
						wr[ks:ks+ns], wi[ks:ks+ns], 0, 0, nil, 1)
				}
				// In case of a rare QR failure use eigenvalues
				// of the trailing 2×2 principal submatrix.
				if ks >= kbot {
					aa := h[(kbot-1)*ldh+kbot-1]
					bb := h[(kbot-1)*ldh+kbot]
					cc := h[kbot*ldh+kbot-1]
					dd := h[kbot*ldh+kbot]
					_, _, _, _, wr[kbot-1], wi[kbot-1], wr[kbot], wi[kbot], _, _ =
						impl.Dlanv2(aa, bb, cc, dd)
					ks = kbot - 1
				}
			}

			if kbot-ks+1 > ns {
				// Sorting the shifts helps a little. Bubble
				// sort keeps complex conjugate pairs together.
				sorted := false
				for k := kbot; k > ks; k-- {
					if sorted {
						break
					}
					sorted = true
					for i := ks; i < k; i++ {
						if math.Abs(wr[i])+math.Abs(wi[i]) >= math.Abs(wr[i+1])+math.Abs(wi[i+1]) {
							continue
						}
						sorted = false
						wr[i], wr[i+1] = wr[i+1], wr[i]
						wi[i], wi[i+1] = wi[i+1], wi[i]
					}
				}
			}

			// Shuffle shifts into pairs of real shifts and pairs of
			// complex conjugate shifts using the fact that complex
			// conjugate shifts are already adjacent to one another.
			// TODO(vladimir-ch): The shuffling here could probably
			// be removed but I'm not sure right now and it's safer
			// to leave it.
			for i := kbot; i > ks+1; i -= 2 {
				if wi[i] == -wi[i-1] {
					continue
				}
				wr[i], wr[i-1], wr[i-2] = wr[i-1], wr[i-2], wr[i]
				wi[i], wi[i-1], wi[i-2] = wi[i-1], wi[i-2], wi[i]
			}
		}

		// If there are only two shifts and both are real, then use only one.
		if kbot-ks+1 == 2 && wi[kbot] == 0 {
			if math.Abs(wr[kbot]-h[kbot*ldh+kbot]) < math.Abs(wr[kbot-1]-h[kbot*ldh+kbot]) {
				wr[kbot-1] = wr[kbot]
			} else {
				wr[kbot] = wr[kbot-1]
			}
		}

		// Use up to ns of the smallest magnitude shifts. If there
		// aren't ns shifts available, then use them all, possibly
		// dropping one to make the number of shifts even.
		ns = min(ns, kbot-ks+1) &^ 1
		ks = kbot - ns + 1

		// Split workspace under the subdiagonal into:
		// - a kdu×kdu work array U in the lower left-hand-corner,
		// - a kdu×nhv horizontal work array WH along the bottom edge
		//   (nhv must be at least kdu but more is better),
		// - an nhv×kdu vertical work array WV along the left-hand-edge
		//   (nhv must be at least kdu but more is better).
		kdu := 3*ns - 3
		ku := n - kdu
		kwh := kdu
		kwv = kdu + 3
		nhv = n - kwv - kdu
		// Small-bulge multi-shift QR sweep.
		impl.Dlaqr5(wantt, wantz, kacc22, n, ktop, kbot, ns,
			wr[ks:ks+ns], wi[ks:ks+ns], h, ldh, iloz, ihiz, z, ldz,
			work, 3, h[ku*ldh:], ldh, nhv, h[kwv*ldh:], ldh, nhv, h[ku*ldh+kwh:], ldh)

		// Note progress (or the lack of it).
		if ld > 0 {
			ndfl = 1
		} else {
			ndfl++
		}
	}

	work[0] = float64(lwkopt)
	return unconverged
}
