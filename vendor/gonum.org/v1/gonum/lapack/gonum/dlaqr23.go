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

// Dlaqr23 performs the orthogonal similarity transformation of an n×n upper
// Hessenberg matrix to detect and deflate fully converged eigenvalues from a
// trailing principal submatrix using aggressive early deflation [1].
//
// On return, H will be overwritten by a new Hessenberg matrix that is a
// perturbation of an orthogonal similarity transformation of H. It is hoped
// that on output H will have many zero subdiagonal entries.
//
// If wantt is true, the matrix H will be fully updated so that the
// quasi-triangular Schur factor can be computed. If wantt is false, then only
// enough of H will be updated to preserve the eigenvalues.
//
// If wantz is true, the orthogonal similarity transformation will be
// accumulated into Z[iloz:ihiz+1,ktop:kbot+1], otherwise Z is not referenced.
//
// ktop and kbot determine a block [ktop:kbot+1,ktop:kbot+1] along the diagonal
// of H. It must hold that
//  0 <= ilo <= ihi < n,     if n > 0,
//  ilo == 0 and ihi == -1,  if n == 0,
// and the block must be isolated, that is, it must hold that
//  ktop == 0   or H[ktop,ktop-1] == 0,
//  kbot == n-1 or H[kbot+1,kbot] == 0,
// otherwise Dlaqr23 will panic.
//
// nw is the deflation window size. It must hold that
//  0 <= nw <= kbot-ktop+1,
// otherwise Dlaqr23 will panic.
//
// iloz and ihiz specify the rows of the n×n matrix Z to which transformations
// will be applied if wantz is true. It must hold that
//  0 <= iloz <= ktop,  and  kbot <= ihiz < n,
// otherwise Dlaqr23 will panic.
//
// sr and si must have length kbot+1, otherwise Dlaqr23 will panic.
//
// v and ldv represent an nw×nw work matrix.
// t and ldt represent an nw×nh work matrix, and nh must be at least nw.
// wv and ldwv represent an nv×nw work matrix.
//
// work must have length at least lwork and lwork must be at least max(1,2*nw),
// otherwise Dlaqr23 will panic. Larger values of lwork may result in greater
// efficiency. On return, work[0] will contain the optimal value of lwork.
//
// If lwork is -1, instead of performing Dlaqr23, the function only estimates the
// optimal workspace size and stores it into work[0]. Neither h nor z are
// accessed.
//
// recur is the non-negative recursion depth. For recur > 0, Dlaqr23 behaves
// as DLAQR3, for recur == 0 it behaves as DLAQR2.
//
// On return, ns and nd will contain respectively the number of unconverged
// (i.e., approximate) eigenvalues and converged eigenvalues that are stored in
// sr and si.
//
// On return, the real and imaginary parts of approximate eigenvalues that may
// be used for shifts will be stored respectively in sr[kbot-nd-ns+1:kbot-nd+1]
// and si[kbot-nd-ns+1:kbot-nd+1].
//
// On return, the real and imaginary parts of converged eigenvalues will be
// stored respectively in sr[kbot-nd+1:kbot+1] and si[kbot-nd+1:kbot+1].
//
// References:
//  [1] K. Braman, R. Byers, R. Mathias. The Multishift QR Algorithm. Part II:
//      Aggressive Early Deflation. SIAM J. Matrix Anal. Appl 23(4) (2002), pp. 948—973
//      URL: http://dx.doi.org/10.1137/S0895479801384585
//
func (impl Implementation) Dlaqr23(wantt, wantz bool, n, ktop, kbot, nw int, h []float64, ldh int, iloz, ihiz int, z []float64, ldz int, sr, si []float64, v []float64, ldv int, nh int, t []float64, ldt int, nv int, wv []float64, ldwv int, work []float64, lwork int, recur int) (ns, nd int) {
	switch {
	case n < 0:
		panic(nLT0)
	case ktop < 0 || max(0, n-1) < ktop:
		panic(badKtop)
	case kbot < min(ktop, n-1) || n <= kbot:
		panic(badKbot)
	case nw < 0 || kbot-ktop+1+1 < nw:
		panic(badNw)
	case ldh < max(1, n):
		panic(badLdH)
	case wantz && (iloz < 0 || ktop < iloz):
		panic(badIloz)
	case wantz && (ihiz < kbot || n <= ihiz):
		panic(badIhiz)
	case ldz < 1, wantz && ldz < n:
		panic(badLdZ)
	case ldv < max(1, nw):
		panic(badLdV)
	case nh < nw:
		panic(badNh)
	case ldt < max(1, nh):
		panic(badLdT)
	case nv < 0:
		panic(nvLT0)
	case ldwv < max(1, nw):
		panic(badLdWV)
	case lwork < max(1, 2*nw) && lwork != -1:
		panic(badLWork)
	case len(work) < max(1, lwork):
		panic(shortWork)
	case recur < 0:
		panic(recurLT0)
	}

	// Quick return for zero window size.
	if nw == 0 {
		work[0] = 1
		return 0, 0
	}

	// LAPACK code does not enforce the documented behavior
	//  nw <= kbot-ktop+1
	// but we do (we panic above).
	jw := nw
	lwkopt := max(1, 2*nw)
	if jw > 2 {
		// Workspace query call to Dgehrd.
		impl.Dgehrd(jw, 0, jw-2, t, ldt, work, work, -1)
		lwk1 := int(work[0])
		// Workspace query call to Dormhr.
		impl.Dormhr(blas.Right, blas.NoTrans, jw, jw, 0, jw-2, t, ldt, work, v, ldv, work, -1)
		lwk2 := int(work[0])
		if recur > 0 {
			// Workspace query call to Dlaqr04.
			impl.Dlaqr04(true, true, jw, 0, jw-1, t, ldt, sr, si, 0, jw-1, v, ldv, work, -1, recur-1)
			lwk3 := int(work[0])
			// Optimal workspace.
			lwkopt = max(jw+max(lwk1, lwk2), lwk3)
		} else {
			// Optimal workspace.
			lwkopt = jw + max(lwk1, lwk2)
		}
	}
	// Quick return in case of workspace query.
	if lwork == -1 {
		work[0] = float64(lwkopt)
		return 0, 0
	}

	// Check input slices only if not doing workspace query.
	switch {
	case len(h) < (n-1)*ldh+n:
		panic(shortH)
	case len(v) < (nw-1)*ldv+nw:
		panic(shortV)
	case len(t) < (nw-1)*ldt+nh:
		panic(shortT)
	case len(wv) < (nv-1)*ldwv+nw:
		panic(shortWV)
	case wantz && len(z) < (n-1)*ldz+n:
		panic(shortZ)
	case len(sr) != kbot+1:
		panic(badLenSr)
	case len(si) != kbot+1:
		panic(badLenSi)
	case ktop > 0 && h[ktop*ldh+ktop-1] != 0:
		panic(notIsolated)
	case kbot+1 < n && h[(kbot+1)*ldh+kbot] != 0:
		panic(notIsolated)
	}

	// Machine constants.
	ulp := dlamchP
	smlnum := float64(n) / ulp * dlamchS

	// Setup deflation window.
	var s float64
	kwtop := kbot - jw + 1
	if kwtop != ktop {
		s = h[kwtop*ldh+kwtop-1]
	}
	if kwtop == kbot {
		// 1×1 deflation window.
		sr[kwtop] = h[kwtop*ldh+kwtop]
		si[kwtop] = 0
		ns = 1
		nd = 0
		if math.Abs(s) <= math.Max(smlnum, ulp*math.Abs(h[kwtop*ldh+kwtop])) {
			ns = 0
			nd = 1
			if kwtop > ktop {
				h[kwtop*ldh+kwtop-1] = 0
			}
		}
		work[0] = 1
		return ns, nd
	}

	// Convert to spike-triangular form. In case of a rare QR failure, this
	// routine continues to do aggressive early deflation using that part of
	// the deflation window that converged using infqr here and there to
	// keep track.
	impl.Dlacpy(blas.Upper, jw, jw, h[kwtop*ldh+kwtop:], ldh, t, ldt)
	bi := blas64.Implementation()
	bi.Dcopy(jw-1, h[(kwtop+1)*ldh+kwtop:], ldh+1, t[ldt:], ldt+1)
	impl.Dlaset(blas.All, jw, jw, 0, 1, v, ldv)
	nmin := impl.Ilaenv(12, "DLAQR3", "SV", jw, 0, jw-1, lwork)
	var infqr int
	if recur > 0 && jw > nmin {
		infqr = impl.Dlaqr04(true, true, jw, 0, jw-1, t, ldt, sr[kwtop:], si[kwtop:], 0, jw-1, v, ldv, work, lwork, recur-1)
	} else {
		infqr = impl.Dlahqr(true, true, jw, 0, jw-1, t, ldt, sr[kwtop:], si[kwtop:], 0, jw-1, v, ldv)
	}
	// Note that ilo == 0 which conveniently coincides with the success
	// value of infqr, that is, infqr as an index always points to the first
	// converged eigenvalue.

	// Dtrexc needs a clean margin near the diagonal.
	for j := 0; j < jw-3; j++ {
		t[(j+2)*ldt+j] = 0
		t[(j+3)*ldt+j] = 0
	}
	if jw >= 3 {
		t[(jw-1)*ldt+jw-3] = 0
	}

	ns = jw
	ilst := infqr
	// Deflation detection loop.
	for ilst < ns {
		bulge := false
		if ns >= 2 {
			bulge = t[(ns-1)*ldt+ns-2] != 0
		}
		if !bulge {
			// Real eigenvalue.
			abst := math.Abs(t[(ns-1)*ldt+ns-1])
			if abst == 0 {
				abst = math.Abs(s)
			}
			if math.Abs(s*v[ns-1]) <= math.Max(smlnum, ulp*abst) {
				// Deflatable.
				ns--
			} else {
				// Undeflatable, move it up out of the way.
				// Dtrexc can not fail in this case.
				_, ilst, _ = impl.Dtrexc(lapack.UpdateSchur, jw, t, ldt, v, ldv, ns-1, ilst, work)
				ilst++
			}
			continue
		}
		// Complex conjugate pair.
		abst := math.Abs(t[(ns-1)*ldt+ns-1]) + math.Sqrt(math.Abs(t[(ns-1)*ldt+ns-2]))*math.Sqrt(math.Abs(t[(ns-2)*ldt+ns-1]))
		if abst == 0 {
			abst = math.Abs(s)
		}
		if math.Max(math.Abs(s*v[ns-1]), math.Abs(s*v[ns-2])) <= math.Max(smlnum, ulp*abst) {
			// Deflatable.
			ns -= 2
		} else {
			// Undeflatable, move them up out of the way.
			// Dtrexc does the right thing with ilst in case of a
			// rare exchange failure.
			_, ilst, _ = impl.Dtrexc(lapack.UpdateSchur, jw, t, ldt, v, ldv, ns-1, ilst, work)
			ilst += 2
		}
	}

	// Return to Hessenberg form.
	if ns == 0 {
		s = 0
	}
	if ns < jw {
		// Sorting diagonal blocks of T improves accuracy for graded
		// matrices. Bubble sort deals well with exchange failures.
		sorted := false
		i := ns
		for !sorted {
			sorted = true
			kend := i - 1
			i = infqr
			var k int
			if i == ns-1 || t[(i+1)*ldt+i] == 0 {
				k = i + 1
			} else {
				k = i + 2
			}
			for k <= kend {
				var evi float64
				if k == i+1 {
					evi = math.Abs(t[i*ldt+i])
				} else {
					evi = math.Abs(t[i*ldt+i]) + math.Sqrt(math.Abs(t[(i+1)*ldt+i]))*math.Sqrt(math.Abs(t[i*ldt+i+1]))
				}

				var evk float64
				if k == kend || t[(k+1)*ldt+k] == 0 {
					evk = math.Abs(t[k*ldt+k])
				} else {
					evk = math.Abs(t[k*ldt+k]) + math.Sqrt(math.Abs(t[(k+1)*ldt+k]))*math.Sqrt(math.Abs(t[k*ldt+k+1]))
				}

				if evi >= evk {
					i = k
				} else {
					sorted = false
					_, ilst, ok := impl.Dtrexc(lapack.UpdateSchur, jw, t, ldt, v, ldv, i, k, work)
					if ok {
						i = ilst
					} else {
						i = k
					}
				}
				if i == kend || t[(i+1)*ldt+i] == 0 {
					k = i + 1
				} else {
					k = i + 2
				}
			}
		}
	}

	// Restore shift/eigenvalue array from T.
	for i := jw - 1; i >= infqr; {
		if i == infqr || t[i*ldt+i-1] == 0 {
			sr[kwtop+i] = t[i*ldt+i]
			si[kwtop+i] = 0
			i--
			continue
		}
		aa := t[(i-1)*ldt+i-1]
		bb := t[(i-1)*ldt+i]
		cc := t[i*ldt+i-1]
		dd := t[i*ldt+i]
		_, _, _, _, sr[kwtop+i-1], si[kwtop+i-1], sr[kwtop+i], si[kwtop+i], _, _ = impl.Dlanv2(aa, bb, cc, dd)
		i -= 2
	}

	if ns < jw || s == 0 {
		if ns > 1 && s != 0 {
			// Reflect spike back into lower triangle.
			bi.Dcopy(ns, v[:ns], 1, work[:ns], 1)
			_, tau := impl.Dlarfg(ns, work[0], work[1:ns], 1)
			work[0] = 1
			impl.Dlaset(blas.Lower, jw-2, jw-2, 0, 0, t[2*ldt:], ldt)
			impl.Dlarf(blas.Left, ns, jw, work[:ns], 1, tau, t, ldt, work[jw:])
			impl.Dlarf(blas.Right, ns, ns, work[:ns], 1, tau, t, ldt, work[jw:])
			impl.Dlarf(blas.Right, jw, ns, work[:ns], 1, tau, v, ldv, work[jw:])
			impl.Dgehrd(jw, 0, ns-1, t, ldt, work[:jw-1], work[jw:], lwork-jw)
		}

		// Copy updated reduced window into place.
		if kwtop > 0 {
			h[kwtop*ldh+kwtop-1] = s * v[0]
		}
		impl.Dlacpy(blas.Upper, jw, jw, t, ldt, h[kwtop*ldh+kwtop:], ldh)
		bi.Dcopy(jw-1, t[ldt:], ldt+1, h[(kwtop+1)*ldh+kwtop:], ldh+1)

		// Accumulate orthogonal matrix in order to update H and Z, if
		// requested.
		if ns > 1 && s != 0 {
			// work[:ns-1] contains the elementary reflectors stored
			// by a call to Dgehrd above.
			impl.Dormhr(blas.Right, blas.NoTrans, jw, ns, 0, ns-1,
				t, ldt, work[:ns-1], v, ldv, work[jw:], lwork-jw)
		}

		// Update vertical slab in H.
		var ltop int
		if !wantt {
			ltop = ktop
		}
		for krow := ltop; krow < kwtop; krow += nv {
			kln := min(nv, kwtop-krow)
			bi.Dgemm(blas.NoTrans, blas.NoTrans, kln, jw, jw,
				1, h[krow*ldh+kwtop:], ldh, v, ldv,
				0, wv, ldwv)
			impl.Dlacpy(blas.All, kln, jw, wv, ldwv, h[krow*ldh+kwtop:], ldh)
		}

		// Update horizontal slab in H.
		if wantt {
			for kcol := kbot + 1; kcol < n; kcol += nh {
				kln := min(nh, n-kcol)
				bi.Dgemm(blas.Trans, blas.NoTrans, jw, kln, jw,
					1, v, ldv, h[kwtop*ldh+kcol:], ldh,
					0, t, ldt)
				impl.Dlacpy(blas.All, jw, kln, t, ldt, h[kwtop*ldh+kcol:], ldh)
			}
		}

		// Update vertical slab in Z.
		if wantz {
			for krow := iloz; krow <= ihiz; krow += nv {
				kln := min(nv, ihiz-krow+1)
				bi.Dgemm(blas.NoTrans, blas.NoTrans, kln, jw, jw,
					1, z[krow*ldz+kwtop:], ldz, v, ldv,
					0, wv, ldwv)
				impl.Dlacpy(blas.All, kln, jw, wv, ldwv, z[krow*ldz+kwtop:], ldz)
			}
		}
	}

	// The number of deflations.
	nd = jw - ns
	// Shifts are converged eigenvalues that could not be deflated.
	// Subtracting infqr from the spike length takes care of the case of a
	// rare QR failure while calculating eigenvalues of the deflation
	// window.
	ns -= infqr
	work[0] = float64(lwkopt)
	return ns, nd
}
