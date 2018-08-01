// Copyright ©2016 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"math"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
)

// Dlaqr5 performs a single small-bulge multi-shift QR sweep on an isolated
// block of a Hessenberg matrix.
//
// wantt and wantz determine whether the quasi-triangular Schur factor and the
// orthogonal Schur factor, respectively, will be computed.
//
// kacc22 specifies the computation mode of far-from-diagonal orthogonal
// updates. Permitted values are:
//  0: Dlaqr5 will not accumulate reflections and will not use matrix-matrix
//     multiply to update far-from-diagonal matrix entries.
//  1: Dlaqr5 will accumulate reflections and use matrix-matrix multiply to
//     update far-from-diagonal matrix entries.
//  2: Dlaqr5 will accumulate reflections, use matrix-matrix multiply to update
//     far-from-diagonal matrix entries, and take advantage of 2×2 block
//     structure during matrix multiplies.
// For other values of kacc2 Dlaqr5 will panic.
//
// n is the order of the Hessenberg matrix H.
//
// ktop and kbot are indices of the first and last row and column of an isolated
// diagonal block upon which the QR sweep will be applied. It must hold that
//  ktop == 0,   or 0 < ktop <= n-1 and H[ktop, ktop-1] == 0, and
//  kbot == n-1, or 0 <= kbot < n-1 and H[kbot+1, kbot] == 0,
// otherwise Dlaqr5 will panic.
//
// nshfts is the number of simultaneous shifts. It must be positive and even,
// otherwise Dlaqr5 will panic.
//
// sr and si contain the real and imaginary parts, respectively, of the shifts
// of origin that define the multi-shift QR sweep. On return both slices may be
// reordered by Dlaqr5. Their length must be equal to nshfts, otherwise Dlaqr5
// will panic.
//
// h and ldh represent the Hessenberg matrix H of size n×n. On return
// multi-shift QR sweep with shifts sr+i*si has been applied to the isolated
// diagonal block in rows and columns ktop through kbot, inclusive.
//
// iloz and ihiz specify the rows of Z to which transformations will be applied
// if wantz is true. It must hold that 0 <= iloz <= ihiz < n, otherwise Dlaqr5
// will panic.
//
// z and ldz represent the matrix Z of size n×n. If wantz is true, the QR sweep
// orthogonal similarity transformation is accumulated into
// z[iloz:ihiz,iloz:ihiz] from the right, otherwise z not referenced.
//
// v and ldv represent an auxiliary matrix V of size (nshfts/2)×3. Note that V
// is transposed with respect to the reference netlib implementation.
//
// u and ldu represent an auxiliary matrix of size (3*nshfts-3)×(3*nshfts-3).
//
// wh and ldwh represent an auxiliary matrix of size (3*nshfts-3)×nh.
//
// wv and ldwv represent an auxiliary matrix of size nv×(3*nshfts-3).
//
// Dlaqr5 is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dlaqr5(wantt, wantz bool, kacc22 int, n, ktop, kbot, nshfts int, sr, si []float64, h []float64, ldh int, iloz, ihiz int, z []float64, ldz int, v []float64, ldv int, u []float64, ldu int, nv int, wv []float64, ldwv int, nh int, wh []float64, ldwh int) {
	checkMatrix(n, n, h, ldh)
	if ktop < 0 || n <= ktop {
		panic("lapack: invalid value of ktop")
	}
	if ktop > 0 && h[ktop*ldh+ktop-1] != 0 {
		panic("lapack: diagonal block is not isolated")
	}
	if kbot < 0 || n <= kbot {
		panic("lapack: invalid value of kbot")
	}
	if kbot < n-1 && h[(kbot+1)*ldh+kbot] != 0 {
		panic("lapack: diagonal block is not isolated")
	}
	if nshfts < 0 || nshfts&0x1 != 0 {
		panic("lapack: invalid number of shifts")
	}
	if len(sr) != nshfts || len(si) != nshfts {
		panic(badSlice) // TODO(vladimir-ch) Another message?
	}
	if wantz {
		if ihiz >= n {
			panic("lapack: invalid value of ihiz")
		}
		if iloz < 0 || ihiz < iloz {
			panic("lapack: invalid value of iloz")
		}
		checkMatrix(n, n, z, ldz)
	}
	checkMatrix(nshfts/2, 3, v, ldv) // Transposed w.r.t. lapack.
	checkMatrix(3*nshfts-3, 3*nshfts-3, u, ldu)
	checkMatrix(nv, 3*nshfts-3, wv, ldwv)
	checkMatrix(3*nshfts-3, nh, wh, ldwh)
	if kacc22 != 0 && kacc22 != 1 && kacc22 != 2 {
		panic("lapack: invalid value of kacc22")
	}

	// If there are no shifts, then there is nothing to do.
	if nshfts < 2 {
		return
	}
	// If the active block is empty or 1×1, then there is nothing to do.
	if ktop >= kbot {
		return
	}

	// Shuffle shifts into pairs of real shifts and pairs of complex
	// conjugate shifts assuming complex conjugate shifts are already
	// adjacent to one another.
	for i := 0; i < nshfts-2; i += 2 {
		if si[i] == -si[i+1] {
			continue
		}
		sr[i], sr[i+1], sr[i+2] = sr[i+1], sr[i+2], sr[i]
		si[i], si[i+1], si[i+2] = si[i+1], si[i+2], si[i]
	}

	// Note: lapack says that nshfts must be even but allows it to be odd
	// anyway. We panic above if nshfts is not even, so reducing it by one
	// is unnecessary. The only caller Dlaqr04 uses only even nshfts.
	//
	// The original comment and code from lapack-3.6.0/SRC/dlaqr5.f:341:
	// *     ==== NSHFTS is supposed to be even, but if it is odd,
	// *     .    then simply reduce it by one.  The shuffle above
	// *     .    ensures that the dropped shift is real and that
	// *     .    the remaining shifts are paired. ====
	// *
	//      NS = NSHFTS - MOD( NSHFTS, 2 )
	ns := nshfts

	safmin := dlamchS
	ulp := dlamchP
	smlnum := safmin * float64(n) / ulp

	// Use accumulated reflections to update far-from-diagonal entries?
	accum := kacc22 == 1 || kacc22 == 2
	// If so, exploit the 2×2 block structure?
	blk22 := ns > 2 && kacc22 == 2

	// Clear trash.
	if ktop+2 <= kbot {
		h[(ktop+2)*ldh+ktop] = 0
	}

	// nbmps = number of 2-shift bulges in the chain.
	nbmps := ns / 2

	// kdu = width of slab.
	kdu := 6*nbmps - 3

	// Create and chase chains of nbmps bulges.
	for incol := 3*(1-nbmps) + ktop - 1; incol <= kbot-2; incol += 3*nbmps - 2 {
		ndcol := incol + kdu
		if accum {
			impl.Dlaset(blas.All, kdu, kdu, 0, 1, u, ldu)
		}

		// Near-the-diagonal bulge chase. The following loop performs
		// the near-the-diagonal part of a small bulge multi-shift QR
		// sweep. Each 6*nbmps-2 column diagonal chunk extends from
		// column incol to column ndcol (including both column incol and
		// column ndcol). The following loop chases a 3*nbmps column
		// long chain of nbmps bulges 3*nbmps-2 columns to the right.
		// (incol may be less than ktop and ndcol may be greater than
		// kbot indicating phantom columns from which to chase bulges
		// before they are actually introduced or to which to chase
		// bulges beyond column kbot.)
		for krcol := incol; krcol <= min(incol+3*nbmps-3, kbot-2); krcol++ {
			// Bulges number mtop to mbot are active double implicit
			// shift bulges. There may or may not also be small 2×2
			// bulge, if there is room. The inactive bulges (if any)
			// must wait until the active bulges have moved down the
			// diagonal to make room. The phantom matrix paradigm
			// described above helps keep track.

			mtop := max(0, ((ktop-1)-krcol+2)/3)
			mbot := min(nbmps, (kbot-krcol)/3) - 1
			m22 := mbot + 1
			bmp22 := (mbot < nbmps-1) && (krcol+3*m22 == kbot-2)

			// Generate reflections to chase the chain right one
			// column. (The minimum value of k is ktop-1.)
			for m := mtop; m <= mbot; m++ {
				k := krcol + 3*m
				if k == ktop-1 {
					impl.Dlaqr1(3, h[ktop*ldh+ktop:], ldh,
						sr[2*m], si[2*m], sr[2*m+1], si[2*m+1],
						v[m*ldv:m*ldv+3])
					alpha := v[m*ldv]
					_, v[m*ldv] = impl.Dlarfg(3, alpha, v[m*ldv+1:m*ldv+3], 1)
					continue
				}
				beta := h[(k+1)*ldh+k]
				v[m*ldv+1] = h[(k+2)*ldh+k]
				v[m*ldv+2] = h[(k+3)*ldh+k]
				beta, v[m*ldv] = impl.Dlarfg(3, beta, v[m*ldv+1:m*ldv+3], 1)

				// A bulge may collapse because of vigilant deflation or
				// destructive underflow. In the underflow case, try the
				// two-small-subdiagonals trick to try to reinflate the
				// bulge.
				if h[(k+3)*ldh+k] != 0 || h[(k+3)*ldh+k+1] != 0 || h[(k+3)*ldh+k+2] == 0 {
					// Typical case: not collapsed (yet).
					h[(k+1)*ldh+k] = beta
					h[(k+2)*ldh+k] = 0
					h[(k+3)*ldh+k] = 0
					continue
				}

				// Atypical case: collapsed. Attempt to reintroduce
				// ignoring H[k+1,k] and H[k+2,k]. If the fill
				// resulting from the new reflector is too large,
				// then abandon it. Otherwise, use the new one.
				var vt [3]float64
				impl.Dlaqr1(3, h[(k+1)*ldh+k+1:], ldh, sr[2*m],
					si[2*m], sr[2*m+1], si[2*m+1], vt[:])
				alpha := vt[0]
				_, vt[0] = impl.Dlarfg(3, alpha, vt[1:3], 1)
				refsum := vt[0] * (h[(k+1)*ldh+k] + vt[1]*h[(k+2)*ldh+k])

				dsum := math.Abs(h[k*ldh+k]) + math.Abs(h[(k+1)*ldh+k+1]) + math.Abs(h[(k+2)*ldh+k+2])
				if math.Abs(h[(k+2)*ldh+k]-refsum*vt[1])+math.Abs(refsum*vt[2]) > ulp*dsum {
					// Starting a new bulge here would create
					// non-negligible fill. Use the old one with
					// trepidation.
					h[(k+1)*ldh+k] = beta
					h[(k+2)*ldh+k] = 0
					h[(k+3)*ldh+k] = 0
					continue
				} else {
					// Starting a new bulge here would create
					// only negligible fill. Replace the old
					// reflector with the new one.
					h[(k+1)*ldh+k] -= refsum
					h[(k+2)*ldh+k] = 0
					h[(k+3)*ldh+k] = 0
					v[m*ldv] = vt[0]
					v[m*ldv+1] = vt[1]
					v[m*ldv+2] = vt[2]
				}
			}

			// Generate a 2×2 reflection, if needed.
			if bmp22 {
				k := krcol + 3*m22
				if k == ktop-1 {
					impl.Dlaqr1(2, h[(k+1)*ldh+k+1:], ldh,
						sr[2*m22], si[2*m22], sr[2*m22+1], si[2*m22+1],
						v[m22*ldv:m22*ldv+2])
					beta := v[m22*ldv]
					_, v[m22*ldv] = impl.Dlarfg(2, beta, v[m22*ldv+1:m22*ldv+2], 1)
				} else {
					beta := h[(k+1)*ldh+k]
					v[m22*ldv+1] = h[(k+2)*ldh+k]
					beta, v[m22*ldv] = impl.Dlarfg(2, beta, v[m22*ldv+1:m22*ldv+2], 1)
					h[(k+1)*ldh+k] = beta
					h[(k+2)*ldh+k] = 0
				}
			}

			// Multiply H by reflections from the left.
			var jbot int
			switch {
			case accum:
				jbot = min(ndcol, kbot)
			case wantt:
				jbot = n - 1
			default:
				jbot = kbot
			}
			for j := max(ktop, krcol); j <= jbot; j++ {
				mend := min(mbot+1, (j-krcol+2)/3) - 1
				for m := mtop; m <= mend; m++ {
					k := krcol + 3*m
					refsum := v[m*ldv] * (h[(k+1)*ldh+j] +
						v[m*ldv+1]*h[(k+2)*ldh+j] + v[m*ldv+2]*h[(k+3)*ldh+j])
					h[(k+1)*ldh+j] -= refsum
					h[(k+2)*ldh+j] -= refsum * v[m*ldv+1]
					h[(k+3)*ldh+j] -= refsum * v[m*ldv+2]
				}
			}
			if bmp22 {
				k := krcol + 3*m22
				for j := max(k+1, ktop); j <= jbot; j++ {
					refsum := v[m22*ldv] * (h[(k+1)*ldh+j] + v[m22*ldv+1]*h[(k+2)*ldh+j])
					h[(k+1)*ldh+j] -= refsum
					h[(k+2)*ldh+j] -= refsum * v[m22*ldv+1]
				}
			}

			// Multiply H by reflections from the right. Delay filling in the last row
			// until the vigilant deflation check is complete.
			var jtop int
			switch {
			case accum:
				jtop = max(ktop, incol)
			case wantt:
				jtop = 0
			default:
				jtop = ktop
			}
			for m := mtop; m <= mbot; m++ {
				if v[m*ldv] == 0 {
					continue
				}
				k := krcol + 3*m
				for j := jtop; j <= min(kbot, k+3); j++ {
					refsum := v[m*ldv] * (h[j*ldh+k+1] +
						v[m*ldv+1]*h[j*ldh+k+2] + v[m*ldv+2]*h[j*ldh+k+3])
					h[j*ldh+k+1] -= refsum
					h[j*ldh+k+2] -= refsum * v[m*ldv+1]
					h[j*ldh+k+3] -= refsum * v[m*ldv+2]
				}
				if accum {
					// Accumulate U. (If necessary, update Z later with with an
					// efficient matrix-matrix multiply.)
					kms := k - incol
					for j := max(0, ktop-incol-1); j < kdu; j++ {
						refsum := v[m*ldv] * (u[j*ldu+kms] +
							v[m*ldv+1]*u[j*ldu+kms+1] + v[m*ldv+2]*u[j*ldu+kms+2])
						u[j*ldu+kms] -= refsum
						u[j*ldu+kms+1] -= refsum * v[m*ldv+1]
						u[j*ldu+kms+2] -= refsum * v[m*ldv+2]
					}
				} else if wantz {
					// U is not accumulated, so update Z now by multiplying by
					// reflections from the right.
					for j := iloz; j <= ihiz; j++ {
						refsum := v[m*ldv] * (z[j*ldz+k+1] +
							v[m*ldv+1]*z[j*ldz+k+2] + v[m*ldv+2]*z[j*ldz+k+3])
						z[j*ldz+k+1] -= refsum
						z[j*ldz+k+2] -= refsum * v[m*ldv+1]
						z[j*ldz+k+3] -= refsum * v[m*ldv+2]
					}
				}
			}

			// Special case: 2×2 reflection (if needed).
			if bmp22 && v[m22*ldv] != 0 {
				k := krcol + 3*m22
				for j := jtop; j <= min(kbot, k+3); j++ {
					refsum := v[m22*ldv] * (h[j*ldh+k+1] + v[m22*ldv+1]*h[j*ldh+k+2])
					h[j*ldh+k+1] -= refsum
					h[j*ldh+k+2] -= refsum * v[m22*ldv+1]
				}
				if accum {
					kms := k - incol
					for j := max(0, ktop-incol-1); j < kdu; j++ {
						refsum := v[m22*ldv] * (u[j*ldu+kms] + v[m22*ldv+1]*u[j*ldu+kms+1])
						u[j*ldu+kms] -= refsum
						u[j*ldu+kms+1] -= refsum * v[m22*ldv+1]
					}
				} else if wantz {
					for j := iloz; j <= ihiz; j++ {
						refsum := v[m22*ldv] * (z[j*ldz+k+1] + v[m22*ldv+1]*z[j*ldz+k+2])
						z[j*ldz+k+1] -= refsum
						z[j*ldz+k+2] -= refsum * v[m22*ldv+1]
					}
				}
			}

			// Vigilant deflation check.
			mstart := mtop
			if krcol+3*mstart < ktop {
				mstart++
			}
			mend := mbot
			if bmp22 {
				mend++
			}
			if krcol == kbot-2 {
				mend++
			}
			for m := mstart; m <= mend; m++ {
				k := min(kbot-1, krcol+3*m)

				// The following convergence test requires that the tradition
				// small-compared-to-nearby-diagonals criterion and the Ahues &
				// Tisseur (LAWN 122, 1997) criteria both be satisfied. The latter
				// improves accuracy in some examples. Falling back on an alternate
				// convergence criterion when tst1 or tst2 is zero (as done here) is
				// traditional but probably unnecessary.

				if h[(k+1)*ldh+k] == 0 {
					continue
				}
				tst1 := math.Abs(h[k*ldh+k]) + math.Abs(h[(k+1)*ldh+k+1])
				if tst1 == 0 {
					if k >= ktop+1 {
						tst1 += math.Abs(h[k*ldh+k-1])
					}
					if k >= ktop+2 {
						tst1 += math.Abs(h[k*ldh+k-2])
					}
					if k >= ktop+3 {
						tst1 += math.Abs(h[k*ldh+k-3])
					}
					if k <= kbot-2 {
						tst1 += math.Abs(h[(k+2)*ldh+k+1])
					}
					if k <= kbot-3 {
						tst1 += math.Abs(h[(k+3)*ldh+k+1])
					}
					if k <= kbot-4 {
						tst1 += math.Abs(h[(k+4)*ldh+k+1])
					}
				}
				if math.Abs(h[(k+1)*ldh+k]) <= math.Max(smlnum, ulp*tst1) {
					h12 := math.Max(math.Abs(h[(k+1)*ldh+k]), math.Abs(h[k*ldh+k+1]))
					h21 := math.Min(math.Abs(h[(k+1)*ldh+k]), math.Abs(h[k*ldh+k+1]))
					h11 := math.Max(math.Abs(h[(k+1)*ldh+k+1]), math.Abs(h[k*ldh+k]-h[(k+1)*ldh+k+1]))
					h22 := math.Min(math.Abs(h[(k+1)*ldh+k+1]), math.Abs(h[k*ldh+k]-h[(k+1)*ldh+k+1]))
					scl := h11 + h12
					tst2 := h22 * (h11 / scl)
					if tst2 == 0 || h21*(h12/scl) <= math.Max(smlnum, ulp*tst2) {
						h[(k+1)*ldh+k] = 0
					}
				}
			}

			// Fill in the last row of each bulge.
			mend = min(nbmps, (kbot-krcol-1)/3) - 1
			for m := mtop; m <= mend; m++ {
				k := krcol + 3*m
				refsum := v[m*ldv] * v[m*ldv+2] * h[(k+4)*ldh+k+3]
				h[(k+4)*ldh+k+1] = -refsum
				h[(k+4)*ldh+k+2] = -refsum * v[m*ldv+1]
				h[(k+4)*ldh+k+3] -= refsum * v[m*ldv+2]
			}
		}

		// Use U (if accumulated) to update far-from-diagonal entries in H.
		// If required, use U to update Z as well.
		if !accum {
			continue
		}
		var jtop, jbot int
		if wantt {
			jtop = 0
			jbot = n - 1
		} else {
			jtop = ktop
			jbot = kbot
		}
		bi := blas64.Implementation()
		if !blk22 || incol < ktop || kbot < ndcol || ns <= 2 {
			// Updates not exploiting the 2×2 block structure of U. k0 and nu keep track
			// of the location and size of U in the special cases of introducing bulges
			// and chasing bulges off the bottom. In these special cases and in case the
			// number of shifts is ns = 2, there is no 2×2 block structure to exploit.

			k0 := max(0, ktop-incol-1)
			nu := kdu - max(0, ndcol-kbot) - k0

			// Horizontal multiply.
			for jcol := min(ndcol, kbot) + 1; jcol <= jbot; jcol += nh {
				jlen := min(nh, jbot-jcol+1)
				bi.Dgemm(blas.Trans, blas.NoTrans, nu, jlen, nu,
					1, u[k0*ldu+k0:], ldu,
					h[(incol+k0+1)*ldh+jcol:], ldh,
					0, wh, ldwh)
				impl.Dlacpy(blas.All, nu, jlen, wh, ldwh, h[(incol+k0+1)*ldh+jcol:], ldh)
			}

			// Vertical multiply.
			for jrow := jtop; jrow <= max(ktop, incol)-1; jrow += nv {
				jlen := min(nv, max(ktop, incol)-jrow)
				bi.Dgemm(blas.NoTrans, blas.NoTrans, jlen, nu, nu,
					1, h[jrow*ldh+incol+k0+1:], ldh,
					u[k0*ldu+k0:], ldu,
					0, wv, ldwv)
				impl.Dlacpy(blas.All, jlen, nu, wv, ldwv, h[jrow*ldh+incol+k0+1:], ldh)
			}

			// Z multiply (also vertical).
			if wantz {
				for jrow := iloz; jrow <= ihiz; jrow += nv {
					jlen := min(nv, ihiz-jrow+1)
					bi.Dgemm(blas.NoTrans, blas.NoTrans, jlen, nu, nu,
						1, z[jrow*ldz+incol+k0+1:], ldz,
						u[k0*ldu+k0:], ldu,
						0, wv, ldwv)
					impl.Dlacpy(blas.All, jlen, nu, wv, ldwv, z[jrow*ldz+incol+k0+1:], ldz)
				}
			}

			continue
		}

		// Updates exploiting U's 2×2 block structure.

		// i2, i4, j2, j4 are the last rows and columns of the blocks.
		i2 := (kdu + 1) / 2
		i4 := kdu
		j2 := i4 - i2
		j4 := kdu

		// kzs and knz deal with the band of zeros along the diagonal of one of the
		// triangular blocks.
		kzs := (j4 - j2) - (ns + 1)
		knz := ns + 1

		// Horizontal multiply.
		for jcol := min(ndcol, kbot) + 1; jcol <= jbot; jcol += nh {
			jlen := min(nh, jbot-jcol+1)

			// Copy bottom of H to top+kzs of scratch (the first kzs
			// rows get multiplied by zero).
			impl.Dlacpy(blas.All, knz, jlen, h[(incol+1+j2)*ldh+jcol:], ldh, wh[kzs*ldwh:], ldwh)

			// Multiply by U21^T.
			impl.Dlaset(blas.All, kzs, jlen, 0, 0, wh, ldwh)
			bi.Dtrmm(blas.Left, blas.Upper, blas.Trans, blas.NonUnit, knz, jlen,
				1, u[j2*ldu+kzs:], ldu, wh[kzs*ldwh:], ldwh)

			// Multiply top of H by U11^T.
			bi.Dgemm(blas.Trans, blas.NoTrans, i2, jlen, j2,
				1, u, ldu, h[(incol+1)*ldh+jcol:], ldh,
				1, wh, ldwh)

			// Copy top of H to bottom of WH.
			impl.Dlacpy(blas.All, j2, jlen, h[(incol+1)*ldh+jcol:], ldh, wh[i2*ldwh:], ldwh)

			// Multiply by U21^T.
			bi.Dtrmm(blas.Left, blas.Lower, blas.Trans, blas.NonUnit, j2, jlen,
				1, u[i2:], ldu, wh[i2*ldwh:], ldwh)

			// Multiply by U22.
			bi.Dgemm(blas.Trans, blas.NoTrans, i4-i2, jlen, j4-j2,
				1, u[j2*ldu+i2:], ldu, h[(incol+1+j2)*ldh+jcol:], ldh,
				1, wh[i2*ldwh:], ldwh)

			// Copy it back.
			impl.Dlacpy(blas.All, kdu, jlen, wh, ldwh, h[(incol+1)*ldh+jcol:], ldh)
		}

		// Vertical multiply.
		for jrow := jtop; jrow <= max(incol, ktop)-1; jrow += nv {
			jlen := min(nv, max(incol, ktop)-jrow)

			// Copy right of H to scratch (the first kzs columns get multiplied
			// by zero).
			impl.Dlacpy(blas.All, jlen, knz, h[jrow*ldh+incol+1+j2:], ldh, wv[kzs:], ldwv)

			// Multiply by U21.
			impl.Dlaset(blas.All, jlen, kzs, 0, 0, wv, ldwv)
			bi.Dtrmm(blas.Right, blas.Upper, blas.NoTrans, blas.NonUnit, jlen, knz,
				1, u[j2*ldu+kzs:], ldu, wv[kzs:], ldwv)

			// Multiply by U11.
			bi.Dgemm(blas.NoTrans, blas.NoTrans, jlen, i2, j2,
				1, h[jrow*ldh+incol+1:], ldh, u, ldu,
				1, wv, ldwv)

			// Copy left of H to right of scratch.
			impl.Dlacpy(blas.All, jlen, j2, h[jrow*ldh+incol+1:], ldh, wv[i2:], ldwv)

			// Multiply by U21.
			bi.Dtrmm(blas.Right, blas.Lower, blas.NoTrans, blas.NonUnit, jlen, i4-i2,
				1, u[i2:], ldu, wv[i2:], ldwv)

			// Multiply by U22.
			bi.Dgemm(blas.NoTrans, blas.NoTrans, jlen, i4-i2, j4-j2,
				1, h[jrow*ldh+incol+1+j2:], ldh, u[j2*ldu+i2:], ldu,
				1, wv[i2:], ldwv)

			// Copy it back.
			impl.Dlacpy(blas.All, jlen, kdu, wv, ldwv, h[jrow*ldh+incol+1:], ldh)
		}

		if !wantz {
			continue
		}
		// Multiply Z (also vertical).
		for jrow := iloz; jrow <= ihiz; jrow += nv {
			jlen := min(nv, ihiz-jrow+1)

			// Copy right of Z to left of scratch (first kzs columns get
			// multiplied by zero).
			impl.Dlacpy(blas.All, jlen, knz, z[jrow*ldz+incol+1+j2:], ldz, wv[kzs:], ldwv)

			// Multiply by U12.
			impl.Dlaset(blas.All, jlen, kzs, 0, 0, wv, ldwv)
			bi.Dtrmm(blas.Right, blas.Upper, blas.NoTrans, blas.NonUnit, jlen, knz,
				1, u[j2*ldu+kzs:], ldu, wv[kzs:], ldwv)

			// Multiply by U11.
			bi.Dgemm(blas.NoTrans, blas.NoTrans, jlen, i2, j2,
				1, z[jrow*ldz+incol+1:], ldz, u, ldu,
				1, wv, ldwv)

			// Copy left of Z to right of scratch.
			impl.Dlacpy(blas.All, jlen, j2, z[jrow*ldz+incol+1:], ldz, wv[i2:], ldwv)

			// Multiply by U21.
			bi.Dtrmm(blas.Right, blas.Lower, blas.NoTrans, blas.NonUnit, jlen, i4-i2,
				1, u[i2:], ldu, wv[i2:], ldwv)

			// Multiply by U22.
			bi.Dgemm(blas.NoTrans, blas.NoTrans, jlen, i4-i2, j4-j2,
				1, z[jrow*ldz+incol+1+j2:], ldz, u[j2*ldu+i2:], ldu,
				1, wv[i2:], ldwv)

			// Copy the result back to Z.
			impl.Dlacpy(blas.All, jlen, kdu, wv, ldwv, z[jrow*ldz+incol+1:], ldz)
		}
	}
}
