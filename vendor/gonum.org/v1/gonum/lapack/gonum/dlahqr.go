// Copyright ©2016 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"math"

	"gonum.org/v1/gonum/blas/blas64"
)

// Dlahqr computes the eigenvalues and Schur factorization of a block of an n×n
// upper Hessenberg matrix H, using the double-shift/single-shift QR algorithm.
//
// h and ldh represent the matrix H. Dlahqr works primarily with the Hessenberg
// submatrix H[ilo:ihi+1,ilo:ihi+1], but applies transformations to all of H if
// wantt is true. It is assumed that H[ihi+1:n,ihi+1:n] is already upper
// quasi-triangular, although this is not checked.
//
// It must hold that
//  0 <= ilo <= max(0,ihi), and ihi < n,
// and that
//  H[ilo,ilo-1] == 0,  if ilo > 0,
// otherwise Dlahqr will panic.
//
// If unconverged is zero on return, wr[ilo:ihi+1] and wi[ilo:ihi+1] will contain
// respectively the real and imaginary parts of the computed eigenvalues ilo
// to ihi. If two eigenvalues are computed as a complex conjugate pair, they are
// stored in consecutive elements of wr and wi, say the i-th and (i+1)th, with
// wi[i] > 0 and wi[i+1] < 0. If wantt is true, the eigenvalues are stored in
// the same order as on the diagonal of the Schur form returned in H, with
// wr[i] = H[i,i], and, if H[i:i+2,i:i+2] is a 2×2 diagonal block,
// wi[i] = sqrt(abs(H[i+1,i]*H[i,i+1])) and wi[i+1] = -wi[i].
//
// wr and wi must have length ihi+1.
//
// z and ldz represent an n×n matrix Z. If wantz is true, the transformations
// will be applied to the submatrix Z[iloz:ihiz+1,ilo:ihi+1] and it must hold that
//  0 <= iloz <= ilo, and ihi <= ihiz < n.
// If wantz is false, z is not referenced.
//
// unconverged indicates whether Dlahqr computed all the eigenvalues ilo to ihi
// in a total of 30 iterations per eigenvalue.
//
// If unconverged is zero, all the eigenvalues ilo to ihi have been computed and
// will be stored on return in wr[ilo:ihi+1] and wi[ilo:ihi+1].
//
// If unconverged is zero and wantt is true, H[ilo:ihi+1,ilo:ihi+1] will be
// overwritten on return by upper quasi-triangular full Schur form with any
// 2×2 diagonal blocks in standard form.
//
// If unconverged is zero and if wantt is false, the contents of h on return is
// unspecified.
//
// If unconverged is positive, some eigenvalues have not converged, and
// wr[unconverged:ihi+1] and wi[unconverged:ihi+1] contain those eigenvalues
// which have been successfully computed.
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
// Dlahqr is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dlahqr(wantt, wantz bool, n, ilo, ihi int, h []float64, ldh int, wr, wi []float64, iloz, ihiz int, z []float64, ldz int) (unconverged int) {
	checkMatrix(n, n, h, ldh)
	switch {
	case ilo < 0 || max(0, ihi) < ilo:
		panic(badIlo)
	case n <= ihi:
		panic(badIhi)
	case len(wr) != ihi+1:
		panic("lapack: bad length of wr")
	case len(wi) != ihi+1:
		panic("lapack: bad length of wi")
	case ilo > 0 && h[ilo*ldh+ilo-1] != 0:
		panic("lapack: block is not isolated")
	}
	if wantz {
		checkMatrix(n, n, z, ldz)
		switch {
		case iloz < 0 || ilo < iloz:
			panic("lapack: iloz out of range")
		case ihiz < ihi || n <= ihiz:
			panic("lapack: ihiz out of range")
		}
	}

	// Quick return if possible.
	if n == 0 {
		return 0
	}
	if ilo == ihi {
		wr[ilo] = h[ilo*ldh+ilo]
		wi[ilo] = 0
		return 0
	}

	// Clear out the trash.
	for j := ilo; j < ihi-2; j++ {
		h[(j+2)*ldh+j] = 0
		h[(j+3)*ldh+j] = 0
	}
	if ilo <= ihi-2 {
		h[ihi*ldh+ihi-2] = 0
	}

	nh := ihi - ilo + 1
	nz := ihiz - iloz + 1

	// Set machine-dependent constants for the stopping criterion.
	ulp := dlamchP
	smlnum := float64(nh) / ulp * dlamchS

	// i1 and i2 are the indices of the first row and last column of H to
	// which transformations must be applied. If eigenvalues only are being
	// computed, i1 and i2 are set inside the main loop.
	var i1, i2 int
	if wantt {
		i1 = 0
		i2 = n - 1
	}

	itmax := 30 * max(10, nh) // Total number of QR iterations allowed.

	// The main loop begins here. i is the loop index and decreases from ihi
	// to ilo in steps of 1 or 2. Each iteration of the loop works with the
	// active submatrix in rows and columns l to i. Eigenvalues i+1 to ihi
	// have already converged. Either l = ilo or H[l,l-1] is negligible so
	// that the matrix splits.
	bi := blas64.Implementation()
	i := ihi
	for i >= ilo {
		l := ilo

		// Perform QR iterations on rows and columns ilo to i until a
		// submatrix of order 1 or 2 splits off at the bottom because a
		// subdiagonal element has become negligible.
		converged := false
		for its := 0; its <= itmax; its++ {
			// Look for a single small subdiagonal element.
			var k int
			for k = i; k > l; k-- {
				if math.Abs(h[k*ldh+k-1]) <= smlnum {
					break
				}
				tst := math.Abs(h[(k-1)*ldh+k-1]) + math.Abs(h[k*ldh+k])
				if tst == 0 {
					if k-2 >= ilo {
						tst += math.Abs(h[(k-1)*ldh+k-2])
					}
					if k+1 <= ihi {
						tst += math.Abs(h[(k+1)*ldh+k])
					}
				}
				// The following is a conservative small
				// subdiagonal deflation criterion due to Ahues
				// & Tisseur (LAWN 122, 1997). It has better
				// mathematical foundation and improves accuracy
				// in some cases.
				if math.Abs(h[k*ldh+k-1]) <= ulp*tst {
					ab := math.Max(math.Abs(h[k*ldh+k-1]), math.Abs(h[(k-1)*ldh+k]))
					ba := math.Min(math.Abs(h[k*ldh+k-1]), math.Abs(h[(k-1)*ldh+k]))
					aa := math.Max(math.Abs(h[k*ldh+k]), math.Abs(h[(k-1)*ldh+k-1]-h[k*ldh+k]))
					bb := math.Min(math.Abs(h[k*ldh+k]), math.Abs(h[(k-1)*ldh+k-1]-h[k*ldh+k]))
					s := aa + ab
					if ab/s*ba <= math.Max(smlnum, aa/s*bb*ulp) {
						break
					}
				}
			}
			l = k
			if l > ilo {
				// H[l,l-1] is negligible.
				h[l*ldh+l-1] = 0
			}
			if l >= i-1 {
				// Break the loop because a submatrix of order 1
				// or 2 has split off.
				converged = true
				break
			}

			// Now the active submatrix is in rows and columns l to
			// i. If eigenvalues only are being computed, only the
			// active submatrix need be transformed.
			if !wantt {
				i1 = l
				i2 = i
			}

			const (
				dat1 = 3.0
				dat2 = -0.4375
			)
			var h11, h21, h12, h22 float64
			switch its {
			case 10: // Exceptional shift.
				s := math.Abs(h[(l+1)*ldh+l]) + math.Abs(h[(l+2)*ldh+l+1])
				h11 = dat1*s + h[l*ldh+l]
				h12 = dat2 * s
				h21 = s
				h22 = h11
			case 20: // Exceptional shift.
				s := math.Abs(h[i*ldh+i-1]) + math.Abs(h[(i-1)*ldh+i-2])
				h11 = dat1*s + h[i*ldh+i]
				h12 = dat2 * s
				h21 = s
				h22 = h11
			default: // Prepare to use Francis' double shift (i.e.,
				// 2nd degree generalized Rayleigh quotient).
				h11 = h[(i-1)*ldh+i-1]
				h21 = h[i*ldh+i-1]
				h12 = h[(i-1)*ldh+i]
				h22 = h[i*ldh+i]
			}
			s := math.Abs(h11) + math.Abs(h12) + math.Abs(h21) + math.Abs(h22)
			var (
				rt1r, rt1i float64
				rt2r, rt2i float64
			)
			if s != 0 {
				h11 /= s
				h21 /= s
				h12 /= s
				h22 /= s
				tr := (h11 + h22) / 2
				det := (h11-tr)*(h22-tr) - h12*h21
				rtdisc := math.Sqrt(math.Abs(det))
				if det >= 0 {
					// Complex conjugate shifts.
					rt1r = tr * s
					rt2r = rt1r
					rt1i = rtdisc * s
					rt2i = -rt1i
				} else {
					// Real shifts (use only one of them).
					rt1r = tr + rtdisc
					rt2r = tr - rtdisc
					if math.Abs(rt1r-h22) <= math.Abs(rt2r-h22) {
						rt1r *= s
						rt2r = rt1r
					} else {
						rt2r *= s
						rt1r = rt2r
					}
					rt1i = 0
					rt2i = 0
				}
			}

			// Look for two consecutive small subdiagonal elements.
			var m int
			var v [3]float64
			for m = i - 2; m >= l; m-- {
				// Determine the effect of starting the
				// double-shift QR iteration at row m, and see
				// if this would make H[m,m-1] negligible. The
				// following uses scaling to avoid overflows and
				// most underflows.
				h21s := h[(m+1)*ldh+m]
				s := math.Abs(h[m*ldh+m]-rt2r) + math.Abs(rt2i) + math.Abs(h21s)
				h21s /= s
				v[0] = h21s*h[m*ldh+m+1] + (h[m*ldh+m]-rt1r)*((h[m*ldh+m]-rt2r)/s) - rt2i/s*rt1i
				v[1] = h21s * (h[m*ldh+m] + h[(m+1)*ldh+m+1] - rt1r - rt2r)
				v[2] = h21s * h[(m+2)*ldh+m+1]
				s = math.Abs(v[0]) + math.Abs(v[1]) + math.Abs(v[2])
				v[0] /= s
				v[1] /= s
				v[2] /= s
				if m == l {
					break
				}
				dsum := math.Abs(h[(m-1)*ldh+m-1]) + math.Abs(h[m*ldh+m]) + math.Abs(h[(m+1)*ldh+m+1])
				if math.Abs(h[m*ldh+m-1])*(math.Abs(v[1])+math.Abs(v[2])) <= ulp*math.Abs(v[0])*dsum {
					break
				}
			}

			// Double-shift QR step.
			for k := m; k < i; k++ {
				// The first iteration of this loop determines a
				// reflection G from the vector V and applies it
				// from left and right to H, thus creating a
				// non-zero bulge below the subdiagonal.
				//
				// Each subsequent iteration determines a
				// reflection G to restore the Hessenberg form
				// in the (k-1)th column, and thus chases the
				// bulge one step toward the bottom of the
				// active submatrix. nr is the order of G.

				nr := min(3, i-k+1)
				if k > m {
					bi.Dcopy(nr, h[k*ldh+k-1:], ldh, v[:], 1)
				}
				var t0 float64
				v[0], t0 = impl.Dlarfg(nr, v[0], v[1:], 1)
				if k > m {
					h[k*ldh+k-1] = v[0]
					h[(k+1)*ldh+k-1] = 0
					if k < i-1 {
						h[(k+2)*ldh+k-1] = 0
					}
				} else if m > l {
					// Use the following instead of H[k,k-1] = -H[k,k-1]
					// to avoid a bug when v[1] and v[2] underflow.
					h[k*ldh+k-1] *= 1 - t0
				}
				t1 := t0 * v[1]
				if nr == 3 {
					t2 := t0 * v[2]

					// Apply G from the left to transform
					// the rows of the matrix in columns k
					// to i2.
					for j := k; j <= i2; j++ {
						sum := h[k*ldh+j] + v[1]*h[(k+1)*ldh+j] + v[2]*h[(k+2)*ldh+j]
						h[k*ldh+j] -= sum * t0
						h[(k+1)*ldh+j] -= sum * t1
						h[(k+2)*ldh+j] -= sum * t2
					}

					// Apply G from the right to transform
					// the columns of the matrix in rows i1
					// to min(k+3,i).
					for j := i1; j <= min(k+3, i); j++ {
						sum := h[j*ldh+k] + v[1]*h[j*ldh+k+1] + v[2]*h[j*ldh+k+2]
						h[j*ldh+k] -= sum * t0
						h[j*ldh+k+1] -= sum * t1
						h[j*ldh+k+2] -= sum * t2
					}

					if wantz {
						// Accumulate transformations in the matrix Z.
						for j := iloz; j <= ihiz; j++ {
							sum := z[j*ldz+k] + v[1]*z[j*ldz+k+1] + v[2]*z[j*ldz+k+2]
							z[j*ldz+k] -= sum * t0
							z[j*ldz+k+1] -= sum * t1
							z[j*ldz+k+2] -= sum * t2
						}
					}
				} else if nr == 2 {
					// Apply G from the left to transform
					// the rows of the matrix in columns k
					// to i2.
					for j := k; j <= i2; j++ {
						sum := h[k*ldh+j] + v[1]*h[(k+1)*ldh+j]
						h[k*ldh+j] -= sum * t0
						h[(k+1)*ldh+j] -= sum * t1
					}

					// Apply G from the right to transform
					// the columns of the matrix in rows i1
					// to min(k+3,i).
					for j := i1; j <= i; j++ {
						sum := h[j*ldh+k] + v[1]*h[j*ldh+k+1]
						h[j*ldh+k] -= sum * t0
						h[j*ldh+k+1] -= sum * t1
					}

					if wantz {
						// Accumulate transformations in the matrix Z.
						for j := iloz; j <= ihiz; j++ {
							sum := z[j*ldz+k] + v[1]*z[j*ldz+k+1]
							z[j*ldz+k] -= sum * t0
							z[j*ldz+k+1] -= sum * t1
						}
					}
				}
			}
		}

		if !converged {
			// The QR iteration finished without splitting off a
			// submatrix of order 1 or 2.
			return i + 1
		}

		if l == i {
			// H[i,i-1] is negligible: one eigenvalue has converged.
			wr[i] = h[i*ldh+i]
			wi[i] = 0
		} else if l == i-1 {
			// H[i-1,i-2] is negligible: a pair of eigenvalues have converged.

			// Transform the 2×2 submatrix to standard Schur form,
			// and compute and store the eigenvalues.
			var cs, sn float64
			a, b := h[(i-1)*ldh+i-1], h[(i-1)*ldh+i]
			c, d := h[i*ldh+i-1], h[i*ldh+i]
			a, b, c, d, wr[i-1], wi[i-1], wr[i], wi[i], cs, sn = impl.Dlanv2(a, b, c, d)
			h[(i-1)*ldh+i-1], h[(i-1)*ldh+i] = a, b
			h[i*ldh+i-1], h[i*ldh+i] = c, d

			if wantt {
				// Apply the transformation to the rest of H.
				if i2 > i {
					bi.Drot(i2-i, h[(i-1)*ldh+i+1:], 1, h[i*ldh+i+1:], 1, cs, sn)
				}
				bi.Drot(i-i1-1, h[i1*ldh+i-1:], ldh, h[i1*ldh+i:], ldh, cs, sn)
			}

			if wantz {
				// Apply the transformation to Z.
				bi.Drot(nz, z[iloz*ldz+i-1:], ldz, z[iloz*ldz+i:], ldz, cs, sn)
			}
		}

		// Return to start of the main loop with new value of i.
		i = l - 1
	}
	return 0
}
