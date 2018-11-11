// Copyright ©2016 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import "math"

// Dlanv2 computes the Schur factorization of a real 2×2 matrix:
//  [ a b ] = [ cs -sn ] * [ aa bb ] * [ cs sn ]
//  [ c d ]   [ sn  cs ]   [ cc dd ] * [-sn cs ]
// If cc is zero, aa and dd are real eigenvalues of the matrix. Otherwise it
// holds that aa = dd and bb*cc < 0, and aa ± sqrt(bb*cc) are complex conjugate
// eigenvalues. The real and imaginary parts of the eigenvalues are returned in
// (rt1r,rt1i) and (rt2r,rt2i).
func (impl Implementation) Dlanv2(a, b, c, d float64) (aa, bb, cc, dd float64, rt1r, rt1i, rt2r, rt2i float64, cs, sn float64) {
	switch {
	case c == 0: // Matrix is already upper triangular.
		aa = a
		bb = b
		cc = 0
		dd = d
		cs = 1
		sn = 0
	case b == 0: // Matrix is lower triangular, swap rows and columns.
		aa = d
		bb = -c
		cc = 0
		dd = a
		cs = 0
		sn = 1
	case a == d && math.Signbit(b) != math.Signbit(c): // Matrix is already in the standard Schur form.
		aa = a
		bb = b
		cc = c
		dd = d
		cs = 1
		sn = 0
	default:
		temp := a - d
		p := temp / 2
		bcmax := math.Max(math.Abs(b), math.Abs(c))
		bcmis := math.Min(math.Abs(b), math.Abs(c))
		if b*c < 0 {
			bcmis *= -1
		}
		scale := math.Max(math.Abs(p), bcmax)
		z := p/scale*p + bcmax/scale*bcmis
		eps := dlamchP

		if z >= 4*eps {
			// Real eigenvalues. Compute aa and dd.
			if p > 0 {
				z = p + math.Sqrt(scale)*math.Sqrt(z)
			} else {
				z = p - math.Sqrt(scale)*math.Sqrt(z)
			}
			aa = d + z
			dd = d - bcmax/z*bcmis
			// Compute bb and the rotation matrix.
			tau := impl.Dlapy2(c, z)
			cs = z / tau
			sn = c / tau
			bb = b - c
			cc = 0
		} else {
			// Complex eigenvalues, or real (almost) equal eigenvalues.
			// Make diagonal elements equal.
			sigma := b + c
			tau := impl.Dlapy2(sigma, temp)
			cs = math.Sqrt((1 + math.Abs(sigma)/tau) / 2)
			sn = -p / (tau * cs)
			if sigma < 0 {
				sn *= -1
			}
			// Compute [ aa bb ] = [ a b ] [ cs -sn ]
			//         [ cc dd ]   [ c d ] [ sn  cs ]
			aa = a*cs + b*sn
			bb = -a*sn + b*cs
			cc = c*cs + d*sn
			dd = -c*sn + d*cs
			// Compute [ a b ] = [ cs sn ] [ aa bb ]
			//         [ c d ]   [-sn cs ] [ cc dd ]
			a = aa*cs + cc*sn
			b = bb*cs + dd*sn
			c = -aa*sn + cc*cs
			d = -bb*sn + dd*cs

			temp = (a + d) / 2
			aa = temp
			bb = b
			cc = c
			dd = temp

			if cc != 0 {
				if bb != 0 {
					if math.Signbit(bb) == math.Signbit(cc) {
						// Real eigenvalues, reduce to
						// upper triangular form.
						sab := math.Sqrt(math.Abs(bb))
						sac := math.Sqrt(math.Abs(cc))
						p = sab * sac
						if cc < 0 {
							p *= -1
						}
						tau = 1 / math.Sqrt(math.Abs(bb+cc))
						aa = temp + p
						bb = bb - cc
						cc = 0
						dd = temp - p
						cs1 := sab * tau
						sn1 := sac * tau
						cs, sn = cs*cs1-sn*sn1, cs*sn1+sn+cs1
					}
				} else {
					bb = -cc
					cc = 0
					cs, sn = -sn, cs
				}
			}
		}
	}

	// Store eigenvalues in (rt1r,rt1i) and (rt2r,rt2i).
	rt1r = aa
	rt2r = dd
	if cc != 0 {
		rt1i = math.Sqrt(math.Abs(bb)) * math.Sqrt(math.Abs(cc))
		rt2i = -rt1i
	}
	return
}
