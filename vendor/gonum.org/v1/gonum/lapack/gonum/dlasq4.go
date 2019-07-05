// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import "math"

// Dlasq4 computes an approximation to the smallest eigenvalue using values of d
// from the previous transform.
// i0, n0, and n0in are zero-indexed.
//
// Dlasq4 is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dlasq4(i0, n0 int, z []float64, pp int, n0in int, dmin, dmin1, dmin2, dn, dn1, dn2, tau float64, ttype int, g float64) (tauOut float64, ttypeOut int, gOut float64) {
	switch {
	case i0 < 0:
		panic(i0LT0)
	case n0 < 0:
		panic(n0LT0)
	case len(z) < 4*n0:
		panic(shortZ)
	case pp != 0 && pp != 1:
		panic(badPp)
	}

	const (
		cnst1 = 0.563
		cnst2 = 1.01
		cnst3 = 1.05

		cnstthird = 0.333 // TODO(btracey): Fix?
	)
	// A negative dmin forces the shift to take that absolute value
	// ttype records the type of shift.
	if dmin <= 0 {
		tau = -dmin
		ttype = -1
		return tau, ttype, g
	}
	nn := 4*(n0+1) + pp - 1 // -1 for zero indexing
	s := math.NaN()         // Poison s so that failure to take a path below is obvious
	if n0in == n0 {
		// No eigenvalues deflated.
		if dmin == dn || dmin == dn1 {
			b1 := math.Sqrt(z[nn-3]) * math.Sqrt(z[nn-5])
			b2 := math.Sqrt(z[nn-7]) * math.Sqrt(z[nn-9])
			a2 := z[nn-7] + z[nn-5]
			if dmin == dn && dmin1 == dn1 {
				gap2 := dmin2 - a2 - dmin2/4
				var gap1 float64
				if gap2 > 0 && gap2 > b2 {
					gap1 = a2 - dn - (b2/gap2)*b2
				} else {
					gap1 = a2 - dn - (b1 + b2)
				}
				if gap1 > 0 && gap1 > b1 {
					s = math.Max(dn-(b1/gap1)*b1, 0.5*dmin)
					ttype = -2
				} else {
					s = 0
					if dn > b1 {
						s = dn - b1
					}
					if a2 > b1+b2 {
						s = math.Min(s, a2-(b1+b2))
					}
					s = math.Max(s, cnstthird*dmin)
					ttype = -3
				}
			} else {
				ttype = -4
				s = dmin / 4
				var gam float64
				var np int
				if dmin == dn {
					gam = dn
					a2 = 0
					if z[nn-5] > z[nn-7] {
						return tau, ttype, g
					}
					b2 = z[nn-5] / z[nn-7]
					np = nn - 9
				} else {
					np = nn - 2*pp
					gam = dn1
					if z[np-4] > z[np-2] {
						return tau, ttype, g
					}
					a2 = z[np-4] / z[np-2]
					if z[nn-9] > z[nn-11] {
						return tau, ttype, g
					}
					b2 = z[nn-9] / z[nn-11]
					np = nn - 13
				}
				// Approximate contribution to norm squared from i < nn-1.
				a2 += b2
				for i4loop := np + 1; i4loop >= 4*(i0+1)-1+pp; i4loop -= 4 {
					i4 := i4loop - 1
					if b2 == 0 {
						break
					}
					b1 = b2
					if z[i4] > z[i4-2] {
						return tau, ttype, g
					}
					b2 *= z[i4] / z[i4-2]
					a2 += b2
					if 100*math.Max(b2, b1) < a2 || cnst1 < a2 {
						break
					}
				}
				a2 *= cnst3
				// Rayleigh quotient residual bound.
				if a2 < cnst1 {
					s = gam * (1 - math.Sqrt(a2)) / (1 + a2)
				}
			}
		} else if dmin == dn2 {
			ttype = -5
			s = dmin / 4
			// Compute contribution to norm squared from i > nn-2.
			np := nn - 2*pp
			b1 := z[np-2]
			b2 := z[np-6]
			gam := dn2
			if z[np-8] > b2 || z[np-4] > b1 {
				return tau, ttype, g
			}
			a2 := (z[np-8] / b2) * (1 + z[np-4]/b1)
			// Approximate contribution to norm squared from i < nn-2.
			if n0-i0 > 2 {
				b2 = z[nn-13] / z[nn-15]
				a2 += b2
				for i4loop := (nn + 1) - 17; i4loop >= 4*(i0+1)-1+pp; i4loop -= 4 {
					i4 := i4loop - 1
					if b2 == 0 {
						break
					}
					b1 = b2
					if z[i4] > z[i4-2] {
						return tau, ttype, g
					}
					b2 *= z[i4] / z[i4-2]
					a2 += b2
					if 100*math.Max(b2, b1) < a2 || cnst1 < a2 {
						break
					}
				}
				a2 *= cnst3
			}
			if a2 < cnst1 {
				s = gam * (1 - math.Sqrt(a2)) / (1 + a2)
			}
		} else {
			// Case 6, no information to guide us.
			if ttype == -6 {
				g += cnstthird * (1 - g)
			} else if ttype == -18 {
				g = cnstthird / 4
			} else {
				g = 1.0 / 4
			}
			s = g * dmin
			ttype = -6
		}
	} else if n0in == (n0 + 1) {
		// One eigenvalue just deflated. Use DMIN1, DN1 for DMIN and DN.
		if dmin1 == dn1 && dmin2 == dn2 {
			ttype = -7
			s = cnstthird * dmin1
			if z[nn-5] > z[nn-7] {
				return tau, ttype, g
			}
			b1 := z[nn-5] / z[nn-7]
			b2 := b1
			if b2 != 0 {
				for i4loop := 4*(n0+1) - 9 + pp; i4loop >= 4*(i0+1)-1+pp; i4loop -= 4 {
					i4 := i4loop - 1
					a2 := b1
					if z[i4] > z[i4-2] {
						return tau, ttype, g
					}
					b1 *= z[i4] / z[i4-2]
					b2 += b1
					if 100*math.Max(b1, a2) < b2 {
						break
					}
				}
			}
			b2 = math.Sqrt(cnst3 * b2)
			a2 := dmin1 / (1 + b2*b2)
			gap2 := 0.5*dmin2 - a2
			if gap2 > 0 && gap2 > b2*a2 {
				s = math.Max(s, a2*(1-cnst2*a2*(b2/gap2)*b2))
			} else {
				s = math.Max(s, a2*(1-cnst2*b2))
				ttype = -8
			}
		} else {
			s = dmin1 / 4
			if dmin1 == dn1 {
				s = 0.5 * dmin1
			}
			ttype = -9
		}
	} else if n0in == (n0 + 2) {
		// Two eigenvalues deflated. Use DMIN2, DN2 for DMIN and DN.
		if dmin2 == dn2 && 2*z[nn-5] < z[nn-7] {
			ttype = -10
			s = cnstthird * dmin2
			if z[nn-5] > z[nn-7] {
				return tau, ttype, g
			}
			b1 := z[nn-5] / z[nn-7]
			b2 := b1
			if b2 != 0 {
				for i4loop := 4*(n0+1) - 9 + pp; i4loop >= 4*(i0+1)-1+pp; i4loop -= 4 {
					i4 := i4loop - 1
					if z[i4] > z[i4-2] {
						return tau, ttype, g
					}
					b1 *= z[i4] / z[i4-2]
					b2 += b1
					if 100*b1 < b2 {
						break
					}
				}
			}
			b2 = math.Sqrt(cnst3 * b2)
			a2 := dmin2 / (1 + b2*b2)
			gap2 := z[nn-7] + z[nn-9] - math.Sqrt(z[nn-11])*math.Sqrt(z[nn-9]) - a2
			if gap2 > 0 && gap2 > b2*a2 {
				s = math.Max(s, a2*(1-cnst2*a2*(b2/gap2)*b2))
			} else {
				s = math.Max(s, a2*(1-cnst2*b2))
			}
		} else {
			s = dmin2 / 4
			ttype = -11
		}
	} else if n0in > n0+2 {
		// Case 12, more than two eigenvalues deflated. No information.
		s = 0
		ttype = -12
	}
	tau = s
	return tau, ttype, g
}
