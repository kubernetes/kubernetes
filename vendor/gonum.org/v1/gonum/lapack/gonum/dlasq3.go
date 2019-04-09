// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import "math"

// Dlasq3 checks for deflation, computes a shift (tau) and calls dqds.
// In case of failure it changes shifts, and tries again until output
// is positive.
//
// Dlasq3 is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dlasq3(i0, n0 int, z []float64, pp int, dmin, sigma, desig, qmax float64, nFail, iter, nDiv int, ttype int, dmin1, dmin2, dn, dn1, dn2, g, tau float64) (
	i0Out, n0Out, ppOut int, dminOut, sigmaOut, desigOut, qmaxOut float64, nFailOut, iterOut, nDivOut, ttypeOut int, dmin1Out, dmin2Out, dnOut, dn1Out, dn2Out, gOut, tauOut float64) {
	switch {
	case i0 < 0:
		panic(i0LT0)
	case n0 < 0:
		panic(n0LT0)
	case len(z) < 4*n0:
		panic(shortZ)
	case pp != 0 && pp != 1 && pp != 2:
		panic(badPp)
	}

	const cbias = 1.5

	n0in := n0
	eps := dlamchP
	tol := eps * 100
	tol2 := tol * tol
	var nn int
	var t float64
	for {
		if n0 < i0 {
			return i0, n0, pp, dmin, sigma, desig, qmax, nFail, iter, nDiv, ttype, dmin1, dmin2, dn, dn1, dn2, g, tau
		}
		if n0 == i0 {
			z[4*(n0+1)-4] = z[4*(n0+1)+pp-4] + sigma
			n0--
			continue
		}
		nn = 4*(n0+1) + pp - 1
		if n0 != i0+1 {
			// Check whether e[n0-1] is negligible, 1 eigenvalue.
			if z[nn-5] > tol2*(sigma+z[nn-3]) && z[nn-2*pp-4] > tol2*z[nn-7] {
				// Check whether e[n0-2] is negligible, 2 eigenvalues.
				if z[nn-9] > tol2*sigma && z[nn-2*pp-8] > tol2*z[nn-11] {
					break
				}
			} else {
				z[4*(n0+1)-4] = z[4*(n0+1)+pp-4] + sigma
				n0--
				continue
			}
		}
		if z[nn-3] > z[nn-7] {
			z[nn-3], z[nn-7] = z[nn-7], z[nn-3]
		}
		t = 0.5 * (z[nn-7] - z[nn-3] + z[nn-5])
		if z[nn-5] > z[nn-3]*tol2 && t != 0 {
			s := z[nn-3] * (z[nn-5] / t)
			if s <= t {
				s = z[nn-3] * (z[nn-5] / (t * (1 + math.Sqrt(1+s/t))))
			} else {
				s = z[nn-3] * (z[nn-5] / (t + math.Sqrt(t)*math.Sqrt(t+s)))
			}
			t = z[nn-7] + (s + z[nn-5])
			z[nn-3] *= z[nn-7] / t
			z[nn-7] = t
		}
		z[4*(n0+1)-8] = z[nn-7] + sigma
		z[4*(n0+1)-4] = z[nn-3] + sigma
		n0 -= 2
	}
	if pp == 2 {
		pp = 0
	}

	// Reverse the qd-array, if warranted.
	if dmin <= 0 || n0 < n0in {
		if cbias*z[4*(i0+1)+pp-4] < z[4*(n0+1)+pp-4] {
			ipn4Out := 4 * (i0 + n0 + 2)
			for j4loop := 4 * (i0 + 1); j4loop <= 2*((i0+1)+(n0+1)-1); j4loop += 4 {
				ipn4 := ipn4Out - 1
				j4 := j4loop - 1

				z[j4-3], z[ipn4-j4-4] = z[ipn4-j4-4], z[j4-3]
				z[j4-2], z[ipn4-j4-3] = z[ipn4-j4-3], z[j4-2]
				z[j4-1], z[ipn4-j4-6] = z[ipn4-j4-6], z[j4-1]
				z[j4], z[ipn4-j4-5] = z[ipn4-j4-5], z[j4]
			}
			if n0-i0 <= 4 {
				z[4*(n0+1)+pp-2] = z[4*(i0+1)+pp-2]
				z[4*(n0+1)-pp-1] = z[4*(i0+1)-pp-1]
			}
			dmin2 = math.Min(dmin2, z[4*(i0+1)-pp-2])
			z[4*(n0+1)+pp-2] = math.Min(math.Min(z[4*(n0+1)+pp-2], z[4*(i0+1)+pp-2]), z[4*(i0+1)+pp+2])
			z[4*(n0+1)-pp-1] = math.Min(math.Min(z[4*(n0+1)-pp-1], z[4*(i0+1)-pp-1]), z[4*(i0+1)-pp+3])
			qmax = math.Max(math.Max(qmax, z[4*(i0+1)+pp-4]), z[4*(i0+1)+pp])
			dmin = math.Copysign(0, -1) // Fortran code has -zero, but -0 in go is 0
		}
	}

	// Choose a shift.
	tau, ttype, g = impl.Dlasq4(i0, n0, z, pp, n0in, dmin, dmin1, dmin2, dn, dn1, dn2, tau, ttype, g)

	// Call dqds until dmin > 0.
loop:
	for {
		i0, n0, pp, tau, sigma, dmin, dmin1, dmin2, dn, dn1, dn2 = impl.Dlasq5(i0, n0, z, pp, tau, sigma)

		nDiv += n0 - i0 + 2
		iter++
		switch {
		case dmin >= 0 && dmin1 >= 0:
			// Success.
			goto done

		case dmin < 0 && dmin1 > 0 && z[4*n0-pp-1] < tol*(sigma+dn1) && math.Abs(dn) < tol*sigma:
			// Convergence hidden by negative dn.
			z[4*n0-pp+1] = 0
			dmin = 0
			goto done

		case dmin < 0:
			// Tau too big. Select new Tau and try again.
			nFail++
			if ttype < -22 {
				// Failed twice. Play it safe.
				tau = 0
			} else if dmin1 > 0 {
				// Late failure. Gives excellent shift.
				tau = (tau + dmin) * (1 - 2*eps)
				ttype -= 11
			} else {
				// Early failure. Divide by 4.
				tau = tau / 4
				ttype -= 12
			}

		case math.IsNaN(dmin):
			if tau == 0 {
				break loop
			}
			tau = 0

		default:
			// Possible underflow. Play it safe.
			break loop
		}
	}

	// Risk of underflow.
	dmin, dmin1, dmin2, dn, dn1, dn2 = impl.Dlasq6(i0, n0, z, pp)
	nDiv += n0 - i0 + 2
	iter++
	tau = 0

done:
	if tau < sigma {
		desig += tau
		t = sigma + desig
		desig -= t - sigma
	} else {
		t = sigma + tau
		desig += sigma - (t - tau)
	}
	sigma = t
	return i0, n0, pp, dmin, sigma, desig, qmax, nFail, iter, nDiv, ttype, dmin1, dmin2, dn, dn1, dn2, g, tau
}
