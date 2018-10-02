// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import "math"

// Dlasq5 computes one dqds transform in ping-pong form.
// i0 and n0 are zero-indexed.
//
// Dlasq5 is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dlasq5(i0, n0 int, z []float64, pp int, tau, sigma float64) (i0Out, n0Out, ppOut int, tauOut, sigmaOut, dmin, dmin1, dmin2, dn, dnm1, dnm2 float64) {
	// The lapack function has inputs for ieee and eps, but Go requires ieee so
	// these are unnecessary.
	if n0-i0-1 <= 0 {
		return i0, n0, pp, tau, sigma, dmin, dmin1, dmin2, dn, dnm1, dnm2
	}
	eps := dlamchP
	dthresh := eps * (sigma + tau)
	if tau < dthresh*0.5 {
		tau = 0
	}
	var j4 int
	var emin float64
	if tau != 0 {
		j4 = 4*i0 + pp
		emin = z[j4+4]
		d := z[j4] - tau
		dmin = d
		// In the reference there are code paths that actually return this value.
		// dmin1 = -z[j4]
		if pp == 0 {
			for j4loop := 4 * (i0 + 1); j4loop <= 4*((n0+1)-3); j4loop += 4 {
				j4 := j4loop - 1
				z[j4-2] = d + z[j4-1]
				tmp := z[j4+1] / z[j4-2]
				d = d*tmp - tau
				dmin = math.Min(dmin, d)
				z[j4] = z[j4-1] * tmp
				emin = math.Min(z[j4], emin)
			}
		} else {
			for j4loop := 4 * (i0 + 1); j4loop <= 4*((n0+1)-3); j4loop += 4 {
				j4 := j4loop - 1
				z[j4-3] = d + z[j4]
				tmp := z[j4+2] / z[j4-3]
				d = d*tmp - tau
				dmin = math.Min(dmin, d)
				z[j4-1] = z[j4] * tmp
				emin = math.Min(z[j4-1], emin)
			}
		}
		// Unroll the last two steps.
		dnm2 = d
		dmin2 = dmin
		j4 = 4*((n0+1)-2) - pp - 1
		j4p2 := j4 + 2*pp - 1
		z[j4-2] = dnm2 + z[j4p2]
		z[j4] = z[j4p2+2] * (z[j4p2] / z[j4-2])
		dnm1 = z[j4p2+2]*(dnm2/z[j4-2]) - tau
		dmin = math.Min(dmin, dnm1)

		dmin1 = dmin
		j4 += 4
		j4p2 = j4 + 2*pp - 1
		z[j4-2] = dnm1 + z[j4p2]
		z[j4] = z[j4p2+2] * (z[j4p2] / z[j4-2])
		dn = z[j4p2+2]*(dnm1/z[j4-2]) - tau
		dmin = math.Min(dmin, dn)
	} else {
		// This is the version that sets d's to zero if they are small enough.
		j4 = 4*(i0+1) + pp - 4
		emin = z[j4+4]
		d := z[j4] - tau
		dmin = d
		// In the reference there are code paths that actually return this value.
		// dmin1 = -z[j4]
		if pp == 0 {
			for j4loop := 4 * (i0 + 1); j4loop <= 4*((n0+1)-3); j4loop += 4 {
				j4 := j4loop - 1
				z[j4-2] = d + z[j4-1]
				tmp := z[j4+1] / z[j4-2]
				d = d*tmp - tau
				if d < dthresh {
					d = 0
				}
				dmin = math.Min(dmin, d)
				z[j4] = z[j4-1] * tmp
				emin = math.Min(z[j4], emin)
			}
		} else {
			for j4loop := 4 * (i0 + 1); j4loop <= 4*((n0+1)-3); j4loop += 4 {
				j4 := j4loop - 1
				z[j4-3] = d + z[j4]
				tmp := z[j4+2] / z[j4-3]
				d = d*tmp - tau
				if d < dthresh {
					d = 0
				}
				dmin = math.Min(dmin, d)
				z[j4-1] = z[j4] * tmp
				emin = math.Min(z[j4-1], emin)
			}
		}
		// Unroll the last two steps.
		dnm2 = d
		dmin2 = dmin
		j4 = 4*((n0+1)-2) - pp - 1
		j4p2 := j4 + 2*pp - 1
		z[j4-2] = dnm2 + z[j4p2]
		z[j4] = z[j4p2+2] * (z[j4p2] / z[j4-2])
		dnm1 = z[j4p2+2]*(dnm2/z[j4-2]) - tau
		dmin = math.Min(dmin, dnm1)

		dmin1 = dmin
		j4 += 4
		j4p2 = j4 + 2*pp - 1
		z[j4-2] = dnm1 + z[j4p2]
		z[j4] = z[j4p2+2] * (z[j4p2] / z[j4-2])
		dn = z[j4p2+2]*(dnm1/z[j4-2]) - tau
		dmin = math.Min(dmin, dn)
	}
	z[j4+2] = dn
	z[4*(n0+1)-pp-1] = emin
	return i0, n0, pp, tau, sigma, dmin, dmin1, dmin2, dn, dnm1, dnm2
}
