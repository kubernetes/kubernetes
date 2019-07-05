// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import "math"

// Dlasq6 computes one dqd transform in ping-pong form with protection against
// overflow and underflow. z has length at least 4*(n0+1) and holds the qd array.
// i0 is the zero-based first index.
// n0 is the zero-based last index.
//
// Dlasq6 is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dlasq6(i0, n0 int, z []float64, pp int) (dmin, dmin1, dmin2, dn, dnm1, dnm2 float64) {
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

	if n0-i0-1 <= 0 {
		return dmin, dmin1, dmin2, dn, dnm1, dnm2
	}

	safmin := dlamchS
	j4 := 4*(i0+1) + pp - 4 // -4 rather than -3 for zero indexing
	emin := z[j4+4]
	d := z[j4]
	dmin = d
	if pp == 0 {
		for j4loop := 4 * (i0 + 1); j4loop <= 4*((n0+1)-3); j4loop += 4 {
			j4 := j4loop - 1 // Translate back to zero-indexed.
			z[j4-2] = d + z[j4-1]
			if z[j4-2] == 0 {
				z[j4] = 0
				d = z[j4+1]
				dmin = d
				emin = 0
			} else if safmin*z[j4+1] < z[j4-2] && safmin*z[j4-2] < z[j4+1] {
				tmp := z[j4+1] / z[j4-2]
				z[j4] = z[j4-1] * tmp
				d *= tmp
			} else {
				z[j4] = z[j4+1] * (z[j4-1] / z[j4-2])
				d = z[j4+1] * (d / z[j4-2])
			}
			dmin = math.Min(dmin, d)
			emin = math.Min(emin, z[j4])
		}
	} else {
		for j4loop := 4 * (i0 + 1); j4loop <= 4*((n0+1)-3); j4loop += 4 {
			j4 := j4loop - 1
			z[j4-3] = d + z[j4]
			if z[j4-3] == 0 {
				z[j4-1] = 0
				d = z[j4+2]
				dmin = d
				emin = 0
			} else if safmin*z[j4+2] < z[j4-3] && safmin*z[j4-3] < z[j4+2] {
				tmp := z[j4+2] / z[j4-3]
				z[j4-1] = z[j4] * tmp
				d *= tmp
			} else {
				z[j4-1] = z[j4+2] * (z[j4] / z[j4-3])
				d = z[j4+2] * (d / z[j4-3])
			}
			dmin = math.Min(dmin, d)
			emin = math.Min(emin, z[j4-1])
		}
	}
	// Unroll last two steps.
	dnm2 = d
	dmin2 = dmin
	j4 = 4*(n0-1) - pp - 1
	j4p2 := j4 + 2*pp - 1
	z[j4-2] = dnm2 + z[j4p2]
	if z[j4-2] == 0 {
		z[j4] = 0
		dnm1 = z[j4p2+2]
		dmin = dnm1
		emin = 0
	} else if safmin*z[j4p2+2] < z[j4-2] && safmin*z[j4-2] < z[j4p2+2] {
		tmp := z[j4p2+2] / z[j4-2]
		z[j4] = z[j4p2] * tmp
		dnm1 = dnm2 * tmp
	} else {
		z[j4] = z[j4p2+2] * (z[j4p2] / z[j4-2])
		dnm1 = z[j4p2+2] * (dnm2 / z[j4-2])
	}
	dmin = math.Min(dmin, dnm1)
	dmin1 = dmin
	j4 += 4
	j4p2 = j4 + 2*pp - 1
	z[j4-2] = dnm1 + z[j4p2]
	if z[j4-2] == 0 {
		z[j4] = 0
		dn = z[j4p2+2]
		dmin = dn
		emin = 0
	} else if safmin*z[j4p2+2] < z[j4-2] && safmin*z[j4-2] < z[j4p2+2] {
		tmp := z[j4p2+2] / z[j4-2]
		z[j4] = z[j4p2] * tmp
		dn = dnm1 * tmp
	} else {
		z[j4] = z[j4p2+2] * (z[j4p2] / z[j4-2])
		dn = z[j4p2+2] * (dnm1 / z[j4-2])
	}
	dmin = math.Min(dmin, dn)
	z[j4+2] = dn
	z[4*(n0+1)-pp-1] = emin
	return dmin, dmin1, dmin2, dn, dnm1, dnm2
}
