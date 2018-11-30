// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import "math"

// Dlasv2 computes the singular value decomposition of a 2×2 matrix.
//  [ csl snl] [f g] [csr -snr] = [ssmax     0]
//  [-snl csl] [0 h] [snr  csr] = [    0 ssmin]
// ssmax is the larger absolute singular value, and ssmin is the smaller absolute
// singular value. [cls, snl] and [csr, snr] are the left and right singular vectors.
//
// Dlasv2 is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dlasv2(f, g, h float64) (ssmin, ssmax, snr, csr, snl, csl float64) {
	ft := f
	fa := math.Abs(ft)
	ht := h
	ha := math.Abs(h)
	// pmax points to the largest element of the matrix in terms of absolute value.
	// 1 if F, 2 if G, 3 if H.
	pmax := 1
	swap := ha > fa
	if swap {
		pmax = 3
		ft, ht = ht, ft
		fa, ha = ha, fa
	}
	gt := g
	ga := math.Abs(gt)
	var clt, crt, slt, srt float64
	if ga == 0 {
		ssmin = ha
		ssmax = fa
		clt = 1
		crt = 1
		slt = 0
		srt = 0
	} else {
		gasmall := true
		if ga > fa {
			pmax = 2
			if (fa / ga) < dlamchE {
				gasmall = false
				ssmax = ga
				if ha > 1 {
					ssmin = fa / (ga / ha)
				} else {
					ssmin = (fa / ga) * ha
				}
				clt = 1
				slt = ht / gt
				srt = 1
				crt = ft / gt
			}
		}
		if gasmall {
			d := fa - ha
			l := d / fa
			if d == fa { // deal with inf
				l = 1
			}
			m := gt / ft
			t := 2 - l
			s := math.Hypot(t, m)
			var r float64
			if l == 0 {
				r = math.Abs(m)
			} else {
				r = math.Hypot(l, m)
			}
			a := 0.5 * (s + r)
			ssmin = ha / a
			ssmax = fa * a
			if m == 0 {
				if l == 0 {
					t = math.Copysign(2, ft) * math.Copysign(1, gt)
				} else {
					t = gt/math.Copysign(d, ft) + m/t
				}
			} else {
				t = (m/(s+t) + m/(r+l)) * (1 + a)
			}
			l = math.Hypot(t, 2)
			crt = 2 / l
			srt = t / l
			clt = (crt + srt*m) / a
			slt = (ht / ft) * srt / a
		}
	}
	if swap {
		csl = srt
		snl = crt
		csr = slt
		snr = clt
	} else {
		csl = clt
		snl = slt
		csr = crt
		snr = srt
	}
	var tsign float64
	switch pmax {
	case 1:
		tsign = math.Copysign(1, csr) * math.Copysign(1, csl) * math.Copysign(1, f)
	case 2:
		tsign = math.Copysign(1, snr) * math.Copysign(1, csl) * math.Copysign(1, g)
	case 3:
		tsign = math.Copysign(1, snr) * math.Copysign(1, snl) * math.Copysign(1, h)
	}
	ssmax = math.Copysign(ssmax, tsign)
	ssmin = math.Copysign(ssmin, tsign*math.Copysign(1, f)*math.Copysign(1, h))
	return ssmin, ssmax, snr, csr, snl, csl
}
