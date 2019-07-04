// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import "math"

// Dlartg generates a plane rotation so that
//  [ cs sn] * [f] = [r]
//  [-sn cs]   [g] = [0]
// This is a more accurate version of BLAS drotg, with the other differences that
// if g = 0, then cs = 1 and sn = 0, and if f = 0 and g != 0, then cs = 0 and sn = 1.
// If abs(f) > abs(g), cs will be positive.
//
// Dlartg is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dlartg(f, g float64) (cs, sn, r float64) {
	safmn2 := math.Pow(dlamchB, math.Trunc(math.Log(dlamchS/dlamchE)/math.Log(dlamchB)/2))
	safmx2 := 1 / safmn2
	if g == 0 {
		cs = 1
		sn = 0
		r = f
		return cs, sn, r
	}
	if f == 0 {
		cs = 0
		sn = 1
		r = g
		return cs, sn, r
	}
	f1 := f
	g1 := g
	scale := math.Max(math.Abs(f1), math.Abs(g1))
	if scale >= safmx2 {
		var count int
		for {
			count++
			f1 *= safmn2
			g1 *= safmn2
			scale = math.Max(math.Abs(f1), math.Abs(g1))
			if scale < safmx2 {
				break
			}
		}
		r = math.Sqrt(f1*f1 + g1*g1)
		cs = f1 / r
		sn = g1 / r
		for i := 0; i < count; i++ {
			r *= safmx2
		}
	} else if scale <= safmn2 {
		var count int
		for {
			count++
			f1 *= safmx2
			g1 *= safmx2
			scale = math.Max(math.Abs(f1), math.Abs(g1))
			if scale >= safmn2 {
				break
			}
		}
		r = math.Sqrt(f1*f1 + g1*g1)
		cs = f1 / r
		sn = g1 / r
		for i := 0; i < count; i++ {
			r *= safmn2
		}
	} else {
		r = math.Sqrt(f1*f1 + g1*g1)
		cs = f1 / r
		sn = g1 / r
	}
	if math.Abs(f) > math.Abs(g) && cs < 0 {
		cs *= -1
		sn *= -1
		r *= -1
	}
	return cs, sn, r
}
