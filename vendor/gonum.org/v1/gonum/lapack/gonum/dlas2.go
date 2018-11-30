// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import "math"

// Dlas2 computes the singular values of the 2×2 matrix defined by
//  [F G]
//  [0 H]
// The smaller and larger singular values are returned in that order.
//
// Dlas2 is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dlas2(f, g, h float64) (ssmin, ssmax float64) {
	fa := math.Abs(f)
	ga := math.Abs(g)
	ha := math.Abs(h)
	fhmin := math.Min(fa, ha)
	fhmax := math.Max(fa, ha)
	if fhmin == 0 {
		if fhmax == 0 {
			return 0, ga
		}
		v := math.Min(fhmax, ga) / math.Max(fhmax, ga)
		return 0, math.Max(fhmax, ga) * math.Sqrt(1+v*v)
	}
	if ga < fhmax {
		as := 1 + fhmin/fhmax
		at := (fhmax - fhmin) / fhmax
		au := (ga / fhmax) * (ga / fhmax)
		c := 2 / (math.Sqrt(as*as+au) + math.Sqrt(at*at+au))
		return fhmin * c, fhmax / c
	}
	au := fhmax / ga
	if au == 0 {
		return fhmin * fhmax / ga, ga
	}
	as := 1 + fhmin/fhmax
	at := (fhmax - fhmin) / fhmax
	c := 1 / (math.Sqrt(1+(as*au)*(as*au)) + math.Sqrt(1+(at*au)*(at*au)))
	return 2 * (fhmin * c) * au, ga / (c + c)
}
