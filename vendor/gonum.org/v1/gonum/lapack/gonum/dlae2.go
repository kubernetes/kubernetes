// Copyright ©2016 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import "math"

// Dlae2 computes the eigenvalues of a 2×2 symmetric matrix
//  [a b]
//  [b c]
// and returns the eigenvalue with the larger absolute value as rt1 and the
// smaller as rt2.
//
// Dlae2 is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dlae2(a, b, c float64) (rt1, rt2 float64) {
	sm := a + c
	df := a - c
	adf := math.Abs(df)
	tb := b + b
	ab := math.Abs(tb)
	acmx := c
	acmn := a
	if math.Abs(a) > math.Abs(c) {
		acmx = a
		acmn = c
	}
	var rt float64
	if adf > ab {
		rt = adf * math.Sqrt(1+(ab/adf)*(ab/adf))
	} else if adf < ab {
		rt = ab * math.Sqrt(1+(adf/ab)*(adf/ab))
	} else {
		rt = ab * math.Sqrt2
	}
	if sm < 0 {
		rt1 = 0.5 * (sm - rt)
		rt2 = (acmx/rt1)*acmn - (b/rt1)*b
		return rt1, rt2
	}
	if sm > 0 {
		rt1 = 0.5 * (sm + rt)
		rt2 = (acmx/rt1)*acmn - (b/rt1)*b
		return rt1, rt2
	}
	rt1 = 0.5 * rt
	rt2 = -0.5 * rt
	return rt1, rt2
}
