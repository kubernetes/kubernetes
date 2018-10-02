// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import "math"

// Dlaev2 computes the Eigen decomposition of a symmetric 2×2 matrix.
// The matrix is given by
//  [a b]
//  [b c]
// Dlaev2 returns rt1 and rt2, the eigenvalues of the matrix where |RT1| > |RT2|,
// and [cs1, sn1] which is the unit right eigenvalue for RT1.
//  [ cs1 sn1] [a b] [cs1 -sn1] = [rt1   0]
//  [-sn1 cs1] [b c] [sn1  cs1]   [  0 rt2]
//
// Dlaev2 is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dlaev2(a, b, c float64) (rt1, rt2, cs1, sn1 float64) {
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
		rt = ab * math.Sqrt(2)
	}
	var sgn1 float64
	if sm < 0 {
		rt1 = 0.5 * (sm - rt)
		sgn1 = -1
		rt2 = (acmx/rt1)*acmn - (b/rt1)*b
	} else if sm > 0 {
		rt1 = 0.5 * (sm + rt)
		sgn1 = 1
		rt2 = (acmx/rt1)*acmn - (b/rt1)*b
	} else {
		rt1 = 0.5 * rt
		rt2 = -0.5 * rt
		sgn1 = 1
	}
	var cs, sgn2 float64
	if df >= 0 {
		cs = df + rt
		sgn2 = 1
	} else {
		cs = df - rt
		sgn2 = -1
	}
	acs := math.Abs(cs)
	if acs > ab {
		ct := -tb / cs
		sn1 = 1 / math.Sqrt(1+ct*ct)
		cs1 = ct * sn1
	} else {
		if ab == 0 {
			cs1 = 1
			sn1 = 0
		} else {
			tn := -cs / tb
			cs1 = 1 / math.Sqrt(1+tn*tn)
			sn1 = tn * cs1
		}
	}
	if sgn1 == sgn2 {
		tn := cs1
		cs1 = -sn1
		sn1 = tn
	}
	return rt1, rt2, cs1, sn1
}
