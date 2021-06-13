// Copyright ©2019 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import "math"

// Dcombssq adds two scaled sum-of-squares quantities, V := V1 + V2,
//  V_scale^2 * V_ssq := V1_scale^2 * V1_ssq + V2_scale^2 * V2_ssq
// and returns the result V.
//
// Dcombssq is an internal routine. It is exported for testing purposes.
func (Implementation) Dcombssq(scale1, ssq1, scale2, ssq2 float64) (scale, ssq float64) {
	if scale1 >= scale2 {
		if scale1 != 0 {
			return scale1, ssq1 + (scale2/scale1)*(scale2/scale1)*ssq2
		}
		// Both scales are zero.
		if math.IsNaN(ssq1) || math.IsNaN(ssq2) {
			return 0, math.NaN()
		}
		return 0, 0
	}
	return scale2, ssq2 + (scale1/scale2)*(scale1/scale2)*ssq1
}
