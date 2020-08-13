// Copyright ©2019 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

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
		// If the input is non-negative and we are here, then scale2 must inevitably be 0, too.
		return 0, 0
	}
	return scale2, ssq2 + (scale1/scale2)*(scale1/scale2)*ssq1
}
