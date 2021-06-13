// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !amd64 noasm gccgo safe

package f64

// DotUnitary is
//  for i, v := range x {
//  	sum += y[i] * v
//  }
//  return sum
func DotUnitary(x, y []float64) (sum float64) {
	for i, v := range x {
		sum += y[i] * v
	}
	return sum
}

// DotInc is
//  for i := 0; i < int(n); i++ {
//  	sum += y[iy] * x[ix]
//  	ix += incX
//  	iy += incY
//  }
//  return sum
func DotInc(x, y []float64, n, incX, incY, ix, iy uintptr) (sum float64) {
	for i := 0; i < int(n); i++ {
		sum += y[iy] * x[ix]
		ix += incX
		iy += incY
	}
	return sum
}
