// Copyright ©2016 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !amd64 noasm gccgo safe

package f64

// ScalUnitary is
//  for i := range x {
//  	x[i] *= alpha
//  }
func ScalUnitary(alpha float64, x []float64) {
	for i := range x {
		x[i] *= alpha
	}
}

// ScalUnitaryTo is
//  for i, v := range x {
//  	dst[i] = alpha * v
//  }
func ScalUnitaryTo(dst []float64, alpha float64, x []float64) {
	for i, v := range x {
		dst[i] = alpha * v
	}
}

// ScalInc is
//  var ix uintptr
//  for i := 0; i < int(n); i++ {
//  	x[ix] *= alpha
//  	ix += incX
//  }
func ScalInc(alpha float64, x []float64, n, incX uintptr) {
	var ix uintptr
	for i := 0; i < int(n); i++ {
		x[ix] *= alpha
		ix += incX
	}
}

// ScalIncTo is
//  var idst, ix uintptr
//  for i := 0; i < int(n); i++ {
//  	dst[idst] = alpha * x[ix]
//  	ix += incX
//  	idst += incDst
//  }
func ScalIncTo(dst []float64, incDst uintptr, alpha float64, x []float64, n, incX uintptr) {
	var idst, ix uintptr
	for i := 0; i < int(n); i++ {
		dst[idst] = alpha * x[ix]
		ix += incX
		idst += incDst
	}
}
