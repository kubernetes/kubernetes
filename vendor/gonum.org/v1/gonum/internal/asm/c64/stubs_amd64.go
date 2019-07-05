// Copyright ©2016 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !noasm,!appengine,!safe

package c64

// AxpyUnitary is
//  for i, v := range x {
//  	y[i] += alpha * v
//  }
func AxpyUnitary(alpha complex64, x, y []complex64)

// AxpyUnitaryTo is
//  for i, v := range x {
//  	dst[i] = alpha*v + y[i]
//  }
func AxpyUnitaryTo(dst []complex64, alpha complex64, x, y []complex64)

// AxpyInc is
//  for i := 0; i < int(n); i++ {
//  	y[iy] += alpha * x[ix]
//  	ix += incX
//  	iy += incY
//  }
func AxpyInc(alpha complex64, x, y []complex64, n, incX, incY, ix, iy uintptr)

// AxpyIncTo is
//  for i := 0; i < int(n); i++ {
//  	dst[idst] = alpha*x[ix] + y[iy]
//  	ix += incX
//  	iy += incY
//  	idst += incDst
//  }
func AxpyIncTo(dst []complex64, incDst, idst uintptr, alpha complex64, x, y []complex64, n, incX, incY, ix, iy uintptr)

// DotcUnitary is
//  for i, v := range x {
//  	sum += y[i] * conj(v)
//  }
//  return sum
func DotcUnitary(x, y []complex64) (sum complex64)

// DotcInc is
//  for i := 0; i < int(n); i++ {
//  	sum += y[iy] * conj(x[ix])
//  	ix += incX
//  	iy += incY
//  }
//  return sum
func DotcInc(x, y []complex64, n, incX, incY, ix, iy uintptr) (sum complex64)

// DotuUnitary is
//  for i, v := range x {
//  	sum += y[i] * v
//  }
//  return sum
func DotuUnitary(x, y []complex64) (sum complex64)

// DotuInc is
//  for i := 0; i < int(n); i++ {
//  	sum += y[iy] * x[ix]
//  	ix += incX
//  	iy += incY
//  }
//  return sum
func DotuInc(x, y []complex64, n, incX, incY, ix, iy uintptr) (sum complex64)
