// Copyright ©2016 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !noasm,!gccgo,!safe

package f32

// AxpyUnitary is
//  for i, v := range x {
//  	y[i] += alpha * v
//  }
func AxpyUnitary(alpha float32, x, y []float32)

// AxpyUnitaryTo is
//  for i, v := range x {
//  	dst[i] = alpha*v + y[i]
//  }
func AxpyUnitaryTo(dst []float32, alpha float32, x, y []float32)

// AxpyInc is
//  for i := 0; i < int(n); i++ {
//  	y[iy] += alpha * x[ix]
//  	ix += incX
//  	iy += incY
//  }
func AxpyInc(alpha float32, x, y []float32, n, incX, incY, ix, iy uintptr)

// AxpyIncTo is
//  for i := 0; i < int(n); i++ {
//  	dst[idst] = alpha*x[ix] + y[iy]
//  	ix += incX
//  	iy += incY
//  	idst += incDst
//  }
func AxpyIncTo(dst []float32, incDst, idst uintptr, alpha float32, x, y []float32, n, incX, incY, ix, iy uintptr)

// DdotUnitary is
//  for i, v := range x {
//  	sum += float64(y[i]) * float64(v)
//  }
//  return
func DdotUnitary(x, y []float32) (sum float64)

// DdotInc is
//  for i := 0; i < int(n); i++ {
//  	sum += float64(y[iy]) * float64(x[ix])
//  	ix += incX
//  	iy += incY
//  }
//  return
func DdotInc(x, y []float32, n, incX, incY, ix, iy uintptr) (sum float64)

// DotUnitary is
//  for i, v := range x {
//  	sum += y[i] * v
//  }
//  return sum
func DotUnitary(x, y []float32) (sum float32)

// DotInc is
//  for i := 0; i < int(n); i++ {
//  	sum += y[iy] * x[ix]
//  	ix += incX
//  	iy += incY
//  }
//  return sum
func DotInc(x, y []float32, n, incX, incY, ix, iy uintptr) (sum float32)

// Sum is
//  var sum float32
//  for _, v := range x {
// 		sum += v
//  }
//  return sum
func Sum(x []float32) float32
