// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !noasm,!appengine,!safe

package f64

// L1Norm is
//  for _, v := range x {
//  	sum += math.Abs(v)
//  }
//  return sum
func L1Norm(x []float64) (sum float64)

// L1NormInc is
//  for i := 0; i < n*incX; i += incX {
//  	sum += math.Abs(x[i])
//  }
//  return sum
func L1NormInc(x []float64, n, incX int) (sum float64)

// AddConst is
//  for i := range x {
//  	x[i] += alpha
//  }
func AddConst(alpha float64, x []float64)

// Add is
//  for i, v := range s {
//  	dst[i] += v
//  }
func Add(dst, s []float64)

// AxpyUnitary is
//  for i, v := range x {
//  	y[i] += alpha * v
//  }
func AxpyUnitary(alpha float64, x, y []float64)

// AxpyUnitaryTo is
//  for i, v := range x {
//  	dst[i] = alpha*v + y[i]
//  }
func AxpyUnitaryTo(dst []float64, alpha float64, x, y []float64)

// AxpyInc is
//  for i := 0; i < int(n); i++ {
//  	y[iy] += alpha * x[ix]
//  	ix += incX
//  	iy += incY
//  }
func AxpyInc(alpha float64, x, y []float64, n, incX, incY, ix, iy uintptr)

// AxpyIncTo is
//  for i := 0; i < int(n); i++ {
//  	dst[idst] = alpha*x[ix] + y[iy]
//  	ix += incX
//  	iy += incY
//  	idst += incDst
//  }
func AxpyIncTo(dst []float64, incDst, idst uintptr, alpha float64, x, y []float64, n, incX, incY, ix, iy uintptr)

// CumSum is
//  if len(s) == 0 {
//  	return dst
//  }
//  dst[0] = s[0]
//  for i, v := range s[1:] {
//  	dst[i+1] = dst[i] + v
//  }
//  return dst
func CumSum(dst, s []float64) []float64

// CumProd is
//  if len(s) == 0 {
//  	return dst
//  }
//  dst[0] = s[0]
//  for i, v := range s[1:] {
//  	dst[i+1] = dst[i] * v
//  }
//  return dst
func CumProd(dst, s []float64) []float64

// Div is
//  for i, v := range s {
//  	dst[i] /= v
//  }
func Div(dst, s []float64)

// DivTo is
//  for i, v := range s {
//  	dst[i] = v / t[i]
//  }
//  return dst
func DivTo(dst, x, y []float64) []float64

// DotUnitary is
//  for i, v := range x {
//  	sum += y[i] * v
//  }
//  return sum
func DotUnitary(x, y []float64) (sum float64)

// DotInc is
//  for i := 0; i < int(n); i++ {
//  	sum += y[iy] * x[ix]
//  	ix += incX
//  	iy += incY
//  }
//  return sum
func DotInc(x, y []float64, n, incX, incY, ix, iy uintptr) (sum float64)

// L1Dist is
//  var norm float64
//  for i, v := range s {
//  	norm += math.Abs(t[i] - v)
//  }
//  return norm
func L1Dist(s, t []float64) float64

// LinfDist is
//  var norm float64
//  if len(s) == 0 {
//  	return 0
//  }
//  norm = math.Abs(t[0] - s[0])
//  for i, v := range s[1:] {
//  	absDiff := math.Abs(t[i+1] - v)
//  	if absDiff > norm || math.IsNaN(norm) {
//  		norm = absDiff
//  	}
//  }
//  return norm
func LinfDist(s, t []float64) float64

// ScalUnitary is
//  for i := range x {
//  	x[i] *= alpha
//  }
func ScalUnitary(alpha float64, x []float64)

// ScalUnitaryTo is
//  for i, v := range x {
//  	dst[i] = alpha * v
//  }
func ScalUnitaryTo(dst []float64, alpha float64, x []float64)

// ScalInc is
//  var ix uintptr
//  for i := 0; i < int(n); i++ {
//  	x[ix] *= alpha
//  	ix += incX
//  }
func ScalInc(alpha float64, x []float64, n, incX uintptr)

// ScalIncTo is
//  var idst, ix uintptr
//  for i := 0; i < int(n); i++ {
//  	dst[idst] = alpha * x[ix]
//  	ix += incX
//  	idst += incDst
//  }
func ScalIncTo(dst []float64, incDst uintptr, alpha float64, x []float64, n, incX uintptr)

// Sum is
//  var sum float64
//  for i := range x {
//      sum += x[i]
//  }
func Sum(x []float64) float64
