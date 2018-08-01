// Copyright ©2017 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"math"

	"gonum.org/v1/gonum/internal/asm/c128"
)

// Dzasum returns the sum of the absolute values of the elements of x
//  \sum_i |Re(x[i])| + |Im(x[i])|
// Dzasum returns 0 if incX is negative.
func (Implementation) Dzasum(n int, x []complex128, incX int) float64 {
	if n < 0 {
		panic(nLT0)
	}
	if incX < 1 {
		if incX == 0 {
			panic(zeroIncX)
		}
		return 0
	}
	var sum float64
	if incX == 1 {
		if len(x) < n {
			panic(badX)
		}
		for _, v := range x[:n] {
			sum += dcabs1(v)
		}
		return sum
	}
	if (n-1)*incX >= len(x) {
		panic(badX)
	}
	for i := 0; i < n; i++ {
		v := x[i*incX]
		sum += dcabs1(v)
	}
	return sum
}

// Dznrm2 computes the Euclidean norm of the complex vector x,
//  ‖x‖_2 = sqrt(\sum_i x[i] * conj(x[i])).
// This function returns 0 if incX is negative.
func (Implementation) Dznrm2(n int, x []complex128, incX int) float64 {
	if incX < 1 {
		if incX == 0 {
			panic(zeroIncX)
		}
		return 0
	}
	if n < 1 {
		if n == 0 {
			return 0
		}
		panic(nLT0)
	}
	if (n-1)*incX >= len(x) {
		panic(badX)
	}
	var (
		scale float64
		ssq   float64 = 1
	)
	if incX == 1 {
		for _, v := range x[:n] {
			re, im := math.Abs(real(v)), math.Abs(imag(v))
			if re != 0 {
				if re > scale {
					ssq = 1 + ssq*(scale/re)*(scale/re)
					scale = re
				} else {
					ssq += (re / scale) * (re / scale)
				}
			}
			if im != 0 {
				if im > scale {
					ssq = 1 + ssq*(scale/im)*(scale/im)
					scale = im
				} else {
					ssq += (im / scale) * (im / scale)
				}
			}
		}
		if math.IsInf(scale, 1) {
			return math.Inf(1)
		}
		return scale * math.Sqrt(ssq)
	}
	for ix := 0; ix < n*incX; ix += incX {
		re, im := math.Abs(real(x[ix])), math.Abs(imag(x[ix]))
		if re != 0 {
			if re > scale {
				ssq = 1 + ssq*(scale/re)*(scale/re)
				scale = re
			} else {
				ssq += (re / scale) * (re / scale)
			}
		}
		if im != 0 {
			if im > scale {
				ssq = 1 + ssq*(scale/im)*(scale/im)
				scale = im
			} else {
				ssq += (im / scale) * (im / scale)
			}
		}
	}
	if math.IsInf(scale, 1) {
		return math.Inf(1)
	}
	return scale * math.Sqrt(ssq)
}

// Izamax returns the index of the first element of x having largest |Re(·)|+|Im(·)|.
// Izamax returns -1 if n is 0 or incX is negative.
func (Implementation) Izamax(n int, x []complex128, incX int) int {
	if incX < 1 {
		if incX == 0 {
			panic(zeroIncX)
		}
		// Return invalid index.
		return -1
	}
	if n < 1 {
		if n == 0 {
			// Return invalid index.
			return -1
		}
		panic(nLT0)
	}
	if len(x) <= (n-1)*incX {
		panic(badX)
	}
	idx := 0
	max := dcabs1(x[0])
	if incX == 1 {
		for i, v := range x[1:n] {
			absV := dcabs1(v)
			if absV > max {
				max = absV
				idx = i + 1
			}
		}
		return idx
	}
	ix := incX
	for i := 1; i < n; i++ {
		absV := dcabs1(x[ix])
		if absV > max {
			max = absV
			idx = i
		}
		ix += incX
	}
	return idx
}

// Zaxpy adds alpha times x to y:
//  y[i] += alpha * x[i] for all i
func (Implementation) Zaxpy(n int, alpha complex128, x []complex128, incX int, y []complex128, incY int) {
	if incX == 0 {
		panic(zeroIncX)
	}
	if incY == 0 {
		panic(zeroIncY)
	}
	if n < 1 {
		if n == 0 {
			return
		}
		panic(nLT0)
	}
	if (incX > 0 && (n-1)*incX >= len(x)) || (incX < 0 && (1-n)*incX >= len(x)) {
		panic(badX)
	}
	if (incY > 0 && (n-1)*incY >= len(y)) || (incY < 0 && (1-n)*incY >= len(y)) {
		panic(badY)
	}
	if alpha == 0 {
		return
	}
	if incX == 1 && incY == 1 {
		c128.AxpyUnitary(alpha, x[:n], y[:n])
		return
	}
	var ix, iy int
	if incX < 0 {
		ix = (1 - n) * incX
	}
	if incY < 0 {
		iy = (1 - n) * incY
	}
	c128.AxpyInc(alpha, x, y, uintptr(n), uintptr(incX), uintptr(incY), uintptr(ix), uintptr(iy))
}

// Zcopy copies the vector x to vector y.
func (Implementation) Zcopy(n int, x []complex128, incX int, y []complex128, incY int) {
	if incX == 0 {
		panic(zeroIncX)
	}
	if incY == 0 {
		panic(zeroIncY)
	}
	if n < 1 {
		if n == 0 {
			return
		}
		panic(nLT0)
	}
	if (incX > 0 && (n-1)*incX >= len(x)) || (incX < 0 && (1-n)*incX >= len(x)) {
		panic(badX)
	}
	if (incY > 0 && (n-1)*incY >= len(y)) || (incY < 0 && (1-n)*incY >= len(y)) {
		panic(badY)
	}
	if incX == 1 && incY == 1 {
		copy(y[:n], x[:n])
		return
	}
	var ix, iy int
	if incX < 0 {
		ix = (-n + 1) * incX
	}
	if incY < 0 {
		iy = (-n + 1) * incY
	}
	for i := 0; i < n; i++ {
		y[iy] = x[ix]
		ix += incX
		iy += incY
	}
}

// Zdotc computes the dot product
//  x^H · y
// of two complex vectors x and y.
func (Implementation) Zdotc(n int, x []complex128, incX int, y []complex128, incY int) complex128 {
	if incX == 0 {
		panic(zeroIncX)
	}
	if incY == 0 {
		panic(zeroIncY)
	}
	if n <= 0 {
		if n == 0 {
			return 0
		}
		panic(nLT0)
	}
	if incX == 1 && incY == 1 {
		if len(x) < n {
			panic(badX)
		}
		if len(y) < n {
			panic(badY)
		}
		return c128.DotcUnitary(x[:n], y[:n])
	}
	var ix, iy int
	if incX < 0 {
		ix = (-n + 1) * incX
	}
	if incY < 0 {
		iy = (-n + 1) * incY
	}
	if ix >= len(x) || (n-1)*incX >= len(x) {
		panic(badX)
	}
	if iy >= len(y) || (n-1)*incY >= len(y) {
		panic(badY)
	}
	return c128.DotcInc(x, y, uintptr(n), uintptr(incX), uintptr(incY), uintptr(ix), uintptr(iy))
}

// Zdotu computes the dot product
//  x^T · y
// of two complex vectors x and y.
func (Implementation) Zdotu(n int, x []complex128, incX int, y []complex128, incY int) complex128 {
	if incX == 0 {
		panic(zeroIncX)
	}
	if incY == 0 {
		panic(zeroIncY)
	}
	if n <= 0 {
		if n == 0 {
			return 0
		}
		panic(nLT0)
	}
	if incX == 1 && incY == 1 {
		if len(x) < n {
			panic(badX)
		}
		if len(y) < n {
			panic(badY)
		}
		return c128.DotuUnitary(x[:n], y[:n])
	}
	var ix, iy int
	if incX < 0 {
		ix = (-n + 1) * incX
	}
	if incY < 0 {
		iy = (-n + 1) * incY
	}
	if ix >= len(x) || (n-1)*incX >= len(x) {
		panic(badX)
	}
	if iy >= len(y) || (n-1)*incY >= len(y) {
		panic(badY)
	}
	return c128.DotuInc(x, y, uintptr(n), uintptr(incX), uintptr(incY), uintptr(ix), uintptr(iy))
}

// Zdscal scales the vector x by a real scalar alpha.
// Zdscal has no effect if incX < 0.
func (Implementation) Zdscal(n int, alpha float64, x []complex128, incX int) {
	if incX < 1 {
		if incX == 0 {
			panic(zeroIncX)
		}
		return
	}
	if (n-1)*incX >= len(x) {
		panic(badX)
	}
	if n < 1 {
		if n == 0 {
			return
		}
		panic(nLT0)
	}
	if alpha == 0 {
		if incX == 1 {
			x = x[:n]
			for i := range x {
				x[i] = 0
			}
			return
		}
		for ix := 0; ix < n*incX; ix += incX {
			x[ix] = 0
		}
		return
	}
	if incX == 1 {
		x = x[:n]
		for i, v := range x {
			x[i] = complex(alpha*real(v), alpha*imag(v))
		}
		return
	}
	for ix := 0; ix < n*incX; ix += incX {
		v := x[ix]
		x[ix] = complex(alpha*real(v), alpha*imag(v))
	}
}

// Zscal scales the vector x by a complex scalar alpha.
// Zscal has no effect if incX < 0.
func (Implementation) Zscal(n int, alpha complex128, x []complex128, incX int) {
	if incX < 1 {
		if incX == 0 {
			panic(zeroIncX)
		}
		return
	}
	if (n-1)*incX >= len(x) {
		panic(badX)
	}
	if n < 1 {
		if n == 0 {
			return
		}
		panic(nLT0)
	}
	if alpha == 0 {
		if incX == 1 {
			x = x[:n]
			for i := range x {
				x[i] = 0
			}
			return
		}
		for ix := 0; ix < n*incX; ix += incX {
			x[ix] = 0
		}
		return
	}
	if incX == 1 {
		c128.ScalUnitary(alpha, x[:n])
		return
	}
	c128.ScalInc(alpha, x, uintptr(n), uintptr(incX))
}

// Zswap exchanges the elements of two complex vectors x and y.
func (Implementation) Zswap(n int, x []complex128, incX int, y []complex128, incY int) {
	if incX == 0 {
		panic(zeroIncX)
	}
	if incY == 0 {
		panic(zeroIncY)
	}
	if n < 1 {
		if n == 0 {
			return
		}
		panic(nLT0)
	}
	if (incX > 0 && (n-1)*incX >= len(x)) || (incX < 0 && (1-n)*incX >= len(x)) {
		panic(badX)
	}
	if (incY > 0 && (n-1)*incY >= len(y)) || (incY < 0 && (1-n)*incY >= len(y)) {
		panic(badY)
	}
	if incX == 1 && incY == 1 {
		x = x[:n]
		for i, v := range x {
			x[i], y[i] = y[i], v
		}
		return
	}
	var ix, iy int
	if incX < 0 {
		ix = (-n + 1) * incX
	}
	if incY < 0 {
		iy = (-n + 1) * incY
	}
	for i := 0; i < n; i++ {
		x[ix], y[iy] = y[iy], x[ix]
		ix += incX
		iy += incY
	}
}
