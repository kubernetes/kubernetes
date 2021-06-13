// Copyright ©2017 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !amd64 noasm gccgo safe

package f32

// Ger performs the rank-one operation
//  A += alpha * x * yᵀ
// where A is an m×n dense matrix, x and y are vectors, and alpha is a scalar.
func Ger(m, n uintptr, alpha float32, x []float32, incX uintptr, y []float32, incY uintptr, a []float32, lda uintptr) {

	if incX == 1 && incY == 1 {
		x = x[:m]
		y = y[:n]
		for i, xv := range x {
			AxpyUnitary(alpha*xv, y, a[uintptr(i)*lda:uintptr(i)*lda+n])
		}
		return
	}

	var ky, kx uintptr
	if int(incY) < 0 {
		ky = uintptr(-int(n-1) * int(incY))
	}
	if int(incX) < 0 {
		kx = uintptr(-int(m-1) * int(incX))
	}

	ix := kx
	for i := 0; i < int(m); i++ {
		AxpyInc(alpha*x[ix], y, a[uintptr(i)*lda:uintptr(i)*lda+n], uintptr(n), uintptr(incY), 1, uintptr(ky), 0)
		ix += incX
	}
}
