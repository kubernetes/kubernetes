// Copyright ©2017 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import "gonum.org/v1/gonum/blas/blas64"

// Dlapll returns the smallest singular value of the n×2 matrix A = [ x y ].
// The function first computes the QR factorization of A = Q*R, and then computes
// the SVD of the 2-by-2 upper triangular matrix r.
//
// The contents of x and y are overwritten during the call.
//
// Dlapll is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dlapll(n int, x []float64, incX int, y []float64, incY int) float64 {
	checkVector(n, x, incX)
	checkVector(n, y, incY)

	if n <= 1 {
		return 0
	}

	// Compute the QR factorization of the N-by-2 matrix [ X Y ].
	a00, tau := impl.Dlarfg(n, x[0], x[incX:], incX)
	x[0] = 1

	bi := blas64.Implementation()
	c := -tau * bi.Ddot(n, x, incX, y, incY)
	bi.Daxpy(n, c, x, incX, y, incY)
	a11, _ := impl.Dlarfg(n-1, y[incY], y[2*incY:], incY)

	// Compute the SVD of 2-by-2 upper triangular matrix.
	ssmin, _ := impl.Dlas2(a00, y[0], a11)
	return ssmin
}
