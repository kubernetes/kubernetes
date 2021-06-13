// Copyright ©2020 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import "math"

// Dgtsv solves the equation
//  A * X = B
// where A is an n×n tridiagonal matrix. It uses Gaussian elimination with
// partial pivoting. The equation Aᵀ * X = B may be solved by swapping the
// arguments for du and dl.
//
// On entry, dl, d and du contain the sub-diagonal, the diagonal and the
// super-diagonal, respectively, of A. On return, the first n-2 elements of dl,
// the first n-1 elements of du and the first n elements of d may be
// overwritten.
//
// On entry, b contains the n×nrhs right-hand side matrix B. On return, b will
// be overwritten. If ok is true, it will be overwritten by the solution matrix X.
//
// Dgtsv returns whether the solution X has been successfuly computed.
func (impl Implementation) Dgtsv(n, nrhs int, dl, d, du []float64, b []float64, ldb int) (ok bool) {
	switch {
	case n < 0:
		panic(nLT0)
	case nrhs < 0:
		panic(nrhsLT0)
	case ldb < max(1, nrhs):
		panic(badLdB)
	}

	if n == 0 || nrhs == 0 {
		return true
	}

	switch {
	case len(dl) < n-1:
		panic(shortDL)
	case len(d) < n:
		panic(shortD)
	case len(du) < n-1:
		panic(shortDU)
	case len(b) < (n-1)*ldb+nrhs:
		panic(shortB)
	}

	dl = dl[:n-1]
	d = d[:n]
	du = du[:n-1]

	for i := 0; i < n-1; i++ {
		if math.Abs(d[i]) >= math.Abs(dl[i]) {
			// No row interchange required.
			if d[i] == 0 {
				return false
			}
			fact := dl[i] / d[i]
			d[i+1] -= fact * du[i]
			for j := 0; j < nrhs; j++ {
				b[(i+1)*ldb+j] -= fact * b[i*ldb+j]
			}
			dl[i] = 0
		} else {
			// Interchange rows i and i+1.
			fact := d[i] / dl[i]
			d[i] = dl[i]
			tmp := d[i+1]
			d[i+1] = du[i] - fact*tmp
			du[i] = tmp
			if i+1 < n-1 {
				dl[i] = du[i+1]
				du[i+1] = -fact * dl[i]
			}
			for j := 0; j < nrhs; j++ {
				tmp = b[i*ldb+j]
				b[i*ldb+j] = b[(i+1)*ldb+j]
				b[(i+1)*ldb+j] = tmp - fact*b[(i+1)*ldb+j]
			}
		}
	}
	if d[n-1] == 0 {
		return false
	}

	// Back solve with the matrix U from the factorization.
	for j := 0; j < nrhs; j++ {
		b[(n-1)*ldb+j] /= d[n-1]
		if n > 1 {
			b[(n-2)*ldb+j] = (b[(n-2)*ldb+j] - du[n-2]*b[(n-1)*ldb+j]) / d[n-2]
		}
		for i := n - 3; i >= 0; i-- {
			b[i*ldb+j] = (b[i*ldb+j] - du[i]*b[(i+1)*ldb+j] - dl[i]*b[(i+2)*ldb+j]) / d[i]
		}
	}

	return true
}
