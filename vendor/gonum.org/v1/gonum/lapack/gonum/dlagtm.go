// Copyright ©2020 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import "gonum.org/v1/gonum/blas"

// Dlagtm performs one of the matrix-matrix operations
//  C = alpha * A * B + beta * C   if trans == blas.NoTrans
//  C = alpha * Aᵀ * B + beta * C  if trans == blas.Trans or blas.ConjTrans
// where A is an m×m tridiagonal matrix represented by its diagonals dl, d, du,
// B and C are m×n dense matrices, and alpha and beta are scalars.
func (impl Implementation) Dlagtm(trans blas.Transpose, m, n int, alpha float64, dl, d, du []float64, b []float64, ldb int, beta float64, c []float64, ldc int) {
	switch {
	case trans != blas.NoTrans && trans != blas.Trans && trans != blas.ConjTrans:
		panic(badTrans)
	case m < 0:
		panic(mLT0)
	case n < 0:
		panic(nLT0)
	case ldb < max(1, n):
		panic(badLdB)
	case ldc < max(1, n):
		panic(badLdC)
	}

	if m == 0 || n == 0 {
		return
	}

	switch {
	case len(dl) < m-1:
		panic(shortDL)
	case len(d) < m:
		panic(shortD)
	case len(du) < m-1:
		panic(shortDU)
	case len(b) < (m-1)*ldb+n:
		panic(shortB)
	case len(c) < (m-1)*ldc+n:
		panic(shortC)
	}

	if beta != 1 {
		if beta == 0 {
			for i := 0; i < m; i++ {
				ci := c[i*ldc : i*ldc+n]
				for j := range ci {
					ci[j] = 0
				}
			}
		} else {
			for i := 0; i < m; i++ {
				ci := c[i*ldc : i*ldc+n]
				for j := range ci {
					ci[j] *= beta
				}
			}
		}
	}

	if alpha == 0 {
		return
	}

	if m == 1 {
		if alpha == 1 {
			for j := 0; j < n; j++ {
				c[j] += d[0] * b[j]
			}
		} else {
			for j := 0; j < n; j++ {
				c[j] += alpha * d[0] * b[j]
			}
		}
		return
	}

	if trans != blas.NoTrans {
		dl, du = du, dl
	}

	if alpha == 1 {
		for j := 0; j < n; j++ {
			c[j] += d[0]*b[j] + du[0]*b[ldb+j]
		}
		for i := 1; i < m-1; i++ {
			for j := 0; j < n; j++ {
				c[i*ldc+j] += dl[i-1]*b[(i-1)*ldb+j] + d[i]*b[i*ldb+j] + du[i]*b[(i+1)*ldb+j]
			}
		}
		for j := 0; j < n; j++ {
			c[(m-1)*ldc+j] += dl[m-2]*b[(m-2)*ldb+j] + d[m-1]*b[(m-1)*ldb+j]
		}
	} else {
		for j := 0; j < n; j++ {
			c[j] += alpha * (d[0]*b[j] + du[0]*b[ldb+j])
		}
		for i := 1; i < m-1; i++ {
			for j := 0; j < n; j++ {
				c[i*ldc+j] += alpha * (dl[i-1]*b[(i-1)*ldb+j] + d[i]*b[i*ldb+j] + du[i]*b[(i+1)*ldb+j])
			}
		}
		for j := 0; j < n; j++ {
			c[(m-1)*ldc+j] += alpha * (dl[m-2]*b[(m-2)*ldb+j] + d[m-1]*b[(m-1)*ldb+j])
		}
	}
}
