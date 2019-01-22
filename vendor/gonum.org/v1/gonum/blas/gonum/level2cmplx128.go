// Copyright ©2017 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"math/cmplx"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/internal/asm/c128"
)

// Zgbmv performs one of the matrix-vector operations
//  y = alpha * A * x + beta * y    if trans = blas.NoTrans
//  y = alpha * A^T * x + beta * y  if trans = blas.Trans
//  y = alpha * A^H * x + beta * y  if trans = blas.ConjTrans
// where alpha and beta are scalars, x and y are vectors, and A is an m×n band matrix
// with kL sub-diagonals and kU super-diagonals.
func (Implementation) Zgbmv(trans blas.Transpose, m, n, kL, kU int, alpha complex128, ab []complex128, ldab int, x []complex128, incX int, beta complex128, y []complex128, incY int) {
	checkZBandMatrix('A', m, n, kL, kU, ab, ldab)
	var lenX, lenY int
	switch trans {
	default:
		panic(badTranspose)
	case blas.NoTrans:
		lenX = n
		lenY = m
	case blas.Trans, blas.ConjTrans:
		lenX = m
		lenY = n
	}
	checkZVector('x', lenX, x, incX)
	checkZVector('y', lenY, y, incY)

	if m == 0 || n == 0 || (alpha == 0 && beta == 1) {
		return
	}

	var kx int
	if incX < 0 {
		kx = (1 - lenX) * incX
	}
	var ky int
	if incY < 0 {
		ky = (1 - lenY) * incY
	}

	// Form y = beta*y.
	if beta != 1 {
		if incY == 1 {
			if beta == 0 {
				for i := range y[:lenY] {
					y[i] = 0
				}
			} else {
				c128.ScalUnitary(beta, y[:lenY])
			}
		} else {
			iy := ky
			if beta == 0 {
				for i := 0; i < lenY; i++ {
					y[iy] = 0
					iy += incY
				}
			} else {
				if incY > 0 {
					c128.ScalInc(beta, y, uintptr(lenY), uintptr(incY))
				} else {
					c128.ScalInc(beta, y, uintptr(lenY), uintptr(-incY))
				}
			}
		}
	}

	nRow := min(m, n+kL)
	nCol := kL + 1 + kU
	switch trans {
	case blas.NoTrans:
		iy := ky
		if incX == 1 {
			for i := 0; i < nRow; i++ {
				l := max(0, kL-i)
				u := min(nCol, n+kL-i)
				aRow := ab[i*ldab+l : i*ldab+u]
				off := max(0, i-kL)
				xtmp := x[off : off+u-l]
				var sum complex128
				for j, v := range aRow {
					sum += xtmp[j] * v
				}
				y[iy] += alpha * sum
				iy += incY
			}
		} else {
			for i := 0; i < nRow; i++ {
				l := max(0, kL-i)
				u := min(nCol, n+kL-i)
				aRow := ab[i*ldab+l : i*ldab+u]
				off := max(0, i-kL) * incX
				jx := kx
				var sum complex128
				for _, v := range aRow {
					sum += x[off+jx] * v
					jx += incX
				}
				y[iy] += alpha * sum
				iy += incY
			}
		}
	case blas.Trans:
		if incX == 1 {
			for i := 0; i < nRow; i++ {
				l := max(0, kL-i)
				u := min(nCol, n+kL-i)
				aRow := ab[i*ldab+l : i*ldab+u]
				off := max(0, i-kL) * incY
				alphaxi := alpha * x[i]
				jy := ky
				for _, v := range aRow {
					y[off+jy] += alphaxi * v
					jy += incY
				}
			}
		} else {
			ix := kx
			for i := 0; i < nRow; i++ {
				l := max(0, kL-i)
				u := min(nCol, n+kL-i)
				aRow := ab[i*ldab+l : i*ldab+u]
				off := max(0, i-kL) * incY
				alphaxi := alpha * x[ix]
				jy := ky
				for _, v := range aRow {
					y[off+jy] += alphaxi * v
					jy += incY
				}
				ix += incX
			}
		}
	case blas.ConjTrans:
		if incX == 1 {
			for i := 0; i < nRow; i++ {
				l := max(0, kL-i)
				u := min(nCol, n+kL-i)
				aRow := ab[i*ldab+l : i*ldab+u]
				off := max(0, i-kL) * incY
				alphaxi := alpha * x[i]
				jy := ky
				for _, v := range aRow {
					y[off+jy] += alphaxi * cmplx.Conj(v)
					jy += incY
				}
			}
		} else {
			ix := kx
			for i := 0; i < nRow; i++ {
				l := max(0, kL-i)
				u := min(nCol, n+kL-i)
				aRow := ab[i*ldab+l : i*ldab+u]
				off := max(0, i-kL) * incY
				alphaxi := alpha * x[ix]
				jy := ky
				for _, v := range aRow {
					y[off+jy] += alphaxi * cmplx.Conj(v)
					jy += incY
				}
				ix += incX
			}
		}
	}
}

// Zgemv performs one of the matrix-vector operations
//  y = alpha * A * x + beta * y    if trans = blas.NoTrans
//  y = alpha * A^T * x + beta * y  if trans = blas.Trans
//  y = alpha * A^H * x + beta * y  if trans = blas.ConjTrans
// where alpha and beta are scalars, x and y are vectors, and A is an m×n dense matrix.
func (Implementation) Zgemv(trans blas.Transpose, m, n int, alpha complex128, a []complex128, lda int, x []complex128, incX int, beta complex128, y []complex128, incY int) {
	checkZMatrix('A', m, n, a, lda)
	switch trans {
	default:
		panic(badTranspose)
	case blas.NoTrans:
		checkZVector('x', n, x, incX)
		checkZVector('y', m, y, incY)
	case blas.Trans, blas.ConjTrans:
		checkZVector('x', m, x, incX)
		checkZVector('y', n, y, incY)
	}

	if m == 0 || n == 0 || (alpha == 0 && beta == 1) {
		return
	}

	var lenX, lenY int
	if trans == blas.NoTrans {
		lenX = n
		lenY = m
	} else {
		lenX = m
		lenY = n
	}
	var kx int
	if incX < 0 {
		kx = (1 - lenX) * incX
	}
	var ky int
	if incY < 0 {
		ky = (1 - lenY) * incY
	}

	// Form y = beta*y.
	if beta != 1 {
		if incY == 1 {
			if beta == 0 {
				for i := range y[:lenY] {
					y[i] = 0
				}
			} else {
				c128.ScalUnitary(beta, y[:lenY])
			}
		} else {
			iy := ky
			if beta == 0 {
				for i := 0; i < lenY; i++ {
					y[iy] = 0
					iy += incY
				}
			} else {
				if incY > 0 {
					c128.ScalInc(beta, y, uintptr(lenY), uintptr(incY))
				} else {
					c128.ScalInc(beta, y, uintptr(lenY), uintptr(-incY))
				}
			}
		}
	}

	if alpha == 0 {
		return
	}

	switch trans {
	default:
		// Form y = alpha*A*x + y.
		iy := ky
		if incX == 1 {
			for i := 0; i < m; i++ {
				y[iy] += alpha * c128.DotuUnitary(a[i*lda:i*lda+n], x[:n])
				iy += incY
			}
			return
		}
		for i := 0; i < m; i++ {
			y[iy] += alpha * c128.DotuInc(a[i*lda:i*lda+n], x, uintptr(n), 1, uintptr(incX), 0, uintptr(kx))
			iy += incY
		}
		return

	case blas.Trans:
		// Form y = alpha*A^T*x + y.
		ix := kx
		if incY == 1 {
			for i := 0; i < m; i++ {
				c128.AxpyUnitary(alpha*x[ix], a[i*lda:i*lda+n], y[:n])
				ix += incX
			}
			return
		}
		for i := 0; i < m; i++ {
			c128.AxpyInc(alpha*x[ix], a[i*lda:i*lda+n], y, uintptr(n), 1, uintptr(incY), 0, uintptr(ky))
			ix += incX
		}
		return

	case blas.ConjTrans:
		// Form y = alpha*A^H*x + y.
		ix := kx
		if incY == 1 {
			for i := 0; i < m; i++ {
				tmp := alpha * x[ix]
				for j := 0; j < n; j++ {
					y[j] += tmp * cmplx.Conj(a[i*lda+j])
				}
				ix += incX
			}
			return
		}
		for i := 0; i < m; i++ {
			tmp := alpha * x[ix]
			jy := ky
			for j := 0; j < n; j++ {
				y[jy] += tmp * cmplx.Conj(a[i*lda+j])
				jy += incY
			}
			ix += incX
		}
		return
	}
}

// Zgerc performs the rank-one operation
//  A += alpha * x * y^H
// where A is an m×n dense matrix, alpha is a scalar, x is an m element vector,
// and y is an n element vector.
func (Implementation) Zgerc(m, n int, alpha complex128, x []complex128, incX int, y []complex128, incY int, a []complex128, lda int) {
	checkZMatrix('A', m, n, a, lda)
	checkZVector('x', m, x, incX)
	checkZVector('y', n, y, incY)

	if m == 0 || n == 0 || alpha == 0 {
		return
	}

	var kx, jy int
	if incX < 0 {
		kx = (1 - m) * incX
	}
	if incY < 0 {
		jy = (1 - n) * incY
	}
	for j := 0; j < n; j++ {
		if y[jy] != 0 {
			tmp := alpha * cmplx.Conj(y[jy])
			c128.AxpyInc(tmp, x, a[j:], uintptr(m), uintptr(incX), uintptr(lda), uintptr(kx), 0)
		}
		jy += incY
	}
}

// Zgeru performs the rank-one operation
//  A += alpha * x * y^T
// where A is an m×n dense matrix, alpha is a scalar, x is an m element vector,
// and y is an n element vector.
func (Implementation) Zgeru(m, n int, alpha complex128, x []complex128, incX int, y []complex128, incY int, a []complex128, lda int) {
	checkZMatrix('A', m, n, a, lda)
	checkZVector('x', m, x, incX)
	checkZVector('y', n, y, incY)

	if m == 0 || n == 0 || alpha == 0 {
		return
	}

	var kx int
	if incX < 0 {
		kx = (1 - m) * incX
	}
	if incY == 1 {
		for i := 0; i < m; i++ {
			if x[kx] != 0 {
				tmp := alpha * x[kx]
				c128.AxpyUnitary(tmp, y[:n], a[i*lda:i*lda+n])
			}
			kx += incX
		}
		return
	}
	var jy int
	if incY < 0 {
		jy = (1 - n) * incY
	}
	for i := 0; i < m; i++ {
		if x[kx] != 0 {
			tmp := alpha * x[kx]
			c128.AxpyInc(tmp, y, a[i*lda:i*lda+n], uintptr(n), uintptr(incY), 1, uintptr(jy), 0)
		}
		kx += incX
	}
}

// Zhbmv performs the matrix-vector operation
//  y = alpha * A * x + beta * y
// where alpha and beta are scalars, x and y are vectors, and A is an n×n
// Hermitian band matrix with k super-diagonals. The imaginary parts of
// the diagonal elements of A are ignored and assumed to be zero.
func (Implementation) Zhbmv(uplo blas.Uplo, n, k int, alpha complex128, ab []complex128, ldab int, x []complex128, incX int, beta complex128, y []complex128, incY int) {
	if uplo != blas.Upper && uplo != blas.Lower {
		panic(badUplo)
	}
	checkZhbMatrix('A', n, k, ab, ldab)
	checkZVector('x', n, x, incX)
	checkZVector('y', n, y, incY)

	if n == 0 || (alpha == 0 && beta == 1) {
		return
	}

	// Set up the start indices in X and Y.
	var kx int
	if incX < 0 {
		kx = (1 - n) * incX
	}
	var ky int
	if incY < 0 {
		ky = (1 - n) * incY
	}

	// Form y = beta*y.
	if beta != 1 {
		if incY == 1 {
			if beta == 0 {
				for i := range y[:n] {
					y[i] = 0
				}
			} else {
				for i, v := range y[:n] {
					y[i] = beta * v
				}
			}
		} else {
			iy := ky
			if beta == 0 {
				for i := 0; i < n; i++ {
					y[iy] = 0
					iy += incY
				}
			} else {
				for i := 0; i < n; i++ {
					y[iy] = beta * y[iy]
					iy += incY
				}
			}
		}
	}

	if alpha == 0 {
		return
	}

	// The elements of A are accessed sequentially with one pass through ab.
	switch uplo {
	case blas.Upper:
		iy := ky
		if incX == 1 {
			for i := 0; i < n; i++ {
				aRow := ab[i*ldab:]
				alphaxi := alpha * x[i]
				sum := alphaxi * complex(real(aRow[0]), 0)
				u := min(k+1, n-i)
				jy := incY
				for j := 1; j < u; j++ {
					v := aRow[j]
					sum += alpha * x[i+j] * v
					y[iy+jy] += alphaxi * cmplx.Conj(v)
					jy += incY
				}
				y[iy] += sum
				iy += incY
			}
		} else {
			ix := kx
			for i := 0; i < n; i++ {
				aRow := ab[i*ldab:]
				alphaxi := alpha * x[ix]
				sum := alphaxi * complex(real(aRow[0]), 0)
				u := min(k+1, n-i)
				jx := incX
				jy := incY
				for j := 1; j < u; j++ {
					v := aRow[j]
					sum += alpha * x[ix+jx] * v
					y[iy+jy] += alphaxi * cmplx.Conj(v)
					jx += incX
					jy += incY
				}
				y[iy] += sum
				ix += incX
				iy += incY
			}
		}
	case blas.Lower:
		iy := ky
		if incX == 1 {
			for i := 0; i < n; i++ {
				l := max(0, k-i)
				alphaxi := alpha * x[i]
				jy := l * incY
				aRow := ab[i*ldab:]
				for j := l; j < k; j++ {
					v := aRow[j]
					y[iy] += alpha * v * x[i-k+j]
					y[iy-k*incY+jy] += alphaxi * cmplx.Conj(v)
					jy += incY
				}
				y[iy] += alphaxi * complex(real(aRow[k]), 0)
				iy += incY
			}
		} else {
			ix := kx
			for i := 0; i < n; i++ {
				l := max(0, k-i)
				alphaxi := alpha * x[ix]
				jx := l * incX
				jy := l * incY
				aRow := ab[i*ldab:]
				for j := l; j < k; j++ {
					v := aRow[j]
					y[iy] += alpha * v * x[ix-k*incX+jx]
					y[iy-k*incY+jy] += alphaxi * cmplx.Conj(v)
					jx += incX
					jy += incY
				}
				y[iy] += alphaxi * complex(real(aRow[k]), 0)
				ix += incX
				iy += incY
			}
		}
	}
}

// Zhemv performs the matrix-vector operation
//  y = alpha * A * x + beta * y
// where alpha and beta are scalars, x and y are vectors, and A is an n×n
// Hermitian matrix. The imaginary parts of the diagonal elements of A are
// ignored and assumed to be zero.
func (Implementation) Zhemv(uplo blas.Uplo, n int, alpha complex128, a []complex128, lda int, x []complex128, incX int, beta complex128, y []complex128, incY int) {
	if uplo != blas.Upper && uplo != blas.Lower {
		panic(badUplo)
	}
	checkZMatrix('A', n, n, a, lda)
	checkZVector('x', n, x, incX)
	checkZVector('y', n, y, incY)

	if n == 0 || (alpha == 0 && beta == 1) {
		return
	}

	// Set up the start indices in X and Y.
	var kx int
	if incX < 0 {
		kx = (1 - n) * incX
	}
	var ky int
	if incY < 0 {
		ky = (1 - n) * incY
	}

	// Form y = beta*y.
	if beta != 1 {
		if incY == 1 {
			if beta == 0 {
				for i := range y[:n] {
					y[i] = 0
				}
			} else {
				for i, v := range y[:n] {
					y[i] = beta * v
				}
			}
		} else {
			iy := ky
			if beta == 0 {
				for i := 0; i < n; i++ {
					y[iy] = 0
					iy += incY
				}
			} else {
				for i := 0; i < n; i++ {
					y[iy] = beta * y[iy]
					iy += incY
				}
			}
		}
	}

	if alpha == 0 {
		return
	}

	// The elements of A are accessed sequentially with one pass through
	// the triangular part of A.

	if uplo == blas.Upper {
		// Form y when A is stored in upper triangle.
		if incX == 1 && incY == 1 {
			for i := 0; i < n; i++ {
				tmp1 := alpha * x[i]
				var tmp2 complex128
				for j := i + 1; j < n; j++ {
					y[j] += tmp1 * cmplx.Conj(a[i*lda+j])
					tmp2 += a[i*lda+j] * x[j]
				}
				aii := complex(real(a[i*lda+i]), 0)
				y[i] += tmp1*aii + alpha*tmp2
			}
		} else {
			ix := kx
			iy := ky
			for i := 0; i < n; i++ {
				tmp1 := alpha * x[ix]
				var tmp2 complex128
				jx := ix
				jy := iy
				for j := i + 1; j < n; j++ {
					jx += incX
					jy += incY
					y[jy] += tmp1 * cmplx.Conj(a[i*lda+j])
					tmp2 += a[i*lda+j] * x[jx]
				}
				aii := complex(real(a[i*lda+i]), 0)
				y[iy] += tmp1*aii + alpha*tmp2
				ix += incX
				iy += incY
			}
		}
		return
	}

	// Form y when A is stored in lower triangle.
	if incX == 1 && incY == 1 {
		for i := 0; i < n; i++ {
			tmp1 := alpha * x[i]
			var tmp2 complex128
			for j := 0; j < i; j++ {
				y[j] += tmp1 * cmplx.Conj(a[i*lda+j])
				tmp2 += a[i*lda+j] * x[j]
			}
			aii := complex(real(a[i*lda+i]), 0)
			y[i] += tmp1*aii + alpha*tmp2
		}
	} else {
		ix := kx
		iy := ky
		for i := 0; i < n; i++ {
			tmp1 := alpha * x[ix]
			var tmp2 complex128
			jx := kx
			jy := ky
			for j := 0; j < i; j++ {
				y[jy] += tmp1 * cmplx.Conj(a[i*lda+j])
				tmp2 += a[i*lda+j] * x[jx]
				jx += incX
				jy += incY
			}
			aii := complex(real(a[i*lda+i]), 0)
			y[iy] += tmp1*aii + alpha*tmp2
			ix += incX
			iy += incY
		}
	}
}

// Zher performs the Hermitian rank-one operation
//  A += alpha * x * x^H
// where A is an n×n Hermitian matrix, alpha is a real scalar, and x is an n
// element vector. On entry, the imaginary parts of the diagonal elements of A
// are ignored and assumed to be zero, on return they will be set to zero.
func (Implementation) Zher(uplo blas.Uplo, n int, alpha float64, x []complex128, incX int, a []complex128, lda int) {
	if uplo != blas.Upper && uplo != blas.Lower {
		panic(badUplo)
	}
	checkZMatrix('A', n, n, a, lda)
	checkZVector('x', n, x, incX)

	if n == 0 || alpha == 0 {
		return
	}

	var kx int
	if incX < 0 {
		kx = (1 - n) * incX
	}
	if uplo == blas.Upper {
		if incX == 1 {
			for i := 0; i < n; i++ {
				if x[i] != 0 {
					tmp := complex(alpha*real(x[i]), alpha*imag(x[i]))
					aii := real(a[i*lda+i])
					xtmp := real(tmp * cmplx.Conj(x[i]))
					a[i*lda+i] = complex(aii+xtmp, 0)
					for j := i + 1; j < n; j++ {
						a[i*lda+j] += tmp * cmplx.Conj(x[j])
					}
				} else {
					aii := real(a[i*lda+i])
					a[i*lda+i] = complex(aii, 0)
				}
			}
			return
		}

		ix := kx
		for i := 0; i < n; i++ {
			if x[ix] != 0 {
				tmp := complex(alpha*real(x[ix]), alpha*imag(x[ix]))
				aii := real(a[i*lda+i])
				xtmp := real(tmp * cmplx.Conj(x[ix]))
				a[i*lda+i] = complex(aii+xtmp, 0)
				jx := ix + incX
				for j := i + 1; j < n; j++ {
					a[i*lda+j] += tmp * cmplx.Conj(x[jx])
					jx += incX
				}
			} else {
				aii := real(a[i*lda+i])
				a[i*lda+i] = complex(aii, 0)
			}
			ix += incX
		}
		return
	}

	if incX == 1 {
		for i := 0; i < n; i++ {
			if x[i] != 0 {
				tmp := complex(alpha*real(x[i]), alpha*imag(x[i]))
				for j := 0; j < i; j++ {
					a[i*lda+j] += tmp * cmplx.Conj(x[j])
				}
				aii := real(a[i*lda+i])
				xtmp := real(tmp * cmplx.Conj(x[i]))
				a[i*lda+i] = complex(aii+xtmp, 0)
			} else {
				aii := real(a[i*lda+i])
				a[i*lda+i] = complex(aii, 0)
			}
		}
		return
	}

	ix := kx
	for i := 0; i < n; i++ {
		if x[ix] != 0 {
			tmp := complex(alpha*real(x[ix]), alpha*imag(x[ix]))
			jx := kx
			for j := 0; j < i; j++ {
				a[i*lda+j] += tmp * cmplx.Conj(x[jx])
				jx += incX
			}
			aii := real(a[i*lda+i])
			xtmp := real(tmp * cmplx.Conj(x[ix]))
			a[i*lda+i] = complex(aii+xtmp, 0)

		} else {
			aii := real(a[i*lda+i])
			a[i*lda+i] = complex(aii, 0)
		}
		ix += incX
	}
}

// Zher2 performs the Hermitian rank-two operation
//  A += alpha * x * y^H + conj(alpha) * y * x^H
// where alpha is a scalar, x and y are n element vectors and A is an n×n
// Hermitian matrix. On entry, the imaginary parts of the diagonal elements are
// ignored and assumed to be zero. On return they will be set to zero.
func (Implementation) Zher2(uplo blas.Uplo, n int, alpha complex128, x []complex128, incX int, y []complex128, incY int, a []complex128, lda int) {
	if uplo != blas.Upper && uplo != blas.Lower {
		panic(badUplo)
	}
	checkZMatrix('A', n, n, a, lda)
	checkZVector('x', n, x, incX)
	checkZVector('y', n, y, incY)

	if n == 0 || alpha == 0 {
		return
	}

	var kx, ky int
	var ix, iy int
	if incX != 1 || incY != 1 {
		if incX < 0 {
			kx = (1 - n) * incX
		}
		if incY < 0 {
			ky = (1 - n) * incY
		}
		ix = kx
		iy = ky
	}
	if uplo == blas.Upper {
		if incX == 1 && incY == 1 {
			for i := 0; i < n; i++ {
				if x[i] != 0 || y[i] != 0 {
					tmp1 := alpha * x[i]
					tmp2 := cmplx.Conj(alpha) * y[i]
					aii := real(a[i*lda+i]) + real(tmp1*cmplx.Conj(y[i])) + real(tmp2*cmplx.Conj(x[i]))
					a[i*lda+i] = complex(aii, 0)
					for j := i + 1; j < n; j++ {
						a[i*lda+j] += tmp1*cmplx.Conj(y[j]) + tmp2*cmplx.Conj(x[j])
					}
				} else {
					aii := real(a[i*lda+i])
					a[i*lda+i] = complex(aii, 0)
				}
			}
			return
		}
		for i := 0; i < n; i++ {
			if x[ix] != 0 || y[iy] != 0 {
				tmp1 := alpha * x[ix]
				tmp2 := cmplx.Conj(alpha) * y[iy]
				aii := real(a[i*lda+i]) + real(tmp1*cmplx.Conj(y[iy])) + real(tmp2*cmplx.Conj(x[ix]))
				a[i*lda+i] = complex(aii, 0)
				jx := ix + incX
				jy := iy + incY
				for j := i + 1; j < n; j++ {
					a[i*lda+j] += tmp1*cmplx.Conj(y[jy]) + tmp2*cmplx.Conj(x[jx])
					jx += incX
					jy += incY
				}
			} else {
				aii := real(a[i*lda+i])
				a[i*lda+i] = complex(aii, 0)
			}
			ix += incX
			iy += incY
		}
		return
	}

	if incX == 1 && incY == 1 {
		for i := 0; i < n; i++ {
			if x[i] != 0 || y[i] != 0 {
				tmp1 := alpha * x[i]
				tmp2 := cmplx.Conj(alpha) * y[i]
				for j := 0; j < i; j++ {
					a[i*lda+j] += tmp1*cmplx.Conj(y[j]) + tmp2*cmplx.Conj(x[j])
				}
				aii := real(a[i*lda+i]) + real(tmp1*cmplx.Conj(y[i])) + real(tmp2*cmplx.Conj(x[i]))
				a[i*lda+i] = complex(aii, 0)
			} else {
				aii := real(a[i*lda+i])
				a[i*lda+i] = complex(aii, 0)
			}
		}
		return
	}
	for i := 0; i < n; i++ {
		if x[ix] != 0 || y[iy] != 0 {
			tmp1 := alpha * x[ix]
			tmp2 := cmplx.Conj(alpha) * y[iy]
			jx := kx
			jy := ky
			for j := 0; j < i; j++ {
				a[i*lda+j] += tmp1*cmplx.Conj(y[jy]) + tmp2*cmplx.Conj(x[jx])
				jx += incX
				jy += incY
			}
			aii := real(a[i*lda+i]) + real(tmp1*cmplx.Conj(y[iy])) + real(tmp2*cmplx.Conj(x[ix]))
			a[i*lda+i] = complex(aii, 0)
		} else {
			aii := real(a[i*lda+i])
			a[i*lda+i] = complex(aii, 0)
		}
		ix += incX
		iy += incY
	}
}

// Zhpmv performs the matrix-vector operation
//  y = alpha * A * x + beta * y
// where alpha and beta are scalars, x and y are vectors, and A is an n×n
// Hermitian matrix in packed form. The imaginary parts of the diagonal
// elements of A are ignored and assumed to be zero.
func (Implementation) Zhpmv(uplo blas.Uplo, n int, alpha complex128, ap []complex128, x []complex128, incX int, beta complex128, y []complex128, incY int) {
	if uplo != blas.Upper && uplo != blas.Lower {
		panic(badUplo)
	}
	checkZVector('x', n, x, incX)
	checkZVector('y', n, y, incY)
	if len(ap) < n*(n+1)/2 {
		panic("blas: insufficient A packed matrix slice length")
	}

	if n == 0 || (alpha == 0 && beta == 1) {
		return
	}

	// Set up the start indices in X and Y.
	var kx int
	if incX < 0 {
		kx = (1 - n) * incX
	}
	var ky int
	if incY < 0 {
		ky = (1 - n) * incY
	}

	// Form y = beta*y.
	if beta != 1 {
		if incY == 1 {
			if beta == 0 {
				for i := range y[:n] {
					y[i] = 0
				}
			} else {
				for i, v := range y[:n] {
					y[i] = beta * v
				}
			}
		} else {
			iy := ky
			if beta == 0 {
				for i := 0; i < n; i++ {
					y[iy] = 0
					iy += incY
				}
			} else {
				for i := 0; i < n; i++ {
					y[iy] *= beta
					iy += incY
				}
			}
		}
	}

	if alpha == 0 {
		return
	}

	// The elements of A are accessed sequentially with one pass through ap.

	var kk int
	if uplo == blas.Upper {
		// Form y when ap contains the upper triangle.
		// Here, kk points to the current diagonal element in ap.
		if incX == 1 && incY == 1 {
			for i := 0; i < n; i++ {
				tmp1 := alpha * x[i]
				y[i] += tmp1 * complex(real(ap[kk]), 0)
				var tmp2 complex128
				k := kk + 1
				for j := i + 1; j < n; j++ {
					y[j] += tmp1 * cmplx.Conj(ap[k])
					tmp2 += ap[k] * x[j]
					k++
				}
				y[i] += alpha * tmp2
				kk += n - i
			}
		} else {
			ix := kx
			iy := ky
			for i := 0; i < n; i++ {
				tmp1 := alpha * x[ix]
				y[iy] += tmp1 * complex(real(ap[kk]), 0)
				var tmp2 complex128
				jx := ix
				jy := iy
				for k := kk + 1; k < kk+n-i; k++ {
					jx += incX
					jy += incY
					y[jy] += tmp1 * cmplx.Conj(ap[k])
					tmp2 += ap[k] * x[jx]
				}
				y[iy] += alpha * tmp2
				ix += incX
				iy += incY
				kk += n - i
			}
		}
		return
	}

	// Form y when ap contains the lower triangle.
	// Here, kk points to the beginning of current row in ap.
	if incX == 1 && incY == 1 {
		for i := 0; i < n; i++ {
			tmp1 := alpha * x[i]
			var tmp2 complex128
			k := kk
			for j := 0; j < i; j++ {
				y[j] += tmp1 * cmplx.Conj(ap[k])
				tmp2 += ap[k] * x[j]
				k++
			}
			aii := complex(real(ap[kk+i]), 0)
			y[i] += tmp1*aii + alpha*tmp2
			kk += i + 1
		}
	} else {
		ix := kx
		iy := ky
		for i := 0; i < n; i++ {
			tmp1 := alpha * x[ix]
			var tmp2 complex128
			jx := kx
			jy := ky
			for k := kk; k < kk+i; k++ {
				y[jy] += tmp1 * cmplx.Conj(ap[k])
				tmp2 += ap[k] * x[jx]
				jx += incX
				jy += incY
			}
			aii := complex(real(ap[kk+i]), 0)
			y[iy] += tmp1*aii + alpha*tmp2
			ix += incX
			iy += incY
			kk += i + 1
		}
	}
}

// Zhpr performs the Hermitian rank-1 operation
//  A += alpha * x * x^H
// where alpha is a real scalar, x is a vector, and A is an n×n hermitian matrix
// in packed form. On entry, the imaginary parts of the diagonal elements are
// assumed to be zero, and on return they are set to zero.
func (Implementation) Zhpr(uplo blas.Uplo, n int, alpha float64, x []complex128, incX int, ap []complex128) {
	if uplo != blas.Upper && uplo != blas.Lower {
		panic(badUplo)
	}
	if n < 0 {
		panic(nLT0)
	}
	checkZVector('x', n, x, incX)
	if len(ap) < n*(n+1)/2 {
		panic("blas: insufficient A packed matrix slice length")
	}

	if n == 0 || alpha == 0 {
		return
	}

	// Set up start index in X.
	var kx int
	if incX < 0 {
		kx = (1 - n) * incX
	}

	// The elements of A are accessed sequentially with one pass through ap.

	var kk int
	if uplo == blas.Upper {
		// Form A when upper triangle is stored in AP.
		// Here, kk points to the current diagonal element in ap.
		if incX == 1 {
			for i := 0; i < n; i++ {
				xi := x[i]
				if xi != 0 {
					aii := real(ap[kk]) + alpha*real(cmplx.Conj(xi)*xi)
					ap[kk] = complex(aii, 0)

					tmp := complex(alpha, 0) * xi
					a := ap[kk+1 : kk+n-i]
					x := x[i+1 : n]
					for j, v := range x {
						a[j] += tmp * cmplx.Conj(v)
					}
				} else {
					ap[kk] = complex(real(ap[kk]), 0)
				}
				kk += n - i
			}
		} else {
			ix := kx
			for i := 0; i < n; i++ {
				xi := x[ix]
				if xi != 0 {
					aii := real(ap[kk]) + alpha*real(cmplx.Conj(xi)*xi)
					ap[kk] = complex(aii, 0)

					tmp := complex(alpha, 0) * xi
					jx := ix + incX
					a := ap[kk+1 : kk+n-i]
					for k := range a {
						a[k] += tmp * cmplx.Conj(x[jx])
						jx += incX
					}
				} else {
					ap[kk] = complex(real(ap[kk]), 0)
				}
				ix += incX
				kk += n - i
			}
		}
		return
	}

	// Form A when lower triangle is stored in AP.
	// Here, kk points to the beginning of current row in ap.
	if incX == 1 {
		for i := 0; i < n; i++ {
			xi := x[i]
			if xi != 0 {
				tmp := complex(alpha, 0) * xi
				a := ap[kk : kk+i]
				for j, v := range x[:i] {
					a[j] += tmp * cmplx.Conj(v)
				}

				aii := real(ap[kk+i]) + alpha*real(cmplx.Conj(xi)*xi)
				ap[kk+i] = complex(aii, 0)
			} else {
				ap[kk+i] = complex(real(ap[kk+i]), 0)
			}
			kk += i + 1
		}
	} else {
		ix := kx
		for i := 0; i < n; i++ {
			xi := x[ix]
			if xi != 0 {
				tmp := complex(alpha, 0) * xi
				a := ap[kk : kk+i]
				jx := kx
				for k := range a {
					a[k] += tmp * cmplx.Conj(x[jx])
					jx += incX
				}

				aii := real(ap[kk+i]) + alpha*real(cmplx.Conj(xi)*xi)
				ap[kk+i] = complex(aii, 0)
			} else {
				ap[kk+i] = complex(real(ap[kk+i]), 0)
			}
			ix += incX
			kk += i + 1
		}
	}
}

// Zhpr2 performs the Hermitian rank-2 operation
//  A += alpha * x * y^H + conj(alpha) * y * x^H
// where alpha is a complex scalar, x and y are n element vectors, and A is an
// n×n Hermitian matrix, supplied in packed form. On entry, the imaginary parts
// of the diagonal elements are assumed to be zero, and on return they are set to zero.
func (Implementation) Zhpr2(uplo blas.Uplo, n int, alpha complex128, x []complex128, incX int, y []complex128, incY int, ap []complex128) {
	if uplo != blas.Upper && uplo != blas.Lower {
		panic(badUplo)
	}
	if n < 0 {
		panic(nLT0)
	}
	checkZVector('x', n, x, incX)
	checkZVector('y', n, y, incY)
	if len(ap) < n*(n+1)/2 {
		panic("blas: insufficient A packed matrix slice length")
	}

	if n == 0 || alpha == 0 {
		return
	}

	// Set up start indices in X and Y.
	var kx int
	if incX < 0 {
		kx = (1 - n) * incX
	}
	var ky int
	if incY < 0 {
		ky = (1 - n) * incY
	}

	// The elements of A are accessed sequentially with one pass through ap.

	var kk int
	if uplo == blas.Upper {
		// Form A when upper triangle is stored in AP.
		// Here, kk points to the current diagonal element in ap.
		if incX == 1 && incY == 1 {
			for i := 0; i < n; i++ {
				if x[i] != 0 || y[i] != 0 {
					tmp1 := alpha * x[i]
					tmp2 := cmplx.Conj(alpha) * y[i]
					aii := real(ap[kk]) + real(tmp1*cmplx.Conj(y[i])) + real(tmp2*cmplx.Conj(x[i]))
					ap[kk] = complex(aii, 0)
					k := kk + 1
					for j := i + 1; j < n; j++ {
						ap[k] += tmp1*cmplx.Conj(y[j]) + tmp2*cmplx.Conj(x[j])
						k++
					}
				} else {
					ap[kk] = complex(real(ap[kk]), 0)
				}
				kk += n - i
			}
		} else {
			ix := kx
			iy := ky
			for i := 0; i < n; i++ {
				if x[ix] != 0 || y[iy] != 0 {
					tmp1 := alpha * x[ix]
					tmp2 := cmplx.Conj(alpha) * y[iy]
					aii := real(ap[kk]) + real(tmp1*cmplx.Conj(y[iy])) + real(tmp2*cmplx.Conj(x[ix]))
					ap[kk] = complex(aii, 0)
					jx := ix + incX
					jy := iy + incY
					for k := kk + 1; k < kk+n-i; k++ {
						ap[k] += tmp1*cmplx.Conj(y[jy]) + tmp2*cmplx.Conj(x[jx])
						jx += incX
						jy += incY
					}
				} else {
					ap[kk] = complex(real(ap[kk]), 0)
				}
				ix += incX
				iy += incY
				kk += n - i
			}
		}
		return
	}

	// Form A when lower triangle is stored in AP.
	// Here, kk points to the beginning of current row in ap.
	if incX == 1 && incY == 1 {
		for i := 0; i < n; i++ {
			if x[i] != 0 || y[i] != 0 {
				tmp1 := alpha * x[i]
				tmp2 := cmplx.Conj(alpha) * y[i]
				k := kk
				for j := 0; j < i; j++ {
					ap[k] += tmp1*cmplx.Conj(y[j]) + tmp2*cmplx.Conj(x[j])
					k++
				}
				aii := real(ap[kk+i]) + real(tmp1*cmplx.Conj(y[i])) + real(tmp2*cmplx.Conj(x[i]))
				ap[kk+i] = complex(aii, 0)
			} else {
				ap[kk+i] = complex(real(ap[kk+i]), 0)
			}
			kk += i + 1
		}
	} else {
		ix := kx
		iy := ky
		for i := 0; i < n; i++ {
			if x[ix] != 0 || y[iy] != 0 {
				tmp1 := alpha * x[ix]
				tmp2 := cmplx.Conj(alpha) * y[iy]
				jx := kx
				jy := ky
				for k := kk; k < kk+i; k++ {
					ap[k] += tmp1*cmplx.Conj(y[jy]) + tmp2*cmplx.Conj(x[jx])
					jx += incX
					jy += incY
				}
				aii := real(ap[kk+i]) + real(tmp1*cmplx.Conj(y[iy])) + real(tmp2*cmplx.Conj(x[ix]))
				ap[kk+i] = complex(aii, 0)
			} else {
				ap[kk+i] = complex(real(ap[kk+i]), 0)
			}
			ix += incX
			iy += incY
			kk += i + 1
		}
	}
}

// Ztbmv performs one of the matrix-vector operations
//  x = A * x    if trans = blas.NoTrans
//  x = A^T * x  if trans = blas.Trans
//  x = A^H * x  if trans = blas.ConjTrans
// where x is an n element vector and A is an n×n triangular band matrix, with
// (k+1) diagonals.
func (Implementation) Ztbmv(uplo blas.Uplo, trans blas.Transpose, diag blas.Diag, n, k int, ab []complex128, ldab int, x []complex128, incX int) {
	if uplo != blas.Upper && uplo != blas.Lower {
		panic(badUplo)
	}
	if trans != blas.NoTrans && trans != blas.Trans && trans != blas.ConjTrans {
		panic(badTranspose)
	}
	if diag != blas.Unit && diag != blas.NonUnit {
		panic(badDiag)
	}
	checkZtbMatrix('A', n, k, ab, ldab)
	checkZVector('x', n, x, incX)

	if n == 0 {
		return
	}

	// Set up start index in X.
	var kx int
	if incX < 0 {
		kx = (1 - n) * incX
	}

	switch trans {
	case blas.NoTrans:
		if uplo == blas.Upper {
			if incX == 1 {
				for i := 0; i < n; i++ {
					xi := x[i]
					if diag == blas.NonUnit {
						xi *= ab[i*ldab]
					}
					kk := min(k, n-i-1)
					for j, aij := range ab[i*ldab+1 : i*ldab+kk+1] {
						xi += x[i+j+1] * aij
					}
					x[i] = xi
				}
			} else {
				ix := kx
				for i := 0; i < n; i++ {
					xi := x[ix]
					if diag == blas.NonUnit {
						xi *= ab[i*ldab]
					}
					kk := min(k, n-i-1)
					jx := ix + incX
					for _, aij := range ab[i*ldab+1 : i*ldab+kk+1] {
						xi += x[jx] * aij
						jx += incX
					}
					x[ix] = xi
					ix += incX
				}
			}
		} else {
			if incX == 1 {
				for i := n - 1; i >= 0; i-- {
					xi := x[i]
					if diag == blas.NonUnit {
						xi *= ab[i*ldab+k]
					}
					kk := min(k, i)
					for j, aij := range ab[i*ldab+k-kk : i*ldab+k] {
						xi += x[i-kk+j] * aij
					}
					x[i] = xi
				}
			} else {
				ix := kx + (n-1)*incX
				for i := n - 1; i >= 0; i-- {
					xi := x[ix]
					if diag == blas.NonUnit {
						xi *= ab[i*ldab+k]
					}
					kk := min(k, i)
					jx := ix - kk*incX
					for _, aij := range ab[i*ldab+k-kk : i*ldab+k] {
						xi += x[jx] * aij
						jx += incX
					}
					x[ix] = xi
					ix -= incX
				}
			}
		}
	case blas.Trans:
		if uplo == blas.Upper {
			if incX == 1 {
				for i := n - 1; i >= 0; i-- {
					kk := min(k, n-i-1)
					xi := x[i]
					for j, aij := range ab[i*ldab+1 : i*ldab+kk+1] {
						x[i+j+1] += xi * aij
					}
					if diag == blas.NonUnit {
						x[i] *= ab[i*ldab]
					}
				}
			} else {
				ix := kx + (n-1)*incX
				for i := n - 1; i >= 0; i-- {
					kk := min(k, n-i-1)
					jx := ix + incX
					xi := x[ix]
					for _, aij := range ab[i*ldab+1 : i*ldab+kk+1] {
						x[jx] += xi * aij
						jx += incX
					}
					if diag == blas.NonUnit {
						x[ix] *= ab[i*ldab]
					}
					ix -= incX
				}
			}
		} else {
			if incX == 1 {
				for i := 0; i < n; i++ {
					kk := min(k, i)
					xi := x[i]
					for j, aij := range ab[i*ldab+k-kk : i*ldab+k] {
						x[i-kk+j] += xi * aij
					}
					if diag == blas.NonUnit {
						x[i] *= ab[i*ldab+k]
					}
				}
			} else {
				ix := kx
				for i := 0; i < n; i++ {
					kk := min(k, i)
					jx := ix - kk*incX
					xi := x[ix]
					for _, aij := range ab[i*ldab+k-kk : i*ldab+k] {
						x[jx] += xi * aij
						jx += incX
					}
					if diag == blas.NonUnit {
						x[ix] *= ab[i*ldab+k]
					}
					ix += incX
				}
			}
		}
	case blas.ConjTrans:
		if uplo == blas.Upper {
			if incX == 1 {
				for i := n - 1; i >= 0; i-- {
					kk := min(k, n-i-1)
					xi := x[i]
					for j, aij := range ab[i*ldab+1 : i*ldab+kk+1] {
						x[i+j+1] += xi * cmplx.Conj(aij)
					}
					if diag == blas.NonUnit {
						x[i] *= cmplx.Conj(ab[i*ldab])
					}
				}
			} else {
				ix := kx + (n-1)*incX
				for i := n - 1; i >= 0; i-- {
					kk := min(k, n-i-1)
					jx := ix + incX
					xi := x[ix]
					for _, aij := range ab[i*ldab+1 : i*ldab+kk+1] {
						x[jx] += xi * cmplx.Conj(aij)
						jx += incX
					}
					if diag == blas.NonUnit {
						x[ix] *= cmplx.Conj(ab[i*ldab])
					}
					ix -= incX
				}
			}
		} else {
			if incX == 1 {
				for i := 0; i < n; i++ {
					kk := min(k, i)
					xi := x[i]
					for j, aij := range ab[i*ldab+k-kk : i*ldab+k] {
						x[i-kk+j] += xi * cmplx.Conj(aij)
					}
					if diag == blas.NonUnit {
						x[i] *= cmplx.Conj(ab[i*ldab+k])
					}
				}
			} else {
				ix := kx
				for i := 0; i < n; i++ {
					kk := min(k, i)
					jx := ix - kk*incX
					xi := x[ix]
					for _, aij := range ab[i*ldab+k-kk : i*ldab+k] {
						x[jx] += xi * cmplx.Conj(aij)
						jx += incX
					}
					if diag == blas.NonUnit {
						x[ix] *= cmplx.Conj(ab[i*ldab+k])
					}
					ix += incX
				}
			}
		}
	}
}

// Ztbsv solves one of the systems of equations
//  A * x = b    if trans == blas.NoTrans
//  A^T * x = b  if trans == blas.Trans
//  A^H * x = b  if trans == blas.ConjTrans
// where b and x are n element vectors and A is an n×n triangular band matrix
// with (k+1) diagonals.
//
// On entry, x contains the values of b, and the solution is
// stored in-place into x.
//
// No test for singularity or near-singularity is included in this
// routine. Such tests must be performed before calling this routine.
func (Implementation) Ztbsv(uplo blas.Uplo, trans blas.Transpose, diag blas.Diag, n, k int, ab []complex128, ldab int, x []complex128, incX int) {
	if uplo != blas.Upper && uplo != blas.Lower {
		panic(badUplo)
	}
	if trans != blas.NoTrans && trans != blas.Trans && trans != blas.ConjTrans {
		panic(badTranspose)
	}
	if diag != blas.Unit && diag != blas.NonUnit {
		panic(badDiag)
	}
	checkZtbMatrix('A', n, k, ab, ldab)
	checkZVector('x', n, x, incX)

	if n == 0 {
		return
	}

	// Set up start index in X.
	var kx int
	if incX < 0 {
		kx = (1 - n) * incX
	}

	switch trans {
	case blas.NoTrans:
		if uplo == blas.Upper {
			if incX == 1 {
				for i := n - 1; i >= 0; i-- {
					kk := min(k, n-i-1)
					var sum complex128
					for j, aij := range ab[i*ldab+1 : i*ldab+kk+1] {
						sum += x[i+1+j] * aij
					}
					x[i] -= sum
					if diag == blas.NonUnit {
						x[i] /= ab[i*ldab]
					}
				}
			} else {
				ix := kx + (n-1)*incX
				for i := n - 1; i >= 0; i-- {
					kk := min(k, n-i-1)
					var sum complex128
					jx := ix + incX
					for _, aij := range ab[i*ldab+1 : i*ldab+kk+1] {
						sum += x[jx] * aij
						jx += incX
					}
					x[ix] -= sum
					if diag == blas.NonUnit {
						x[ix] /= ab[i*ldab]
					}
					ix -= incX
				}
			}
		} else {
			if incX == 1 {
				for i := 0; i < n; i++ {
					kk := min(k, i)
					var sum complex128
					for j, aij := range ab[i*ldab+k-kk : i*ldab+k] {
						sum += x[i-kk+j] * aij
					}
					x[i] -= sum
					if diag == blas.NonUnit {
						x[i] /= ab[i*ldab+k]
					}
				}
			} else {
				ix := kx
				for i := 0; i < n; i++ {
					kk := min(k, i)
					var sum complex128
					jx := ix - kk*incX
					for _, aij := range ab[i*ldab+k-kk : i*ldab+k] {
						sum += x[jx] * aij
						jx += incX
					}
					x[ix] -= sum
					if diag == blas.NonUnit {
						x[ix] /= ab[i*ldab+k]
					}
					ix += incX
				}
			}
		}
	case blas.Trans:
		if uplo == blas.Upper {
			if incX == 1 {
				for i := 0; i < n; i++ {
					if diag == blas.NonUnit {
						x[i] /= ab[i*ldab]
					}
					kk := min(k, n-i-1)
					xi := x[i]
					for j, aij := range ab[i*ldab+1 : i*ldab+kk+1] {
						x[i+1+j] -= xi * aij
					}
				}
			} else {
				ix := kx
				for i := 0; i < n; i++ {
					if diag == blas.NonUnit {
						x[ix] /= ab[i*ldab]
					}
					kk := min(k, n-i-1)
					xi := x[ix]
					jx := ix + incX
					for _, aij := range ab[i*ldab+1 : i*ldab+kk+1] {
						x[jx] -= xi * aij
						jx += incX
					}
					ix += incX
				}
			}
		} else {
			if incX == 1 {
				for i := n - 1; i >= 0; i-- {
					if diag == blas.NonUnit {
						x[i] /= ab[i*ldab+k]
					}
					kk := min(k, i)
					xi := x[i]
					for j, aij := range ab[i*ldab+k-kk : i*ldab+k] {
						x[i-kk+j] -= xi * aij
					}
				}
			} else {
				ix := kx + (n-1)*incX
				for i := n - 1; i >= 0; i-- {
					if diag == blas.NonUnit {
						x[ix] /= ab[i*ldab+k]
					}
					kk := min(k, i)
					xi := x[ix]
					jx := ix - kk*incX
					for _, aij := range ab[i*ldab+k-kk : i*ldab+k] {
						x[jx] -= xi * aij
						jx += incX
					}
					ix -= incX
				}
			}
		}
	case blas.ConjTrans:
		if uplo == blas.Upper {
			if incX == 1 {
				for i := 0; i < n; i++ {
					if diag == blas.NonUnit {
						x[i] /= cmplx.Conj(ab[i*ldab])
					}
					kk := min(k, n-i-1)
					xi := x[i]
					for j, aij := range ab[i*ldab+1 : i*ldab+kk+1] {
						x[i+1+j] -= xi * cmplx.Conj(aij)
					}
				}
			} else {
				ix := kx
				for i := 0; i < n; i++ {
					if diag == blas.NonUnit {
						x[ix] /= cmplx.Conj(ab[i*ldab])
					}
					kk := min(k, n-i-1)
					xi := x[ix]
					jx := ix + incX
					for _, aij := range ab[i*ldab+1 : i*ldab+kk+1] {
						x[jx] -= xi * cmplx.Conj(aij)
						jx += incX
					}
					ix += incX
				}
			}
		} else {
			if incX == 1 {
				for i := n - 1; i >= 0; i-- {
					if diag == blas.NonUnit {
						x[i] /= cmplx.Conj(ab[i*ldab+k])
					}
					kk := min(k, i)
					xi := x[i]
					for j, aij := range ab[i*ldab+k-kk : i*ldab+k] {
						x[i-kk+j] -= xi * cmplx.Conj(aij)
					}
				}
			} else {
				ix := kx + (n-1)*incX
				for i := n - 1; i >= 0; i-- {
					if diag == blas.NonUnit {
						x[ix] /= cmplx.Conj(ab[i*ldab+k])
					}
					kk := min(k, i)
					xi := x[ix]
					jx := ix - kk*incX
					for _, aij := range ab[i*ldab+k-kk : i*ldab+k] {
						x[jx] -= xi * cmplx.Conj(aij)
						jx += incX
					}
					ix -= incX
				}
			}
		}
	}
}

// Ztpmv performs one of the matrix-vector operations
//  x = A * x    if trans = blas.NoTrans
//  x = A^T * x  if trans = blas.Trans
//  x = A^H * x  if trans = blas.ConjTrans
// where x is an n element vector and A is an n×n triangular matrix, supplied in
// packed form.
func (Implementation) Ztpmv(uplo blas.Uplo, trans blas.Transpose, diag blas.Diag, n int, ap []complex128, x []complex128, incX int) {
	if uplo != blas.Upper && uplo != blas.Lower {
		panic(badUplo)
	}
	if trans != blas.NoTrans && trans != blas.Trans && trans != blas.ConjTrans {
		panic(badTranspose)
	}
	if diag != blas.Unit && diag != blas.NonUnit {
		panic(badDiag)
	}
	checkZVector('x', n, x, incX)
	if len(ap) < n*(n+1)/2 {
		panic("blas: insufficient A packed matrix slice length")
	}

	if n == 0 {
		return
	}

	// Set up start index in X.
	var kx int
	if incX < 0 {
		kx = (1 - n) * incX
	}

	// The elements of A are accessed sequentially with one pass through A.

	if trans == blas.NoTrans {
		// Form x = A*x.
		if uplo == blas.Upper {
			// kk points to the current diagonal element in ap.
			kk := 0
			if incX == 1 {
				x = x[:n]
				for i := range x {
					if diag == blas.NonUnit {
						x[i] *= ap[kk]
					}
					if n-i-1 > 0 {
						x[i] += c128.DotuUnitary(ap[kk+1:kk+n-i], x[i+1:])
					}
					kk += n - i
				}
			} else {
				ix := kx
				for i := 0; i < n; i++ {
					if diag == blas.NonUnit {
						x[ix] *= ap[kk]
					}
					if n-i-1 > 0 {
						x[ix] += c128.DotuInc(ap[kk+1:kk+n-i], x, uintptr(n-i-1), 1, uintptr(incX), 0, uintptr(ix+incX))
					}
					ix += incX
					kk += n - i
				}
			}
		} else {
			// kk points to the beginning of current row in ap.
			kk := n*(n+1)/2 - n
			if incX == 1 {
				for i := n - 1; i >= 0; i-- {
					if diag == blas.NonUnit {
						x[i] *= ap[kk+i]
					}
					if i > 0 {
						x[i] += c128.DotuUnitary(ap[kk:kk+i], x[:i])
					}
					kk -= i
				}
			} else {
				ix := kx + (n-1)*incX
				for i := n - 1; i >= 0; i-- {
					if diag == blas.NonUnit {
						x[ix] *= ap[kk+i]
					}
					if i > 0 {
						x[ix] += c128.DotuInc(ap[kk:kk+i], x, uintptr(i), 1, uintptr(incX), 0, uintptr(kx))
					}
					ix -= incX
					kk -= i
				}
			}
		}
		return
	}

	if trans == blas.Trans {
		// Form x = A^T*x.
		if uplo == blas.Upper {
			// kk points to the current diagonal element in ap.
			kk := n*(n+1)/2 - 1
			if incX == 1 {
				for i := n - 1; i >= 0; i-- {
					xi := x[i]
					if diag == blas.NonUnit {
						x[i] *= ap[kk]
					}
					if n-i-1 > 0 {
						c128.AxpyUnitary(xi, ap[kk+1:kk+n-i], x[i+1:n])
					}
					kk -= n - i + 1
				}
			} else {
				ix := kx + (n-1)*incX
				for i := n - 1; i >= 0; i-- {
					xi := x[ix]
					if diag == blas.NonUnit {
						x[ix] *= ap[kk]
					}
					if n-i-1 > 0 {
						c128.AxpyInc(xi, ap[kk+1:kk+n-i], x, uintptr(n-i-1), 1, uintptr(incX), 0, uintptr(ix+incX))
					}
					ix -= incX
					kk -= n - i + 1
				}
			}
		} else {
			// kk points to the beginning of current row in ap.
			kk := 0
			if incX == 1 {
				x = x[:n]
				for i := range x {
					if i > 0 {
						c128.AxpyUnitary(x[i], ap[kk:kk+i], x[:i])
					}
					if diag == blas.NonUnit {
						x[i] *= ap[kk+i]
					}
					kk += i + 1
				}
			} else {
				ix := kx
				for i := 0; i < n; i++ {
					if i > 0 {
						c128.AxpyInc(x[ix], ap[kk:kk+i], x, uintptr(i), 1, uintptr(incX), 0, uintptr(kx))
					}
					if diag == blas.NonUnit {
						x[ix] *= ap[kk+i]
					}
					ix += incX
					kk += i + 1
				}
			}
		}
		return
	}

	// Form x = A^H*x.
	if uplo == blas.Upper {
		// kk points to the current diagonal element in ap.
		kk := n*(n+1)/2 - 1
		if incX == 1 {
			for i := n - 1; i >= 0; i-- {
				xi := x[i]
				if diag == blas.NonUnit {
					x[i] *= cmplx.Conj(ap[kk])
				}
				k := kk + 1
				for j := i + 1; j < n; j++ {
					x[j] += xi * cmplx.Conj(ap[k])
					k++
				}
				kk -= n - i + 1
			}
		} else {
			ix := kx + (n-1)*incX
			for i := n - 1; i >= 0; i-- {
				xi := x[ix]
				if diag == blas.NonUnit {
					x[ix] *= cmplx.Conj(ap[kk])
				}
				jx := ix + incX
				k := kk + 1
				for j := i + 1; j < n; j++ {
					x[jx] += xi * cmplx.Conj(ap[k])
					jx += incX
					k++
				}
				ix -= incX
				kk -= n - i + 1
			}
		}
	} else {
		// kk points to the beginning of current row in ap.
		kk := 0
		if incX == 1 {
			x = x[:n]
			for i, xi := range x {
				for j := 0; j < i; j++ {
					x[j] += xi * cmplx.Conj(ap[kk+j])
				}
				if diag == blas.NonUnit {
					x[i] *= cmplx.Conj(ap[kk+i])
				}
				kk += i + 1
			}
		} else {
			ix := kx
			for i := 0; i < n; i++ {
				xi := x[ix]
				jx := kx
				for j := 0; j < i; j++ {
					x[jx] += xi * cmplx.Conj(ap[kk+j])
					jx += incX
				}
				if diag == blas.NonUnit {
					x[ix] *= cmplx.Conj(ap[kk+i])
				}
				ix += incX
				kk += i + 1
			}
		}
	}
}

// Ztpsv solves one of the systems of equations
//  A * x = b    if trans == blas.NoTrans
//  A^T * x = b  if trans == blas.Trans
//  A^H * x = b  if trans == blas.ConjTrans
// where b and x are n element vectors and A is an n×n triangular matrix in
// packed form.
//
// On entry, x contains the values of b, and the solution is
// stored in-place into x.
//
// No test for singularity or near-singularity is included in this
// routine. Such tests must be performed before calling this routine.
func (Implementation) Ztpsv(uplo blas.Uplo, trans blas.Transpose, diag blas.Diag, n int, ap []complex128, x []complex128, incX int) {
	if uplo != blas.Upper && uplo != blas.Lower {
		panic(badUplo)
	}
	if trans != blas.NoTrans && trans != blas.Trans && trans != blas.ConjTrans {
		panic(badTranspose)
	}
	if diag != blas.Unit && diag != blas.NonUnit {
		panic(badDiag)
	}
	if len(ap) < n*(n+1)/2 {
		panic("blas: insufficient A packed matrix slice length")
	}
	checkZVector('x', n, x, incX)

	if n == 0 {
		return
	}

	// Set up start index in X.
	var kx int
	if incX < 0 {
		kx = (1 - n) * incX
	}

	// The elements of A are accessed sequentially with one pass through ap.

	if trans == blas.NoTrans {
		// Form x = inv(A)*x.
		if uplo == blas.Upper {
			kk := n*(n+1)/2 - 1
			if incX == 1 {
				for i := n - 1; i >= 0; i-- {
					aii := ap[kk]
					if n-i-1 > 0 {
						x[i] -= c128.DotuUnitary(x[i+1:n], ap[kk+1:kk+n-i])
					}
					if diag == blas.NonUnit {
						x[i] /= aii
					}
					kk -= n - i + 1
				}
			} else {
				ix := kx + (n-1)*incX
				for i := n - 1; i >= 0; i-- {
					aii := ap[kk]
					if n-i-1 > 0 {
						x[ix] -= c128.DotuInc(x, ap[kk+1:kk+n-i], uintptr(n-i-1), uintptr(incX), 1, uintptr(ix+incX), 0)
					}
					if diag == blas.NonUnit {
						x[ix] /= aii
					}
					ix -= incX
					kk -= n - i + 1
				}
			}
		} else {
			kk := 0
			if incX == 1 {
				for i := 0; i < n; i++ {
					if i > 0 {
						x[i] -= c128.DotuUnitary(x[:i], ap[kk:kk+i+1])
					}
					if diag == blas.NonUnit {
						x[i] /= ap[kk+i]
					}
					kk += i + 1
				}
			} else {
				ix := kx
				for i := 0; i < n; i++ {
					if i > 0 {
						x[ix] -= c128.DotuInc(x, ap[kk:kk+i+1], uintptr(i), uintptr(incX), 1, uintptr(kx), 0)
					}
					if diag == blas.NonUnit {
						x[ix] /= ap[kk+i]
					}
					ix += incX
					kk += i + 1
				}
			}
		}
		return
	}

	if trans == blas.Trans {
		// Form x = inv(A^T)*x.
		if uplo == blas.Upper {
			kk := 0
			if incX == 1 {
				for j := 0; j < n; j++ {
					if diag == blas.NonUnit {
						x[j] /= ap[kk]
					}
					if n-j-1 > 0 {
						c128.AxpyUnitary(-x[j], ap[kk+1:kk+n-j], x[j+1:n])
					}
					kk += n - j
				}
			} else {
				jx := kx
				for j := 0; j < n; j++ {
					if diag == blas.NonUnit {
						x[jx] /= ap[kk]
					}
					if n-j-1 > 0 {
						c128.AxpyInc(-x[jx], ap[kk+1:kk+n-j], x, uintptr(n-j-1), 1, uintptr(incX), 0, uintptr(jx+incX))
					}
					jx += incX
					kk += n - j
				}
			}
		} else {
			kk := n*(n+1)/2 - n
			if incX == 1 {
				for j := n - 1; j >= 0; j-- {
					if diag == blas.NonUnit {
						x[j] /= ap[kk+j]
					}
					if j > 0 {
						c128.AxpyUnitary(-x[j], ap[kk:kk+j], x[:j])
					}
					kk -= j
				}
			} else {
				jx := kx + (n-1)*incX
				for j := n - 1; j >= 0; j-- {
					if diag == blas.NonUnit {
						x[jx] /= ap[kk+j]
					}
					if j > 0 {
						c128.AxpyInc(-x[jx], ap[kk:kk+j], x, uintptr(j), 1, uintptr(incX), 0, uintptr(kx))
					}
					jx -= incX
					kk -= j
				}
			}
		}
		return
	}

	// Form x = inv(A^H)*x.
	if uplo == blas.Upper {
		kk := 0
		if incX == 1 {
			for j := 0; j < n; j++ {
				if diag == blas.NonUnit {
					x[j] /= cmplx.Conj(ap[kk])
				}
				xj := x[j]
				k := kk + 1
				for i := j + 1; i < n; i++ {
					x[i] -= xj * cmplx.Conj(ap[k])
					k++
				}
				kk += n - j
			}
		} else {
			jx := kx
			for j := 0; j < n; j++ {
				if diag == blas.NonUnit {
					x[jx] /= cmplx.Conj(ap[kk])
				}
				xj := x[jx]
				ix := jx + incX
				k := kk + 1
				for i := j + 1; i < n; i++ {
					x[ix] -= xj * cmplx.Conj(ap[k])
					ix += incX
					k++
				}
				jx += incX
				kk += n - j
			}
		}
	} else {
		kk := n*(n+1)/2 - n
		if incX == 1 {
			for j := n - 1; j >= 0; j-- {
				if diag == blas.NonUnit {
					x[j] /= cmplx.Conj(ap[kk+j])
				}
				xj := x[j]
				for i := 0; i < j; i++ {
					x[i] -= xj * cmplx.Conj(ap[kk+i])
				}
				kk -= j
			}
		} else {
			jx := kx + (n-1)*incX
			for j := n - 1; j >= 0; j-- {
				if diag == blas.NonUnit {
					x[jx] /= cmplx.Conj(ap[kk+j])
				}
				xj := x[jx]
				ix := kx
				for i := 0; i < j; i++ {
					x[ix] -= xj * cmplx.Conj(ap[kk+i])
					ix += incX
				}
				jx -= incX
				kk -= j
			}
		}
	}
}

// Ztrmv performs one of the matrix-vector operations
//  x = A * x    if trans = blas.NoTrans
//  x = A^T * x  if trans = blas.Trans
//  x = A^H * x  if trans = blas.ConjTrans
// where x is a vector, and A is an n×n triangular matrix.
func (Implementation) Ztrmv(uplo blas.Uplo, trans blas.Transpose, diag blas.Diag, n int, a []complex128, lda int, x []complex128, incX int) {
	if uplo != blas.Upper && uplo != blas.Lower {
		panic(badUplo)
	}
	if trans != blas.NoTrans && trans != blas.Trans && trans != blas.ConjTrans {
		panic(badTranspose)
	}
	if diag != blas.Unit && diag != blas.NonUnit {
		panic(badDiag)
	}
	checkZMatrix('A', n, n, a, lda)
	checkZVector('x', n, x, incX)

	if n == 0 {
		return
	}

	// Set up start index in X.
	var kx int
	if incX < 0 {
		kx = (1 - n) * incX
	}

	// The elements of A are accessed sequentially with one pass through A.

	if trans == blas.NoTrans {
		// Form x = A*x.
		if uplo == blas.Upper {
			if incX == 1 {
				for i := 0; i < n; i++ {
					if diag == blas.NonUnit {
						x[i] *= a[i*lda+i]
					}
					if n-i-1 > 0 {
						x[i] += c128.DotuUnitary(a[i*lda+i+1:i*lda+n], x[i+1:n])
					}
				}
			} else {
				ix := kx
				for i := 0; i < n; i++ {
					if diag == blas.NonUnit {
						x[ix] *= a[i*lda+i]
					}
					if n-i-1 > 0 {
						x[ix] += c128.DotuInc(a[i*lda+i+1:i*lda+n], x, uintptr(n-i-1), 1, uintptr(incX), 0, uintptr(ix+incX))
					}
					ix += incX
				}
			}
		} else {
			if incX == 1 {
				for i := n - 1; i >= 0; i-- {
					if diag == blas.NonUnit {
						x[i] *= a[i*lda+i]
					}
					if i > 0 {
						x[i] += c128.DotuUnitary(a[i*lda:i*lda+i], x[:i])
					}
				}
			} else {
				ix := kx + (n-1)*incX
				for i := n - 1; i >= 0; i-- {
					if diag == blas.NonUnit {
						x[ix] *= a[i*lda+i]
					}
					if i > 0 {
						x[ix] += c128.DotuInc(a[i*lda:i*lda+i], x, uintptr(i), 1, uintptr(incX), 0, uintptr(kx))
					}
					ix -= incX
				}
			}
		}
		return
	}

	if trans == blas.Trans {
		// Form x = A^T*x.
		if uplo == blas.Upper {
			if incX == 1 {
				for i := n - 1; i >= 0; i-- {
					xi := x[i]
					if diag == blas.NonUnit {
						x[i] *= a[i*lda+i]
					}
					if n-i-1 > 0 {
						c128.AxpyUnitary(xi, a[i*lda+i+1:i*lda+n], x[i+1:n])
					}
				}
			} else {
				ix := kx + (n-1)*incX
				for i := n - 1; i >= 0; i-- {
					xi := x[ix]
					if diag == blas.NonUnit {
						x[ix] *= a[i*lda+i]
					}
					if n-i-1 > 0 {
						c128.AxpyInc(xi, a[i*lda+i+1:i*lda+n], x, uintptr(n-i-1), 1, uintptr(incX), 0, uintptr(ix+incX))
					}
					ix -= incX
				}
			}
		} else {
			if incX == 1 {
				for i := 0; i < n; i++ {
					if i > 0 {
						c128.AxpyUnitary(x[i], a[i*lda:i*lda+i], x[:i])
					}
					if diag == blas.NonUnit {
						x[i] *= a[i*lda+i]
					}
				}
			} else {
				ix := kx
				for i := 0; i < n; i++ {
					if i > 0 {
						c128.AxpyInc(x[ix], a[i*lda:i*lda+i], x, uintptr(i), 1, uintptr(incX), 0, uintptr(kx))
					}
					if diag == blas.NonUnit {
						x[ix] *= a[i*lda+i]
					}
					ix += incX
				}
			}
		}
		return
	}

	// Form x = A^H*x.
	if uplo == blas.Upper {
		if incX == 1 {
			for i := n - 1; i >= 0; i-- {
				xi := x[i]
				if diag == blas.NonUnit {
					x[i] *= cmplx.Conj(a[i*lda+i])
				}
				for j := i + 1; j < n; j++ {
					x[j] += xi * cmplx.Conj(a[i*lda+j])
				}
			}
		} else {
			ix := kx + (n-1)*incX
			for i := n - 1; i >= 0; i-- {
				xi := x[ix]
				if diag == blas.NonUnit {
					x[ix] *= cmplx.Conj(a[i*lda+i])
				}
				jx := ix + incX
				for j := i + 1; j < n; j++ {
					x[jx] += xi * cmplx.Conj(a[i*lda+j])
					jx += incX
				}
				ix -= incX
			}
		}
	} else {
		if incX == 1 {
			for i := 0; i < n; i++ {
				for j := 0; j < i; j++ {
					x[j] += x[i] * cmplx.Conj(a[i*lda+j])
				}
				if diag == blas.NonUnit {
					x[i] *= cmplx.Conj(a[i*lda+i])
				}
			}
		} else {
			ix := kx
			for i := 0; i < n; i++ {
				jx := kx
				for j := 0; j < i; j++ {
					x[jx] += x[ix] * cmplx.Conj(a[i*lda+j])
					jx += incX
				}
				if diag == blas.NonUnit {
					x[ix] *= cmplx.Conj(a[i*lda+i])
				}
				ix += incX
			}
		}
	}
}

// Ztrsv solves one of the systems of equations
//  A * x = b    if trans == blas.NoTrans
//  A^T * x = b  if trans == blas.Trans
//  A^H * x = b  if trans == blas.ConjTrans
// where b and x are n element vectors and A is an n×n triangular matrix.
//
// On entry, x contains the values of b, and the solution is
// stored in-place into x.
//
// No test for singularity or near-singularity is included in this
// routine. Such tests must be performed before calling this routine.
func (Implementation) Ztrsv(uplo blas.Uplo, trans blas.Transpose, diag blas.Diag, n int, a []complex128, lda int, x []complex128, incX int) {
	if uplo != blas.Upper && uplo != blas.Lower {
		panic(badUplo)
	}
	if trans != blas.NoTrans && trans != blas.Trans && trans != blas.ConjTrans {
		panic(badTranspose)
	}
	if diag != blas.Unit && diag != blas.NonUnit {
		panic(badDiag)
	}
	checkZMatrix('A', n, n, a, lda)
	checkZVector('x', n, x, incX)

	if n == 0 {
		return
	}

	// Set up start index in X.
	var kx int
	if incX < 0 {
		kx = (1 - n) * incX
	}

	// The elements of A are accessed sequentially with one pass through A.

	if trans == blas.NoTrans {
		// Form x = inv(A)*x.
		if uplo == blas.Upper {
			if incX == 1 {
				for i := n - 1; i >= 0; i-- {
					aii := a[i*lda+i]
					if n-i-1 > 0 {
						x[i] -= c128.DotuUnitary(x[i+1:n], a[i*lda+i+1:i*lda+n])
					}
					if diag == blas.NonUnit {
						x[i] /= aii
					}
				}
			} else {
				ix := kx + (n-1)*incX
				for i := n - 1; i >= 0; i-- {
					aii := a[i*lda+i]
					if n-i-1 > 0 {
						x[ix] -= c128.DotuInc(x, a[i*lda+i+1:i*lda+n], uintptr(n-i-1), uintptr(incX), 1, uintptr(ix+incX), 0)
					}
					if diag == blas.NonUnit {
						x[ix] /= aii
					}
					ix -= incX
				}
			}
		} else {
			if incX == 1 {
				for i := 0; i < n; i++ {
					if i > 0 {
						x[i] -= c128.DotuUnitary(x[:i], a[i*lda:i*lda+i])
					}
					if diag == blas.NonUnit {
						x[i] /= a[i*lda+i]
					}
				}
			} else {
				ix := kx
				for i := 0; i < n; i++ {
					if i > 0 {
						x[ix] -= c128.DotuInc(x, a[i*lda:i*lda+i], uintptr(i), uintptr(incX), 1, uintptr(kx), 0)
					}
					if diag == blas.NonUnit {
						x[ix] /= a[i*lda+i]
					}
					ix += incX
				}
			}
		}
		return
	}

	if trans == blas.Trans {
		// Form x = inv(A^T)*x.
		if uplo == blas.Upper {
			if incX == 1 {
				for j := 0; j < n; j++ {
					if diag == blas.NonUnit {
						x[j] /= a[j*lda+j]
					}
					if n-j-1 > 0 {
						c128.AxpyUnitary(-x[j], a[j*lda+j+1:j*lda+n], x[j+1:n])
					}
				}
			} else {
				jx := kx
				for j := 0; j < n; j++ {
					if diag == blas.NonUnit {
						x[jx] /= a[j*lda+j]
					}
					if n-j-1 > 0 {
						c128.AxpyInc(-x[jx], a[j*lda+j+1:j*lda+n], x, uintptr(n-j-1), 1, uintptr(incX), 0, uintptr(jx+incX))
					}
					jx += incX
				}
			}
		} else {
			if incX == 1 {
				for j := n - 1; j >= 0; j-- {
					if diag == blas.NonUnit {
						x[j] /= a[j*lda+j]
					}
					xj := x[j]
					if j > 0 {
						c128.AxpyUnitary(-xj, a[j*lda:j*lda+j], x[:j])
					}
				}
			} else {
				jx := kx + (n-1)*incX
				for j := n - 1; j >= 0; j-- {
					if diag == blas.NonUnit {
						x[jx] /= a[j*lda+j]
					}
					if j > 0 {
						c128.AxpyInc(-x[jx], a[j*lda:j*lda+j], x, uintptr(j), 1, uintptr(incX), 0, uintptr(kx))
					}
					jx -= incX
				}
			}
		}
		return
	}

	// Form x = inv(A^H)*x.
	if uplo == blas.Upper {
		if incX == 1 {
			for j := 0; j < n; j++ {
				if diag == blas.NonUnit {
					x[j] /= cmplx.Conj(a[j*lda+j])
				}
				xj := x[j]
				for i := j + 1; i < n; i++ {
					x[i] -= xj * cmplx.Conj(a[j*lda+i])
				}
			}
		} else {
			jx := kx
			for j := 0; j < n; j++ {
				if diag == blas.NonUnit {
					x[jx] /= cmplx.Conj(a[j*lda+j])
				}
				xj := x[jx]
				ix := jx + incX
				for i := j + 1; i < n; i++ {
					x[ix] -= xj * cmplx.Conj(a[j*lda+i])
					ix += incX
				}
				jx += incX
			}
		}
	} else {
		if incX == 1 {
			for j := n - 1; j >= 0; j-- {
				if diag == blas.NonUnit {
					x[j] /= cmplx.Conj(a[j*lda+j])
				}
				xj := x[j]
				for i := 0; i < j; i++ {
					x[i] -= xj * cmplx.Conj(a[j*lda+i])
				}
			}
		} else {
			jx := kx + (n-1)*incX
			for j := n - 1; j >= 0; j-- {
				if diag == blas.NonUnit {
					x[jx] /= cmplx.Conj(a[j*lda+j])
				}
				xj := x[jx]
				ix := kx
				for i := 0; i < j; i++ {
					x[ix] -= xj * cmplx.Conj(a[j*lda+i])
					ix += incX
				}
				jx -= incX
			}
		}
	}
}
