// Copyright ©2014 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/internal/asm/f64"
)

var _ blas.Float64Level2 = Implementation{}

// Dger performs the rank-one operation
//  A += alpha * x * yᵀ
// where A is an m×n dense matrix, x and y are vectors, and alpha is a scalar.
func (Implementation) Dger(m, n int, alpha float64, x []float64, incX int, y []float64, incY int, a []float64, lda int) {
	if m < 0 {
		panic(mLT0)
	}
	if n < 0 {
		panic(nLT0)
	}
	if lda < max(1, n) {
		panic(badLdA)
	}
	if incX == 0 {
		panic(zeroIncX)
	}
	if incY == 0 {
		panic(zeroIncY)
	}

	// Quick return if possible.
	if m == 0 || n == 0 {
		return
	}

	// For zero matrix size the following slice length checks are trivially satisfied.
	if (incX > 0 && len(x) <= (m-1)*incX) || (incX < 0 && len(x) <= (1-m)*incX) {
		panic(shortX)
	}
	if (incY > 0 && len(y) <= (n-1)*incY) || (incY < 0 && len(y) <= (1-n)*incY) {
		panic(shortY)
	}
	if len(a) < lda*(m-1)+n {
		panic(shortA)
	}

	// Quick return if possible.
	if alpha == 0 {
		return
	}
	f64.Ger(uintptr(m), uintptr(n),
		alpha,
		x, uintptr(incX),
		y, uintptr(incY),
		a, uintptr(lda))
}

// Dgbmv performs one of the matrix-vector operations
//  y = alpha * A * x + beta * y   if tA == blas.NoTrans
//  y = alpha * Aᵀ * x + beta * y  if tA == blas.Trans or blas.ConjTrans
// where A is an m×n band matrix with kL sub-diagonals and kU super-diagonals,
// x and y are vectors, and alpha and beta are scalars.
func (Implementation) Dgbmv(tA blas.Transpose, m, n, kL, kU int, alpha float64, a []float64, lda int, x []float64, incX int, beta float64, y []float64, incY int) {
	if tA != blas.NoTrans && tA != blas.Trans && tA != blas.ConjTrans {
		panic(badTranspose)
	}
	if m < 0 {
		panic(mLT0)
	}
	if n < 0 {
		panic(nLT0)
	}
	if kL < 0 {
		panic(kLLT0)
	}
	if kU < 0 {
		panic(kULT0)
	}
	if lda < kL+kU+1 {
		panic(badLdA)
	}
	if incX == 0 {
		panic(zeroIncX)
	}
	if incY == 0 {
		panic(zeroIncY)
	}

	// Quick return if possible.
	if m == 0 || n == 0 {
		return
	}

	// For zero matrix size the following slice length checks are trivially satisfied.
	if len(a) < lda*(min(m, n+kL)-1)+kL+kU+1 {
		panic(shortA)
	}
	lenX := m
	lenY := n
	if tA == blas.NoTrans {
		lenX = n
		lenY = m
	}
	if (incX > 0 && len(x) <= (lenX-1)*incX) || (incX < 0 && len(x) <= (1-lenX)*incX) {
		panic(shortX)
	}
	if (incY > 0 && len(y) <= (lenY-1)*incY) || (incY < 0 && len(y) <= (1-lenY)*incY) {
		panic(shortY)
	}

	// Quick return if possible.
	if alpha == 0 && beta == 1 {
		return
	}

	var kx, ky int
	if incX < 0 {
		kx = -(lenX - 1) * incX
	}
	if incY < 0 {
		ky = -(lenY - 1) * incY
	}

	// Form y = beta * y.
	if beta != 1 {
		if incY == 1 {
			if beta == 0 {
				for i := range y[:lenY] {
					y[i] = 0
				}
			} else {
				f64.ScalUnitary(beta, y[:lenY])
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
					f64.ScalInc(beta, y, uintptr(lenY), uintptr(incY))
				} else {
					f64.ScalInc(beta, y, uintptr(lenY), uintptr(-incY))
				}
			}
		}
	}

	if alpha == 0 {
		return
	}

	// i and j are indices of the compacted banded matrix.
	// off is the offset into the dense matrix (off + j = densej)
	nCol := kU + 1 + kL
	if tA == blas.NoTrans {
		iy := ky
		if incX == 1 {
			for i := 0; i < min(m, n+kL); i++ {
				l := max(0, kL-i)
				u := min(nCol, n+kL-i)
				off := max(0, i-kL)
				atmp := a[i*lda+l : i*lda+u]
				xtmp := x[off : off+u-l]
				var sum float64
				for j, v := range atmp {
					sum += xtmp[j] * v
				}
				y[iy] += sum * alpha
				iy += incY
			}
			return
		}
		for i := 0; i < min(m, n+kL); i++ {
			l := max(0, kL-i)
			u := min(nCol, n+kL-i)
			off := max(0, i-kL)
			atmp := a[i*lda+l : i*lda+u]
			jx := kx
			var sum float64
			for _, v := range atmp {
				sum += x[off*incX+jx] * v
				jx += incX
			}
			y[iy] += sum * alpha
			iy += incY
		}
		return
	}
	if incX == 1 {
		for i := 0; i < min(m, n+kL); i++ {
			l := max(0, kL-i)
			u := min(nCol, n+kL-i)
			off := max(0, i-kL)
			atmp := a[i*lda+l : i*lda+u]
			tmp := alpha * x[i]
			jy := ky
			for _, v := range atmp {
				y[jy+off*incY] += tmp * v
				jy += incY
			}
		}
		return
	}
	ix := kx
	for i := 0; i < min(m, n+kL); i++ {
		l := max(0, kL-i)
		u := min(nCol, n+kL-i)
		off := max(0, i-kL)
		atmp := a[i*lda+l : i*lda+u]
		tmp := alpha * x[ix]
		jy := ky
		for _, v := range atmp {
			y[jy+off*incY] += tmp * v
			jy += incY
		}
		ix += incX
	}
}

// Dtrmv performs one of the matrix-vector operations
//  x = A * x   if tA == blas.NoTrans
//  x = Aᵀ * x  if tA == blas.Trans or blas.ConjTrans
// where A is an n×n triangular matrix, and x is a vector.
func (Implementation) Dtrmv(ul blas.Uplo, tA blas.Transpose, d blas.Diag, n int, a []float64, lda int, x []float64, incX int) {
	if ul != blas.Lower && ul != blas.Upper {
		panic(badUplo)
	}
	if tA != blas.NoTrans && tA != blas.Trans && tA != blas.ConjTrans {
		panic(badTranspose)
	}
	if d != blas.NonUnit && d != blas.Unit {
		panic(badDiag)
	}
	if n < 0 {
		panic(nLT0)
	}
	if lda < max(1, n) {
		panic(badLdA)
	}
	if incX == 0 {
		panic(zeroIncX)
	}

	// Quick return if possible.
	if n == 0 {
		return
	}

	// For zero matrix size the following slice length checks are trivially satisfied.
	if len(a) < lda*(n-1)+n {
		panic(shortA)
	}
	if (incX > 0 && len(x) <= (n-1)*incX) || (incX < 0 && len(x) <= (1-n)*incX) {
		panic(shortX)
	}

	nonUnit := d != blas.Unit
	if n == 1 {
		if nonUnit {
			x[0] *= a[0]
		}
		return
	}
	var kx int
	if incX <= 0 {
		kx = -(n - 1) * incX
	}
	if tA == blas.NoTrans {
		if ul == blas.Upper {
			if incX == 1 {
				for i := 0; i < n; i++ {
					ilda := i * lda
					var tmp float64
					if nonUnit {
						tmp = a[ilda+i] * x[i]
					} else {
						tmp = x[i]
					}
					x[i] = tmp + f64.DotUnitary(a[ilda+i+1:ilda+n], x[i+1:n])
				}
				return
			}
			ix := kx
			for i := 0; i < n; i++ {
				ilda := i * lda
				var tmp float64
				if nonUnit {
					tmp = a[ilda+i] * x[ix]
				} else {
					tmp = x[ix]
				}
				x[ix] = tmp + f64.DotInc(x, a[ilda+i+1:ilda+n], uintptr(n-i-1), uintptr(incX), 1, uintptr(ix+incX), 0)
				ix += incX
			}
			return
		}
		if incX == 1 {
			for i := n - 1; i >= 0; i-- {
				ilda := i * lda
				var tmp float64
				if nonUnit {
					tmp += a[ilda+i] * x[i]
				} else {
					tmp = x[i]
				}
				x[i] = tmp + f64.DotUnitary(a[ilda:ilda+i], x[:i])
			}
			return
		}
		ix := kx + (n-1)*incX
		for i := n - 1; i >= 0; i-- {
			ilda := i * lda
			var tmp float64
			if nonUnit {
				tmp = a[ilda+i] * x[ix]
			} else {
				tmp = x[ix]
			}
			x[ix] = tmp + f64.DotInc(x, a[ilda:ilda+i], uintptr(i), uintptr(incX), 1, uintptr(kx), 0)
			ix -= incX
		}
		return
	}
	// Cases where a is transposed.
	if ul == blas.Upper {
		if incX == 1 {
			for i := n - 1; i >= 0; i-- {
				ilda := i * lda
				xi := x[i]
				f64.AxpyUnitary(xi, a[ilda+i+1:ilda+n], x[i+1:n])
				if nonUnit {
					x[i] *= a[ilda+i]
				}
			}
			return
		}
		ix := kx + (n-1)*incX
		for i := n - 1; i >= 0; i-- {
			ilda := i * lda
			xi := x[ix]
			f64.AxpyInc(xi, a[ilda+i+1:ilda+n], x, uintptr(n-i-1), 1, uintptr(incX), 0, uintptr(kx+(i+1)*incX))
			if nonUnit {
				x[ix] *= a[ilda+i]
			}
			ix -= incX
		}
		return
	}
	if incX == 1 {
		for i := 0; i < n; i++ {
			ilda := i * lda
			xi := x[i]
			f64.AxpyUnitary(xi, a[ilda:ilda+i], x[:i])
			if nonUnit {
				x[i] *= a[i*lda+i]
			}
		}
		return
	}
	ix := kx
	for i := 0; i < n; i++ {
		ilda := i * lda
		xi := x[ix]
		f64.AxpyInc(xi, a[ilda:ilda+i], x, uintptr(i), 1, uintptr(incX), 0, uintptr(kx))
		if nonUnit {
			x[ix] *= a[ilda+i]
		}
		ix += incX
	}
}

// Dtrsv solves one of the systems of equations
//  A * x = b   if tA == blas.NoTrans
//  Aᵀ * x = b  if tA == blas.Trans or blas.ConjTrans
// where A is an n×n triangular matrix, and x and b are vectors.
//
// At entry to the function, x contains the values of b, and the result is
// stored in-place into x.
//
// No test for singularity or near-singularity is included in this
// routine. Such tests must be performed before calling this routine.
func (Implementation) Dtrsv(ul blas.Uplo, tA blas.Transpose, d blas.Diag, n int, a []float64, lda int, x []float64, incX int) {
	if ul != blas.Lower && ul != blas.Upper {
		panic(badUplo)
	}
	if tA != blas.NoTrans && tA != blas.Trans && tA != blas.ConjTrans {
		panic(badTranspose)
	}
	if d != blas.NonUnit && d != blas.Unit {
		panic(badDiag)
	}
	if n < 0 {
		panic(nLT0)
	}
	if lda < max(1, n) {
		panic(badLdA)
	}
	if incX == 0 {
		panic(zeroIncX)
	}

	// Quick return if possible.
	if n == 0 {
		return
	}

	// For zero matrix size the following slice length checks are trivially satisfied.
	if len(a) < lda*(n-1)+n {
		panic(shortA)
	}
	if (incX > 0 && len(x) <= (n-1)*incX) || (incX < 0 && len(x) <= (1-n)*incX) {
		panic(shortX)
	}

	if n == 1 {
		if d == blas.NonUnit {
			x[0] /= a[0]
		}
		return
	}

	var kx int
	if incX < 0 {
		kx = -(n - 1) * incX
	}
	nonUnit := d == blas.NonUnit
	if tA == blas.NoTrans {
		if ul == blas.Upper {
			if incX == 1 {
				for i := n - 1; i >= 0; i-- {
					var sum float64
					atmp := a[i*lda+i+1 : i*lda+n]
					for j, v := range atmp {
						jv := i + j + 1
						sum += x[jv] * v
					}
					x[i] -= sum
					if nonUnit {
						x[i] /= a[i*lda+i]
					}
				}
				return
			}
			ix := kx + (n-1)*incX
			for i := n - 1; i >= 0; i-- {
				var sum float64
				jx := ix + incX
				atmp := a[i*lda+i+1 : i*lda+n]
				for _, v := range atmp {
					sum += x[jx] * v
					jx += incX
				}
				x[ix] -= sum
				if nonUnit {
					x[ix] /= a[i*lda+i]
				}
				ix -= incX
			}
			return
		}
		if incX == 1 {
			for i := 0; i < n; i++ {
				var sum float64
				atmp := a[i*lda : i*lda+i]
				for j, v := range atmp {
					sum += x[j] * v
				}
				x[i] -= sum
				if nonUnit {
					x[i] /= a[i*lda+i]
				}
			}
			return
		}
		ix := kx
		for i := 0; i < n; i++ {
			jx := kx
			var sum float64
			atmp := a[i*lda : i*lda+i]
			for _, v := range atmp {
				sum += x[jx] * v
				jx += incX
			}
			x[ix] -= sum
			if nonUnit {
				x[ix] /= a[i*lda+i]
			}
			ix += incX
		}
		return
	}
	// Cases where a is transposed.
	if ul == blas.Upper {
		if incX == 1 {
			for i := 0; i < n; i++ {
				if nonUnit {
					x[i] /= a[i*lda+i]
				}
				xi := x[i]
				atmp := a[i*lda+i+1 : i*lda+n]
				for j, v := range atmp {
					jv := j + i + 1
					x[jv] -= v * xi
				}
			}
			return
		}
		ix := kx
		for i := 0; i < n; i++ {
			if nonUnit {
				x[ix] /= a[i*lda+i]
			}
			xi := x[ix]
			jx := kx + (i+1)*incX
			atmp := a[i*lda+i+1 : i*lda+n]
			for _, v := range atmp {
				x[jx] -= v * xi
				jx += incX
			}
			ix += incX
		}
		return
	}
	if incX == 1 {
		for i := n - 1; i >= 0; i-- {
			if nonUnit {
				x[i] /= a[i*lda+i]
			}
			xi := x[i]
			atmp := a[i*lda : i*lda+i]
			for j, v := range atmp {
				x[j] -= v * xi
			}
		}
		return
	}
	ix := kx + (n-1)*incX
	for i := n - 1; i >= 0; i-- {
		if nonUnit {
			x[ix] /= a[i*lda+i]
		}
		xi := x[ix]
		jx := kx
		atmp := a[i*lda : i*lda+i]
		for _, v := range atmp {
			x[jx] -= v * xi
			jx += incX
		}
		ix -= incX
	}
}

// Dsymv performs the matrix-vector operation
//  y = alpha * A * x + beta * y
// where A is an n×n symmetric matrix, x and y are vectors, and alpha and
// beta are scalars.
func (Implementation) Dsymv(ul blas.Uplo, n int, alpha float64, a []float64, lda int, x []float64, incX int, beta float64, y []float64, incY int) {
	if ul != blas.Lower && ul != blas.Upper {
		panic(badUplo)
	}
	if n < 0 {
		panic(nLT0)
	}
	if lda < max(1, n) {
		panic(badLdA)
	}
	if incX == 0 {
		panic(zeroIncX)
	}
	if incY == 0 {
		panic(zeroIncY)
	}

	// Quick return if possible.
	if n == 0 {
		return
	}

	// For zero matrix size the following slice length checks are trivially satisfied.
	if len(a) < lda*(n-1)+n {
		panic(shortA)
	}
	if (incX > 0 && len(x) <= (n-1)*incX) || (incX < 0 && len(x) <= (1-n)*incX) {
		panic(shortX)
	}
	if (incY > 0 && len(y) <= (n-1)*incY) || (incY < 0 && len(y) <= (1-n)*incY) {
		panic(shortY)
	}

	// Quick return if possible.
	if alpha == 0 && beta == 1 {
		return
	}

	// Set up start points
	var kx, ky int
	if incX < 0 {
		kx = -(n - 1) * incX
	}
	if incY < 0 {
		ky = -(n - 1) * incY
	}

	// Form y = beta * y
	if beta != 1 {
		if incY == 1 {
			if beta == 0 {
				for i := range y[:n] {
					y[i] = 0
				}
			} else {
				f64.ScalUnitary(beta, y[:n])
			}
		} else {
			iy := ky
			if beta == 0 {
				for i := 0; i < n; i++ {
					y[iy] = 0
					iy += incY
				}
			} else {
				if incY > 0 {
					f64.ScalInc(beta, y, uintptr(n), uintptr(incY))
				} else {
					f64.ScalInc(beta, y, uintptr(n), uintptr(-incY))
				}
			}
		}
	}

	if alpha == 0 {
		return
	}

	if n == 1 {
		y[0] += alpha * a[0] * x[0]
		return
	}

	if ul == blas.Upper {
		if incX == 1 {
			iy := ky
			for i := 0; i < n; i++ {
				xv := x[i] * alpha
				sum := x[i] * a[i*lda+i]
				jy := ky + (i+1)*incY
				atmp := a[i*lda+i+1 : i*lda+n]
				for j, v := range atmp {
					jp := j + i + 1
					sum += x[jp] * v
					y[jy] += xv * v
					jy += incY
				}
				y[iy] += alpha * sum
				iy += incY
			}
			return
		}
		ix := kx
		iy := ky
		for i := 0; i < n; i++ {
			xv := x[ix] * alpha
			sum := x[ix] * a[i*lda+i]
			jx := kx + (i+1)*incX
			jy := ky + (i+1)*incY
			atmp := a[i*lda+i+1 : i*lda+n]
			for _, v := range atmp {
				sum += x[jx] * v
				y[jy] += xv * v
				jx += incX
				jy += incY
			}
			y[iy] += alpha * sum
			ix += incX
			iy += incY
		}
		return
	}
	// Cases where a is lower triangular.
	if incX == 1 {
		iy := ky
		for i := 0; i < n; i++ {
			jy := ky
			xv := alpha * x[i]
			atmp := a[i*lda : i*lda+i]
			var sum float64
			for j, v := range atmp {
				sum += x[j] * v
				y[jy] += xv * v
				jy += incY
			}
			sum += x[i] * a[i*lda+i]
			sum *= alpha
			y[iy] += sum
			iy += incY
		}
		return
	}
	ix := kx
	iy := ky
	for i := 0; i < n; i++ {
		jx := kx
		jy := ky
		xv := alpha * x[ix]
		atmp := a[i*lda : i*lda+i]
		var sum float64
		for _, v := range atmp {
			sum += x[jx] * v
			y[jy] += xv * v
			jx += incX
			jy += incY
		}
		sum += x[ix] * a[i*lda+i]
		sum *= alpha
		y[iy] += sum
		ix += incX
		iy += incY
	}
}

// Dtbmv performs one of the matrix-vector operations
//  x = A * x   if tA == blas.NoTrans
//  x = Aᵀ * x  if tA == blas.Trans or blas.ConjTrans
// where A is an n×n triangular band matrix with k+1 diagonals, and x is a vector.
func (Implementation) Dtbmv(ul blas.Uplo, tA blas.Transpose, d blas.Diag, n, k int, a []float64, lda int, x []float64, incX int) {
	if ul != blas.Lower && ul != blas.Upper {
		panic(badUplo)
	}
	if tA != blas.NoTrans && tA != blas.Trans && tA != blas.ConjTrans {
		panic(badTranspose)
	}
	if d != blas.NonUnit && d != blas.Unit {
		panic(badDiag)
	}
	if n < 0 {
		panic(nLT0)
	}
	if k < 0 {
		panic(kLT0)
	}
	if lda < k+1 {
		panic(badLdA)
	}
	if incX == 0 {
		panic(zeroIncX)
	}

	// Quick return if possible.
	if n == 0 {
		return
	}

	// For zero matrix size the following slice length checks are trivially satisfied.
	if len(a) < lda*(n-1)+k+1 {
		panic(shortA)
	}
	if (incX > 0 && len(x) <= (n-1)*incX) || (incX < 0 && len(x) <= (1-n)*incX) {
		panic(shortX)
	}

	var kx int
	if incX < 0 {
		kx = -(n - 1) * incX
	}

	nonunit := d != blas.Unit

	if tA == blas.NoTrans {
		if ul == blas.Upper {
			if incX == 1 {
				for i := 0; i < n; i++ {
					u := min(1+k, n-i)
					var sum float64
					atmp := a[i*lda:]
					xtmp := x[i:]
					for j := 1; j < u; j++ {
						sum += xtmp[j] * atmp[j]
					}
					if nonunit {
						sum += xtmp[0] * atmp[0]
					} else {
						sum += xtmp[0]
					}
					x[i] = sum
				}
				return
			}
			ix := kx
			for i := 0; i < n; i++ {
				u := min(1+k, n-i)
				var sum float64
				atmp := a[i*lda:]
				jx := incX
				for j := 1; j < u; j++ {
					sum += x[ix+jx] * atmp[j]
					jx += incX
				}
				if nonunit {
					sum += x[ix] * atmp[0]
				} else {
					sum += x[ix]
				}
				x[ix] = sum
				ix += incX
			}
			return
		}
		if incX == 1 {
			for i := n - 1; i >= 0; i-- {
				l := max(0, k-i)
				atmp := a[i*lda:]
				var sum float64
				for j := l; j < k; j++ {
					sum += x[i-k+j] * atmp[j]
				}
				if nonunit {
					sum += x[i] * atmp[k]
				} else {
					sum += x[i]
				}
				x[i] = sum
			}
			return
		}
		ix := kx + (n-1)*incX
		for i := n - 1; i >= 0; i-- {
			l := max(0, k-i)
			atmp := a[i*lda:]
			var sum float64
			jx := l * incX
			for j := l; j < k; j++ {
				sum += x[ix-k*incX+jx] * atmp[j]
				jx += incX
			}
			if nonunit {
				sum += x[ix] * atmp[k]
			} else {
				sum += x[ix]
			}
			x[ix] = sum
			ix -= incX
		}
		return
	}
	if ul == blas.Upper {
		if incX == 1 {
			for i := n - 1; i >= 0; i-- {
				u := k + 1
				if i < u {
					u = i + 1
				}
				var sum float64
				for j := 1; j < u; j++ {
					sum += x[i-j] * a[(i-j)*lda+j]
				}
				if nonunit {
					sum += x[i] * a[i*lda]
				} else {
					sum += x[i]
				}
				x[i] = sum
			}
			return
		}
		ix := kx + (n-1)*incX
		for i := n - 1; i >= 0; i-- {
			u := k + 1
			if i < u {
				u = i + 1
			}
			var sum float64
			jx := incX
			for j := 1; j < u; j++ {
				sum += x[ix-jx] * a[(i-j)*lda+j]
				jx += incX
			}
			if nonunit {
				sum += x[ix] * a[i*lda]
			} else {
				sum += x[ix]
			}
			x[ix] = sum
			ix -= incX
		}
		return
	}
	if incX == 1 {
		for i := 0; i < n; i++ {
			u := k
			if i+k >= n {
				u = n - i - 1
			}
			var sum float64
			for j := 0; j < u; j++ {
				sum += x[i+j+1] * a[(i+j+1)*lda+k-j-1]
			}
			if nonunit {
				sum += x[i] * a[i*lda+k]
			} else {
				sum += x[i]
			}
			x[i] = sum
		}
		return
	}
	ix := kx
	for i := 0; i < n; i++ {
		u := k
		if i+k >= n {
			u = n - i - 1
		}
		var (
			sum float64
			jx  int
		)
		for j := 0; j < u; j++ {
			sum += x[ix+jx+incX] * a[(i+j+1)*lda+k-j-1]
			jx += incX
		}
		if nonunit {
			sum += x[ix] * a[i*lda+k]
		} else {
			sum += x[ix]
		}
		x[ix] = sum
		ix += incX
	}
}

// Dtpmv performs one of the matrix-vector operations
//  x = A * x   if tA == blas.NoTrans
//  x = Aᵀ * x  if tA == blas.Trans or blas.ConjTrans
// where A is an n×n triangular matrix in packed format, and x is a vector.
func (Implementation) Dtpmv(ul blas.Uplo, tA blas.Transpose, d blas.Diag, n int, ap []float64, x []float64, incX int) {
	if ul != blas.Lower && ul != blas.Upper {
		panic(badUplo)
	}
	if tA != blas.NoTrans && tA != blas.Trans && tA != blas.ConjTrans {
		panic(badTranspose)
	}
	if d != blas.NonUnit && d != blas.Unit {
		panic(badDiag)
	}
	if n < 0 {
		panic(nLT0)
	}
	if incX == 0 {
		panic(zeroIncX)
	}

	// Quick return if possible.
	if n == 0 {
		return
	}

	// For zero matrix size the following slice length checks are trivially satisfied.
	if len(ap) < n*(n+1)/2 {
		panic(shortAP)
	}
	if (incX > 0 && len(x) <= (n-1)*incX) || (incX < 0 && len(x) <= (1-n)*incX) {
		panic(shortX)
	}

	var kx int
	if incX < 0 {
		kx = -(n - 1) * incX
	}

	nonUnit := d == blas.NonUnit
	var offset int // Offset is the index of (i,i)
	if tA == blas.NoTrans {
		if ul == blas.Upper {
			if incX == 1 {
				for i := 0; i < n; i++ {
					xi := x[i]
					if nonUnit {
						xi *= ap[offset]
					}
					atmp := ap[offset+1 : offset+n-i]
					xtmp := x[i+1:]
					for j, v := range atmp {
						xi += v * xtmp[j]
					}
					x[i] = xi
					offset += n - i
				}
				return
			}
			ix := kx
			for i := 0; i < n; i++ {
				xix := x[ix]
				if nonUnit {
					xix *= ap[offset]
				}
				atmp := ap[offset+1 : offset+n-i]
				jx := kx + (i+1)*incX
				for _, v := range atmp {
					xix += v * x[jx]
					jx += incX
				}
				x[ix] = xix
				offset += n - i
				ix += incX
			}
			return
		}
		if incX == 1 {
			offset = n*(n+1)/2 - 1
			for i := n - 1; i >= 0; i-- {
				xi := x[i]
				if nonUnit {
					xi *= ap[offset]
				}
				atmp := ap[offset-i : offset]
				for j, v := range atmp {
					xi += v * x[j]
				}
				x[i] = xi
				offset -= i + 1
			}
			return
		}
		ix := kx + (n-1)*incX
		offset = n*(n+1)/2 - 1
		for i := n - 1; i >= 0; i-- {
			xix := x[ix]
			if nonUnit {
				xix *= ap[offset]
			}
			atmp := ap[offset-i : offset]
			jx := kx
			for _, v := range atmp {
				xix += v * x[jx]
				jx += incX
			}
			x[ix] = xix
			offset -= i + 1
			ix -= incX
		}
		return
	}
	// Cases where ap is transposed.
	if ul == blas.Upper {
		if incX == 1 {
			offset = n*(n+1)/2 - 1
			for i := n - 1; i >= 0; i-- {
				xi := x[i]
				atmp := ap[offset+1 : offset+n-i]
				xtmp := x[i+1:]
				for j, v := range atmp {
					xtmp[j] += v * xi
				}
				if nonUnit {
					x[i] *= ap[offset]
				}
				offset -= n - i + 1
			}
			return
		}
		ix := kx + (n-1)*incX
		offset = n*(n+1)/2 - 1
		for i := n - 1; i >= 0; i-- {
			xix := x[ix]
			jx := kx + (i+1)*incX
			atmp := ap[offset+1 : offset+n-i]
			for _, v := range atmp {
				x[jx] += v * xix
				jx += incX
			}
			if nonUnit {
				x[ix] *= ap[offset]
			}
			offset -= n - i + 1
			ix -= incX
		}
		return
	}
	if incX == 1 {
		for i := 0; i < n; i++ {
			xi := x[i]
			atmp := ap[offset-i : offset]
			for j, v := range atmp {
				x[j] += v * xi
			}
			if nonUnit {
				x[i] *= ap[offset]
			}
			offset += i + 2
		}
		return
	}
	ix := kx
	for i := 0; i < n; i++ {
		xix := x[ix]
		jx := kx
		atmp := ap[offset-i : offset]
		for _, v := range atmp {
			x[jx] += v * xix
			jx += incX
		}
		if nonUnit {
			x[ix] *= ap[offset]
		}
		ix += incX
		offset += i + 2
	}
}

// Dtbsv solves one of the systems of equations
//  A * x = b   if tA == blas.NoTrans
//  Aᵀ * x = b  if tA == blas.Trans or tA == blas.ConjTrans
// where A is an n×n triangular band matrix with k+1 diagonals,
// and x and b are vectors.
//
// At entry to the function, x contains the values of b, and the result is
// stored in-place into x.
//
// No test for singularity or near-singularity is included in this
// routine. Such tests must be performed before calling this routine.
func (Implementation) Dtbsv(ul blas.Uplo, tA blas.Transpose, d blas.Diag, n, k int, a []float64, lda int, x []float64, incX int) {
	if ul != blas.Lower && ul != blas.Upper {
		panic(badUplo)
	}
	if tA != blas.NoTrans && tA != blas.Trans && tA != blas.ConjTrans {
		panic(badTranspose)
	}
	if d != blas.NonUnit && d != blas.Unit {
		panic(badDiag)
	}
	if n < 0 {
		panic(nLT0)
	}
	if k < 0 {
		panic(kLT0)
	}
	if lda < k+1 {
		panic(badLdA)
	}
	if incX == 0 {
		panic(zeroIncX)
	}

	// Quick return if possible.
	if n == 0 {
		return
	}

	// For zero matrix size the following slice length checks are trivially satisfied.
	if len(a) < lda*(n-1)+k+1 {
		panic(shortA)
	}
	if (incX > 0 && len(x) <= (n-1)*incX) || (incX < 0 && len(x) <= (1-n)*incX) {
		panic(shortX)
	}

	var kx int
	if incX < 0 {
		kx = -(n - 1) * incX
	}
	nonUnit := d == blas.NonUnit
	// Form x = A^-1 x.
	// Several cases below use subslices for speed improvement.
	// The incX != 1 cases usually do not because incX may be negative.
	if tA == blas.NoTrans {
		if ul == blas.Upper {
			if incX == 1 {
				for i := n - 1; i >= 0; i-- {
					bands := k
					if i+bands >= n {
						bands = n - i - 1
					}
					atmp := a[i*lda+1:]
					xtmp := x[i+1 : i+bands+1]
					var sum float64
					for j, v := range xtmp {
						sum += v * atmp[j]
					}
					x[i] -= sum
					if nonUnit {
						x[i] /= a[i*lda]
					}
				}
				return
			}
			ix := kx + (n-1)*incX
			for i := n - 1; i >= 0; i-- {
				max := k + 1
				if i+max > n {
					max = n - i
				}
				atmp := a[i*lda:]
				var (
					jx  int
					sum float64
				)
				for j := 1; j < max; j++ {
					jx += incX
					sum += x[ix+jx] * atmp[j]
				}
				x[ix] -= sum
				if nonUnit {
					x[ix] /= atmp[0]
				}
				ix -= incX
			}
			return
		}
		if incX == 1 {
			for i := 0; i < n; i++ {
				bands := k
				if i-k < 0 {
					bands = i
				}
				atmp := a[i*lda+k-bands:]
				xtmp := x[i-bands : i]
				var sum float64
				for j, v := range xtmp {
					sum += v * atmp[j]
				}
				x[i] -= sum
				if nonUnit {
					x[i] /= atmp[bands]
				}
			}
			return
		}
		ix := kx
		for i := 0; i < n; i++ {
			bands := k
			if i-k < 0 {
				bands = i
			}
			atmp := a[i*lda+k-bands:]
			var (
				sum float64
				jx  int
			)
			for j := 0; j < bands; j++ {
				sum += x[ix-bands*incX+jx] * atmp[j]
				jx += incX
			}
			x[ix] -= sum
			if nonUnit {
				x[ix] /= atmp[bands]
			}
			ix += incX
		}
		return
	}
	// Cases where a is transposed.
	if ul == blas.Upper {
		if incX == 1 {
			for i := 0; i < n; i++ {
				bands := k
				if i-k < 0 {
					bands = i
				}
				var sum float64
				for j := 0; j < bands; j++ {
					sum += x[i-bands+j] * a[(i-bands+j)*lda+bands-j]
				}
				x[i] -= sum
				if nonUnit {
					x[i] /= a[i*lda]
				}
			}
			return
		}
		ix := kx
		for i := 0; i < n; i++ {
			bands := k
			if i-k < 0 {
				bands = i
			}
			var (
				sum float64
				jx  int
			)
			for j := 0; j < bands; j++ {
				sum += x[ix-bands*incX+jx] * a[(i-bands+j)*lda+bands-j]
				jx += incX
			}
			x[ix] -= sum
			if nonUnit {
				x[ix] /= a[i*lda]
			}
			ix += incX
		}
		return
	}
	if incX == 1 {
		for i := n - 1; i >= 0; i-- {
			bands := k
			if i+bands >= n {
				bands = n - i - 1
			}
			var sum float64
			xtmp := x[i+1 : i+1+bands]
			for j, v := range xtmp {
				sum += v * a[(i+j+1)*lda+k-j-1]
			}
			x[i] -= sum
			if nonUnit {
				x[i] /= a[i*lda+k]
			}
		}
		return
	}
	ix := kx + (n-1)*incX
	for i := n - 1; i >= 0; i-- {
		bands := k
		if i+bands >= n {
			bands = n - i - 1
		}
		var (
			sum float64
			jx  int
		)
		for j := 0; j < bands; j++ {
			sum += x[ix+jx+incX] * a[(i+j+1)*lda+k-j-1]
			jx += incX
		}
		x[ix] -= sum
		if nonUnit {
			x[ix] /= a[i*lda+k]
		}
		ix -= incX
	}
}

// Dsbmv performs the matrix-vector operation
//  y = alpha * A * x + beta * y
// where A is an n×n symmetric band matrix with k super-diagonals, x and y are
// vectors, and alpha and beta are scalars.
func (Implementation) Dsbmv(ul blas.Uplo, n, k int, alpha float64, a []float64, lda int, x []float64, incX int, beta float64, y []float64, incY int) {
	if ul != blas.Lower && ul != blas.Upper {
		panic(badUplo)
	}
	if n < 0 {
		panic(nLT0)
	}
	if k < 0 {
		panic(kLT0)
	}
	if lda < k+1 {
		panic(badLdA)
	}
	if incX == 0 {
		panic(zeroIncX)
	}
	if incY == 0 {
		panic(zeroIncY)
	}

	// Quick return if possible.
	if n == 0 {
		return
	}

	// For zero matrix size the following slice length checks are trivially satisfied.
	if len(a) < lda*(n-1)+k+1 {
		panic(shortA)
	}
	if (incX > 0 && len(x) <= (n-1)*incX) || (incX < 0 && len(x) <= (1-n)*incX) {
		panic(shortX)
	}
	if (incY > 0 && len(y) <= (n-1)*incY) || (incY < 0 && len(y) <= (1-n)*incY) {
		panic(shortY)
	}

	// Quick return if possible.
	if alpha == 0 && beta == 1 {
		return
	}

	// Set up indexes
	lenX := n
	lenY := n
	var kx, ky int
	if incX < 0 {
		kx = -(lenX - 1) * incX
	}
	if incY < 0 {
		ky = -(lenY - 1) * incY
	}

	// Form y = beta * y.
	if beta != 1 {
		if incY == 1 {
			if beta == 0 {
				for i := range y[:n] {
					y[i] = 0
				}
			} else {
				f64.ScalUnitary(beta, y[:n])
			}
		} else {
			iy := ky
			if beta == 0 {
				for i := 0; i < n; i++ {
					y[iy] = 0
					iy += incY
				}
			} else {
				if incY > 0 {
					f64.ScalInc(beta, y, uintptr(n), uintptr(incY))
				} else {
					f64.ScalInc(beta, y, uintptr(n), uintptr(-incY))
				}
			}
		}
	}

	if alpha == 0 {
		return
	}

	if ul == blas.Upper {
		if incX == 1 {
			iy := ky
			for i := 0; i < n; i++ {
				atmp := a[i*lda:]
				tmp := alpha * x[i]
				sum := tmp * atmp[0]
				u := min(k, n-i-1)
				jy := incY
				for j := 1; j <= u; j++ {
					v := atmp[j]
					sum += alpha * x[i+j] * v
					y[iy+jy] += tmp * v
					jy += incY
				}
				y[iy] += sum
				iy += incY
			}
			return
		}
		ix := kx
		iy := ky
		for i := 0; i < n; i++ {
			atmp := a[i*lda:]
			tmp := alpha * x[ix]
			sum := tmp * atmp[0]
			u := min(k, n-i-1)
			jx := incX
			jy := incY
			for j := 1; j <= u; j++ {
				v := atmp[j]
				sum += alpha * x[ix+jx] * v
				y[iy+jy] += tmp * v
				jx += incX
				jy += incY
			}
			y[iy] += sum
			ix += incX
			iy += incY
		}
		return
	}

	// Casses where a has bands below the diagonal.
	if incX == 1 {
		iy := ky
		for i := 0; i < n; i++ {
			l := max(0, k-i)
			tmp := alpha * x[i]
			jy := l * incY
			atmp := a[i*lda:]
			for j := l; j < k; j++ {
				v := atmp[j]
				y[iy] += alpha * v * x[i-k+j]
				y[iy-k*incY+jy] += tmp * v
				jy += incY
			}
			y[iy] += tmp * atmp[k]
			iy += incY
		}
		return
	}
	ix := kx
	iy := ky
	for i := 0; i < n; i++ {
		l := max(0, k-i)
		tmp := alpha * x[ix]
		jx := l * incX
		jy := l * incY
		atmp := a[i*lda:]
		for j := l; j < k; j++ {
			v := atmp[j]
			y[iy] += alpha * v * x[ix-k*incX+jx]
			y[iy-k*incY+jy] += tmp * v
			jx += incX
			jy += incY
		}
		y[iy] += tmp * atmp[k]
		ix += incX
		iy += incY
	}
}

// Dsyr performs the symmetric rank-one update
//  A += alpha * x * xᵀ
// where A is an n×n symmetric matrix, and x is a vector.
func (Implementation) Dsyr(ul blas.Uplo, n int, alpha float64, x []float64, incX int, a []float64, lda int) {
	if ul != blas.Lower && ul != blas.Upper {
		panic(badUplo)
	}
	if n < 0 {
		panic(nLT0)
	}
	if lda < max(1, n) {
		panic(badLdA)
	}
	if incX == 0 {
		panic(zeroIncX)
	}

	// Quick return if possible.
	if n == 0 {
		return
	}

	// For zero matrix size the following slice length checks are trivially satisfied.
	if (incX > 0 && len(x) <= (n-1)*incX) || (incX < 0 && len(x) <= (1-n)*incX) {
		panic(shortX)
	}
	if len(a) < lda*(n-1)+n {
		panic(shortA)
	}

	// Quick return if possible.
	if alpha == 0 {
		return
	}

	lenX := n
	var kx int
	if incX < 0 {
		kx = -(lenX - 1) * incX
	}
	if ul == blas.Upper {
		if incX == 1 {
			for i := 0; i < n; i++ {
				tmp := x[i] * alpha
				if tmp != 0 {
					atmp := a[i*lda+i : i*lda+n]
					xtmp := x[i:n]
					for j, v := range xtmp {
						atmp[j] += v * tmp
					}
				}
			}
			return
		}
		ix := kx
		for i := 0; i < n; i++ {
			tmp := x[ix] * alpha
			if tmp != 0 {
				jx := ix
				atmp := a[i*lda:]
				for j := i; j < n; j++ {
					atmp[j] += x[jx] * tmp
					jx += incX
				}
			}
			ix += incX
		}
		return
	}
	// Cases where a is lower triangular.
	if incX == 1 {
		for i := 0; i < n; i++ {
			tmp := x[i] * alpha
			if tmp != 0 {
				atmp := a[i*lda:]
				xtmp := x[:i+1]
				for j, v := range xtmp {
					atmp[j] += tmp * v
				}
			}
		}
		return
	}
	ix := kx
	for i := 0; i < n; i++ {
		tmp := x[ix] * alpha
		if tmp != 0 {
			atmp := a[i*lda:]
			jx := kx
			for j := 0; j < i+1; j++ {
				atmp[j] += tmp * x[jx]
				jx += incX
			}
		}
		ix += incX
	}
}

// Dsyr2 performs the symmetric rank-two update
//  A += alpha * x * yᵀ + alpha * y * xᵀ
// where A is an n×n symmetric matrix, x and y are vectors, and alpha is a scalar.
func (Implementation) Dsyr2(ul blas.Uplo, n int, alpha float64, x []float64, incX int, y []float64, incY int, a []float64, lda int) {
	if ul != blas.Lower && ul != blas.Upper {
		panic(badUplo)
	}
	if n < 0 {
		panic(nLT0)
	}
	if lda < max(1, n) {
		panic(badLdA)
	}
	if incX == 0 {
		panic(zeroIncX)
	}
	if incY == 0 {
		panic(zeroIncY)
	}

	// Quick return if possible.
	if n == 0 {
		return
	}

	// For zero matrix size the following slice length checks are trivially satisfied.
	if (incX > 0 && len(x) <= (n-1)*incX) || (incX < 0 && len(x) <= (1-n)*incX) {
		panic(shortX)
	}
	if (incY > 0 && len(y) <= (n-1)*incY) || (incY < 0 && len(y) <= (1-n)*incY) {
		panic(shortY)
	}
	if len(a) < lda*(n-1)+n {
		panic(shortA)
	}

	// Quick return if possible.
	if alpha == 0 {
		return
	}

	var ky, kx int
	if incY < 0 {
		ky = -(n - 1) * incY
	}
	if incX < 0 {
		kx = -(n - 1) * incX
	}
	if ul == blas.Upper {
		if incX == 1 && incY == 1 {
			for i := 0; i < n; i++ {
				xi := x[i]
				yi := y[i]
				atmp := a[i*lda:]
				for j := i; j < n; j++ {
					atmp[j] += alpha * (xi*y[j] + x[j]*yi)
				}
			}
			return
		}
		ix := kx
		iy := ky
		for i := 0; i < n; i++ {
			jx := kx + i*incX
			jy := ky + i*incY
			xi := x[ix]
			yi := y[iy]
			atmp := a[i*lda:]
			for j := i; j < n; j++ {
				atmp[j] += alpha * (xi*y[jy] + x[jx]*yi)
				jx += incX
				jy += incY
			}
			ix += incX
			iy += incY
		}
		return
	}
	if incX == 1 && incY == 1 {
		for i := 0; i < n; i++ {
			xi := x[i]
			yi := y[i]
			atmp := a[i*lda:]
			for j := 0; j <= i; j++ {
				atmp[j] += alpha * (xi*y[j] + x[j]*yi)
			}
		}
		return
	}
	ix := kx
	iy := ky
	for i := 0; i < n; i++ {
		jx := kx
		jy := ky
		xi := x[ix]
		yi := y[iy]
		atmp := a[i*lda:]
		for j := 0; j <= i; j++ {
			atmp[j] += alpha * (xi*y[jy] + x[jx]*yi)
			jx += incX
			jy += incY
		}
		ix += incX
		iy += incY
	}
}

// Dtpsv solves one of the systems of equations
//  A * x = b   if tA == blas.NoTrans
//  Aᵀ * x = b  if tA == blas.Trans or blas.ConjTrans
// where A is an n×n triangular matrix in packed format, and x and b are vectors.
//
// At entry to the function, x contains the values of b, and the result is
// stored in-place into x.
//
// No test for singularity or near-singularity is included in this
// routine. Such tests must be performed before calling this routine.
func (Implementation) Dtpsv(ul blas.Uplo, tA blas.Transpose, d blas.Diag, n int, ap []float64, x []float64, incX int) {
	if ul != blas.Lower && ul != blas.Upper {
		panic(badUplo)
	}
	if tA != blas.NoTrans && tA != blas.Trans && tA != blas.ConjTrans {
		panic(badTranspose)
	}
	if d != blas.NonUnit && d != blas.Unit {
		panic(badDiag)
	}
	if n < 0 {
		panic(nLT0)
	}
	if incX == 0 {
		panic(zeroIncX)
	}

	// Quick return if possible.
	if n == 0 {
		return
	}

	// For zero matrix size the following slice length checks are trivially satisfied.
	if len(ap) < n*(n+1)/2 {
		panic(shortAP)
	}
	if (incX > 0 && len(x) <= (n-1)*incX) || (incX < 0 && len(x) <= (1-n)*incX) {
		panic(shortX)
	}

	var kx int
	if incX < 0 {
		kx = -(n - 1) * incX
	}

	nonUnit := d == blas.NonUnit
	var offset int // Offset is the index of (i,i)
	if tA == blas.NoTrans {
		if ul == blas.Upper {
			offset = n*(n+1)/2 - 1
			if incX == 1 {
				for i := n - 1; i >= 0; i-- {
					atmp := ap[offset+1 : offset+n-i]
					xtmp := x[i+1:]
					var sum float64
					for j, v := range atmp {
						sum += v * xtmp[j]
					}
					x[i] -= sum
					if nonUnit {
						x[i] /= ap[offset]
					}
					offset -= n - i + 1
				}
				return
			}
			ix := kx + (n-1)*incX
			for i := n - 1; i >= 0; i-- {
				atmp := ap[offset+1 : offset+n-i]
				jx := kx + (i+1)*incX
				var sum float64
				for _, v := range atmp {
					sum += v * x[jx]
					jx += incX
				}
				x[ix] -= sum
				if nonUnit {
					x[ix] /= ap[offset]
				}
				ix -= incX
				offset -= n - i + 1
			}
			return
		}
		if incX == 1 {
			for i := 0; i < n; i++ {
				atmp := ap[offset-i : offset]
				var sum float64
				for j, v := range atmp {
					sum += v * x[j]
				}
				x[i] -= sum
				if nonUnit {
					x[i] /= ap[offset]
				}
				offset += i + 2
			}
			return
		}
		ix := kx
		for i := 0; i < n; i++ {
			jx := kx
			atmp := ap[offset-i : offset]
			var sum float64
			for _, v := range atmp {
				sum += v * x[jx]
				jx += incX
			}
			x[ix] -= sum
			if nonUnit {
				x[ix] /= ap[offset]
			}
			ix += incX
			offset += i + 2
		}
		return
	}
	// Cases where ap is transposed.
	if ul == blas.Upper {
		if incX == 1 {
			for i := 0; i < n; i++ {
				if nonUnit {
					x[i] /= ap[offset]
				}
				xi := x[i]
				atmp := ap[offset+1 : offset+n-i]
				xtmp := x[i+1:]
				for j, v := range atmp {
					xtmp[j] -= v * xi
				}
				offset += n - i
			}
			return
		}
		ix := kx
		for i := 0; i < n; i++ {
			if nonUnit {
				x[ix] /= ap[offset]
			}
			xix := x[ix]
			atmp := ap[offset+1 : offset+n-i]
			jx := kx + (i+1)*incX
			for _, v := range atmp {
				x[jx] -= v * xix
				jx += incX
			}
			ix += incX
			offset += n - i
		}
		return
	}
	if incX == 1 {
		offset = n*(n+1)/2 - 1
		for i := n - 1; i >= 0; i-- {
			if nonUnit {
				x[i] /= ap[offset]
			}
			xi := x[i]
			atmp := ap[offset-i : offset]
			for j, v := range atmp {
				x[j] -= v * xi
			}
			offset -= i + 1
		}
		return
	}
	ix := kx + (n-1)*incX
	offset = n*(n+1)/2 - 1
	for i := n - 1; i >= 0; i-- {
		if nonUnit {
			x[ix] /= ap[offset]
		}
		xix := x[ix]
		atmp := ap[offset-i : offset]
		jx := kx
		for _, v := range atmp {
			x[jx] -= v * xix
			jx += incX
		}
		ix -= incX
		offset -= i + 1
	}
}

// Dspmv performs the matrix-vector operation
//  y = alpha * A * x + beta * y
// where A is an n×n symmetric matrix in packed format, x and y are vectors,
// and alpha and beta are scalars.
func (Implementation) Dspmv(ul blas.Uplo, n int, alpha float64, ap []float64, x []float64, incX int, beta float64, y []float64, incY int) {
	if ul != blas.Lower && ul != blas.Upper {
		panic(badUplo)
	}
	if n < 0 {
		panic(nLT0)
	}
	if incX == 0 {
		panic(zeroIncX)
	}
	if incY == 0 {
		panic(zeroIncY)
	}

	// Quick return if possible.
	if n == 0 {
		return
	}

	// For zero matrix size the following slice length checks are trivially satisfied.
	if len(ap) < n*(n+1)/2 {
		panic(shortAP)
	}
	if (incX > 0 && len(x) <= (n-1)*incX) || (incX < 0 && len(x) <= (1-n)*incX) {
		panic(shortX)
	}
	if (incY > 0 && len(y) <= (n-1)*incY) || (incY < 0 && len(y) <= (1-n)*incY) {
		panic(shortY)
	}

	// Quick return if possible.
	if alpha == 0 && beta == 1 {
		return
	}

	// Set up start points
	var kx, ky int
	if incX < 0 {
		kx = -(n - 1) * incX
	}
	if incY < 0 {
		ky = -(n - 1) * incY
	}

	// Form y = beta * y.
	if beta != 1 {
		if incY == 1 {
			if beta == 0 {
				for i := range y[:n] {
					y[i] = 0
				}
			} else {
				f64.ScalUnitary(beta, y[:n])
			}
		} else {
			iy := ky
			if beta == 0 {
				for i := 0; i < n; i++ {
					y[iy] = 0
					iy += incY
				}
			} else {
				if incY > 0 {
					f64.ScalInc(beta, y, uintptr(n), uintptr(incY))
				} else {
					f64.ScalInc(beta, y, uintptr(n), uintptr(-incY))
				}
			}
		}
	}

	if alpha == 0 {
		return
	}

	if n == 1 {
		y[0] += alpha * ap[0] * x[0]
		return
	}
	var offset int // Offset is the index of (i,i).
	if ul == blas.Upper {
		if incX == 1 {
			iy := ky
			for i := 0; i < n; i++ {
				xv := x[i] * alpha
				sum := ap[offset] * x[i]
				atmp := ap[offset+1 : offset+n-i]
				xtmp := x[i+1:]
				jy := ky + (i+1)*incY
				for j, v := range atmp {
					sum += v * xtmp[j]
					y[jy] += v * xv
					jy += incY
				}
				y[iy] += alpha * sum
				iy += incY
				offset += n - i
			}
			return
		}
		ix := kx
		iy := ky
		for i := 0; i < n; i++ {
			xv := x[ix] * alpha
			sum := ap[offset] * x[ix]
			atmp := ap[offset+1 : offset+n-i]
			jx := kx + (i+1)*incX
			jy := ky + (i+1)*incY
			for _, v := range atmp {
				sum += v * x[jx]
				y[jy] += v * xv
				jx += incX
				jy += incY
			}
			y[iy] += alpha * sum
			ix += incX
			iy += incY
			offset += n - i
		}
		return
	}
	if incX == 1 {
		iy := ky
		for i := 0; i < n; i++ {
			xv := x[i] * alpha
			atmp := ap[offset-i : offset]
			jy := ky
			var sum float64
			for j, v := range atmp {
				sum += v * x[j]
				y[jy] += v * xv
				jy += incY
			}
			sum += ap[offset] * x[i]
			y[iy] += alpha * sum
			iy += incY
			offset += i + 2
		}
		return
	}
	ix := kx
	iy := ky
	for i := 0; i < n; i++ {
		xv := x[ix] * alpha
		atmp := ap[offset-i : offset]
		jx := kx
		jy := ky
		var sum float64
		for _, v := range atmp {
			sum += v * x[jx]
			y[jy] += v * xv
			jx += incX
			jy += incY
		}

		sum += ap[offset] * x[ix]
		y[iy] += alpha * sum
		ix += incX
		iy += incY
		offset += i + 2
	}
}

// Dspr performs the symmetric rank-one operation
//  A += alpha * x * xᵀ
// where A is an n×n symmetric matrix in packed format, x is a vector, and
// alpha is a scalar.
func (Implementation) Dspr(ul blas.Uplo, n int, alpha float64, x []float64, incX int, ap []float64) {
	if ul != blas.Lower && ul != blas.Upper {
		panic(badUplo)
	}
	if n < 0 {
		panic(nLT0)
	}
	if incX == 0 {
		panic(zeroIncX)
	}

	// Quick return if possible.
	if n == 0 {
		return
	}

	// For zero matrix size the following slice length checks are trivially satisfied.
	if (incX > 0 && len(x) <= (n-1)*incX) || (incX < 0 && len(x) <= (1-n)*incX) {
		panic(shortX)
	}
	if len(ap) < n*(n+1)/2 {
		panic(shortAP)
	}

	// Quick return if possible.
	if alpha == 0 {
		return
	}

	lenX := n
	var kx int
	if incX < 0 {
		kx = -(lenX - 1) * incX
	}
	var offset int // Offset is the index of (i,i).
	if ul == blas.Upper {
		if incX == 1 {
			for i := 0; i < n; i++ {
				atmp := ap[offset:]
				xv := alpha * x[i]
				xtmp := x[i:n]
				for j, v := range xtmp {
					atmp[j] += xv * v
				}
				offset += n - i
			}
			return
		}
		ix := kx
		for i := 0; i < n; i++ {
			jx := kx + i*incX
			atmp := ap[offset:]
			xv := alpha * x[ix]
			for j := 0; j < n-i; j++ {
				atmp[j] += xv * x[jx]
				jx += incX
			}
			ix += incX
			offset += n - i
		}
		return
	}
	if incX == 1 {
		for i := 0; i < n; i++ {
			atmp := ap[offset-i:]
			xv := alpha * x[i]
			xtmp := x[:i+1]
			for j, v := range xtmp {
				atmp[j] += xv * v
			}
			offset += i + 2
		}
		return
	}
	ix := kx
	for i := 0; i < n; i++ {
		jx := kx
		atmp := ap[offset-i:]
		xv := alpha * x[ix]
		for j := 0; j <= i; j++ {
			atmp[j] += xv * x[jx]
			jx += incX
		}
		ix += incX
		offset += i + 2
	}
}

// Dspr2 performs the symmetric rank-2 update
//  A += alpha * x * yᵀ + alpha * y * xᵀ
// where A is an n×n symmetric matrix in packed format, x and y are vectors,
// and alpha is a scalar.
func (Implementation) Dspr2(ul blas.Uplo, n int, alpha float64, x []float64, incX int, y []float64, incY int, ap []float64) {
	if ul != blas.Lower && ul != blas.Upper {
		panic(badUplo)
	}
	if n < 0 {
		panic(nLT0)
	}
	if incX == 0 {
		panic(zeroIncX)
	}
	if incY == 0 {
		panic(zeroIncY)
	}

	// Quick return if possible.
	if n == 0 {
		return
	}

	// For zero matrix size the following slice length checks are trivially satisfied.
	if (incX > 0 && len(x) <= (n-1)*incX) || (incX < 0 && len(x) <= (1-n)*incX) {
		panic(shortX)
	}
	if (incY > 0 && len(y) <= (n-1)*incY) || (incY < 0 && len(y) <= (1-n)*incY) {
		panic(shortY)
	}
	if len(ap) < n*(n+1)/2 {
		panic(shortAP)
	}

	// Quick return if possible.
	if alpha == 0 {
		return
	}

	var ky, kx int
	if incY < 0 {
		ky = -(n - 1) * incY
	}
	if incX < 0 {
		kx = -(n - 1) * incX
	}
	var offset int // Offset is the index of (i,i).
	if ul == blas.Upper {
		if incX == 1 && incY == 1 {
			for i := 0; i < n; i++ {
				atmp := ap[offset:]
				xi := x[i]
				yi := y[i]
				xtmp := x[i:n]
				ytmp := y[i:n]
				for j, v := range xtmp {
					atmp[j] += alpha * (xi*ytmp[j] + v*yi)
				}
				offset += n - i
			}
			return
		}
		ix := kx
		iy := ky
		for i := 0; i < n; i++ {
			jx := kx + i*incX
			jy := ky + i*incY
			atmp := ap[offset:]
			xi := x[ix]
			yi := y[iy]
			for j := 0; j < n-i; j++ {
				atmp[j] += alpha * (xi*y[jy] + x[jx]*yi)
				jx += incX
				jy += incY
			}
			ix += incX
			iy += incY
			offset += n - i
		}
		return
	}
	if incX == 1 && incY == 1 {
		for i := 0; i < n; i++ {
			atmp := ap[offset-i:]
			xi := x[i]
			yi := y[i]
			xtmp := x[:i+1]
			for j, v := range xtmp {
				atmp[j] += alpha * (xi*y[j] + v*yi)
			}
			offset += i + 2
		}
		return
	}
	ix := kx
	iy := ky
	for i := 0; i < n; i++ {
		jx := kx
		jy := ky
		atmp := ap[offset-i:]
		for j := 0; j <= i; j++ {
			atmp[j] += alpha * (x[ix]*y[jy] + x[jx]*y[iy])
			jx += incX
			jy += incY
		}
		ix += incX
		iy += incY
		offset += i + 2
	}
}
