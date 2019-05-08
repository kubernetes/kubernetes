// Copyright ©2014 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/internal/asm/f64"
)

var _ blas.Float64Level3 = Implementation{}

// Dtrsm solves one of the matrix equations
//  A * X = alpha * B    if tA == blas.NoTrans and side == blas.Left
//  A^T * X = alpha * B  if tA == blas.Trans or blas.ConjTrans, and side == blas.Left
//  X * A = alpha * B    if tA == blas.NoTrans and side == blas.Right
//  X * A^T = alpha * B  if tA == blas.Trans or blas.ConjTrans, and side == blas.Right
// where A is an n×n or m×m triangular matrix, X and B are m×n matrices, and alpha is a
// scalar.
//
// At entry to the function, X contains the values of B, and the result is
// stored in-place into X.
//
// No check is made that A is invertible.
func (Implementation) Dtrsm(s blas.Side, ul blas.Uplo, tA blas.Transpose, d blas.Diag, m, n int, alpha float64, a []float64, lda int, b []float64, ldb int) {
	if s != blas.Left && s != blas.Right {
		panic(badSide)
	}
	if ul != blas.Lower && ul != blas.Upper {
		panic(badUplo)
	}
	if tA != blas.NoTrans && tA != blas.Trans && tA != blas.ConjTrans {
		panic(badTranspose)
	}
	if d != blas.NonUnit && d != blas.Unit {
		panic(badDiag)
	}
	if m < 0 {
		panic(mLT0)
	}
	if n < 0 {
		panic(nLT0)
	}
	k := n
	if s == blas.Left {
		k = m
	}
	if lda < max(1, k) {
		panic(badLdA)
	}
	if ldb < max(1, n) {
		panic(badLdB)
	}

	// Quick return if possible.
	if m == 0 || n == 0 {
		return
	}

	// For zero matrix size the following slice length checks are trivially satisfied.
	if len(a) < lda*(k-1)+k {
		panic(shortA)
	}
	if len(b) < ldb*(m-1)+n {
		panic(shortB)
	}

	if alpha == 0 {
		for i := 0; i < m; i++ {
			btmp := b[i*ldb : i*ldb+n]
			for j := range btmp {
				btmp[j] = 0
			}
		}
		return
	}
	nonUnit := d == blas.NonUnit
	if s == blas.Left {
		if tA == blas.NoTrans {
			if ul == blas.Upper {
				for i := m - 1; i >= 0; i-- {
					btmp := b[i*ldb : i*ldb+n]
					if alpha != 1 {
						f64.ScalUnitary(alpha, btmp)
					}
					for ka, va := range a[i*lda+i+1 : i*lda+m] {
						if va != 0 {
							k := ka + i + 1
							f64.AxpyUnitary(-va, b[k*ldb:k*ldb+n], btmp)
						}
					}
					if nonUnit {
						tmp := 1 / a[i*lda+i]
						f64.ScalUnitary(tmp, btmp)
					}
				}
				return
			}
			for i := 0; i < m; i++ {
				btmp := b[i*ldb : i*ldb+n]
				if alpha != 1 {
					f64.ScalUnitary(alpha, btmp)
				}
				for k, va := range a[i*lda : i*lda+i] {
					if va != 0 {
						f64.AxpyUnitary(-va, b[k*ldb:k*ldb+n], btmp)
					}
				}
				if nonUnit {
					tmp := 1 / a[i*lda+i]
					f64.ScalUnitary(tmp, btmp)
				}
			}
			return
		}
		// Cases where a is transposed
		if ul == blas.Upper {
			for k := 0; k < m; k++ {
				btmpk := b[k*ldb : k*ldb+n]
				if nonUnit {
					tmp := 1 / a[k*lda+k]
					f64.ScalUnitary(tmp, btmpk)
				}
				for ia, va := range a[k*lda+k+1 : k*lda+m] {
					if va != 0 {
						i := ia + k + 1
						f64.AxpyUnitary(-va, btmpk, b[i*ldb:i*ldb+n])
					}
				}
				if alpha != 1 {
					f64.ScalUnitary(alpha, btmpk)
				}
			}
			return
		}
		for k := m - 1; k >= 0; k-- {
			btmpk := b[k*ldb : k*ldb+n]
			if nonUnit {
				tmp := 1 / a[k*lda+k]
				f64.ScalUnitary(tmp, btmpk)
			}
			for i, va := range a[k*lda : k*lda+k] {
				if va != 0 {
					f64.AxpyUnitary(-va, btmpk, b[i*ldb:i*ldb+n])
				}
			}
			if alpha != 1 {
				f64.ScalUnitary(alpha, btmpk)
			}
		}
		return
	}
	// Cases where a is to the right of X.
	if tA == blas.NoTrans {
		if ul == blas.Upper {
			for i := 0; i < m; i++ {
				btmp := b[i*ldb : i*ldb+n]
				if alpha != 1 {
					f64.ScalUnitary(alpha, btmp)
				}
				for k, vb := range btmp {
					if vb == 0 {
						continue
					}
					if nonUnit {
						btmp[k] /= a[k*lda+k]
					}
					f64.AxpyUnitary(-btmp[k], a[k*lda+k+1:k*lda+n], btmp[k+1:n])
				}
			}
			return
		}
		for i := 0; i < m; i++ {
			btmp := b[i*ldb : i*ldb+n]
			if alpha != 1 {
				f64.ScalUnitary(alpha, btmp)
			}
			for k := n - 1; k >= 0; k-- {
				if btmp[k] == 0 {
					continue
				}
				if nonUnit {
					btmp[k] /= a[k*lda+k]
				}
				f64.AxpyUnitary(-btmp[k], a[k*lda:k*lda+k], btmp[:k])
			}
		}
		return
	}
	// Cases where a is transposed.
	if ul == blas.Upper {
		for i := 0; i < m; i++ {
			btmp := b[i*ldb : i*ldb+n]
			for j := n - 1; j >= 0; j-- {
				tmp := alpha*btmp[j] - f64.DotUnitary(a[j*lda+j+1:j*lda+n], btmp[j+1:])
				if nonUnit {
					tmp /= a[j*lda+j]
				}
				btmp[j] = tmp
			}
		}
		return
	}
	for i := 0; i < m; i++ {
		btmp := b[i*ldb : i*ldb+n]
		for j := 0; j < n; j++ {
			tmp := alpha*btmp[j] - f64.DotUnitary(a[j*lda:j*lda+j], btmp[:j])
			if nonUnit {
				tmp /= a[j*lda+j]
			}
			btmp[j] = tmp
		}
	}
}

// Dsymm performs one of the matrix-matrix operations
//  C = alpha * A * B + beta * C  if side == blas.Left
//  C = alpha * B * A + beta * C  if side == blas.Right
// where A is an n×n or m×m symmetric matrix, B and C are m×n matrices, and alpha
// is a scalar.
func (Implementation) Dsymm(s blas.Side, ul blas.Uplo, m, n int, alpha float64, a []float64, lda int, b []float64, ldb int, beta float64, c []float64, ldc int) {
	if s != blas.Right && s != blas.Left {
		panic(badSide)
	}
	if ul != blas.Lower && ul != blas.Upper {
		panic(badUplo)
	}
	if m < 0 {
		panic(mLT0)
	}
	if n < 0 {
		panic(nLT0)
	}
	k := n
	if s == blas.Left {
		k = m
	}
	if lda < max(1, k) {
		panic(badLdA)
	}
	if ldb < max(1, n) {
		panic(badLdB)
	}
	if ldc < max(1, n) {
		panic(badLdC)
	}

	// Quick return if possible.
	if m == 0 || n == 0 {
		return
	}

	// For zero matrix size the following slice length checks are trivially satisfied.
	if len(a) < lda*(k-1)+k {
		panic(shortA)
	}
	if len(b) < ldb*(m-1)+n {
		panic(shortB)
	}
	if len(c) < ldc*(m-1)+n {
		panic(shortC)
	}

	// Quick return if possible.
	if alpha == 0 && beta == 1 {
		return
	}

	if alpha == 0 {
		if beta == 0 {
			for i := 0; i < m; i++ {
				ctmp := c[i*ldc : i*ldc+n]
				for j := range ctmp {
					ctmp[j] = 0
				}
			}
			return
		}
		for i := 0; i < m; i++ {
			ctmp := c[i*ldc : i*ldc+n]
			for j := 0; j < n; j++ {
				ctmp[j] *= beta
			}
		}
		return
	}

	isUpper := ul == blas.Upper
	if s == blas.Left {
		for i := 0; i < m; i++ {
			atmp := alpha * a[i*lda+i]
			btmp := b[i*ldb : i*ldb+n]
			ctmp := c[i*ldc : i*ldc+n]
			for j, v := range btmp {
				ctmp[j] *= beta
				ctmp[j] += atmp * v
			}

			for k := 0; k < i; k++ {
				var atmp float64
				if isUpper {
					atmp = a[k*lda+i]
				} else {
					atmp = a[i*lda+k]
				}
				atmp *= alpha
				f64.AxpyUnitary(atmp, b[k*ldb:k*ldb+n], ctmp)
			}
			for k := i + 1; k < m; k++ {
				var atmp float64
				if isUpper {
					atmp = a[i*lda+k]
				} else {
					atmp = a[k*lda+i]
				}
				atmp *= alpha
				f64.AxpyUnitary(atmp, b[k*ldb:k*ldb+n], ctmp)
			}
		}
		return
	}
	if isUpper {
		for i := 0; i < m; i++ {
			for j := n - 1; j >= 0; j-- {
				tmp := alpha * b[i*ldb+j]
				var tmp2 float64
				atmp := a[j*lda+j+1 : j*lda+n]
				btmp := b[i*ldb+j+1 : i*ldb+n]
				ctmp := c[i*ldc+j+1 : i*ldc+n]
				for k, v := range atmp {
					ctmp[k] += tmp * v
					tmp2 += btmp[k] * v
				}
				c[i*ldc+j] *= beta
				c[i*ldc+j] += tmp*a[j*lda+j] + alpha*tmp2
			}
		}
		return
	}
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			tmp := alpha * b[i*ldb+j]
			var tmp2 float64
			atmp := a[j*lda : j*lda+j]
			btmp := b[i*ldb : i*ldb+j]
			ctmp := c[i*ldc : i*ldc+j]
			for k, v := range atmp {
				ctmp[k] += tmp * v
				tmp2 += btmp[k] * v
			}
			c[i*ldc+j] *= beta
			c[i*ldc+j] += tmp*a[j*lda+j] + alpha*tmp2
		}
	}
}

// Dsyrk performs one of the symmetric rank-k operations
//  C = alpha * A * A^T + beta * C  if tA == blas.NoTrans
//  C = alpha * A^T * A + beta * C  if tA == blas.Trans or tA == blas.ConjTrans
// where A is an n×k or k×n matrix, C is an n×n symmetric matrix, and alpha and
// beta are scalars.
func (Implementation) Dsyrk(ul blas.Uplo, tA blas.Transpose, n, k int, alpha float64, a []float64, lda int, beta float64, c []float64, ldc int) {
	if ul != blas.Lower && ul != blas.Upper {
		panic(badUplo)
	}
	if tA != blas.Trans && tA != blas.NoTrans && tA != blas.ConjTrans {
		panic(badTranspose)
	}
	if n < 0 {
		panic(nLT0)
	}
	if k < 0 {
		panic(kLT0)
	}
	row, col := k, n
	if tA == blas.NoTrans {
		row, col = n, k
	}
	if lda < max(1, col) {
		panic(badLdA)
	}
	if ldc < max(1, n) {
		panic(badLdC)
	}

	// Quick return if possible.
	if n == 0 {
		return
	}

	// For zero matrix size the following slice length checks are trivially satisfied.
	if len(a) < lda*(row-1)+col {
		panic(shortA)
	}
	if len(c) < ldc*(n-1)+n {
		panic(shortC)
	}

	if alpha == 0 {
		if beta == 0 {
			if ul == blas.Upper {
				for i := 0; i < n; i++ {
					ctmp := c[i*ldc+i : i*ldc+n]
					for j := range ctmp {
						ctmp[j] = 0
					}
				}
				return
			}
			for i := 0; i < n; i++ {
				ctmp := c[i*ldc : i*ldc+i+1]
				for j := range ctmp {
					ctmp[j] = 0
				}
			}
			return
		}
		if ul == blas.Upper {
			for i := 0; i < n; i++ {
				ctmp := c[i*ldc+i : i*ldc+n]
				for j := range ctmp {
					ctmp[j] *= beta
				}
			}
			return
		}
		for i := 0; i < n; i++ {
			ctmp := c[i*ldc : i*ldc+i+1]
			for j := range ctmp {
				ctmp[j] *= beta
			}
		}
		return
	}
	if tA == blas.NoTrans {
		if ul == blas.Upper {
			for i := 0; i < n; i++ {
				ctmp := c[i*ldc+i : i*ldc+n]
				atmp := a[i*lda : i*lda+k]
				if beta == 0 {
					for jc := range ctmp {
						j := jc + i
						ctmp[jc] = alpha * f64.DotUnitary(atmp, a[j*lda:j*lda+k])
					}
				} else {
					for jc, vc := range ctmp {
						j := jc + i
						ctmp[jc] = vc*beta + alpha*f64.DotUnitary(atmp, a[j*lda:j*lda+k])
					}
				}
			}
			return
		}
		for i := 0; i < n; i++ {
			ctmp := c[i*ldc : i*ldc+i+1]
			atmp := a[i*lda : i*lda+k]
			if beta == 0 {
				for j := range ctmp {
					ctmp[j] = alpha * f64.DotUnitary(a[j*lda:j*lda+k], atmp)
				}
			} else {
				for j, vc := range ctmp {
					ctmp[j] = vc*beta + alpha*f64.DotUnitary(a[j*lda:j*lda+k], atmp)
				}
			}
		}
		return
	}
	// Cases where a is transposed.
	if ul == blas.Upper {
		for i := 0; i < n; i++ {
			ctmp := c[i*ldc+i : i*ldc+n]
			if beta == 0 {
				for j := range ctmp {
					ctmp[j] = 0
				}
			} else if beta != 1 {
				for j := range ctmp {
					ctmp[j] *= beta
				}
			}
			for l := 0; l < k; l++ {
				tmp := alpha * a[l*lda+i]
				if tmp != 0 {
					f64.AxpyUnitary(tmp, a[l*lda+i:l*lda+n], ctmp)
				}
			}
		}
		return
	}
	for i := 0; i < n; i++ {
		ctmp := c[i*ldc : i*ldc+i+1]
		if beta != 1 {
			for j := range ctmp {
				ctmp[j] *= beta
			}
		}
		for l := 0; l < k; l++ {
			tmp := alpha * a[l*lda+i]
			if tmp != 0 {
				f64.AxpyUnitary(tmp, a[l*lda:l*lda+i+1], ctmp)
			}
		}
	}
}

// Dsyr2k performs one of the symmetric rank 2k operations
//  C = alpha * A * B^T + alpha * B * A^T + beta * C  if tA == blas.NoTrans
//  C = alpha * A^T * B + alpha * B^T * A + beta * C  if tA == blas.Trans or tA == blas.ConjTrans
// where A and B are n×k or k×n matrices, C is an n×n symmetric matrix, and
// alpha and beta are scalars.
func (Implementation) Dsyr2k(ul blas.Uplo, tA blas.Transpose, n, k int, alpha float64, a []float64, lda int, b []float64, ldb int, beta float64, c []float64, ldc int) {
	if ul != blas.Lower && ul != blas.Upper {
		panic(badUplo)
	}
	if tA != blas.Trans && tA != blas.NoTrans && tA != blas.ConjTrans {
		panic(badTranspose)
	}
	if n < 0 {
		panic(nLT0)
	}
	if k < 0 {
		panic(kLT0)
	}
	row, col := k, n
	if tA == blas.NoTrans {
		row, col = n, k
	}
	if lda < max(1, col) {
		panic(badLdA)
	}
	if ldb < max(1, col) {
		panic(badLdB)
	}
	if ldc < max(1, n) {
		panic(badLdC)
	}

	// Quick return if possible.
	if n == 0 {
		return
	}

	// For zero matrix size the following slice length checks are trivially satisfied.
	if len(a) < lda*(row-1)+col {
		panic(shortA)
	}
	if len(b) < ldb*(row-1)+col {
		panic(shortB)
	}
	if len(c) < ldc*(n-1)+n {
		panic(shortC)
	}

	if alpha == 0 {
		if beta == 0 {
			if ul == blas.Upper {
				for i := 0; i < n; i++ {
					ctmp := c[i*ldc+i : i*ldc+n]
					for j := range ctmp {
						ctmp[j] = 0
					}
				}
				return
			}
			for i := 0; i < n; i++ {
				ctmp := c[i*ldc : i*ldc+i+1]
				for j := range ctmp {
					ctmp[j] = 0
				}
			}
			return
		}
		if ul == blas.Upper {
			for i := 0; i < n; i++ {
				ctmp := c[i*ldc+i : i*ldc+n]
				for j := range ctmp {
					ctmp[j] *= beta
				}
			}
			return
		}
		for i := 0; i < n; i++ {
			ctmp := c[i*ldc : i*ldc+i+1]
			for j := range ctmp {
				ctmp[j] *= beta
			}
		}
		return
	}
	if tA == blas.NoTrans {
		if ul == blas.Upper {
			for i := 0; i < n; i++ {
				atmp := a[i*lda : i*lda+k]
				btmp := b[i*ldb : i*ldb+k]
				ctmp := c[i*ldc+i : i*ldc+n]
				for jc := range ctmp {
					j := i + jc
					var tmp1, tmp2 float64
					binner := b[j*ldb : j*ldb+k]
					for l, v := range a[j*lda : j*lda+k] {
						tmp1 += v * btmp[l]
						tmp2 += atmp[l] * binner[l]
					}
					ctmp[jc] *= beta
					ctmp[jc] += alpha * (tmp1 + tmp2)
				}
			}
			return
		}
		for i := 0; i < n; i++ {
			atmp := a[i*lda : i*lda+k]
			btmp := b[i*ldb : i*ldb+k]
			ctmp := c[i*ldc : i*ldc+i+1]
			for j := 0; j <= i; j++ {
				var tmp1, tmp2 float64
				binner := b[j*ldb : j*ldb+k]
				for l, v := range a[j*lda : j*lda+k] {
					tmp1 += v * btmp[l]
					tmp2 += atmp[l] * binner[l]
				}
				ctmp[j] *= beta
				ctmp[j] += alpha * (tmp1 + tmp2)
			}
		}
		return
	}
	if ul == blas.Upper {
		for i := 0; i < n; i++ {
			ctmp := c[i*ldc+i : i*ldc+n]
			if beta != 1 {
				for j := range ctmp {
					ctmp[j] *= beta
				}
			}
			for l := 0; l < k; l++ {
				tmp1 := alpha * b[l*ldb+i]
				tmp2 := alpha * a[l*lda+i]
				btmp := b[l*ldb+i : l*ldb+n]
				if tmp1 != 0 || tmp2 != 0 {
					for j, v := range a[l*lda+i : l*lda+n] {
						ctmp[j] += v*tmp1 + btmp[j]*tmp2
					}
				}
			}
		}
		return
	}
	for i := 0; i < n; i++ {
		ctmp := c[i*ldc : i*ldc+i+1]
		if beta != 1 {
			for j := range ctmp {
				ctmp[j] *= beta
			}
		}
		for l := 0; l < k; l++ {
			tmp1 := alpha * b[l*ldb+i]
			tmp2 := alpha * a[l*lda+i]
			btmp := b[l*ldb : l*ldb+i+1]
			if tmp1 != 0 || tmp2 != 0 {
				for j, v := range a[l*lda : l*lda+i+1] {
					ctmp[j] += v*tmp1 + btmp[j]*tmp2
				}
			}
		}
	}
}

// Dtrmm performs one of the matrix-matrix operations
//  B = alpha * A * B    if tA == blas.NoTrans and side == blas.Left
//  B = alpha * A^T * B  if tA == blas.Trans or blas.ConjTrans, and side == blas.Left
//  B = alpha * B * A    if tA == blas.NoTrans and side == blas.Right
//  B = alpha * B * A^T  if tA == blas.Trans or blas.ConjTrans, and side == blas.Right
// where A is an n×n or m×m triangular matrix, B is an m×n matrix, and alpha is a scalar.
func (Implementation) Dtrmm(s blas.Side, ul blas.Uplo, tA blas.Transpose, d blas.Diag, m, n int, alpha float64, a []float64, lda int, b []float64, ldb int) {
	if s != blas.Left && s != blas.Right {
		panic(badSide)
	}
	if ul != blas.Lower && ul != blas.Upper {
		panic(badUplo)
	}
	if tA != blas.NoTrans && tA != blas.Trans && tA != blas.ConjTrans {
		panic(badTranspose)
	}
	if d != blas.NonUnit && d != blas.Unit {
		panic(badDiag)
	}
	if m < 0 {
		panic(mLT0)
	}
	if n < 0 {
		panic(nLT0)
	}
	k := n
	if s == blas.Left {
		k = m
	}
	if lda < max(1, k) {
		panic(badLdA)
	}
	if ldb < max(1, n) {
		panic(badLdB)
	}

	// Quick return if possible.
	if m == 0 || n == 0 {
		return
	}

	// For zero matrix size the following slice length checks are trivially satisfied.
	if len(a) < lda*(k-1)+k {
		panic(shortA)
	}
	if len(b) < ldb*(m-1)+n {
		panic(shortB)
	}

	if alpha == 0 {
		for i := 0; i < m; i++ {
			btmp := b[i*ldb : i*ldb+n]
			for j := range btmp {
				btmp[j] = 0
			}
		}
		return
	}

	nonUnit := d == blas.NonUnit
	if s == blas.Left {
		if tA == blas.NoTrans {
			if ul == blas.Upper {
				for i := 0; i < m; i++ {
					tmp := alpha
					if nonUnit {
						tmp *= a[i*lda+i]
					}
					btmp := b[i*ldb : i*ldb+n]
					f64.ScalUnitary(tmp, btmp)
					for ka, va := range a[i*lda+i+1 : i*lda+m] {
						k := ka + i + 1
						if va != 0 {
							f64.AxpyUnitary(alpha*va, b[k*ldb:k*ldb+n], btmp)
						}
					}
				}
				return
			}
			for i := m - 1; i >= 0; i-- {
				tmp := alpha
				if nonUnit {
					tmp *= a[i*lda+i]
				}
				btmp := b[i*ldb : i*ldb+n]
				f64.ScalUnitary(tmp, btmp)
				for k, va := range a[i*lda : i*lda+i] {
					if va != 0 {
						f64.AxpyUnitary(alpha*va, b[k*ldb:k*ldb+n], btmp)
					}
				}
			}
			return
		}
		// Cases where a is transposed.
		if ul == blas.Upper {
			for k := m - 1; k >= 0; k-- {
				btmpk := b[k*ldb : k*ldb+n]
				for ia, va := range a[k*lda+k+1 : k*lda+m] {
					i := ia + k + 1
					btmp := b[i*ldb : i*ldb+n]
					if va != 0 {
						f64.AxpyUnitary(alpha*va, btmpk, btmp)
					}
				}
				tmp := alpha
				if nonUnit {
					tmp *= a[k*lda+k]
				}
				if tmp != 1 {
					f64.ScalUnitary(tmp, btmpk)
				}
			}
			return
		}
		for k := 0; k < m; k++ {
			btmpk := b[k*ldb : k*ldb+n]
			for i, va := range a[k*lda : k*lda+k] {
				btmp := b[i*ldb : i*ldb+n]
				if va != 0 {
					f64.AxpyUnitary(alpha*va, btmpk, btmp)
				}
			}
			tmp := alpha
			if nonUnit {
				tmp *= a[k*lda+k]
			}
			if tmp != 1 {
				f64.ScalUnitary(tmp, btmpk)
			}
		}
		return
	}
	// Cases where a is on the right
	if tA == blas.NoTrans {
		if ul == blas.Upper {
			for i := 0; i < m; i++ {
				btmp := b[i*ldb : i*ldb+n]
				for k := n - 1; k >= 0; k-- {
					tmp := alpha * btmp[k]
					if tmp == 0 {
						continue
					}
					btmp[k] = tmp
					if nonUnit {
						btmp[k] *= a[k*lda+k]
					}
					f64.AxpyUnitary(tmp, a[k*lda+k+1:k*lda+n], btmp[k+1:n])
				}
			}
			return
		}
		for i := 0; i < m; i++ {
			btmp := b[i*ldb : i*ldb+n]
			for k := 0; k < n; k++ {
				tmp := alpha * btmp[k]
				if tmp == 0 {
					continue
				}
				btmp[k] = tmp
				if nonUnit {
					btmp[k] *= a[k*lda+k]
				}
				f64.AxpyUnitary(tmp, a[k*lda:k*lda+k], btmp[:k])
			}
		}
		return
	}
	// Cases where a is transposed.
	if ul == blas.Upper {
		for i := 0; i < m; i++ {
			btmp := b[i*ldb : i*ldb+n]
			for j, vb := range btmp {
				tmp := vb
				if nonUnit {
					tmp *= a[j*lda+j]
				}
				tmp += f64.DotUnitary(a[j*lda+j+1:j*lda+n], btmp[j+1:n])
				btmp[j] = alpha * tmp
			}
		}
		return
	}
	for i := 0; i < m; i++ {
		btmp := b[i*ldb : i*ldb+n]
		for j := n - 1; j >= 0; j-- {
			tmp := btmp[j]
			if nonUnit {
				tmp *= a[j*lda+j]
			}
			tmp += f64.DotUnitary(a[j*lda:j*lda+j], btmp[:j])
			btmp[j] = alpha * tmp
		}
	}
}
