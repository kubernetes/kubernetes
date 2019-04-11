// Copyright ©2019 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"math/cmplx"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/internal/asm/c128"
)

var _ blas.Complex128Level3 = Implementation{}

// Zgemm performs one of the matrix-matrix operations
//  C = alpha * op(A) * op(B) + beta * C
// where op(X) is one of
//  op(X) = X  or  op(X) = X^T  or  op(X) = X^H,
// alpha and beta are scalars, and A, B and C are matrices, with op(A) an m×k matrix,
// op(B) a k×n matrix and C an m×n matrix.
func (Implementation) Zgemm(tA, tB blas.Transpose, m, n, k int, alpha complex128, a []complex128, lda int, b []complex128, ldb int, beta complex128, c []complex128, ldc int) {
	switch tA {
	default:
		panic(badTranspose)
	case blas.NoTrans, blas.Trans, blas.ConjTrans:
	}
	switch tB {
	default:
		panic(badTranspose)
	case blas.NoTrans, blas.Trans, blas.ConjTrans:
	}
	switch {
	case m < 0:
		panic(mLT0)
	case n < 0:
		panic(nLT0)
	case k < 0:
		panic(kLT0)
	}
	rowA, colA := m, k
	if tA != blas.NoTrans {
		rowA, colA = k, m
	}
	if lda < max(1, colA) {
		panic(badLdA)
	}
	rowB, colB := k, n
	if tB != blas.NoTrans {
		rowB, colB = n, k
	}
	if ldb < max(1, colB) {
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
	if len(a) < (rowA-1)*lda+colA {
		panic(shortA)
	}
	if len(b) < (rowB-1)*ldb+colB {
		panic(shortB)
	}
	if len(c) < (m-1)*ldc+n {
		panic(shortC)
	}

	// Quick return if possible.
	if (alpha == 0 || k == 0) && beta == 1 {
		return
	}

	if alpha == 0 {
		if beta == 0 {
			for i := 0; i < m; i++ {
				for j := 0; j < n; j++ {
					c[i*ldc+j] = 0
				}
			}
		} else {
			for i := 0; i < m; i++ {
				for j := 0; j < n; j++ {
					c[i*ldc+j] *= beta
				}
			}
		}
		return
	}

	switch tA {
	case blas.NoTrans:
		switch tB {
		case blas.NoTrans:
			// Form  C = alpha * A * B + beta * C.
			for i := 0; i < m; i++ {
				switch {
				case beta == 0:
					for j := 0; j < n; j++ {
						c[i*ldc+j] = 0
					}
				case beta != 1:
					for j := 0; j < n; j++ {
						c[i*ldc+j] *= beta
					}
				}
				for l := 0; l < k; l++ {
					tmp := alpha * a[i*lda+l]
					for j := 0; j < n; j++ {
						c[i*ldc+j] += tmp * b[l*ldb+j]
					}
				}
			}
		case blas.Trans:
			// Form  C = alpha * A * B^T + beta * C.
			for i := 0; i < m; i++ {
				switch {
				case beta == 0:
					for j := 0; j < n; j++ {
						c[i*ldc+j] = 0
					}
				case beta != 1:
					for j := 0; j < n; j++ {
						c[i*ldc+j] *= beta
					}
				}
				for l := 0; l < k; l++ {
					tmp := alpha * a[i*lda+l]
					for j := 0; j < n; j++ {
						c[i*ldc+j] += tmp * b[j*ldb+l]
					}
				}
			}
		case blas.ConjTrans:
			// Form  C = alpha * A * B^H + beta * C.
			for i := 0; i < m; i++ {
				switch {
				case beta == 0:
					for j := 0; j < n; j++ {
						c[i*ldc+j] = 0
					}
				case beta != 1:
					for j := 0; j < n; j++ {
						c[i*ldc+j] *= beta
					}
				}
				for l := 0; l < k; l++ {
					tmp := alpha * a[i*lda+l]
					for j := 0; j < n; j++ {
						c[i*ldc+j] += tmp * cmplx.Conj(b[j*ldb+l])
					}
				}
			}
		}
	case blas.Trans:
		switch tB {
		case blas.NoTrans:
			// Form  C = alpha * A^T * B + beta * C.
			for i := 0; i < m; i++ {
				for j := 0; j < n; j++ {
					var tmp complex128
					for l := 0; l < k; l++ {
						tmp += a[l*lda+i] * b[l*ldb+j]
					}
					if beta == 0 {
						c[i*ldc+j] = alpha * tmp
					} else {
						c[i*ldc+j] = alpha*tmp + beta*c[i*ldc+j]
					}
				}
			}
		case blas.Trans:
			// Form  C = alpha * A^T * B^T + beta * C.
			for i := 0; i < m; i++ {
				for j := 0; j < n; j++ {
					var tmp complex128
					for l := 0; l < k; l++ {
						tmp += a[l*lda+i] * b[j*ldb+l]
					}
					if beta == 0 {
						c[i*ldc+j] = alpha * tmp
					} else {
						c[i*ldc+j] = alpha*tmp + beta*c[i*ldc+j]
					}
				}
			}
		case blas.ConjTrans:
			// Form  C = alpha * A^T * B^H + beta * C.
			for i := 0; i < m; i++ {
				for j := 0; j < n; j++ {
					var tmp complex128
					for l := 0; l < k; l++ {
						tmp += a[l*lda+i] * cmplx.Conj(b[j*ldb+l])
					}
					if beta == 0 {
						c[i*ldc+j] = alpha * tmp
					} else {
						c[i*ldc+j] = alpha*tmp + beta*c[i*ldc+j]
					}
				}
			}
		}
	case blas.ConjTrans:
		switch tB {
		case blas.NoTrans:
			// Form  C = alpha * A^H * B + beta * C.
			for i := 0; i < m; i++ {
				for j := 0; j < n; j++ {
					var tmp complex128
					for l := 0; l < k; l++ {
						tmp += cmplx.Conj(a[l*lda+i]) * b[l*ldb+j]
					}
					if beta == 0 {
						c[i*ldc+j] = alpha * tmp
					} else {
						c[i*ldc+j] = alpha*tmp + beta*c[i*ldc+j]
					}
				}
			}
		case blas.Trans:
			// Form  C = alpha * A^H * B^T + beta * C.
			for i := 0; i < m; i++ {
				for j := 0; j < n; j++ {
					var tmp complex128
					for l := 0; l < k; l++ {
						tmp += cmplx.Conj(a[l*lda+i]) * b[j*ldb+l]
					}
					if beta == 0 {
						c[i*ldc+j] = alpha * tmp
					} else {
						c[i*ldc+j] = alpha*tmp + beta*c[i*ldc+j]
					}
				}
			}
		case blas.ConjTrans:
			// Form  C = alpha * A^H * B^H + beta * C.
			for i := 0; i < m; i++ {
				for j := 0; j < n; j++ {
					var tmp complex128
					for l := 0; l < k; l++ {
						tmp += cmplx.Conj(a[l*lda+i]) * cmplx.Conj(b[j*ldb+l])
					}
					if beta == 0 {
						c[i*ldc+j] = alpha * tmp
					} else {
						c[i*ldc+j] = alpha*tmp + beta*c[i*ldc+j]
					}
				}
			}
		}
	}
}

// Zhemm performs one of the matrix-matrix operations
//  C = alpha*A*B + beta*C  if side == blas.Left
//  C = alpha*B*A + beta*C  if side == blas.Right
// where alpha and beta are scalars, A is an m×m or n×n hermitian matrix and B
// and C are m×n matrices. The imaginary parts of the diagonal elements of A are
// assumed to be zero.
func (Implementation) Zhemm(side blas.Side, uplo blas.Uplo, m, n int, alpha complex128, a []complex128, lda int, b []complex128, ldb int, beta complex128, c []complex128, ldc int) {
	na := m
	if side == blas.Right {
		na = n
	}
	switch {
	case side != blas.Left && side != blas.Right:
		panic(badSide)
	case uplo != blas.Lower && uplo != blas.Upper:
		panic(badUplo)
	case m < 0:
		panic(mLT0)
	case n < 0:
		panic(nLT0)
	case lda < max(1, na):
		panic(badLdA)
	case ldb < max(1, n):
		panic(badLdB)
	case ldc < max(1, n):
		panic(badLdC)
	}

	// Quick return if possible.
	if m == 0 || n == 0 {
		return
	}

	// For zero matrix size the following slice length checks are trivially satisfied.
	if len(a) < lda*(na-1)+na {
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
				ci := c[i*ldc : i*ldc+n]
				for j := range ci {
					ci[j] = 0
				}
			}
		} else {
			for i := 0; i < m; i++ {
				ci := c[i*ldc : i*ldc+n]
				c128.ScalUnitary(beta, ci)
			}
		}
		return
	}

	if side == blas.Left {
		// Form  C = alpha*A*B + beta*C.
		for i := 0; i < m; i++ {
			atmp := alpha * complex(real(a[i*lda+i]), 0)
			bi := b[i*ldb : i*ldb+n]
			ci := c[i*ldc : i*ldc+n]
			if beta == 0 {
				for j, bij := range bi {
					ci[j] = atmp * bij
				}
			} else {
				for j, bij := range bi {
					ci[j] = atmp*bij + beta*ci[j]
				}
			}
			if uplo == blas.Upper {
				for k := 0; k < i; k++ {
					atmp = alpha * cmplx.Conj(a[k*lda+i])
					c128.AxpyUnitary(atmp, b[k*ldb:k*ldb+n], ci)
				}
				for k := i + 1; k < m; k++ {
					atmp = alpha * a[i*lda+k]
					c128.AxpyUnitary(atmp, b[k*ldb:k*ldb+n], ci)
				}
			} else {
				for k := 0; k < i; k++ {
					atmp = alpha * a[i*lda+k]
					c128.AxpyUnitary(atmp, b[k*ldb:k*ldb+n], ci)
				}
				for k := i + 1; k < m; k++ {
					atmp = alpha * cmplx.Conj(a[k*lda+i])
					c128.AxpyUnitary(atmp, b[k*ldb:k*ldb+n], ci)
				}
			}
		}
	} else {
		// Form  C = alpha*B*A + beta*C.
		if uplo == blas.Upper {
			for i := 0; i < m; i++ {
				for j := n - 1; j >= 0; j-- {
					abij := alpha * b[i*ldb+j]
					aj := a[j*lda+j+1 : j*lda+n]
					bi := b[i*ldb+j+1 : i*ldb+n]
					ci := c[i*ldc+j+1 : i*ldc+n]
					var tmp complex128
					for k, ajk := range aj {
						ci[k] += abij * ajk
						tmp += bi[k] * cmplx.Conj(ajk)
					}
					ajj := complex(real(a[j*lda+j]), 0)
					if beta == 0 {
						c[i*ldc+j] = abij*ajj + alpha*tmp
					} else {
						c[i*ldc+j] = abij*ajj + alpha*tmp + beta*c[i*ldc+j]
					}
				}
			}
		} else {
			for i := 0; i < m; i++ {
				for j := 0; j < n; j++ {
					abij := alpha * b[i*ldb+j]
					aj := a[j*lda : j*lda+j]
					bi := b[i*ldb : i*ldb+j]
					ci := c[i*ldc : i*ldc+j]
					var tmp complex128
					for k, ajk := range aj {
						ci[k] += abij * ajk
						tmp += bi[k] * cmplx.Conj(ajk)
					}
					ajj := complex(real(a[j*lda+j]), 0)
					if beta == 0 {
						c[i*ldc+j] = abij*ajj + alpha*tmp
					} else {
						c[i*ldc+j] = abij*ajj + alpha*tmp + beta*c[i*ldc+j]
					}
				}
			}
		}
	}
}

// Zherk performs one of the hermitian rank-k operations
//  C = alpha*A*A^H + beta*C  if trans == blas.NoTrans
//  C = alpha*A^H*A + beta*C  if trans == blas.ConjTrans
// where alpha and beta are real scalars, C is an n×n hermitian matrix and A is
// an n×k matrix in the first case and a k×n matrix in the second case.
//
// The imaginary parts of the diagonal elements of C are assumed to be zero, and
// on return they will be set to zero.
func (Implementation) Zherk(uplo blas.Uplo, trans blas.Transpose, n, k int, alpha float64, a []complex128, lda int, beta float64, c []complex128, ldc int) {
	var rowA, colA int
	switch trans {
	default:
		panic(badTranspose)
	case blas.NoTrans:
		rowA, colA = n, k
	case blas.ConjTrans:
		rowA, colA = k, n
	}
	switch {
	case uplo != blas.Lower && uplo != blas.Upper:
		panic(badUplo)
	case n < 0:
		panic(nLT0)
	case k < 0:
		panic(kLT0)
	case lda < max(1, colA):
		panic(badLdA)
	case ldc < max(1, n):
		panic(badLdC)
	}

	// Quick return if possible.
	if n == 0 {
		return
	}

	// For zero matrix size the following slice length checks are trivially satisfied.
	if len(a) < (rowA-1)*lda+colA {
		panic(shortA)
	}
	if len(c) < (n-1)*ldc+n {
		panic(shortC)
	}

	// Quick return if possible.
	if (alpha == 0 || k == 0) && beta == 1 {
		return
	}

	if alpha == 0 {
		if uplo == blas.Upper {
			if beta == 0 {
				for i := 0; i < n; i++ {
					ci := c[i*ldc+i : i*ldc+n]
					for j := range ci {
						ci[j] = 0
					}
				}
			} else {
				for i := 0; i < n; i++ {
					ci := c[i*ldc+i : i*ldc+n]
					ci[0] = complex(beta*real(ci[0]), 0)
					if i != n-1 {
						c128.DscalUnitary(beta, ci[1:])
					}
				}
			}
		} else {
			if beta == 0 {
				for i := 0; i < n; i++ {
					ci := c[i*ldc : i*ldc+i+1]
					for j := range ci {
						ci[j] = 0
					}
				}
			} else {
				for i := 0; i < n; i++ {
					ci := c[i*ldc : i*ldc+i+1]
					if i != 0 {
						c128.DscalUnitary(beta, ci[:i])
					}
					ci[i] = complex(beta*real(ci[i]), 0)
				}
			}
		}
		return
	}

	calpha := complex(alpha, 0)
	if trans == blas.NoTrans {
		// Form  C = alpha*A*A^H + beta*C.
		cbeta := complex(beta, 0)
		if uplo == blas.Upper {
			for i := 0; i < n; i++ {
				ci := c[i*ldc+i : i*ldc+n]
				ai := a[i*lda : i*lda+k]
				switch {
				case beta == 0:
					// Handle the i-th diagonal element of C.
					ci[0] = complex(alpha*real(c128.DotcUnitary(ai, ai)), 0)
					// Handle the remaining elements on the i-th row of C.
					for jc := range ci[1:] {
						j := i + 1 + jc
						ci[jc+1] = calpha * c128.DotcUnitary(a[j*lda:j*lda+k], ai)
					}
				case beta != 1:
					cii := calpha*c128.DotcUnitary(ai, ai) + cbeta*ci[0]
					ci[0] = complex(real(cii), 0)
					for jc, cij := range ci[1:] {
						j := i + 1 + jc
						ci[jc+1] = calpha*c128.DotcUnitary(a[j*lda:j*lda+k], ai) + cbeta*cij
					}
				default:
					cii := calpha*c128.DotcUnitary(ai, ai) + ci[0]
					ci[0] = complex(real(cii), 0)
					for jc, cij := range ci[1:] {
						j := i + 1 + jc
						ci[jc+1] = calpha*c128.DotcUnitary(a[j*lda:j*lda+k], ai) + cij
					}
				}
			}
		} else {
			for i := 0; i < n; i++ {
				ci := c[i*ldc : i*ldc+i+1]
				ai := a[i*lda : i*lda+k]
				switch {
				case beta == 0:
					// Handle the first i-1 elements on the i-th row of C.
					for j := range ci[:i] {
						ci[j] = calpha * c128.DotcUnitary(a[j*lda:j*lda+k], ai)
					}
					// Handle the i-th diagonal element of C.
					ci[i] = complex(alpha*real(c128.DotcUnitary(ai, ai)), 0)
				case beta != 1:
					for j, cij := range ci[:i] {
						ci[j] = calpha*c128.DotcUnitary(a[j*lda:j*lda+k], ai) + cbeta*cij
					}
					cii := calpha*c128.DotcUnitary(ai, ai) + cbeta*ci[i]
					ci[i] = complex(real(cii), 0)
				default:
					for j, cij := range ci[:i] {
						ci[j] = calpha*c128.DotcUnitary(a[j*lda:j*lda+k], ai) + cij
					}
					cii := calpha*c128.DotcUnitary(ai, ai) + ci[i]
					ci[i] = complex(real(cii), 0)
				}
			}
		}
	} else {
		// Form  C = alpha*A^H*A + beta*C.
		if uplo == blas.Upper {
			for i := 0; i < n; i++ {
				ci := c[i*ldc+i : i*ldc+n]
				switch {
				case beta == 0:
					for jc := range ci {
						ci[jc] = 0
					}
				case beta != 1:
					c128.DscalUnitary(beta, ci)
					ci[0] = complex(real(ci[0]), 0)
				default:
					ci[0] = complex(real(ci[0]), 0)
				}
				for j := 0; j < k; j++ {
					aji := cmplx.Conj(a[j*lda+i])
					if aji != 0 {
						c128.AxpyUnitary(calpha*aji, a[j*lda+i:j*lda+n], ci)
					}
				}
				c[i*ldc+i] = complex(real(c[i*ldc+i]), 0)
			}
		} else {
			for i := 0; i < n; i++ {
				ci := c[i*ldc : i*ldc+i+1]
				switch {
				case beta == 0:
					for j := range ci {
						ci[j] = 0
					}
				case beta != 1:
					c128.DscalUnitary(beta, ci)
					ci[i] = complex(real(ci[i]), 0)
				default:
					ci[i] = complex(real(ci[i]), 0)
				}
				for j := 0; j < k; j++ {
					aji := cmplx.Conj(a[j*lda+i])
					if aji != 0 {
						c128.AxpyUnitary(calpha*aji, a[j*lda:j*lda+i+1], ci)
					}
				}
				c[i*ldc+i] = complex(real(c[i*ldc+i]), 0)
			}
		}
	}
}

// Zher2k performs one of the hermitian rank-2k operations
//  C = alpha*A*B^H + conj(alpha)*B*A^H + beta*C  if trans == blas.NoTrans
//  C = alpha*A^H*B + conj(alpha)*B^H*A + beta*C  if trans == blas.ConjTrans
// where alpha and beta are scalars with beta real, C is an n×n hermitian matrix
// and A and B are n×k matrices in the first case and k×n matrices in the second case.
//
// The imaginary parts of the diagonal elements of C are assumed to be zero, and
// on return they will be set to zero.
func (Implementation) Zher2k(uplo blas.Uplo, trans blas.Transpose, n, k int, alpha complex128, a []complex128, lda int, b []complex128, ldb int, beta float64, c []complex128, ldc int) {
	var row, col int
	switch trans {
	default:
		panic(badTranspose)
	case blas.NoTrans:
		row, col = n, k
	case blas.ConjTrans:
		row, col = k, n
	}
	switch {
	case uplo != blas.Lower && uplo != blas.Upper:
		panic(badUplo)
	case n < 0:
		panic(nLT0)
	case k < 0:
		panic(kLT0)
	case lda < max(1, col):
		panic(badLdA)
	case ldb < max(1, col):
		panic(badLdB)
	case ldc < max(1, n):
		panic(badLdC)
	}

	// Quick return if possible.
	if n == 0 {
		return
	}

	// For zero matrix size the following slice length checks are trivially satisfied.
	if len(a) < (row-1)*lda+col {
		panic(shortA)
	}
	if len(b) < (row-1)*ldb+col {
		panic(shortB)
	}
	if len(c) < (n-1)*ldc+n {
		panic(shortC)
	}

	// Quick return if possible.
	if (alpha == 0 || k == 0) && beta == 1 {
		return
	}

	if alpha == 0 {
		if uplo == blas.Upper {
			if beta == 0 {
				for i := 0; i < n; i++ {
					ci := c[i*ldc+i : i*ldc+n]
					for j := range ci {
						ci[j] = 0
					}
				}
			} else {
				for i := 0; i < n; i++ {
					ci := c[i*ldc+i : i*ldc+n]
					ci[0] = complex(beta*real(ci[0]), 0)
					if i != n-1 {
						c128.DscalUnitary(beta, ci[1:])
					}
				}
			}
		} else {
			if beta == 0 {
				for i := 0; i < n; i++ {
					ci := c[i*ldc : i*ldc+i+1]
					for j := range ci {
						ci[j] = 0
					}
				}
			} else {
				for i := 0; i < n; i++ {
					ci := c[i*ldc : i*ldc+i+1]
					if i != 0 {
						c128.DscalUnitary(beta, ci[:i])
					}
					ci[i] = complex(beta*real(ci[i]), 0)
				}
			}
		}
		return
	}

	conjalpha := cmplx.Conj(alpha)
	cbeta := complex(beta, 0)
	if trans == blas.NoTrans {
		// Form  C = alpha*A*B^H + conj(alpha)*B*A^H + beta*C.
		if uplo == blas.Upper {
			for i := 0; i < n; i++ {
				ci := c[i*ldc+i+1 : i*ldc+n]
				ai := a[i*lda : i*lda+k]
				bi := b[i*ldb : i*ldb+k]
				if beta == 0 {
					cii := alpha*c128.DotcUnitary(bi, ai) + conjalpha*c128.DotcUnitary(ai, bi)
					c[i*ldc+i] = complex(real(cii), 0)
					for jc := range ci {
						j := i + 1 + jc
						ci[jc] = alpha*c128.DotcUnitary(b[j*ldb:j*ldb+k], ai) + conjalpha*c128.DotcUnitary(a[j*lda:j*lda+k], bi)
					}
				} else {
					cii := alpha*c128.DotcUnitary(bi, ai) + conjalpha*c128.DotcUnitary(ai, bi) + cbeta*c[i*ldc+i]
					c[i*ldc+i] = complex(real(cii), 0)
					for jc, cij := range ci {
						j := i + 1 + jc
						ci[jc] = alpha*c128.DotcUnitary(b[j*ldb:j*ldb+k], ai) + conjalpha*c128.DotcUnitary(a[j*lda:j*lda+k], bi) + cbeta*cij
					}
				}
			}
		} else {
			for i := 0; i < n; i++ {
				ci := c[i*ldc : i*ldc+i]
				ai := a[i*lda : i*lda+k]
				bi := b[i*ldb : i*ldb+k]
				if beta == 0 {
					for j := range ci {
						ci[j] = alpha*c128.DotcUnitary(b[j*ldb:j*ldb+k], ai) + conjalpha*c128.DotcUnitary(a[j*lda:j*lda+k], bi)
					}
					cii := alpha*c128.DotcUnitary(bi, ai) + conjalpha*c128.DotcUnitary(ai, bi)
					c[i*ldc+i] = complex(real(cii), 0)
				} else {
					for j, cij := range ci {
						ci[j] = alpha*c128.DotcUnitary(b[j*ldb:j*ldb+k], ai) + conjalpha*c128.DotcUnitary(a[j*lda:j*lda+k], bi) + cbeta*cij
					}
					cii := alpha*c128.DotcUnitary(bi, ai) + conjalpha*c128.DotcUnitary(ai, bi) + cbeta*c[i*ldc+i]
					c[i*ldc+i] = complex(real(cii), 0)
				}
			}
		}
	} else {
		// Form  C = alpha*A^H*B + conj(alpha)*B^H*A + beta*C.
		if uplo == blas.Upper {
			for i := 0; i < n; i++ {
				ci := c[i*ldc+i : i*ldc+n]
				switch {
				case beta == 0:
					for jc := range ci {
						ci[jc] = 0
					}
				case beta != 1:
					c128.DscalUnitary(beta, ci)
					ci[0] = complex(real(ci[0]), 0)
				default:
					ci[0] = complex(real(ci[0]), 0)
				}
				for j := 0; j < k; j++ {
					aji := a[j*lda+i]
					bji := b[j*ldb+i]
					if aji != 0 {
						c128.AxpyUnitary(alpha*cmplx.Conj(aji), b[j*ldb+i:j*ldb+n], ci)
					}
					if bji != 0 {
						c128.AxpyUnitary(conjalpha*cmplx.Conj(bji), a[j*lda+i:j*lda+n], ci)
					}
				}
				ci[0] = complex(real(ci[0]), 0)
			}
		} else {
			for i := 0; i < n; i++ {
				ci := c[i*ldc : i*ldc+i+1]
				switch {
				case beta == 0:
					for j := range ci {
						ci[j] = 0
					}
				case beta != 1:
					c128.DscalUnitary(beta, ci)
					ci[i] = complex(real(ci[i]), 0)
				default:
					ci[i] = complex(real(ci[i]), 0)
				}
				for j := 0; j < k; j++ {
					aji := a[j*lda+i]
					bji := b[j*ldb+i]
					if aji != 0 {
						c128.AxpyUnitary(alpha*cmplx.Conj(aji), b[j*ldb:j*ldb+i+1], ci)
					}
					if bji != 0 {
						c128.AxpyUnitary(conjalpha*cmplx.Conj(bji), a[j*lda:j*lda+i+1], ci)
					}
				}
				ci[i] = complex(real(ci[i]), 0)
			}
		}
	}
}

// Zsymm performs one of the matrix-matrix operations
//  C = alpha*A*B + beta*C  if side == blas.Left
//  C = alpha*B*A + beta*C  if side == blas.Right
// where alpha and beta are scalars, A is an m×m or n×n symmetric matrix and B
// and C are m×n matrices.
func (Implementation) Zsymm(side blas.Side, uplo blas.Uplo, m, n int, alpha complex128, a []complex128, lda int, b []complex128, ldb int, beta complex128, c []complex128, ldc int) {
	na := m
	if side == blas.Right {
		na = n
	}
	switch {
	case side != blas.Left && side != blas.Right:
		panic(badSide)
	case uplo != blas.Lower && uplo != blas.Upper:
		panic(badUplo)
	case m < 0:
		panic(mLT0)
	case n < 0:
		panic(nLT0)
	case lda < max(1, na):
		panic(badLdA)
	case ldb < max(1, n):
		panic(badLdB)
	case ldc < max(1, n):
		panic(badLdC)
	}

	// Quick return if possible.
	if m == 0 || n == 0 {
		return
	}

	// For zero matrix size the following slice length checks are trivially satisfied.
	if len(a) < lda*(na-1)+na {
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
				ci := c[i*ldc : i*ldc+n]
				for j := range ci {
					ci[j] = 0
				}
			}
		} else {
			for i := 0; i < m; i++ {
				ci := c[i*ldc : i*ldc+n]
				c128.ScalUnitary(beta, ci)
			}
		}
		return
	}

	if side == blas.Left {
		// Form  C = alpha*A*B + beta*C.
		for i := 0; i < m; i++ {
			atmp := alpha * a[i*lda+i]
			bi := b[i*ldb : i*ldb+n]
			ci := c[i*ldc : i*ldc+n]
			if beta == 0 {
				for j, bij := range bi {
					ci[j] = atmp * bij
				}
			} else {
				for j, bij := range bi {
					ci[j] = atmp*bij + beta*ci[j]
				}
			}
			if uplo == blas.Upper {
				for k := 0; k < i; k++ {
					atmp = alpha * a[k*lda+i]
					c128.AxpyUnitary(atmp, b[k*ldb:k*ldb+n], ci)
				}
				for k := i + 1; k < m; k++ {
					atmp = alpha * a[i*lda+k]
					c128.AxpyUnitary(atmp, b[k*ldb:k*ldb+n], ci)
				}
			} else {
				for k := 0; k < i; k++ {
					atmp = alpha * a[i*lda+k]
					c128.AxpyUnitary(atmp, b[k*ldb:k*ldb+n], ci)
				}
				for k := i + 1; k < m; k++ {
					atmp = alpha * a[k*lda+i]
					c128.AxpyUnitary(atmp, b[k*ldb:k*ldb+n], ci)
				}
			}
		}
	} else {
		// Form  C = alpha*B*A + beta*C.
		if uplo == blas.Upper {
			for i := 0; i < m; i++ {
				for j := n - 1; j >= 0; j-- {
					abij := alpha * b[i*ldb+j]
					aj := a[j*lda+j+1 : j*lda+n]
					bi := b[i*ldb+j+1 : i*ldb+n]
					ci := c[i*ldc+j+1 : i*ldc+n]
					var tmp complex128
					for k, ajk := range aj {
						ci[k] += abij * ajk
						tmp += bi[k] * ajk
					}
					if beta == 0 {
						c[i*ldc+j] = abij*a[j*lda+j] + alpha*tmp
					} else {
						c[i*ldc+j] = abij*a[j*lda+j] + alpha*tmp + beta*c[i*ldc+j]
					}
				}
			}
		} else {
			for i := 0; i < m; i++ {
				for j := 0; j < n; j++ {
					abij := alpha * b[i*ldb+j]
					aj := a[j*lda : j*lda+j]
					bi := b[i*ldb : i*ldb+j]
					ci := c[i*ldc : i*ldc+j]
					var tmp complex128
					for k, ajk := range aj {
						ci[k] += abij * ajk
						tmp += bi[k] * ajk
					}
					if beta == 0 {
						c[i*ldc+j] = abij*a[j*lda+j] + alpha*tmp
					} else {
						c[i*ldc+j] = abij*a[j*lda+j] + alpha*tmp + beta*c[i*ldc+j]
					}
				}
			}
		}
	}
}

// Zsyrk performs one of the symmetric rank-k operations
//  C = alpha*A*A^T + beta*C  if trans == blas.NoTrans
//  C = alpha*A^T*A + beta*C  if trans == blas.Trans
// where alpha and beta are scalars, C is an n×n symmetric matrix and A is
// an n×k matrix in the first case and a k×n matrix in the second case.
func (Implementation) Zsyrk(uplo blas.Uplo, trans blas.Transpose, n, k int, alpha complex128, a []complex128, lda int, beta complex128, c []complex128, ldc int) {
	var rowA, colA int
	switch trans {
	default:
		panic(badTranspose)
	case blas.NoTrans:
		rowA, colA = n, k
	case blas.Trans:
		rowA, colA = k, n
	}
	switch {
	case uplo != blas.Lower && uplo != blas.Upper:
		panic(badUplo)
	case n < 0:
		panic(nLT0)
	case k < 0:
		panic(kLT0)
	case lda < max(1, colA):
		panic(badLdA)
	case ldc < max(1, n):
		panic(badLdC)
	}

	// Quick return if possible.
	if n == 0 {
		return
	}

	// For zero matrix size the following slice length checks are trivially satisfied.
	if len(a) < (rowA-1)*lda+colA {
		panic(shortA)
	}
	if len(c) < (n-1)*ldc+n {
		panic(shortC)
	}

	// Quick return if possible.
	if (alpha == 0 || k == 0) && beta == 1 {
		return
	}

	if alpha == 0 {
		if uplo == blas.Upper {
			if beta == 0 {
				for i := 0; i < n; i++ {
					ci := c[i*ldc+i : i*ldc+n]
					for j := range ci {
						ci[j] = 0
					}
				}
			} else {
				for i := 0; i < n; i++ {
					ci := c[i*ldc+i : i*ldc+n]
					c128.ScalUnitary(beta, ci)
				}
			}
		} else {
			if beta == 0 {
				for i := 0; i < n; i++ {
					ci := c[i*ldc : i*ldc+i+1]
					for j := range ci {
						ci[j] = 0
					}
				}
			} else {
				for i := 0; i < n; i++ {
					ci := c[i*ldc : i*ldc+i+1]
					c128.ScalUnitary(beta, ci)
				}
			}
		}
		return
	}

	if trans == blas.NoTrans {
		// Form  C = alpha*A*A^T + beta*C.
		if uplo == blas.Upper {
			for i := 0; i < n; i++ {
				ci := c[i*ldc+i : i*ldc+n]
				ai := a[i*lda : i*lda+k]
				for jc, cij := range ci {
					j := i + jc
					ci[jc] = beta*cij + alpha*c128.DotuUnitary(ai, a[j*lda:j*lda+k])
				}
			}
		} else {
			for i := 0; i < n; i++ {
				ci := c[i*ldc : i*ldc+i+1]
				ai := a[i*lda : i*lda+k]
				for j, cij := range ci {
					ci[j] = beta*cij + alpha*c128.DotuUnitary(ai, a[j*lda:j*lda+k])
				}
			}
		}
	} else {
		// Form  C = alpha*A^T*A + beta*C.
		if uplo == blas.Upper {
			for i := 0; i < n; i++ {
				ci := c[i*ldc+i : i*ldc+n]
				switch {
				case beta == 0:
					for jc := range ci {
						ci[jc] = 0
					}
				case beta != 1:
					for jc := range ci {
						ci[jc] *= beta
					}
				}
				for j := 0; j < k; j++ {
					aji := a[j*lda+i]
					if aji != 0 {
						c128.AxpyUnitary(alpha*aji, a[j*lda+i:j*lda+n], ci)
					}
				}
			}
		} else {
			for i := 0; i < n; i++ {
				ci := c[i*ldc : i*ldc+i+1]
				switch {
				case beta == 0:
					for j := range ci {
						ci[j] = 0
					}
				case beta != 1:
					for j := range ci {
						ci[j] *= beta
					}
				}
				for j := 0; j < k; j++ {
					aji := a[j*lda+i]
					if aji != 0 {
						c128.AxpyUnitary(alpha*aji, a[j*lda:j*lda+i+1], ci)
					}
				}
			}
		}
	}
}

// Zsyr2k performs one of the symmetric rank-2k operations
//  C = alpha*A*B^T + alpha*B*A^T + beta*C  if trans == blas.NoTrans
//  C = alpha*A^T*B + alpha*B^T*A + beta*C  if trans == blas.Trans
// where alpha and beta are scalars, C is an n×n symmetric matrix and A and B
// are n×k matrices in the first case and k×n matrices in the second case.
func (Implementation) Zsyr2k(uplo blas.Uplo, trans blas.Transpose, n, k int, alpha complex128, a []complex128, lda int, b []complex128, ldb int, beta complex128, c []complex128, ldc int) {
	var row, col int
	switch trans {
	default:
		panic(badTranspose)
	case blas.NoTrans:
		row, col = n, k
	case blas.Trans:
		row, col = k, n
	}
	switch {
	case uplo != blas.Lower && uplo != blas.Upper:
		panic(badUplo)
	case n < 0:
		panic(nLT0)
	case k < 0:
		panic(kLT0)
	case lda < max(1, col):
		panic(badLdA)
	case ldb < max(1, col):
		panic(badLdB)
	case ldc < max(1, n):
		panic(badLdC)
	}

	// Quick return if possible.
	if n == 0 {
		return
	}

	// For zero matrix size the following slice length checks are trivially satisfied.
	if len(a) < (row-1)*lda+col {
		panic(shortA)
	}
	if len(b) < (row-1)*ldb+col {
		panic(shortB)
	}
	if len(c) < (n-1)*ldc+n {
		panic(shortC)
	}

	// Quick return if possible.
	if (alpha == 0 || k == 0) && beta == 1 {
		return
	}

	if alpha == 0 {
		if uplo == blas.Upper {
			if beta == 0 {
				for i := 0; i < n; i++ {
					ci := c[i*ldc+i : i*ldc+n]
					for j := range ci {
						ci[j] = 0
					}
				}
			} else {
				for i := 0; i < n; i++ {
					ci := c[i*ldc+i : i*ldc+n]
					c128.ScalUnitary(beta, ci)
				}
			}
		} else {
			if beta == 0 {
				for i := 0; i < n; i++ {
					ci := c[i*ldc : i*ldc+i+1]
					for j := range ci {
						ci[j] = 0
					}
				}
			} else {
				for i := 0; i < n; i++ {
					ci := c[i*ldc : i*ldc+i+1]
					c128.ScalUnitary(beta, ci)
				}
			}
		}
		return
	}

	if trans == blas.NoTrans {
		// Form  C = alpha*A*B^T + alpha*B*A^T + beta*C.
		if uplo == blas.Upper {
			for i := 0; i < n; i++ {
				ci := c[i*ldc+i : i*ldc+n]
				ai := a[i*lda : i*lda+k]
				bi := b[i*ldb : i*ldb+k]
				if beta == 0 {
					for jc := range ci {
						j := i + jc
						ci[jc] = alpha*c128.DotuUnitary(ai, b[j*ldb:j*ldb+k]) + alpha*c128.DotuUnitary(bi, a[j*lda:j*lda+k])
					}
				} else {
					for jc, cij := range ci {
						j := i + jc
						ci[jc] = alpha*c128.DotuUnitary(ai, b[j*ldb:j*ldb+k]) + alpha*c128.DotuUnitary(bi, a[j*lda:j*lda+k]) + beta*cij
					}
				}
			}
		} else {
			for i := 0; i < n; i++ {
				ci := c[i*ldc : i*ldc+i+1]
				ai := a[i*lda : i*lda+k]
				bi := b[i*ldb : i*ldb+k]
				if beta == 0 {
					for j := range ci {
						ci[j] = alpha*c128.DotuUnitary(ai, b[j*ldb:j*ldb+k]) + alpha*c128.DotuUnitary(bi, a[j*lda:j*lda+k])
					}
				} else {
					for j, cij := range ci {
						ci[j] = alpha*c128.DotuUnitary(ai, b[j*ldb:j*ldb+k]) + alpha*c128.DotuUnitary(bi, a[j*lda:j*lda+k]) + beta*cij
					}
				}
			}
		}
	} else {
		// Form  C = alpha*A^T*B + alpha*B^T*A + beta*C.
		if uplo == blas.Upper {
			for i := 0; i < n; i++ {
				ci := c[i*ldc+i : i*ldc+n]
				switch {
				case beta == 0:
					for jc := range ci {
						ci[jc] = 0
					}
				case beta != 1:
					for jc := range ci {
						ci[jc] *= beta
					}
				}
				for j := 0; j < k; j++ {
					aji := a[j*lda+i]
					bji := b[j*ldb+i]
					if aji != 0 {
						c128.AxpyUnitary(alpha*aji, b[j*ldb+i:j*ldb+n], ci)
					}
					if bji != 0 {
						c128.AxpyUnitary(alpha*bji, a[j*lda+i:j*lda+n], ci)
					}
				}
			}
		} else {
			for i := 0; i < n; i++ {
				ci := c[i*ldc : i*ldc+i+1]
				switch {
				case beta == 0:
					for j := range ci {
						ci[j] = 0
					}
				case beta != 1:
					for j := range ci {
						ci[j] *= beta
					}
				}
				for j := 0; j < k; j++ {
					aji := a[j*lda+i]
					bji := b[j*ldb+i]
					if aji != 0 {
						c128.AxpyUnitary(alpha*aji, b[j*ldb:j*ldb+i+1], ci)
					}
					if bji != 0 {
						c128.AxpyUnitary(alpha*bji, a[j*lda:j*lda+i+1], ci)
					}
				}
			}
		}
	}
}

// Ztrmm performs one of the matrix-matrix operations
//  B = alpha * op(A) * B  if side == blas.Left,
//  B = alpha * B * op(A)  if side == blas.Right,
// where alpha is a scalar, B is an m×n matrix, A is a unit, or non-unit,
// upper or lower triangular matrix and op(A) is one of
//  op(A) = A    if trans == blas.NoTrans,
//  op(A) = A^T  if trans == blas.Trans,
//  op(A) = A^H  if trans == blas.ConjTrans.
func (Implementation) Ztrmm(side blas.Side, uplo blas.Uplo, trans blas.Transpose, diag blas.Diag, m, n int, alpha complex128, a []complex128, lda int, b []complex128, ldb int) {
	na := m
	if side == blas.Right {
		na = n
	}
	switch {
	case side != blas.Left && side != blas.Right:
		panic(badSide)
	case uplo != blas.Lower && uplo != blas.Upper:
		panic(badUplo)
	case trans != blas.NoTrans && trans != blas.Trans && trans != blas.ConjTrans:
		panic(badTranspose)
	case diag != blas.Unit && diag != blas.NonUnit:
		panic(badDiag)
	case m < 0:
		panic(mLT0)
	case n < 0:
		panic(nLT0)
	case lda < max(1, na):
		panic(badLdA)
	case ldb < max(1, n):
		panic(badLdB)
	}

	// Quick return if possible.
	if m == 0 || n == 0 {
		return
	}

	// For zero matrix size the following slice length checks are trivially satisfied.
	if len(a) < (na-1)*lda+na {
		panic(shortA)
	}
	if len(b) < (m-1)*ldb+n {
		panic(shortB)
	}

	// Quick return if possible.
	if alpha == 0 {
		for i := 0; i < m; i++ {
			bi := b[i*ldb : i*ldb+n]
			for j := range bi {
				bi[j] = 0
			}
		}
		return
	}

	noConj := trans != blas.ConjTrans
	noUnit := diag == blas.NonUnit
	if side == blas.Left {
		if trans == blas.NoTrans {
			// Form B = alpha*A*B.
			if uplo == blas.Upper {
				for i := 0; i < m; i++ {
					aii := alpha
					if noUnit {
						aii *= a[i*lda+i]
					}
					bi := b[i*ldb : i*ldb+n]
					for j := range bi {
						bi[j] *= aii
					}
					for ja, aij := range a[i*lda+i+1 : i*lda+m] {
						j := ja + i + 1
						if aij != 0 {
							c128.AxpyUnitary(alpha*aij, b[j*ldb:j*ldb+n], bi)
						}
					}
				}
			} else {
				for i := m - 1; i >= 0; i-- {
					aii := alpha
					if noUnit {
						aii *= a[i*lda+i]
					}
					bi := b[i*ldb : i*ldb+n]
					for j := range bi {
						bi[j] *= aii
					}
					for j, aij := range a[i*lda : i*lda+i] {
						if aij != 0 {
							c128.AxpyUnitary(alpha*aij, b[j*ldb:j*ldb+n], bi)
						}
					}
				}
			}
		} else {
			// Form B = alpha*A^T*B  or  B = alpha*A^H*B.
			if uplo == blas.Upper {
				for k := m - 1; k >= 0; k-- {
					bk := b[k*ldb : k*ldb+n]
					for ja, ajk := range a[k*lda+k+1 : k*lda+m] {
						if ajk == 0 {
							continue
						}
						j := k + 1 + ja
						if noConj {
							c128.AxpyUnitary(alpha*ajk, bk, b[j*ldb:j*ldb+n])
						} else {
							c128.AxpyUnitary(alpha*cmplx.Conj(ajk), bk, b[j*ldb:j*ldb+n])
						}
					}
					akk := alpha
					if noUnit {
						if noConj {
							akk *= a[k*lda+k]
						} else {
							akk *= cmplx.Conj(a[k*lda+k])
						}
					}
					if akk != 1 {
						c128.ScalUnitary(akk, bk)
					}
				}
			} else {
				for k := 0; k < m; k++ {
					bk := b[k*ldb : k*ldb+n]
					for j, ajk := range a[k*lda : k*lda+k] {
						if ajk == 0 {
							continue
						}
						if noConj {
							c128.AxpyUnitary(alpha*ajk, bk, b[j*ldb:j*ldb+n])
						} else {
							c128.AxpyUnitary(alpha*cmplx.Conj(ajk), bk, b[j*ldb:j*ldb+n])
						}
					}
					akk := alpha
					if noUnit {
						if noConj {
							akk *= a[k*lda+k]
						} else {
							akk *= cmplx.Conj(a[k*lda+k])
						}
					}
					if akk != 1 {
						c128.ScalUnitary(akk, bk)
					}
				}
			}
		}
	} else {
		if trans == blas.NoTrans {
			// Form B = alpha*B*A.
			if uplo == blas.Upper {
				for i := 0; i < m; i++ {
					bi := b[i*ldb : i*ldb+n]
					for k := n - 1; k >= 0; k-- {
						abik := alpha * bi[k]
						if abik == 0 {
							continue
						}
						bi[k] = abik
						if noUnit {
							bi[k] *= a[k*lda+k]
						}
						c128.AxpyUnitary(abik, a[k*lda+k+1:k*lda+n], bi[k+1:])
					}
				}
			} else {
				for i := 0; i < m; i++ {
					bi := b[i*ldb : i*ldb+n]
					for k := 0; k < n; k++ {
						abik := alpha * bi[k]
						if abik == 0 {
							continue
						}
						bi[k] = abik
						if noUnit {
							bi[k] *= a[k*lda+k]
						}
						c128.AxpyUnitary(abik, a[k*lda:k*lda+k], bi[:k])
					}
				}
			}
		} else {
			// Form B = alpha*B*A^T  or  B = alpha*B*A^H.
			if uplo == blas.Upper {
				for i := 0; i < m; i++ {
					bi := b[i*ldb : i*ldb+n]
					for j, bij := range bi {
						if noConj {
							if noUnit {
								bij *= a[j*lda+j]
							}
							bij += c128.DotuUnitary(a[j*lda+j+1:j*lda+n], bi[j+1:n])
						} else {
							if noUnit {
								bij *= cmplx.Conj(a[j*lda+j])
							}
							bij += c128.DotcUnitary(a[j*lda+j+1:j*lda+n], bi[j+1:n])
						}
						bi[j] = alpha * bij
					}
				}
			} else {
				for i := 0; i < m; i++ {
					bi := b[i*ldb : i*ldb+n]
					for j := n - 1; j >= 0; j-- {
						bij := bi[j]
						if noConj {
							if noUnit {
								bij *= a[j*lda+j]
							}
							bij += c128.DotuUnitary(a[j*lda:j*lda+j], bi[:j])
						} else {
							if noUnit {
								bij *= cmplx.Conj(a[j*lda+j])
							}
							bij += c128.DotcUnitary(a[j*lda:j*lda+j], bi[:j])
						}
						bi[j] = alpha * bij
					}
				}
			}
		}
	}
}

// Ztrsm solves one of the matrix equations
//  op(A) * X = alpha * B  if side == blas.Left,
//  X * op(A) = alpha * B  if side == blas.Right,
// where alpha is a scalar, X and B are m×n matrices, A is a unit or
// non-unit, upper or lower triangular matrix and op(A) is one of
//  op(A) = A    if transA == blas.NoTrans,
//  op(A) = A^T  if transA == blas.Trans,
//  op(A) = A^H  if transA == blas.ConjTrans.
// On return the matrix X is overwritten on B.
func (Implementation) Ztrsm(side blas.Side, uplo blas.Uplo, transA blas.Transpose, diag blas.Diag, m, n int, alpha complex128, a []complex128, lda int, b []complex128, ldb int) {
	na := m
	if side == blas.Right {
		na = n
	}
	switch {
	case side != blas.Left && side != blas.Right:
		panic(badSide)
	case uplo != blas.Lower && uplo != blas.Upper:
		panic(badUplo)
	case transA != blas.NoTrans && transA != blas.Trans && transA != blas.ConjTrans:
		panic(badTranspose)
	case diag != blas.Unit && diag != blas.NonUnit:
		panic(badDiag)
	case m < 0:
		panic(mLT0)
	case n < 0:
		panic(nLT0)
	case lda < max(1, na):
		panic(badLdA)
	case ldb < max(1, n):
		panic(badLdB)
	}

	// Quick return if possible.
	if m == 0 || n == 0 {
		return
	}

	// For zero matrix size the following slice length checks are trivially satisfied.
	if len(a) < (na-1)*lda+na {
		panic(shortA)
	}
	if len(b) < (m-1)*ldb+n {
		panic(shortB)
	}

	if alpha == 0 {
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				b[i*ldb+j] = 0
			}
		}
		return
	}

	noConj := transA != blas.ConjTrans
	noUnit := diag == blas.NonUnit
	if side == blas.Left {
		if transA == blas.NoTrans {
			// Form  B = alpha*inv(A)*B.
			if uplo == blas.Upper {
				for i := m - 1; i >= 0; i-- {
					bi := b[i*ldb : i*ldb+n]
					if alpha != 1 {
						c128.ScalUnitary(alpha, bi)
					}
					for ka, aik := range a[i*lda+i+1 : i*lda+m] {
						k := i + 1 + ka
						if aik != 0 {
							c128.AxpyUnitary(-aik, b[k*ldb:k*ldb+n], bi)
						}
					}
					if noUnit {
						c128.ScalUnitary(1/a[i*lda+i], bi)
					}
				}
			} else {
				for i := 0; i < m; i++ {
					bi := b[i*ldb : i*ldb+n]
					if alpha != 1 {
						c128.ScalUnitary(alpha, bi)
					}
					for j, aij := range a[i*lda : i*lda+i] {
						if aij != 0 {
							c128.AxpyUnitary(-aij, b[j*ldb:j*ldb+n], bi)
						}
					}
					if noUnit {
						c128.ScalUnitary(1/a[i*lda+i], bi)
					}
				}
			}
		} else {
			// Form  B = alpha*inv(A^T)*B  or  B = alpha*inv(A^H)*B.
			if uplo == blas.Upper {
				for i := 0; i < m; i++ {
					bi := b[i*ldb : i*ldb+n]
					if noUnit {
						if noConj {
							c128.ScalUnitary(1/a[i*lda+i], bi)
						} else {
							c128.ScalUnitary(1/cmplx.Conj(a[i*lda+i]), bi)
						}
					}
					for ja, aij := range a[i*lda+i+1 : i*lda+m] {
						if aij == 0 {
							continue
						}
						j := i + 1 + ja
						if noConj {
							c128.AxpyUnitary(-aij, bi, b[j*ldb:j*ldb+n])
						} else {
							c128.AxpyUnitary(-cmplx.Conj(aij), bi, b[j*ldb:j*ldb+n])
						}
					}
					if alpha != 1 {
						c128.ScalUnitary(alpha, bi)
					}
				}
			} else {
				for i := m - 1; i >= 0; i-- {
					bi := b[i*ldb : i*ldb+n]
					if noUnit {
						if noConj {
							c128.ScalUnitary(1/a[i*lda+i], bi)
						} else {
							c128.ScalUnitary(1/cmplx.Conj(a[i*lda+i]), bi)
						}
					}
					for j, aij := range a[i*lda : i*lda+i] {
						if aij == 0 {
							continue
						}
						if noConj {
							c128.AxpyUnitary(-aij, bi, b[j*ldb:j*ldb+n])
						} else {
							c128.AxpyUnitary(-cmplx.Conj(aij), bi, b[j*ldb:j*ldb+n])
						}
					}
					if alpha != 1 {
						c128.ScalUnitary(alpha, bi)
					}
				}
			}
		}
	} else {
		if transA == blas.NoTrans {
			// Form  B = alpha*B*inv(A).
			if uplo == blas.Upper {
				for i := 0; i < m; i++ {
					bi := b[i*ldb : i*ldb+n]
					if alpha != 1 {
						c128.ScalUnitary(alpha, bi)
					}
					for j, bij := range bi {
						if bij == 0 {
							continue
						}
						if noUnit {
							bi[j] /= a[j*lda+j]
						}
						c128.AxpyUnitary(-bi[j], a[j*lda+j+1:j*lda+n], bi[j+1:n])
					}
				}
			} else {
				for i := 0; i < m; i++ {
					bi := b[i*ldb : i*ldb+n]
					if alpha != 1 {
						c128.ScalUnitary(alpha, bi)
					}
					for j := n - 1; j >= 0; j-- {
						if bi[j] == 0 {
							continue
						}
						if noUnit {
							bi[j] /= a[j*lda+j]
						}
						c128.AxpyUnitary(-bi[j], a[j*lda:j*lda+j], bi[:j])
					}
				}
			}
		} else {
			// Form  B = alpha*B*inv(A^T)  or   B = alpha*B*inv(A^H).
			if uplo == blas.Upper {
				for i := 0; i < m; i++ {
					bi := b[i*ldb : i*ldb+n]
					for j := n - 1; j >= 0; j-- {
						bij := alpha * bi[j]
						if noConj {
							bij -= c128.DotuUnitary(a[j*lda+j+1:j*lda+n], bi[j+1:n])
							if noUnit {
								bij /= a[j*lda+j]
							}
						} else {
							bij -= c128.DotcUnitary(a[j*lda+j+1:j*lda+n], bi[j+1:n])
							if noUnit {
								bij /= cmplx.Conj(a[j*lda+j])
							}
						}
						bi[j] = bij
					}
				}
			} else {
				for i := 0; i < m; i++ {
					bi := b[i*ldb : i*ldb+n]
					for j, bij := range bi {
						bij *= alpha
						if noConj {
							bij -= c128.DotuUnitary(a[j*lda:j*lda+j], bi[:j])
							if noUnit {
								bij /= a[j*lda+j]
							}
						} else {
							bij -= c128.DotcUnitary(a[j*lda:j*lda+j], bi[:j])
							if noUnit {
								bij /= cmplx.Conj(a[j*lda+j])
							}
						}
						bi[j] = bij
					}
				}
			}
		}
	}
}
