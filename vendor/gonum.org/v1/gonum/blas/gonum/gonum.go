// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:generate ./single_precision.bash

package gonum

import "math"

type Implementation struct{}

// The following are panic strings used during parameter checks.
const (
	zeroIncX = "blas: zero x index increment"
	zeroIncY = "blas: zero y index increment"

	mLT0  = "blas: m < 0"
	nLT0  = "blas: n < 0"
	kLT0  = "blas: k < 0"
	kLLT0 = "blas: kL < 0"
	kULT0 = "blas: kU < 0"

	badUplo      = "blas: illegal triangle"
	badTranspose = "blas: illegal transpose"
	badDiag      = "blas: illegal diagonal"
	badSide      = "blas: illegal side"

	badLdA = "blas: bad leading dimension of A"
	badLdB = "blas: bad leading dimension of B"
	badLdC = "blas: bad leading dimension of C"

	badX = "blas: bad length of x"
	badY = "blas: bad length of y"
)

// [SD]gemm behavior constants. These are kept here to keep them out of the
// way during single precision code genration.
const (
	blockSize   = 64 // b x b matrix
	minParBlock = 4  // minimum number of blocks needed to go parallel
	buffMul     = 4  // how big is the buffer relative to the number of workers
)

// subMul is a common type shared by [SD]gemm.
type subMul struct {
	i, j int // index of block
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func min(a, b int) int {
	if a > b {
		return b
	}
	return a
}

func checkSMatrix(name byte, m, n int, a []float32, lda int) {
	if m < 0 {
		panic(mLT0)
	}
	if n < 0 {
		panic(nLT0)
	}
	if lda < n {
		panic("blas: illegal stride of " + string(name))
	}
	if len(a) < (m-1)*lda+n {
		panic("blas: index of " + string(name) + " out of range")
	}
}

func checkDMatrix(name byte, m, n int, a []float64, lda int) {
	if m < 0 {
		panic(mLT0)
	}
	if n < 0 {
		panic(nLT0)
	}
	if lda < n {
		panic("blas: illegal stride of " + string(name))
	}
	if len(a) < (m-1)*lda+n {
		panic("blas: index of " + string(name) + " out of range")
	}
}

func checkZMatrix(name byte, m, n int, a []complex128, lda int) {
	if m < 0 {
		panic(mLT0)
	}
	if n < 0 {
		panic(nLT0)
	}
	if lda < max(1, n) {
		panic("blas: illegal stride of " + string(name))
	}
	if len(a) < (m-1)*lda+n {
		panic("blas: insufficient " + string(name) + " matrix slice length")
	}
}

func checkZBandMatrix(name byte, m, n, kL, kU int, ab []complex128, ldab int) {
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
	if ldab < kL+kU+1 {
		panic("blas: illegal stride of band matrix " + string(name))
	}
	nRow := min(m, n+kL)
	if len(ab) < (nRow-1)*ldab+kL+1+kU {
		panic("blas: insufficient " + string(name) + " band matrix slice length")
	}
}

func checkZhbMatrix(name byte, n, k int, ab []complex128, ldab int) {
	if n < 0 {
		panic(nLT0)
	}
	if k < 0 {
		panic(kLT0)
	}
	if ldab < k+1 {
		panic("blas: illegal stride of Hermitian band matrix " + string(name))
	}
	if len(ab) < (n-1)*ldab+k+1 {
		panic("blas: insufficient " + string(name) + " Hermitian band matrix slice length")
	}
}

func checkZtbMatrix(name byte, n, k int, ab []complex128, ldab int) {
	if n < 0 {
		panic(nLT0)
	}
	if k < 0 {
		panic(kLT0)
	}
	if ldab < k+1 {
		panic("blas: illegal stride of triangular band matrix " + string(name))
	}
	if len(ab) < (n-1)*ldab+k+1 {
		panic("blas: insufficient " + string(name) + " triangular band matrix slice length")
	}
}

func checkZVector(name byte, n int, x []complex128, incX int) {
	if n < 0 {
		panic(nLT0)
	}
	if incX == 0 {
		panic(zeroIncX)
	}
	if (incX > 0 && (n-1)*incX >= len(x)) || (incX < 0 && (1-n)*incX >= len(x)) {
		panic("blas: insufficient " + string(name) + " vector slice length")
	}
}

// blocks returns the number of divisions of the dimension length with the given
// block size.
func blocks(dim, bsize int) int {
	return (dim + bsize - 1) / bsize
}

// dcabs1 returns |real(z)|+|imag(z)|.
func dcabs1(z complex128) float64 {
	return math.Abs(real(z)) + math.Abs(imag(z))
}
