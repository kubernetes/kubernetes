// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

// Panic strings used during parameter checks.
// This list is duplicated in netlib/blas/netlib. Keep in sync.
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
	badFlag      = "blas: illegal rotm flag"

	badLdA = "blas: bad leading dimension of A"
	badLdB = "blas: bad leading dimension of B"
	badLdC = "blas: bad leading dimension of C"

	shortX  = "blas: insufficient length of x"
	shortY  = "blas: insufficient length of y"
	shortAP = "blas: insufficient length of ap"
	shortA  = "blas: insufficient length of a"
	shortB  = "blas: insufficient length of b"
	shortC  = "blas: insufficient length of c"
)
