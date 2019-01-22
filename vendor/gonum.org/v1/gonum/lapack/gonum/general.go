// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"gonum.org/v1/gonum/lapack"
)

// Implementation is the native Go implementation of LAPACK routines. It
// is built on top of calls to the return of blas64.Implementation(), so while
// this code is in pure Go, the underlying BLAS implementation may not be.
type Implementation struct{}

var _ lapack.Float64 = Implementation{}

// This list is duplicated in lapack/cgo. Keep in sync.
const (
	absIncNotOne    = "lapack: increment not one or negative one"
	badAlpha        = "lapack: bad alpha length"
	badAuxv         = "lapack: auxv has insufficient length"
	badBeta         = "lapack: bad beta length"
	badD            = "lapack: d has insufficient length"
	badDecompUpdate = "lapack: bad decomp update"
	badDiag         = "lapack: bad diag"
	badDims         = "lapack: bad input dimensions"
	badDirect       = "lapack: bad direct"
	badE            = "lapack: e has insufficient length"
	badEVComp       = "lapack: bad EVComp"
	badEVJob        = "lapack: bad EVJob"
	badEVSide       = "lapack: bad EVSide"
	badGSVDJob      = "lapack: bad GSVDJob"
	badHowMany      = "lapack: bad HowMany"
	badIlo          = "lapack: ilo out of range"
	badIhi          = "lapack: ihi out of range"
	badIpiv         = "lapack: bad permutation length"
	badJob          = "lapack: bad Job"
	badK1           = "lapack: k1 out of range"
	badK2           = "lapack: k2 out of range"
	badKperm        = "lapack: incorrect permutation length"
	badLdA          = "lapack: index of a out of range"
	badNb           = "lapack: nb out of range"
	badNorm         = "lapack: bad norm"
	badPivot        = "lapack: bad pivot"
	badS            = "lapack: s has insufficient length"
	badShifts       = "lapack: bad shifts"
	badSide         = "lapack: bad side"
	badSlice        = "lapack: bad input slice length"
	badSort         = "lapack: bad Sort"
	badStore        = "lapack: bad store"
	badTau          = "lapack: tau has insufficient length"
	badTauQ         = "lapack: tauQ has insufficient length"
	badTauP         = "lapack: tauP has insufficient length"
	badTrans        = "lapack: bad trans"
	badVn1          = "lapack: vn1 has insufficient length"
	badVn2          = "lapack: vn2 has insufficient length"
	badUplo         = "lapack: illegal triangle"
	badWork         = "lapack: insufficient working memory"
	badZ            = "lapack: insufficient z length"
	kGTM            = "lapack: k > m"
	kGTN            = "lapack: k > n"
	kLT0            = "lapack: k < 0"
	mLTN            = "lapack: m < n"
	nanScale        = "lapack: NaN scale factor"
	negDimension    = "lapack: negative matrix dimension"
	negZ            = "lapack: negative z value"
	nLT0            = "lapack: n < 0"
	nLTM            = "lapack: n < m"
	offsetGTM       = "lapack: offset > m"
	shortWork       = "lapack: working array shorter than declared"
	zeroDiv         = "lapack: zero divisor"
)

// checkMatrix verifies the parameters of a matrix input.
func checkMatrix(m, n int, a []float64, lda int) {
	if m < 0 {
		panic("lapack: has negative number of rows")
	}
	if n < 0 {
		panic("lapack: has negative number of columns")
	}
	if lda < n {
		panic("lapack: stride less than number of columns")
	}
	if len(a) < (m-1)*lda+n {
		panic("lapack: insufficient matrix slice length")
	}
}

func checkVector(n int, v []float64, inc int) {
	if n < 0 {
		panic("lapack: negative vector length")
	}
	if (inc > 0 && (n-1)*inc >= len(v)) || (inc < 0 && (1-n)*inc >= len(v)) {
		panic("lapack: insufficient vector slice length")
	}
}

func checkSymBanded(ab []float64, n, kd, lda int) {
	if n < 0 {
		panic("lapack: negative banded length")
	}
	if kd < 0 {
		panic("lapack: negative bandwidth value")
	}
	if lda < kd+1 {
		panic("lapack: stride less than number of bands")
	}
	if len(ab) < (n-1)*lda+kd {
		panic("lapack: insufficient banded vector length")
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

const (
	// dlamchE is the machine epsilon. For IEEE this is 2^{-53}.
	dlamchE = 1.0 / (1 << 53)

	// dlamchB is the radix of the machine (the base of the number system).
	dlamchB = 2

	// dlamchP is base * eps.
	dlamchP = dlamchB * dlamchE

	// dlamchS is the "safe minimum", that is, the lowest number such that
	// 1/dlamchS does not overflow, or also the smallest normal number.
	// For IEEE this is 2^{-1022}.
	dlamchS = 1.0 / (1 << 256) / (1 << 256) / (1 << 256) / (1 << 254)
)
