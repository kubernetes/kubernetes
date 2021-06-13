// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO(kortschak): Generate this file from shadow.go when all complex type are available.

package mat

import "gonum.org/v1/gonum/blas/cblas128"

// checkOverlapComplex returns false if the receiver does not overlap data elements
// referenced by the parameter and panics otherwise.
//
// checkOverlapComplex methods return a boolean to allow the check call to be added to a
// boolean expression, making use of short-circuit operators.
func checkOverlapComplex(a, b cblas128.General) bool {
	if cap(a.Data) == 0 || cap(b.Data) == 0 {
		return false
	}

	off := offsetComplex(a.Data[:1], b.Data[:1])

	if off == 0 {
		// At least one element overlaps.
		if a.Cols == b.Cols && a.Rows == b.Rows && a.Stride == b.Stride {
			panic(regionIdentity)
		}
		panic(regionOverlap)
	}

	if off > 0 && len(a.Data) <= off {
		// We know a is completely before b.
		return false
	}
	if off < 0 && len(b.Data) <= -off {
		// We know a is completely after b.
		return false
	}

	if a.Stride != b.Stride && a.Stride != 1 && b.Stride != 1 {
		// Too hard, so assume the worst; if either stride
		// is one it will be caught in rectanglesOverlap.
		panic(mismatchedStrides)
	}

	if off < 0 {
		off = -off
		a.Cols, b.Cols = b.Cols, a.Cols
	}
	if rectanglesOverlap(off, a.Cols, b.Cols, min(a.Stride, b.Stride)) {
		panic(regionOverlap)
	}
	return false
}

func (m *CDense) checkOverlap(a cblas128.General) bool {
	return checkOverlapComplex(m.RawCMatrix(), a)
}

func (m *CDense) checkOverlapMatrix(a CMatrix) bool {
	if m == a {
		return false
	}
	var amat cblas128.General
	switch ar := a.(type) {
	default:
		return false
	case RawCMatrixer:
		amat = ar.RawCMatrix()
	}
	return m.checkOverlap(amat)
}
