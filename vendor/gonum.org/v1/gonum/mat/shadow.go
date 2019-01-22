// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat

import (
	"gonum.org/v1/gonum/blas/blas64"
)

const (
	// regionOverlap is the panic string used for the general case
	// of a matrix region overlap between a source and destination.
	regionOverlap = "mat: bad region: overlap"

	// regionIdentity is the panic string used for the specific
	// case of complete agreement between a source and a destination.
	regionIdentity = "mat: bad region: identical"

	// mismatchedStrides is the panic string used for overlapping
	// data slices with differing strides.
	mismatchedStrides = "mat: bad region: different strides"
)

// checkOverlap returns false if the receiver does not overlap data elements
// referenced by the parameter and panics otherwise.
//
// checkOverlap methods return a boolean to allow the check call to be added to a
// boolean expression, making use of short-circuit operators.
func checkOverlap(a, b blas64.General) bool {
	if cap(a.Data) == 0 || cap(b.Data) == 0 {
		return false
	}

	off := offset(a.Data[:1], b.Data[:1])

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

	if a.Stride != b.Stride {
		// Too hard, so assume the worst.
		panic(mismatchedStrides)
	}

	if off < 0 {
		off = -off
		a.Cols, b.Cols = b.Cols, a.Cols
	}
	if rectanglesOverlap(off, a.Cols, b.Cols, a.Stride) {
		panic(regionOverlap)
	}
	return false
}

func (m *Dense) checkOverlap(a blas64.General) bool {
	return checkOverlap(m.RawMatrix(), a)
}

func (m *Dense) checkOverlapMatrix(a Matrix) bool {
	if m == a {
		return false
	}
	var amat blas64.General
	switch a := a.(type) {
	default:
		return false
	case RawMatrixer:
		amat = a.RawMatrix()
	case RawSymmetricer:
		amat = generalFromSymmetric(a.RawSymmetric())
	case RawTriangular:
		amat = generalFromTriangular(a.RawTriangular())
	}
	return m.checkOverlap(amat)
}

func (s *SymDense) checkOverlap(a blas64.General) bool {
	return checkOverlap(generalFromSymmetric(s.RawSymmetric()), a)
}

func (s *SymDense) checkOverlapMatrix(a Matrix) bool {
	if s == a {
		return false
	}
	var amat blas64.General
	switch a := a.(type) {
	default:
		return false
	case RawMatrixer:
		amat = a.RawMatrix()
	case RawSymmetricer:
		amat = generalFromSymmetric(a.RawSymmetric())
	case RawTriangular:
		amat = generalFromTriangular(a.RawTriangular())
	}
	return s.checkOverlap(amat)
}

// generalFromSymmetric returns a blas64.General with the backing
// data and dimensions of a.
func generalFromSymmetric(a blas64.Symmetric) blas64.General {
	return blas64.General{
		Rows:   a.N,
		Cols:   a.N,
		Stride: a.Stride,
		Data:   a.Data,
	}
}

func (t *TriDense) checkOverlap(a blas64.General) bool {
	return checkOverlap(generalFromTriangular(t.RawTriangular()), a)
}

func (t *TriDense) checkOverlapMatrix(a Matrix) bool {
	if t == a {
		return false
	}
	var amat blas64.General
	switch a := a.(type) {
	default:
		return false
	case RawMatrixer:
		amat = a.RawMatrix()
	case RawSymmetricer:
		amat = generalFromSymmetric(a.RawSymmetric())
	case RawTriangular:
		amat = generalFromTriangular(a.RawTriangular())
	}
	return t.checkOverlap(amat)
}

// generalFromTriangular returns a blas64.General with the backing
// data and dimensions of a.
func generalFromTriangular(a blas64.Triangular) blas64.General {
	return blas64.General{
		Rows:   a.N,
		Cols:   a.N,
		Stride: a.Stride,
		Data:   a.Data,
	}
}

func (v *VecDense) checkOverlap(a blas64.Vector) bool {
	mat := v.mat
	if cap(mat.Data) == 0 || cap(a.Data) == 0 {
		return false
	}

	off := offset(mat.Data[:1], a.Data[:1])

	if off == 0 {
		// At least one element overlaps.
		if mat.Inc == a.Inc && len(mat.Data) == len(a.Data) {
			panic(regionIdentity)
		}
		panic(regionOverlap)
	}

	if off > 0 && len(mat.Data) <= off {
		// We know v is completely before a.
		return false
	}
	if off < 0 && len(a.Data) <= -off {
		// We know v is completely after a.
		return false
	}

	if mat.Inc != a.Inc {
		// Too hard, so assume the worst.
		panic(mismatchedStrides)
	}

	if mat.Inc == 1 || off&mat.Inc == 0 {
		panic(regionOverlap)
	}
	return false
}

// rectanglesOverlap returns whether the strided rectangles a and b overlap
// when b is offset by off elements after a but has at least one element before
// the end of a. off must be positive. a and b have aCols and bCols respectively.
//
// rectanglesOverlap works by shifting both matrices left such that the left
// column of a is at 0. The column indexes are flattened by obtaining the shifted
// relative left and right column positions modulo the common stride. This allows
// direct comparison of the column offsets when the matrix backing data slices
// are known to overlap.
func rectanglesOverlap(off, aCols, bCols, stride int) bool {
	if stride == 1 {
		// Unit stride means overlapping data
		// slices must overlap as matrices.
		return true
	}

	// Flatten the shifted matrix column positions
	// so a starts at 0, modulo the common stride.
	aTo := aCols
	// The mod stride operations here make the from
	// and to indexes comparable between a and b when
	// the data slices of a and b overlap.
	bFrom := off % stride
	bTo := (bFrom + bCols) % stride

	if bTo == 0 || bFrom < bTo {
		// b matrix is not wrapped: compare for
		// simple overlap.
		return bFrom < aTo
	}

	// b strictly wraps and so must overlap with a.
	return true
}
