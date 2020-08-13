// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat

import "gonum.org/v1/gonum/blas/blas64"

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

func (m *Dense) checkOverlap(a blas64.General) bool {
	return checkOverlap(m.RawMatrix(), a)
}

func (m *Dense) checkOverlapMatrix(a Matrix) bool {
	if m == a {
		return false
	}
	var amat blas64.General
	switch ar := a.(type) {
	default:
		return false
	case RawMatrixer:
		amat = ar.RawMatrix()
	case RawSymmetricer:
		amat = generalFromSymmetric(ar.RawSymmetric())
	case RawSymBander:
		amat = generalFromSymmetricBand(ar.RawSymBand())
	case RawTriangular:
		amat = generalFromTriangular(ar.RawTriangular())
	case RawVectorer:
		r, c := a.Dims()
		amat = generalFromVector(ar.RawVector(), r, c)
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
	switch ar := a.(type) {
	default:
		return false
	case RawMatrixer:
		amat = ar.RawMatrix()
	case RawSymmetricer:
		amat = generalFromSymmetric(ar.RawSymmetric())
	case RawSymBander:
		amat = generalFromSymmetricBand(ar.RawSymBand())
	case RawTriangular:
		amat = generalFromTriangular(ar.RawTriangular())
	case RawVectorer:
		r, c := a.Dims()
		amat = generalFromVector(ar.RawVector(), r, c)
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
	switch ar := a.(type) {
	default:
		return false
	case RawMatrixer:
		amat = ar.RawMatrix()
	case RawSymmetricer:
		amat = generalFromSymmetric(ar.RawSymmetric())
	case RawSymBander:
		amat = generalFromSymmetricBand(ar.RawSymBand())
	case RawTriangular:
		amat = generalFromTriangular(ar.RawTriangular())
	case RawVectorer:
		r, c := a.Dims()
		amat = generalFromVector(ar.RawVector(), r, c)
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

	if mat.Inc != a.Inc && mat.Inc != 1 && a.Inc != 1 {
		// Too hard, so assume the worst; if either
		// increment is one it will be caught below.
		panic(mismatchedStrides)
	}
	inc := min(mat.Inc, a.Inc)

	if inc == 1 || off&inc == 0 {
		panic(regionOverlap)
	}
	return false
}

// generalFromVector returns a blas64.General with the backing
// data and dimensions of a.
func generalFromVector(a blas64.Vector, r, c int) blas64.General {
	return blas64.General{
		Rows:   r,
		Cols:   c,
		Stride: a.Inc,
		Data:   a.Data,
	}
}

func (s *SymBandDense) checkOverlap(a blas64.General) bool {
	return checkOverlap(generalFromSymmetricBand(s.RawSymBand()), a)
}

func (s *SymBandDense) checkOverlapMatrix(a Matrix) bool {
	if s == a {
		return false
	}
	var amat blas64.General
	switch ar := a.(type) {
	default:
		return false
	case RawMatrixer:
		amat = ar.RawMatrix()
	case RawSymmetricer:
		amat = generalFromSymmetric(ar.RawSymmetric())
	case RawSymBander:
		amat = generalFromSymmetricBand(ar.RawSymBand())
	case RawTriangular:
		amat = generalFromTriangular(ar.RawTriangular())
	case RawVectorer:
		r, c := a.Dims()
		amat = generalFromVector(ar.RawVector(), r, c)
	}
	return s.checkOverlap(amat)
}

// generalFromSymmetricBand returns a blas64.General with the backing
// data and dimensions of a.
func generalFromSymmetricBand(a blas64.SymmetricBand) blas64.General {
	return blas64.General{
		Rows:   a.N,
		Cols:   a.K + 1,
		Data:   a.Data,
		Stride: a.Stride,
	}
}
