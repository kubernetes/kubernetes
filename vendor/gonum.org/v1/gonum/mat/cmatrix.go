// Copyright ©2013 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat

import (
	"math"
	"math/cmplx"

	"gonum.org/v1/gonum/blas/cblas128"
	"gonum.org/v1/gonum/floats/scalar"
)

// CMatrix is the basic matrix interface type for complex matrices.
type CMatrix interface {
	// Dims returns the dimensions of a CMatrix.
	Dims() (r, c int)

	// At returns the value of a matrix element at row i, column j.
	// It will panic if i or j are out of bounds for the matrix.
	At(i, j int) complex128

	// H returns the conjugate transpose of the CMatrix. Whether H
	// returns a copy of the underlying data is implementation dependent.
	// This method may be implemented using the ConjTranspose type, which
	// provides an implicit matrix conjugate transpose.
	H() CMatrix

	// T returns the transpose of the CMatrix. Whether T returns a copy of the
	// underlying data is implementation dependent.
	// This method may be implemented using the CTranspose type, which
	// provides an implicit matrix transpose.
	T() CMatrix
}

// A RawCMatrixer can return a cblas128.General representation of the receiver. Changes to the cblas128.General.Data
// slice will be reflected in the original matrix, changes to the Rows, Cols and Stride fields will not.
type RawCMatrixer interface {
	RawCMatrix() cblas128.General
}

var (
	_ CMatrix          = ConjTranspose{}
	_ UnConjTransposer = ConjTranspose{}
)

// ConjTranspose is a type for performing an implicit matrix conjugate transpose.
// It implements the CMatrix interface, returning values from the conjugate
// transpose of the matrix within.
type ConjTranspose struct {
	CMatrix CMatrix
}

// At returns the value of the element at row i and column j of the conjugate
// transposed matrix, that is, row j and column i of the CMatrix field.
func (t ConjTranspose) At(i, j int) complex128 {
	z := t.CMatrix.At(j, i)
	return cmplx.Conj(z)
}

// Dims returns the dimensions of the transposed matrix. The number of rows returned
// is the number of columns in the CMatrix field, and the number of columns is
// the number of rows in the CMatrix field.
func (t ConjTranspose) Dims() (r, c int) {
	c, r = t.CMatrix.Dims()
	return r, c
}

// H performs an implicit conjugate transpose by returning the CMatrix field.
func (t ConjTranspose) H() CMatrix {
	return t.CMatrix
}

// T performs an implicit transpose by returning the receiver inside a
// CTranspose.
func (t ConjTranspose) T() CMatrix {
	return CTranspose{t}
}

// UnConjTranspose returns the CMatrix field.
func (t ConjTranspose) UnConjTranspose() CMatrix {
	return t.CMatrix
}

// CTranspose is a type for performing an implicit matrix conjugate transpose.
// It implements the CMatrix interface, returning values from the conjugate
// transpose of the matrix within.
type CTranspose struct {
	CMatrix CMatrix
}

// At returns the value of the element at row i and column j of the conjugate
// transposed matrix, that is, row j and column i of the CMatrix field.
func (t CTranspose) At(i, j int) complex128 {
	return t.CMatrix.At(j, i)
}

// Dims returns the dimensions of the transposed matrix. The number of rows returned
// is the number of columns in the CMatrix field, and the number of columns is
// the number of rows in the CMatrix field.
func (t CTranspose) Dims() (r, c int) {
	c, r = t.CMatrix.Dims()
	return r, c
}

// H performs an implicit transpose by returning the receiver inside a
// ConjTranspose.
func (t CTranspose) H() CMatrix {
	return ConjTranspose{t}
}

// T performs an implicit conjugate transpose by returning the CMatrix field.
func (t CTranspose) T() CMatrix {
	return t.CMatrix
}

// Untranspose returns the CMatrix field.
func (t CTranspose) Untranspose() CMatrix {
	return t.CMatrix
}

// UnConjTransposer is a type that can undo an implicit conjugate transpose.
type UnConjTransposer interface {
	// UnConjTranspose returns the underlying CMatrix stored for the implicit
	// conjugate transpose.
	UnConjTranspose() CMatrix

	// Note: This interface is needed to unify all of the Conjugate types. In
	// the cmat128 methods, we need to test if the CMatrix has been implicitly
	// transposed. If this is checked by testing for the specific Conjugate type
	// then the behavior will be different if the user uses H() or HTri() for a
	// triangular matrix.
}

// CUntransposer is a type that can undo an implicit transpose.
type CUntransposer interface {
	// Untranspose returns the underlying CMatrix stored for the implicit
	// transpose.
	Untranspose() CMatrix

	// Note: This interface is needed to unify all of the CTranspose types. In
	// the cmat128 methods, we need to test if the CMatrix has been implicitly
	// transposed. If this is checked by testing for the specific CTranspose type
	// then the behavior will be different if the user uses T() or TTri() for a
	// triangular matrix.
}

// useC returns a complex128 slice with l elements, using c if it
// has the necessary capacity, otherwise creating a new slice.
func useC(c []complex128, l int) []complex128 {
	if l <= cap(c) {
		return c[:l]
	}
	return make([]complex128, l)
}

// useZeroedC returns a complex128 slice with l elements, using c if it
// has the necessary capacity, otherwise creating a new slice. The
// elements of the returned slice are guaranteed to be zero.
func useZeroedC(c []complex128, l int) []complex128 {
	if l <= cap(c) {
		c = c[:l]
		zeroC(c)
		return c
	}
	return make([]complex128, l)
}

// zeroC zeros the given slice's elements.
func zeroC(c []complex128) {
	for i := range c {
		c[i] = 0
	}
}

// untransposeCmplx untransposes a matrix if applicable. If a is an CUntransposer
// or an UnConjTransposer, then untranspose returns the underlying matrix and true for
// the kind of transpose (potentially both).
// If it is not, then it returns the input matrix and false for trans and conj.
func untransposeCmplx(a CMatrix) (u CMatrix, trans, conj bool) {
	switch ut := a.(type) {
	case CUntransposer:
		trans = true
		u := ut.Untranspose()
		if uc, ok := u.(UnConjTransposer); ok {
			return uc.UnConjTranspose(), trans, true
		}
		return u, trans, false
	case UnConjTransposer:
		conj = true
		u := ut.UnConjTranspose()
		if ut, ok := u.(CUntransposer); ok {
			return ut.Untranspose(), true, conj
		}
		return u, false, conj
	default:
		return a, false, false
	}
}

// untransposeExtractCmplx returns an untransposed matrix in a built-in matrix type.
//
// The untransposed matrix is returned unaltered if it is a built-in matrix type.
// Otherwise, if it implements a Raw method, an appropriate built-in type value
// is returned holding the raw matrix value of the input. If neither of these
// is possible, the untransposed matrix is returned.
func untransposeExtractCmplx(a CMatrix) (u CMatrix, trans, conj bool) {
	ut, trans, conj := untransposeCmplx(a)
	switch m := ut.(type) {
	case *CDense:
		return m, trans, conj
	case RawCMatrixer:
		var d CDense
		d.SetRawCMatrix(m.RawCMatrix())
		return &d, trans, conj
	default:
		return ut, trans, conj
	}
}

// CEqual returns whether the matrices a and b have the same size
// and are element-wise equal.
func CEqual(a, b CMatrix) bool {
	ar, ac := a.Dims()
	br, bc := b.Dims()
	if ar != br || ac != bc {
		return false
	}
	// TODO(btracey): Add in fast-paths.
	for i := 0; i < ar; i++ {
		for j := 0; j < ac; j++ {
			if a.At(i, j) != b.At(i, j) {
				return false
			}
		}
	}
	return true
}

// CEqualApprox returns whether the matrices a and b have the same size and contain all equal
// elements with tolerance for element-wise equality specified by epsilon. Matrices
// with non-equal shapes are not equal.
func CEqualApprox(a, b CMatrix, epsilon float64) bool {
	// TODO(btracey):
	ar, ac := a.Dims()
	br, bc := b.Dims()
	if ar != br || ac != bc {
		return false
	}
	for i := 0; i < ar; i++ {
		for j := 0; j < ac; j++ {
			if !cEqualWithinAbsOrRel(a.At(i, j), b.At(i, j), epsilon, epsilon) {
				return false
			}
		}
	}
	return true
}

// TODO(btracey): Move these into a cmplxs if/when we have one.

func cEqualWithinAbsOrRel(a, b complex128, absTol, relTol float64) bool {
	if cEqualWithinAbs(a, b, absTol) {
		return true
	}
	return cEqualWithinRel(a, b, relTol)
}

// cEqualWithinAbs returns true if a and b have an absolute
// difference of less than tol.
func cEqualWithinAbs(a, b complex128, tol float64) bool {
	return a == b || cmplx.Abs(a-b) <= tol
}

const minNormalFloat64 = 2.2250738585072014e-308

// cEqualWithinRel returns true if the difference between a and b
// is not greater than tol times the greater value.
func cEqualWithinRel(a, b complex128, tol float64) bool {
	if a == b {
		return true
	}
	if cmplx.IsNaN(a) || cmplx.IsNaN(b) {
		return false
	}
	// Cannot play the same trick as in floats/scalar because there are multiple
	// possible infinities.
	if cmplx.IsInf(a) {
		if !cmplx.IsInf(b) {
			return false
		}
		ra := real(a)
		if math.IsInf(ra, 0) {
			if ra == real(b) {
				return scalar.EqualWithinRel(imag(a), imag(b), tol)
			}
			return false
		}
		if imag(a) == imag(b) {
			return scalar.EqualWithinRel(ra, real(b), tol)
		}
		return false
	}
	if cmplx.IsInf(b) {
		return false
	}

	delta := cmplx.Abs(a - b)
	if delta <= minNormalFloat64 {
		return delta <= tol*minNormalFloat64
	}
	return delta/math.Max(cmplx.Abs(a), cmplx.Abs(b)) <= tol
}
