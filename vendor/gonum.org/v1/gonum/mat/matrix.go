// Copyright ©2013 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat

import (
	"math"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/lapack"
	"gonum.org/v1/gonum/lapack/lapack64"
)

// Matrix is the basic matrix interface type.
type Matrix interface {
	// Dims returns the dimensions of a Matrix.
	Dims() (r, c int)

	// At returns the value of a matrix element at row i, column j.
	// It will panic if i or j are out of bounds for the matrix.
	At(i, j int) float64

	// T returns the transpose of the Matrix. Whether T returns a copy of the
	// underlying data is implementation dependent.
	// This method may be implemented using the Transpose type, which
	// provides an implicit matrix transpose.
	T() Matrix
}

// allMatrix represents the extra set of methods that all mat Matrix types
// should satisfy. This is used to enforce compile-time consistency between the
// Dense types, especially helpful when adding new features.
type allMatrix interface {
	Reseter
	IsEmpty() bool
	Zero()
}

// denseMatrix represents the extra set of methods that all Dense Matrix types
// should satisfy. This is used to enforce compile-time consistency between the
// Dense types, especially helpful when adding new features.
type denseMatrix interface {
	DiagView() Diagonal
	Tracer
}

var (
	_ Matrix       = Transpose{}
	_ Untransposer = Transpose{}
)

// Transpose is a type for performing an implicit matrix transpose. It implements
// the Matrix interface, returning values from the transpose of the matrix within.
type Transpose struct {
	Matrix Matrix
}

// At returns the value of the element at row i and column j of the transposed
// matrix, that is, row j and column i of the Matrix field.
func (t Transpose) At(i, j int) float64 {
	return t.Matrix.At(j, i)
}

// Dims returns the dimensions of the transposed matrix. The number of rows returned
// is the number of columns in the Matrix field, and the number of columns is
// the number of rows in the Matrix field.
func (t Transpose) Dims() (r, c int) {
	c, r = t.Matrix.Dims()
	return r, c
}

// T performs an implicit transpose by returning the Matrix field.
func (t Transpose) T() Matrix {
	return t.Matrix
}

// Untranspose returns the Matrix field.
func (t Transpose) Untranspose() Matrix {
	return t.Matrix
}

// Untransposer is a type that can undo an implicit transpose.
type Untransposer interface {
	// Note: This interface is needed to unify all of the Transpose types. In
	// the mat methods, we need to test if the Matrix has been implicitly
	// transposed. If this is checked by testing for the specific Transpose type
	// then the behavior will be different if the user uses T() or TTri() for a
	// triangular matrix.

	// Untranspose returns the underlying Matrix stored for the implicit transpose.
	Untranspose() Matrix
}

// UntransposeBander is a type that can undo an implicit band transpose.
type UntransposeBander interface {
	// Untranspose returns the underlying Banded stored for the implicit transpose.
	UntransposeBand() Banded
}

// UntransposeTrier is a type that can undo an implicit triangular transpose.
type UntransposeTrier interface {
	// Untranspose returns the underlying Triangular stored for the implicit transpose.
	UntransposeTri() Triangular
}

// UntransposeTriBander is a type that can undo an implicit triangular banded
// transpose.
type UntransposeTriBander interface {
	// Untranspose returns the underlying Triangular stored for the implicit transpose.
	UntransposeTriBand() TriBanded
}

// Mutable is a matrix interface type that allows elements to be altered.
type Mutable interface {
	// Set alters the matrix element at row i, column j to v.
	// It will panic if i or j are out of bounds for the matrix.
	Set(i, j int, v float64)

	Matrix
}

// A RowViewer can return a Vector reflecting a row that is backed by the matrix
// data. The Vector returned will have length equal to the number of columns.
type RowViewer interface {
	RowView(i int) Vector
}

// A RawRowViewer can return a slice of float64 reflecting a row that is backed by the matrix
// data.
type RawRowViewer interface {
	RawRowView(i int) []float64
}

// A ColViewer can return a Vector reflecting a column that is backed by the matrix
// data. The Vector returned will have length equal to the number of rows.
type ColViewer interface {
	ColView(j int) Vector
}

// A RawColViewer can return a slice of float64 reflecting a column that is backed by the matrix
// data.
type RawColViewer interface {
	RawColView(j int) []float64
}

// A ClonerFrom can make a copy of a into the receiver, overwriting the previous value of the
// receiver. The clone operation does not make any restriction on shape and will not cause
// shadowing.
type ClonerFrom interface {
	CloneFrom(a Matrix)
}

// A Reseter can reset the matrix so that it can be reused as the receiver of a dimensionally
// restricted operation. This is commonly used when the matrix is being used as a workspace
// or temporary matrix.
//
// If the matrix is a view, using Reset may result in data corruption in elements outside
// the view. Similarly, if the matrix shares backing data with another variable, using
// Reset may lead to unexpected changes in data values.
type Reseter interface {
	Reset()
}

// A Copier can make a copy of elements of a into the receiver. The submatrix copied
// starts at row and column 0 and has dimensions equal to the minimum dimensions of
// the two matrices. The number of row and columns copied is returned.
// Copy will copy from a source that aliases the receiver unless the source is transposed;
// an aliasing transpose copy will panic with the exception for a special case when
// the source data has a unitary increment or stride.
type Copier interface {
	Copy(a Matrix) (r, c int)
}

// A Grower can grow the size of the represented matrix by the given number of rows and columns.
// Growing beyond the size given by the Caps method will result in the allocation of a new
// matrix and copying of the elements. If Grow is called with negative increments it will
// panic with ErrIndexOutOfRange.
type Grower interface {
	Caps() (r, c int)
	Grow(r, c int) Matrix
}

// A BandWidther represents a banded matrix and can return the left and right half-bandwidths, k1 and
// k2.
type BandWidther interface {
	BandWidth() (k1, k2 int)
}

// A RawMatrixSetter can set the underlying blas64.General used by the receiver. There is no restriction
// on the shape of the receiver. Changes to the receiver's elements will be reflected in the blas64.General.Data.
type RawMatrixSetter interface {
	SetRawMatrix(a blas64.General)
}

// A RawMatrixer can return a blas64.General representation of the receiver. Changes to the blas64.General.Data
// slice will be reflected in the original matrix, changes to the Rows, Cols and Stride fields will not.
type RawMatrixer interface {
	RawMatrix() blas64.General
}

// A RawVectorer can return a blas64.Vector representation of the receiver. Changes to the blas64.Vector.Data
// slice will be reflected in the original matrix, changes to the Inc field will not.
type RawVectorer interface {
	RawVector() blas64.Vector
}

// A NonZeroDoer can call a function for each non-zero element of the receiver.
// The parameters of the function are the element indices and its value.
type NonZeroDoer interface {
	DoNonZero(func(i, j int, v float64))
}

// A RowNonZeroDoer can call a function for each non-zero element of a row of the receiver.
// The parameters of the function are the element indices and its value.
type RowNonZeroDoer interface {
	DoRowNonZero(i int, fn func(i, j int, v float64))
}

// A ColNonZeroDoer can call a function for each non-zero element of a column of the receiver.
// The parameters of the function are the element indices and its value.
type ColNonZeroDoer interface {
	DoColNonZero(j int, fn func(i, j int, v float64))
}

// untranspose untransposes a matrix if applicable. If a is an Untransposer, then
// untranspose returns the underlying matrix and true. If it is not, then it returns
// the input matrix and false.
func untranspose(a Matrix) (Matrix, bool) {
	if ut, ok := a.(Untransposer); ok {
		return ut.Untranspose(), true
	}
	return a, false
}

// untransposeExtract returns an untransposed matrix in a built-in matrix type.
//
// The untransposed matrix is returned unaltered if it is a built-in matrix type.
// Otherwise, if it implements a Raw method, an appropriate built-in type value
// is returned holding the raw matrix value of the input. If neither of these
// is possible, the untransposed matrix is returned.
func untransposeExtract(a Matrix) (Matrix, bool) {
	ut, trans := untranspose(a)
	switch m := ut.(type) {
	case *DiagDense, *SymBandDense, *TriBandDense, *BandDense, *TriDense, *SymDense, *Dense:
		return m, trans
	// TODO(btracey): Add here if we ever have an equivalent of RawDiagDense.
	case RawSymBander:
		rsb := m.RawSymBand()
		if rsb.Uplo != blas.Upper {
			return ut, trans
		}
		var sb SymBandDense
		sb.SetRawSymBand(rsb)
		return &sb, trans
	case RawTriBander:
		rtb := m.RawTriBand()
		if rtb.Diag == blas.Unit {
			return ut, trans
		}
		var tb TriBandDense
		tb.SetRawTriBand(rtb)
		return &tb, trans
	case RawBander:
		var b BandDense
		b.SetRawBand(m.RawBand())
		return &b, trans
	case RawTriangular:
		rt := m.RawTriangular()
		if rt.Diag == blas.Unit {
			return ut, trans
		}
		var t TriDense
		t.SetRawTriangular(rt)
		return &t, trans
	case RawSymmetricer:
		rs := m.RawSymmetric()
		if rs.Uplo != blas.Upper {
			return ut, trans
		}
		var s SymDense
		s.SetRawSymmetric(rs)
		return &s, trans
	case RawMatrixer:
		var d Dense
		d.SetRawMatrix(m.RawMatrix())
		return &d, trans
	default:
		return ut, trans
	}
}

// TODO(btracey): Consider adding CopyCol/CopyRow if the behavior seems useful.
// TODO(btracey): Add in fast paths to Row/Col for the other concrete types
// (TriDense, etc.) as well as relevant interfaces (RowColer, RawRowViewer, etc.)

// Col copies the elements in the jth column of the matrix into the slice dst.
// The length of the provided slice must equal the number of rows, unless the
// slice is nil in which case a new slice is first allocated.
func Col(dst []float64, j int, a Matrix) []float64 {
	r, c := a.Dims()
	if j < 0 || j >= c {
		panic(ErrColAccess)
	}
	if dst == nil {
		dst = make([]float64, r)
	} else {
		if len(dst) != r {
			panic(ErrColLength)
		}
	}
	aU, aTrans := untranspose(a)
	if rm, ok := aU.(RawMatrixer); ok {
		m := rm.RawMatrix()
		if aTrans {
			copy(dst, m.Data[j*m.Stride:j*m.Stride+m.Cols])
			return dst
		}
		blas64.Copy(blas64.Vector{N: r, Inc: m.Stride, Data: m.Data[j:]},
			blas64.Vector{N: r, Inc: 1, Data: dst},
		)
		return dst
	}
	for i := 0; i < r; i++ {
		dst[i] = a.At(i, j)
	}
	return dst
}

// Row copies the elements in the ith row of the matrix into the slice dst.
// The length of the provided slice must equal the number of columns, unless the
// slice is nil in which case a new slice is first allocated.
func Row(dst []float64, i int, a Matrix) []float64 {
	r, c := a.Dims()
	if i < 0 || i >= r {
		panic(ErrColAccess)
	}
	if dst == nil {
		dst = make([]float64, c)
	} else {
		if len(dst) != c {
			panic(ErrRowLength)
		}
	}
	aU, aTrans := untranspose(a)
	if rm, ok := aU.(RawMatrixer); ok {
		m := rm.RawMatrix()
		if aTrans {
			blas64.Copy(blas64.Vector{N: c, Inc: m.Stride, Data: m.Data[i:]},
				blas64.Vector{N: c, Inc: 1, Data: dst},
			)
			return dst
		}
		copy(dst, m.Data[i*m.Stride:i*m.Stride+m.Cols])
		return dst
	}
	for j := 0; j < c; j++ {
		dst[j] = a.At(i, j)
	}
	return dst
}

// Cond returns the condition number of the given matrix under the given norm.
// The condition number must be based on the 1-norm, 2-norm or ∞-norm.
// Cond will panic with matrix.ErrShape if the matrix has zero size.
//
// BUG(btracey): The computation of the 1-norm and ∞-norm for non-square matrices
// is innacurate, although is typically the right order of magnitude. See
// https://github.com/xianyi/OpenBLAS/issues/636. While the value returned will
// change with the resolution of this bug, the result from Cond will match the
// condition number used internally.
func Cond(a Matrix, norm float64) float64 {
	m, n := a.Dims()
	if m == 0 || n == 0 {
		panic(ErrShape)
	}
	var lnorm lapack.MatrixNorm
	switch norm {
	default:
		panic("mat: bad norm value")
	case 1:
		lnorm = lapack.MaxColumnSum
	case 2:
		var svd SVD
		ok := svd.Factorize(a, SVDNone)
		if !ok {
			return math.Inf(1)
		}
		return svd.Cond()
	case math.Inf(1):
		lnorm = lapack.MaxRowSum
	}

	if m == n {
		// Use the LU decomposition to compute the condition number.
		var lu LU
		lu.factorize(a, lnorm)
		return lu.Cond()
	}
	if m > n {
		// Use the QR factorization to compute the condition number.
		var qr QR
		qr.factorize(a, lnorm)
		return qr.Cond()
	}
	// Use the LQ factorization to compute the condition number.
	var lq LQ
	lq.factorize(a, lnorm)
	return lq.Cond()
}

// Det returns the determinant of the matrix a. In many expressions using LogDet
// will be more numerically stable.
func Det(a Matrix) float64 {
	det, sign := LogDet(a)
	return math.Exp(det) * sign
}

// Dot returns the sum of the element-wise product of a and b.
// Dot panics if the matrix sizes are unequal.
func Dot(a, b Vector) float64 {
	la := a.Len()
	lb := b.Len()
	if la != lb {
		panic(ErrShape)
	}
	if arv, ok := a.(RawVectorer); ok {
		if brv, ok := b.(RawVectorer); ok {
			return blas64.Dot(arv.RawVector(), brv.RawVector())
		}
	}
	var sum float64
	for i := 0; i < la; i++ {
		sum += a.At(i, 0) * b.At(i, 0)
	}
	return sum
}

// Equal returns whether the matrices a and b have the same size
// and are element-wise equal.
func Equal(a, b Matrix) bool {
	ar, ac := a.Dims()
	br, bc := b.Dims()
	if ar != br || ac != bc {
		return false
	}
	aU, aTrans := untranspose(a)
	bU, bTrans := untranspose(b)
	if rma, ok := aU.(RawMatrixer); ok {
		if rmb, ok := bU.(RawMatrixer); ok {
			ra := rma.RawMatrix()
			rb := rmb.RawMatrix()
			if aTrans == bTrans {
				for i := 0; i < ra.Rows; i++ {
					for j := 0; j < ra.Cols; j++ {
						if ra.Data[i*ra.Stride+j] != rb.Data[i*rb.Stride+j] {
							return false
						}
					}
				}
				return true
			}
			for i := 0; i < ra.Rows; i++ {
				for j := 0; j < ra.Cols; j++ {
					if ra.Data[i*ra.Stride+j] != rb.Data[j*rb.Stride+i] {
						return false
					}
				}
			}
			return true
		}
	}
	if rma, ok := aU.(RawSymmetricer); ok {
		if rmb, ok := bU.(RawSymmetricer); ok {
			ra := rma.RawSymmetric()
			rb := rmb.RawSymmetric()
			// Symmetric matrices are always upper and equal to their transpose.
			for i := 0; i < ra.N; i++ {
				for j := i; j < ra.N; j++ {
					if ra.Data[i*ra.Stride+j] != rb.Data[i*rb.Stride+j] {
						return false
					}
				}
			}
			return true
		}
	}
	if ra, ok := aU.(*VecDense); ok {
		if rb, ok := bU.(*VecDense); ok {
			// If the raw vectors are the same length they must either both be
			// transposed or both not transposed (or have length 1).
			for i := 0; i < ra.mat.N; i++ {
				if ra.mat.Data[i*ra.mat.Inc] != rb.mat.Data[i*rb.mat.Inc] {
					return false
				}
			}
			return true
		}
	}
	for i := 0; i < ar; i++ {
		for j := 0; j < ac; j++ {
			if a.At(i, j) != b.At(i, j) {
				return false
			}
		}
	}
	return true
}

// EqualApprox returns whether the matrices a and b have the same size and contain all equal
// elements with tolerance for element-wise equality specified by epsilon. Matrices
// with non-equal shapes are not equal.
func EqualApprox(a, b Matrix, epsilon float64) bool {
	ar, ac := a.Dims()
	br, bc := b.Dims()
	if ar != br || ac != bc {
		return false
	}
	aU, aTrans := untranspose(a)
	bU, bTrans := untranspose(b)
	if rma, ok := aU.(RawMatrixer); ok {
		if rmb, ok := bU.(RawMatrixer); ok {
			ra := rma.RawMatrix()
			rb := rmb.RawMatrix()
			if aTrans == bTrans {
				for i := 0; i < ra.Rows; i++ {
					for j := 0; j < ra.Cols; j++ {
						if !floats.EqualWithinAbsOrRel(ra.Data[i*ra.Stride+j], rb.Data[i*rb.Stride+j], epsilon, epsilon) {
							return false
						}
					}
				}
				return true
			}
			for i := 0; i < ra.Rows; i++ {
				for j := 0; j < ra.Cols; j++ {
					if !floats.EqualWithinAbsOrRel(ra.Data[i*ra.Stride+j], rb.Data[j*rb.Stride+i], epsilon, epsilon) {
						return false
					}
				}
			}
			return true
		}
	}
	if rma, ok := aU.(RawSymmetricer); ok {
		if rmb, ok := bU.(RawSymmetricer); ok {
			ra := rma.RawSymmetric()
			rb := rmb.RawSymmetric()
			// Symmetric matrices are always upper and equal to their transpose.
			for i := 0; i < ra.N; i++ {
				for j := i; j < ra.N; j++ {
					if !floats.EqualWithinAbsOrRel(ra.Data[i*ra.Stride+j], rb.Data[i*rb.Stride+j], epsilon, epsilon) {
						return false
					}
				}
			}
			return true
		}
	}
	if ra, ok := aU.(*VecDense); ok {
		if rb, ok := bU.(*VecDense); ok {
			// If the raw vectors are the same length they must either both be
			// transposed or both not transposed (or have length 1).
			for i := 0; i < ra.mat.N; i++ {
				if !floats.EqualWithinAbsOrRel(ra.mat.Data[i*ra.mat.Inc], rb.mat.Data[i*rb.mat.Inc], epsilon, epsilon) {
					return false
				}
			}
			return true
		}
	}
	for i := 0; i < ar; i++ {
		for j := 0; j < ac; j++ {
			if !floats.EqualWithinAbsOrRel(a.At(i, j), b.At(i, j), epsilon, epsilon) {
				return false
			}
		}
	}
	return true
}

// LogDet returns the log of the determinant and the sign of the determinant
// for the matrix that has been factorized. Numerical stability in product and
// division expressions is generally improved by working in log space.
func LogDet(a Matrix) (det float64, sign float64) {
	// TODO(btracey): Add specialized routines for TriDense, etc.
	var lu LU
	lu.Factorize(a)
	return lu.LogDet()
}

// Max returns the largest element value of the matrix A.
// Max will panic with matrix.ErrShape if the matrix has zero size.
func Max(a Matrix) float64 {
	r, c := a.Dims()
	if r == 0 || c == 0 {
		panic(ErrShape)
	}
	// Max(A) = Max(Aᵀ)
	aU, _ := untranspose(a)
	switch m := aU.(type) {
	case RawMatrixer:
		rm := m.RawMatrix()
		max := math.Inf(-1)
		for i := 0; i < rm.Rows; i++ {
			for _, v := range rm.Data[i*rm.Stride : i*rm.Stride+rm.Cols] {
				if v > max {
					max = v
				}
			}
		}
		return max
	case RawTriangular:
		rm := m.RawTriangular()
		// The max of a triangular is at least 0 unless the size is 1.
		if rm.N == 1 {
			return rm.Data[0]
		}
		max := 0.0
		if rm.Uplo == blas.Upper {
			for i := 0; i < rm.N; i++ {
				for _, v := range rm.Data[i*rm.Stride+i : i*rm.Stride+rm.N] {
					if v > max {
						max = v
					}
				}
			}
			return max
		}
		for i := 0; i < rm.N; i++ {
			for _, v := range rm.Data[i*rm.Stride : i*rm.Stride+i+1] {
				if v > max {
					max = v
				}
			}
		}
		return max
	case RawSymmetricer:
		rm := m.RawSymmetric()
		if rm.Uplo != blas.Upper {
			panic(badSymTriangle)
		}
		max := math.Inf(-1)
		for i := 0; i < rm.N; i++ {
			for _, v := range rm.Data[i*rm.Stride+i : i*rm.Stride+rm.N] {
				if v > max {
					max = v
				}
			}
		}
		return max
	default:
		r, c := aU.Dims()
		max := math.Inf(-1)
		for i := 0; i < r; i++ {
			for j := 0; j < c; j++ {
				v := aU.At(i, j)
				if v > max {
					max = v
				}
			}
		}
		return max
	}
}

// Min returns the smallest element value of the matrix A.
// Min will panic with matrix.ErrShape if the matrix has zero size.
func Min(a Matrix) float64 {
	r, c := a.Dims()
	if r == 0 || c == 0 {
		panic(ErrShape)
	}
	// Min(A) = Min(Aᵀ)
	aU, _ := untranspose(a)
	switch m := aU.(type) {
	case RawMatrixer:
		rm := m.RawMatrix()
		min := math.Inf(1)
		for i := 0; i < rm.Rows; i++ {
			for _, v := range rm.Data[i*rm.Stride : i*rm.Stride+rm.Cols] {
				if v < min {
					min = v
				}
			}
		}
		return min
	case RawTriangular:
		rm := m.RawTriangular()
		// The min of a triangular is at most 0 unless the size is 1.
		if rm.N == 1 {
			return rm.Data[0]
		}
		min := 0.0
		if rm.Uplo == blas.Upper {
			for i := 0; i < rm.N; i++ {
				for _, v := range rm.Data[i*rm.Stride+i : i*rm.Stride+rm.N] {
					if v < min {
						min = v
					}
				}
			}
			return min
		}
		for i := 0; i < rm.N; i++ {
			for _, v := range rm.Data[i*rm.Stride : i*rm.Stride+i+1] {
				if v < min {
					min = v
				}
			}
		}
		return min
	case RawSymmetricer:
		rm := m.RawSymmetric()
		if rm.Uplo != blas.Upper {
			panic(badSymTriangle)
		}
		min := math.Inf(1)
		for i := 0; i < rm.N; i++ {
			for _, v := range rm.Data[i*rm.Stride+i : i*rm.Stride+rm.N] {
				if v < min {
					min = v
				}
			}
		}
		return min
	default:
		r, c := aU.Dims()
		min := math.Inf(1)
		for i := 0; i < r; i++ {
			for j := 0; j < c; j++ {
				v := aU.At(i, j)
				if v < min {
					min = v
				}
			}
		}
		return min
	}
}

// Norm returns the specified (induced) norm of the matrix a. See
// https://en.wikipedia.org/wiki/Matrix_norm for the definition of an induced norm.
//
// Valid norms are:
//    1 - The maximum absolute column sum
//    2 - Frobenius norm, the square root of the sum of the squares of the elements.
//  Inf - The maximum absolute row sum.
// Norm will panic with ErrNormOrder if an illegal norm order is specified and
// with matrix.ErrShape if the matrix has zero size.
func Norm(a Matrix, norm float64) float64 {
	r, c := a.Dims()
	if r == 0 || c == 0 {
		panic(ErrShape)
	}
	aU, aTrans := untranspose(a)
	var work []float64
	switch rma := aU.(type) {
	case RawMatrixer:
		rm := rma.RawMatrix()
		n := normLapack(norm, aTrans)
		if n == lapack.MaxColumnSum {
			work = getFloats(rm.Cols, false)
			defer putFloats(work)
		}
		return lapack64.Lange(n, rm, work)
	case RawTriangular:
		rm := rma.RawTriangular()
		n := normLapack(norm, aTrans)
		if n == lapack.MaxRowSum || n == lapack.MaxColumnSum {
			work = getFloats(rm.N, false)
			defer putFloats(work)
		}
		return lapack64.Lantr(n, rm, work)
	case RawSymmetricer:
		rm := rma.RawSymmetric()
		n := normLapack(norm, aTrans)
		if n == lapack.MaxRowSum || n == lapack.MaxColumnSum {
			work = getFloats(rm.N, false)
			defer putFloats(work)
		}
		return lapack64.Lansy(n, rm, work)
	case *VecDense:
		rv := rma.RawVector()
		switch norm {
		default:
			panic(ErrNormOrder)
		case 1:
			if aTrans {
				imax := blas64.Iamax(rv)
				return math.Abs(rma.At(imax, 0))
			}
			return blas64.Asum(rv)
		case 2:
			return blas64.Nrm2(rv)
		case math.Inf(1):
			if aTrans {
				return blas64.Asum(rv)
			}
			imax := blas64.Iamax(rv)
			return math.Abs(rma.At(imax, 0))
		}
	}
	switch norm {
	default:
		panic(ErrNormOrder)
	case 1:
		var max float64
		for j := 0; j < c; j++ {
			var sum float64
			for i := 0; i < r; i++ {
				sum += math.Abs(a.At(i, j))
			}
			if sum > max {
				max = sum
			}
		}
		return max
	case 2:
		var sum float64
		for i := 0; i < r; i++ {
			for j := 0; j < c; j++ {
				v := a.At(i, j)
				sum += v * v
			}
		}
		return math.Sqrt(sum)
	case math.Inf(1):
		var max float64
		for i := 0; i < r; i++ {
			var sum float64
			for j := 0; j < c; j++ {
				sum += math.Abs(a.At(i, j))
			}
			if sum > max {
				max = sum
			}
		}
		return max
	}
}

// normLapack converts the float64 norm input in Norm to a lapack.MatrixNorm.
func normLapack(norm float64, aTrans bool) lapack.MatrixNorm {
	switch norm {
	case 1:
		n := lapack.MaxColumnSum
		if aTrans {
			n = lapack.MaxRowSum
		}
		return n
	case 2:
		return lapack.Frobenius
	case math.Inf(1):
		n := lapack.MaxRowSum
		if aTrans {
			n = lapack.MaxColumnSum
		}
		return n
	default:
		panic(ErrNormOrder)
	}
}

// Sum returns the sum of the elements of the matrix.
func Sum(a Matrix) float64 {

	var sum float64
	aU, _ := untranspose(a)
	switch rma := aU.(type) {
	case RawSymmetricer:
		rm := rma.RawSymmetric()
		for i := 0; i < rm.N; i++ {
			// Diagonals count once while off-diagonals count twice.
			sum += rm.Data[i*rm.Stride+i]
			var s float64
			for _, v := range rm.Data[i*rm.Stride+i+1 : i*rm.Stride+rm.N] {
				s += v
			}
			sum += 2 * s
		}
		return sum
	case RawTriangular:
		rm := rma.RawTriangular()
		var startIdx, endIdx int
		for i := 0; i < rm.N; i++ {
			// Start and end index for this triangle-row.
			switch rm.Uplo {
			case blas.Upper:
				startIdx = i
				endIdx = rm.N
			case blas.Lower:
				startIdx = 0
				endIdx = i + 1
			default:
				panic(badTriangle)
			}
			for _, v := range rm.Data[i*rm.Stride+startIdx : i*rm.Stride+endIdx] {
				sum += v
			}
		}
		return sum
	case RawMatrixer:
		rm := rma.RawMatrix()
		for i := 0; i < rm.Rows; i++ {
			for _, v := range rm.Data[i*rm.Stride : i*rm.Stride+rm.Cols] {
				sum += v
			}
		}
		return sum
	case *VecDense:
		rm := rma.RawVector()
		for i := 0; i < rm.N; i++ {
			sum += rm.Data[i*rm.Inc]
		}
		return sum
	default:
		r, c := a.Dims()
		for i := 0; i < r; i++ {
			for j := 0; j < c; j++ {
				sum += a.At(i, j)
			}
		}
		return sum
	}
}

// A Tracer can compute the trace of the matrix. Trace must panic if the
// matrix is not square.
type Tracer interface {
	Trace() float64
}

// Trace returns the trace of the matrix. Trace will panic if the
// matrix is not square.
func Trace(a Matrix) float64 {
	m, _ := untransposeExtract(a)
	if t, ok := m.(Tracer); ok {
		return t.Trace()
	}
	r, c := a.Dims()
	if r != c {
		panic(ErrSquare)
	}
	var v float64
	for i := 0; i < r; i++ {
		v += a.At(i, i)
	}
	return v
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

// use returns a float64 slice with l elements, using f if it
// has the necessary capacity, otherwise creating a new slice.
func use(f []float64, l int) []float64 {
	if l <= cap(f) {
		return f[:l]
	}
	return make([]float64, l)
}

// useZeroed returns a float64 slice with l elements, using f if it
// has the necessary capacity, otherwise creating a new slice. The
// elements of the returned slice are guaranteed to be zero.
func useZeroed(f []float64, l int) []float64 {
	if l <= cap(f) {
		f = f[:l]
		zero(f)
		return f
	}
	return make([]float64, l)
}

// zero zeros the given slice's elements.
func zero(f []float64) {
	for i := range f {
		f[i] = 0
	}
}

// useInt returns an int slice with l elements, using i if it
// has the necessary capacity, otherwise creating a new slice.
func useInt(i []int, l int) []int {
	if l <= cap(i) {
		return i[:l]
	}
	return make([]int, l)
}
