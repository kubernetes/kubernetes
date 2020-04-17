// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat

import (
	"math"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/gonum/lapack/lapack64"
)

var (
	triDense *TriDense
	_        Matrix            = triDense
	_        allMatrix         = triDense
	_        denseMatrix       = triDense
	_        Triangular        = triDense
	_        RawTriangular     = triDense
	_        MutableTriangular = triDense

	_ NonZeroDoer    = triDense
	_ RowNonZeroDoer = triDense
	_ ColNonZeroDoer = triDense
)

const badTriCap = "mat: bad capacity for TriDense"

// TriDense represents an upper or lower triangular matrix in dense storage
// format.
type TriDense struct {
	mat blas64.Triangular
	cap int
}

// Triangular represents a triangular matrix. Triangular matrices are always square.
type Triangular interface {
	Matrix
	// Triangle returns the number of rows/columns in the matrix and its
	// orientation.
	Triangle() (n int, kind TriKind)

	// TTri is the equivalent of the T() method in the Matrix interface but
	// guarantees the transpose is of triangular type.
	TTri() Triangular
}

// A RawTriangular can return a blas64.Triangular representation of the receiver.
// Changes to the blas64.Triangular.Data slice will be reflected in the original
// matrix, changes to the N, Stride, Uplo and Diag fields will not.
type RawTriangular interface {
	RawTriangular() blas64.Triangular
}

// A MutableTriangular can set elements of a triangular matrix.
type MutableTriangular interface {
	Triangular
	SetTri(i, j int, v float64)
}

var (
	_ Matrix           = TransposeTri{}
	_ Triangular       = TransposeTri{}
	_ UntransposeTrier = TransposeTri{}
)

// TransposeTri is a type for performing an implicit transpose of a Triangular
// matrix. It implements the Triangular interface, returning values from the
// transpose of the matrix within.
type TransposeTri struct {
	Triangular Triangular
}

// At returns the value of the element at row i and column j of the transposed
// matrix, that is, row j and column i of the Triangular field.
func (t TransposeTri) At(i, j int) float64 {
	return t.Triangular.At(j, i)
}

// Dims returns the dimensions of the transposed matrix. Triangular matrices are
// square and thus this is the same size as the original Triangular.
func (t TransposeTri) Dims() (r, c int) {
	c, r = t.Triangular.Dims()
	return r, c
}

// T performs an implicit transpose by returning the Triangular field.
func (t TransposeTri) T() Matrix {
	return t.Triangular
}

// Triangle returns the number of rows/columns in the matrix and its orientation.
func (t TransposeTri) Triangle() (int, TriKind) {
	n, upper := t.Triangular.Triangle()
	return n, !upper
}

// TTri performs an implicit transpose by returning the Triangular field.
func (t TransposeTri) TTri() Triangular {
	return t.Triangular
}

// Untranspose returns the Triangular field.
func (t TransposeTri) Untranspose() Matrix {
	return t.Triangular
}

func (t TransposeTri) UntransposeTri() Triangular {
	return t.Triangular
}

// NewTriDense creates a new Triangular matrix with n rows and columns. If data == nil,
// a new slice is allocated for the backing slice. If len(data) == n*n, data is
// used as the backing slice, and changes to the elements of the returned TriDense
// will be reflected in data. If neither of these is true, NewTriDense will panic.
// NewTriDense will panic if n is zero.
//
// The data must be arranged in row-major order, i.e. the (i*c + j)-th
// element in the data slice is the {i, j}-th element in the matrix.
// Only the values in the triangular portion corresponding to kind are used.
func NewTriDense(n int, kind TriKind, data []float64) *TriDense {
	if n <= 0 {
		if n == 0 {
			panic(ErrZeroLength)
		}
		panic("mat: negative dimension")
	}
	if data != nil && len(data) != n*n {
		panic(ErrShape)
	}
	if data == nil {
		data = make([]float64, n*n)
	}
	uplo := blas.Lower
	if kind == Upper {
		uplo = blas.Upper
	}
	return &TriDense{
		mat: blas64.Triangular{
			N:      n,
			Stride: n,
			Data:   data,
			Uplo:   uplo,
			Diag:   blas.NonUnit,
		},
		cap: n,
	}
}

func (t *TriDense) Dims() (r, c int) {
	return t.mat.N, t.mat.N
}

// Triangle returns the dimension of t and its orientation. The returned
// orientation is only valid when n is not empty.
func (t *TriDense) Triangle() (n int, kind TriKind) {
	return t.mat.N, t.triKind()
}

func (t *TriDense) isUpper() bool {
	return isUpperUplo(t.mat.Uplo)
}

func (t *TriDense) triKind() TriKind {
	return TriKind(isUpperUplo(t.mat.Uplo))
}

func isUpperUplo(u blas.Uplo) bool {
	switch u {
	case blas.Upper:
		return true
	case blas.Lower:
		return false
	default:
		panic(badTriangle)
	}
}

func uploToTriKind(u blas.Uplo) TriKind {
	switch u {
	case blas.Upper:
		return Upper
	case blas.Lower:
		return Lower
	default:
		panic(badTriangle)
	}
}

// asSymBlas returns the receiver restructured as a blas64.Symmetric with the
// same backing memory. Panics if the receiver is unit.
// This returns a blas64.Symmetric and not a *SymDense because SymDense can only
// be upper triangular.
func (t *TriDense) asSymBlas() blas64.Symmetric {
	if t.mat.Diag == blas.Unit {
		panic("mat: cannot convert unit TriDense into blas64.Symmetric")
	}
	return blas64.Symmetric{
		N:      t.mat.N,
		Stride: t.mat.Stride,
		Data:   t.mat.Data,
		Uplo:   t.mat.Uplo,
	}
}

// T performs an implicit transpose by returning the receiver inside a Transpose.
func (t *TriDense) T() Matrix {
	return Transpose{t}
}

// TTri performs an implicit transpose by returning the receiver inside a TransposeTri.
func (t *TriDense) TTri() Triangular {
	return TransposeTri{t}
}

func (t *TriDense) RawTriangular() blas64.Triangular {
	return t.mat
}

// SetRawTriangular sets the underlying blas64.Triangular used by the receiver.
// Changes to elements in the receiver following the call will be reflected
// in the input.
//
// The supplied Triangular must not use blas.Unit storage format.
func (t *TriDense) SetRawTriangular(mat blas64.Triangular) {
	if mat.Diag == blas.Unit {
		panic("mat: cannot set TriDense with Unit storage format")
	}
	t.cap = mat.N
	t.mat = mat
}

// Reset empties the matrix so that it can be reused as the
// receiver of a dimensionally restricted operation.
//
// Reset should not be used when the matrix shares backing data.
// See the Reseter interface for more information.
func (t *TriDense) Reset() {
	// N and Stride must be zeroed in unison.
	t.mat.N, t.mat.Stride = 0, 0
	// Defensively zero Uplo to ensure
	// it is set correctly later.
	t.mat.Uplo = 0
	t.mat.Data = t.mat.Data[:0]
}

// Zero sets all of the matrix elements to zero.
func (t *TriDense) Zero() {
	if t.isUpper() {
		for i := 0; i < t.mat.N; i++ {
			zero(t.mat.Data[i*t.mat.Stride+i : i*t.mat.Stride+t.mat.N])
		}
		return
	}
	for i := 0; i < t.mat.N; i++ {
		zero(t.mat.Data[i*t.mat.Stride : i*t.mat.Stride+i+1])
	}
}

// IsEmpty returns whether the receiver is empty. Empty matrices can be the
// receiver for size-restricted operations. The receiver can be emptied using
// Reset.
func (t *TriDense) IsEmpty() bool {
	// It must be the case that t.Dims() returns
	// zeros in this case. See comment in Reset().
	return t.mat.Stride == 0
}

// untranspose untransposes a matrix if applicable. If a is an Untransposer, then
// untranspose returns the underlying matrix and true. If it is not, then it returns
// the input matrix and false.
func untransposeTri(a Triangular) (Triangular, bool) {
	if ut, ok := a.(UntransposeTrier); ok {
		return ut.UntransposeTri(), true
	}
	return a, false
}

// ReuseAsTri changes the receiver if it IsEmpty() to be of size n×n.
//
// ReuseAsTri re-uses the backing data slice if it has sufficient capacity,
// otherwise a new slice is allocated. The backing data is zero on return.
//
// ReuseAsTri panics if the receiver is not empty, and panics if
// the input size is less than one. To empty the receiver for re-use,
// Reset should be used.
func (t *TriDense) ReuseAsTri(n int, kind TriKind) {
	if n <= 0 {
		if n == 0 {
			panic(ErrZeroLength)
		}
		panic(ErrNegativeDimension)
	}
	if !t.IsEmpty() {
		panic(ErrReuseNonEmpty)
	}
	t.reuseAsZeroed(n, kind)
}

// reuseAsNonZeroed resizes a zero receiver to an n×n triangular matrix with the given
// orientation. If the receiver is non-zero, reuseAsNonZeroed checks that the receiver
// is the correct size and orientation.
func (t *TriDense) reuseAsNonZeroed(n int, kind TriKind) {
	// reuseAsNonZeroed must be kept in sync with reuseAsZeroed.
	if n == 0 {
		panic(ErrZeroLength)
	}
	ul := blas.Lower
	if kind == Upper {
		ul = blas.Upper
	}
	if t.mat.N > t.cap {
		panic(badTriCap)
	}
	if t.IsEmpty() {
		t.mat = blas64.Triangular{
			N:      n,
			Stride: n,
			Diag:   blas.NonUnit,
			Data:   use(t.mat.Data, n*n),
			Uplo:   ul,
		}
		t.cap = n
		return
	}
	if t.mat.N != n {
		panic(ErrShape)
	}
	if t.mat.Uplo != ul {
		panic(ErrTriangle)
	}
}

// reuseAsZeroed resizes a zero receiver to an n×n triangular matrix with the given
// orientation. If the receiver is non-zero, reuseAsZeroed checks that the receiver
// is the correct size and orientation. It then zeros out the matrix data.
func (t *TriDense) reuseAsZeroed(n int, kind TriKind) {
	// reuseAsZeroed must be kept in sync with reuseAsNonZeroed.
	if n == 0 {
		panic(ErrZeroLength)
	}
	ul := blas.Lower
	if kind == Upper {
		ul = blas.Upper
	}
	if t.mat.N > t.cap {
		panic(badTriCap)
	}
	if t.IsEmpty() {
		t.mat = blas64.Triangular{
			N:      n,
			Stride: n,
			Diag:   blas.NonUnit,
			Data:   useZeroed(t.mat.Data, n*n),
			Uplo:   ul,
		}
		t.cap = n
		return
	}
	if t.mat.N != n {
		panic(ErrShape)
	}
	if t.mat.Uplo != ul {
		panic(ErrTriangle)
	}
	t.Zero()
}

// isolatedWorkspace returns a new TriDense matrix w with the size of a and
// returns a callback to defer which performs cleanup at the return of the call.
// This should be used when a method receiver is the same pointer as an input argument.
func (t *TriDense) isolatedWorkspace(a Triangular) (w *TriDense, restore func()) {
	n, kind := a.Triangle()
	if n == 0 {
		panic(ErrZeroLength)
	}
	w = getWorkspaceTri(n, kind, false)
	return w, func() {
		t.Copy(w)
		putWorkspaceTri(w)
	}
}

// DiagView returns the diagonal as a matrix backed by the original data.
func (t *TriDense) DiagView() Diagonal {
	if t.mat.Diag == blas.Unit {
		panic("mat: cannot take view of Unit diagonal")
	}
	n := t.mat.N
	return &DiagDense{
		mat: blas64.Vector{
			N:    n,
			Inc:  t.mat.Stride + 1,
			Data: t.mat.Data[:(n-1)*t.mat.Stride+n],
		},
	}
}

// Copy makes a copy of elements of a into the receiver. It is similar to the
// built-in copy; it copies as much as the overlap between the two matrices and
// returns the number of rows and columns it copied. Only elements within the
// receiver's non-zero triangle are set.
//
// See the Copier interface for more information.
func (t *TriDense) Copy(a Matrix) (r, c int) {
	r, c = a.Dims()
	r = min(r, t.mat.N)
	c = min(c, t.mat.N)
	if r == 0 || c == 0 {
		return 0, 0
	}

	switch a := a.(type) {
	case RawMatrixer:
		amat := a.RawMatrix()
		if t.isUpper() {
			for i := 0; i < r; i++ {
				copy(t.mat.Data[i*t.mat.Stride+i:i*t.mat.Stride+c], amat.Data[i*amat.Stride+i:i*amat.Stride+c])
			}
		} else {
			for i := 0; i < r; i++ {
				copy(t.mat.Data[i*t.mat.Stride:i*t.mat.Stride+i+1], amat.Data[i*amat.Stride:i*amat.Stride+i+1])
			}
		}
	case RawTriangular:
		amat := a.RawTriangular()
		aIsUpper := isUpperUplo(amat.Uplo)
		tIsUpper := t.isUpper()
		switch {
		case tIsUpper && aIsUpper:
			for i := 0; i < r; i++ {
				copy(t.mat.Data[i*t.mat.Stride+i:i*t.mat.Stride+c], amat.Data[i*amat.Stride+i:i*amat.Stride+c])
			}
		case !tIsUpper && !aIsUpper:
			for i := 0; i < r; i++ {
				copy(t.mat.Data[i*t.mat.Stride:i*t.mat.Stride+i+1], amat.Data[i*amat.Stride:i*amat.Stride+i+1])
			}
		default:
			for i := 0; i < r; i++ {
				t.set(i, i, amat.Data[i*amat.Stride+i])
			}
		}
	default:
		isUpper := t.isUpper()
		for i := 0; i < r; i++ {
			if isUpper {
				for j := i; j < c; j++ {
					t.set(i, j, a.At(i, j))
				}
			} else {
				for j := 0; j <= i; j++ {
					t.set(i, j, a.At(i, j))
				}
			}
		}
	}

	return r, c
}

// InverseTri computes the inverse of the triangular matrix a, storing the result
// into the receiver. If a is ill-conditioned, a Condition error will be returned.
// Note that matrix inversion is numerically unstable, and should generally be
// avoided where possible, for example by using the Solve routines.
func (t *TriDense) InverseTri(a Triangular) error {
	t.checkOverlapMatrix(a)
	n, _ := a.Triangle()
	t.reuseAsNonZeroed(a.Triangle())
	t.Copy(a)
	work := getFloats(3*n, false)
	iwork := getInts(n, false)
	cond := lapack64.Trcon(CondNorm, t.mat, work, iwork)
	putFloats(work)
	putInts(iwork)
	if math.IsInf(cond, 1) {
		return Condition(cond)
	}
	ok := lapack64.Trtri(t.mat)
	if !ok {
		return Condition(math.Inf(1))
	}
	if cond > ConditionTolerance {
		return Condition(cond)
	}
	return nil
}

// MulTri takes the product of triangular matrices a and b and places the result
// in the receiver. The size of a and b must match, and they both must have the
// same TriKind, or Mul will panic.
func (t *TriDense) MulTri(a, b Triangular) {
	n, kind := a.Triangle()
	nb, kindb := b.Triangle()
	if n != nb {
		panic(ErrShape)
	}
	if kind != kindb {
		panic(ErrTriangle)
	}

	aU, _ := untransposeTri(a)
	bU, _ := untransposeTri(b)
	t.checkOverlapMatrix(bU)
	t.checkOverlapMatrix(aU)
	t.reuseAsNonZeroed(n, kind)
	var restore func()
	if t == aU {
		t, restore = t.isolatedWorkspace(aU)
		defer restore()
	} else if t == bU {
		t, restore = t.isolatedWorkspace(bU)
		defer restore()
	}

	// Inspect types here, helps keep the loops later clean(er).
	_, aDiag := aU.(Diagonal)
	_, bDiag := bU.(Diagonal)
	// If they are both diagonal only need 1 loop.
	// All diagonal matrices are Upper.
	// TODO: Add fast paths for DiagDense.
	if aDiag && bDiag {
		t.Zero()
		for i := 0; i < n; i++ {
			t.SetTri(i, i, a.At(i, i)*b.At(i, i))
		}
		return
	}

	// Now we know at least one matrix is non-diagonal.
	// And all diagonal matrices are all Upper.
	// The both-diagonal case is handled above.
	// TODO: Add fast paths for Dense variants.
	if kind == Upper {
		for i := 0; i < n; i++ {
			for j := i; j < n; j++ {
				switch {
				case aDiag:
					t.SetTri(i, j, a.At(i, i)*b.At(i, j))
				case bDiag:
					t.SetTri(i, j, a.At(i, j)*b.At(j, j))
				default:
					var v float64
					for k := i; k <= j; k++ {
						v += a.At(i, k) * b.At(k, j)
					}
					t.SetTri(i, j, v)
				}
			}
		}
		return
	}
	for i := 0; i < n; i++ {
		for j := 0; j <= i; j++ {
			var v float64
			for k := j; k <= i; k++ {
				v += a.At(i, k) * b.At(k, j)
			}
			t.SetTri(i, j, v)
		}
	}
}

// ScaleTri multiplies the elements of a by f, placing the result in the receiver.
// If the receiver is non-zero, the size and kind of the receiver must match
// the input, or ScaleTri will panic.
func (t *TriDense) ScaleTri(f float64, a Triangular) {
	n, kind := a.Triangle()
	t.reuseAsNonZeroed(n, kind)

	// TODO(btracey): Improve the set of fast-paths.
	switch a := a.(type) {
	case RawTriangular:
		amat := a.RawTriangular()
		if t != a {
			t.checkOverlap(generalFromTriangular(amat))
		}
		if kind == Upper {
			for i := 0; i < n; i++ {
				ts := t.mat.Data[i*t.mat.Stride+i : i*t.mat.Stride+n]
				as := amat.Data[i*amat.Stride+i : i*amat.Stride+n]
				for i, v := range as {
					ts[i] = v * f
				}
			}
			return
		}
		for i := 0; i < n; i++ {
			ts := t.mat.Data[i*t.mat.Stride : i*t.mat.Stride+i+1]
			as := amat.Data[i*amat.Stride : i*amat.Stride+i+1]
			for i, v := range as {
				ts[i] = v * f
			}
		}
		return
	default:
		t.checkOverlapMatrix(a)
		isUpper := kind == Upper
		for i := 0; i < n; i++ {
			if isUpper {
				for j := i; j < n; j++ {
					t.set(i, j, f*a.At(i, j))
				}
			} else {
				for j := 0; j <= i; j++ {
					t.set(i, j, f*a.At(i, j))
				}
			}
		}
	}
}

// Trace returns the trace of the matrix.
func (t *TriDense) Trace() float64 {
	// TODO(btracey): could use internal asm sum routine.
	var v float64
	for i := 0; i < t.mat.N; i++ {
		v += t.mat.Data[i*t.mat.Stride+i]
	}
	return v
}

// copySymIntoTriangle copies a symmetric matrix into a TriDense
func copySymIntoTriangle(t *TriDense, s Symmetric) {
	n, upper := t.Triangle()
	ns := s.Symmetric()
	if n != ns {
		panic("mat: triangle size mismatch")
	}
	ts := t.mat.Stride
	if rs, ok := s.(RawSymmetricer); ok {
		sd := rs.RawSymmetric()
		ss := sd.Stride
		if upper {
			if sd.Uplo == blas.Upper {
				for i := 0; i < n; i++ {
					copy(t.mat.Data[i*ts+i:i*ts+n], sd.Data[i*ss+i:i*ss+n])
				}
				return
			}
			for i := 0; i < n; i++ {
				for j := i; j < n; j++ {
					t.mat.Data[i*ts+j] = sd.Data[j*ss+i]
				}
			}
			return
		}
		if sd.Uplo == blas.Upper {
			for i := 0; i < n; i++ {
				for j := 0; j <= i; j++ {
					t.mat.Data[i*ts+j] = sd.Data[j*ss+i]
				}
			}
			return
		}
		for i := 0; i < n; i++ {
			copy(t.mat.Data[i*ts:i*ts+i+1], sd.Data[i*ss:i*ss+i+1])
		}
		return
	}
	if upper {
		for i := 0; i < n; i++ {
			for j := i; j < n; j++ {
				t.mat.Data[i*ts+j] = s.At(i, j)
			}
		}
		return
	}
	for i := 0; i < n; i++ {
		for j := 0; j <= i; j++ {
			t.mat.Data[i*ts+j] = s.At(i, j)
		}
	}
}

// DoNonZero calls the function fn for each of the non-zero elements of t. The function fn
// takes a row/column index and the element value of t at (i, j).
func (t *TriDense) DoNonZero(fn func(i, j int, v float64)) {
	if t.isUpper() {
		for i := 0; i < t.mat.N; i++ {
			for j := i; j < t.mat.N; j++ {
				v := t.at(i, j)
				if v != 0 {
					fn(i, j, v)
				}
			}
		}
		return
	}
	for i := 0; i < t.mat.N; i++ {
		for j := 0; j <= i; j++ {
			v := t.at(i, j)
			if v != 0 {
				fn(i, j, v)
			}
		}
	}
}

// DoRowNonZero calls the function fn for each of the non-zero elements of row i of t. The function fn
// takes a row/column index and the element value of t at (i, j).
func (t *TriDense) DoRowNonZero(i int, fn func(i, j int, v float64)) {
	if i < 0 || t.mat.N <= i {
		panic(ErrRowAccess)
	}
	if t.isUpper() {
		for j := i; j < t.mat.N; j++ {
			v := t.at(i, j)
			if v != 0 {
				fn(i, j, v)
			}
		}
		return
	}
	for j := 0; j <= i; j++ {
		v := t.at(i, j)
		if v != 0 {
			fn(i, j, v)
		}
	}
}

// DoColNonZero calls the function fn for each of the non-zero elements of column j of t. The function fn
// takes a row/column index and the element value of t at (i, j).
func (t *TriDense) DoColNonZero(j int, fn func(i, j int, v float64)) {
	if j < 0 || t.mat.N <= j {
		panic(ErrColAccess)
	}
	if t.isUpper() {
		for i := 0; i <= j; i++ {
			v := t.at(i, j)
			if v != 0 {
				fn(i, j, v)
			}
		}
		return
	}
	for i := j; i < t.mat.N; i++ {
		v := t.at(i, j)
		if v != 0 {
			fn(i, j, v)
		}
	}
}
