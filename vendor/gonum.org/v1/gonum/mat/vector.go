// Copyright ©2013 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat

import (
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/gonum/internal/asm/f64"
)

var (
	vector *VecDense

	_ Matrix        = vector
	_ allMatrix     = vector
	_ Vector        = vector
	_ Reseter       = vector
	_ MutableVector = vector
)

// Vector is a vector.
type Vector interface {
	Matrix
	AtVec(int) float64
	Len() int
}

// A MutableVector can set elements of a vector.
type MutableVector interface {
	Vector
	SetVec(i int, v float64)
}

// TransposeVec is a type for performing an implicit transpose of a Vector.
// It implements the Vector interface, returning values from the transpose
// of the vector within.
type TransposeVec struct {
	Vector Vector
}

// At returns the value of the element at row i and column j of the transposed
// matrix, that is, row j and column i of the Vector field.
func (t TransposeVec) At(i, j int) float64 {
	return t.Vector.At(j, i)
}

// AtVec returns the element at position i. It panics if i is out of bounds.
func (t TransposeVec) AtVec(i int) float64 {
	return t.Vector.AtVec(i)
}

// Dims returns the dimensions of the transposed vector.
func (t TransposeVec) Dims() (r, c int) {
	c, r = t.Vector.Dims()
	return r, c
}

// T performs an implicit transpose by returning the Vector field.
func (t TransposeVec) T() Matrix {
	return t.Vector
}

// Len returns the number of columns in the vector.
func (t TransposeVec) Len() int {
	return t.Vector.Len()
}

// TVec performs an implicit transpose by returning the Vector field.
func (t TransposeVec) TVec() Vector {
	return t.Vector
}

// Untranspose returns the Vector field.
func (t TransposeVec) Untranspose() Matrix {
	return t.Vector
}

func (t TransposeVec) UntransposeVec() Vector {
	return t.Vector
}

// VecDense represents a column vector.
type VecDense struct {
	mat blas64.Vector
	// A BLAS vector can have a negative increment, but allowing this
	// in the mat type complicates a lot of code, and doesn't gain anything.
	// VecDense must have positive increment in this package.
}

// NewVecDense creates a new VecDense of length n. If data == nil,
// a new slice is allocated for the backing slice. If len(data) == n, data is
// used as the backing slice, and changes to the elements of the returned VecDense
// will be reflected in data. If neither of these is true, NewVecDense will panic.
// NewVecDense will panic if n is zero.
func NewVecDense(n int, data []float64) *VecDense {
	if n <= 0 {
		if n == 0 {
			panic(ErrZeroLength)
		}
		panic("mat: negative dimension")
	}
	if len(data) != n && data != nil {
		panic(ErrShape)
	}
	if data == nil {
		data = make([]float64, n)
	}
	return &VecDense{
		mat: blas64.Vector{
			N:    n,
			Inc:  1,
			Data: data,
		},
	}
}

// SliceVec returns a new Vector that shares backing data with the receiver.
// The returned matrix starts at i of the receiver and extends k-i elements.
// SliceVec panics with ErrIndexOutOfRange if the slice is outside the capacity
// of the receiver.
func (v *VecDense) SliceVec(i, k int) Vector {
	return v.sliceVec(i, k)
}

func (v *VecDense) sliceVec(i, k int) *VecDense {
	if i < 0 || k <= i || v.Cap() < k {
		panic(ErrIndexOutOfRange)
	}
	return &VecDense{
		mat: blas64.Vector{
			N:    k - i,
			Inc:  v.mat.Inc,
			Data: v.mat.Data[i*v.mat.Inc : (k-1)*v.mat.Inc+1],
		},
	}
}

// Dims returns the number of rows and columns in the matrix. Columns is always 1
// for a non-Reset vector.
func (v *VecDense) Dims() (r, c int) {
	if v.IsEmpty() {
		return 0, 0
	}
	return v.mat.N, 1
}

// Caps returns the number of rows and columns in the backing matrix. Columns is always 1
// for a non-Reset vector.
func (v *VecDense) Caps() (r, c int) {
	if v.IsEmpty() {
		return 0, 0
	}
	return v.Cap(), 1
}

// Len returns the length of the vector.
func (v *VecDense) Len() int {
	return v.mat.N
}

// Cap returns the capacity of the vector.
func (v *VecDense) Cap() int {
	if v.IsEmpty() {
		return 0
	}
	return (cap(v.mat.Data)-1)/v.mat.Inc + 1
}

// T performs an implicit transpose by returning the receiver inside a Transpose.
func (v *VecDense) T() Matrix {
	return Transpose{v}
}

// TVec performs an implicit transpose by returning the receiver inside a TransposeVec.
func (v *VecDense) TVec() Vector {
	return TransposeVec{v}
}

// Reset empties the matrix so that it can be reused as the
// receiver of a dimensionally restricted operation.
//
// Reset should not be used when the matrix shares backing data.
// See the Reseter interface for more information.
func (v *VecDense) Reset() {
	// No change of Inc or N to 0 may be
	// made unless both are set to 0.
	v.mat.Inc = 0
	v.mat.N = 0
	v.mat.Data = v.mat.Data[:0]
}

// Zero sets all of the matrix elements to zero.
func (v *VecDense) Zero() {
	for i := 0; i < v.mat.N; i++ {
		v.mat.Data[v.mat.Inc*i] = 0
	}
}

// CloneFromVec makes a copy of a into the receiver, overwriting the previous value
// of the receiver.
func (v *VecDense) CloneFromVec(a Vector) {
	if v == a {
		return
	}
	n := a.Len()
	v.mat = blas64.Vector{
		N:    n,
		Inc:  1,
		Data: use(v.mat.Data, n),
	}
	if r, ok := a.(RawVectorer); ok {
		blas64.Copy(r.RawVector(), v.mat)
		return
	}
	for i := 0; i < a.Len(); i++ {
		v.setVec(i, a.AtVec(i))
	}
}

// VecDenseCopyOf returns a newly allocated copy of the elements of a.
func VecDenseCopyOf(a Vector) *VecDense {
	v := &VecDense{}
	v.CloneFromVec(a)
	return v
}

// RawVector returns the underlying blas64.Vector used by the receiver.
// Changes to elements in the receiver following the call will be reflected
// in returned blas64.Vector.
func (v *VecDense) RawVector() blas64.Vector {
	return v.mat
}

// SetRawVector sets the underlying blas64.Vector used by the receiver.
// Changes to elements in the receiver following the call will be reflected
// in the input.
func (v *VecDense) SetRawVector(a blas64.Vector) {
	v.mat = a
}

// CopyVec makes a copy of elements of a into the receiver. It is similar to the
// built-in copy; it copies as much as the overlap between the two vectors and
// returns the number of elements it copied.
func (v *VecDense) CopyVec(a Vector) int {
	n := min(v.Len(), a.Len())
	if v == a {
		return n
	}
	if r, ok := a.(RawVectorer); ok {
		src := r.RawVector()
		src.N = n
		dst := v.mat
		dst.N = n
		blas64.Copy(src, dst)
		return n
	}
	for i := 0; i < n; i++ {
		v.setVec(i, a.AtVec(i))
	}
	return n
}

// ScaleVec scales the vector a by alpha, placing the result in the receiver.
func (v *VecDense) ScaleVec(alpha float64, a Vector) {
	n := a.Len()

	if v == a {
		if v.mat.Inc == 1 {
			f64.ScalUnitary(alpha, v.mat.Data)
			return
		}
		f64.ScalInc(alpha, v.mat.Data, uintptr(n), uintptr(v.mat.Inc))
		return
	}

	v.reuseAsNonZeroed(n)

	if rv, ok := a.(RawVectorer); ok {
		mat := rv.RawVector()
		v.checkOverlap(mat)
		if v.mat.Inc == 1 && mat.Inc == 1 {
			f64.ScalUnitaryTo(v.mat.Data, alpha, mat.Data)
			return
		}
		f64.ScalIncTo(v.mat.Data, uintptr(v.mat.Inc),
			alpha, mat.Data, uintptr(n), uintptr(mat.Inc))
		return
	}

	for i := 0; i < n; i++ {
		v.setVec(i, alpha*a.AtVec(i))
	}
}

// AddScaledVec adds the vectors a and alpha*b, placing the result in the receiver.
func (v *VecDense) AddScaledVec(a Vector, alpha float64, b Vector) {
	if alpha == 1 {
		v.AddVec(a, b)
		return
	}
	if alpha == -1 {
		v.SubVec(a, b)
		return
	}

	ar := a.Len()
	br := b.Len()

	if ar != br {
		panic(ErrShape)
	}

	var amat, bmat blas64.Vector
	fast := true
	aU, _ := untransposeExtract(a)
	if rv, ok := aU.(*VecDense); ok {
		amat = rv.mat
		if v != a {
			v.checkOverlap(amat)
		}
	} else {
		fast = false
	}
	bU, _ := untransposeExtract(b)
	if rv, ok := bU.(*VecDense); ok {
		bmat = rv.mat
		if v != b {
			v.checkOverlap(bmat)
		}
	} else {
		fast = false
	}

	v.reuseAsNonZeroed(ar)

	switch {
	case alpha == 0: // v <- a
		if v == a {
			return
		}
		v.CopyVec(a)
	case v == a && v == b: // v <- v + alpha * v = (alpha + 1) * v
		blas64.Scal(alpha+1, v.mat)
	case !fast: // v <- a + alpha * b without blas64 support.
		for i := 0; i < ar; i++ {
			v.setVec(i, a.AtVec(i)+alpha*b.AtVec(i))
		}
	case v == a && v != b: // v <- v + alpha * b
		if v.mat.Inc == 1 && bmat.Inc == 1 {
			// Fast path for a common case.
			f64.AxpyUnitaryTo(v.mat.Data, alpha, bmat.Data, amat.Data)
		} else {
			f64.AxpyInc(alpha, bmat.Data, v.mat.Data,
				uintptr(ar), uintptr(bmat.Inc), uintptr(v.mat.Inc), 0, 0)
		}
	default: // v <- a + alpha * b or v <- a + alpha * v
		if v.mat.Inc == 1 && amat.Inc == 1 && bmat.Inc == 1 {
			// Fast path for a common case.
			f64.AxpyUnitaryTo(v.mat.Data, alpha, bmat.Data, amat.Data)
		} else {
			f64.AxpyIncTo(v.mat.Data, uintptr(v.mat.Inc), 0,
				alpha, bmat.Data, amat.Data,
				uintptr(ar), uintptr(bmat.Inc), uintptr(amat.Inc), 0, 0)
		}
	}
}

// AddVec adds the vectors a and b, placing the result in the receiver.
func (v *VecDense) AddVec(a, b Vector) {
	ar := a.Len()
	br := b.Len()

	if ar != br {
		panic(ErrShape)
	}

	v.reuseAsNonZeroed(ar)

	aU, _ := untransposeExtract(a)
	bU, _ := untransposeExtract(b)

	if arv, ok := aU.(*VecDense); ok {
		if brv, ok := bU.(*VecDense); ok {
			amat := arv.mat
			bmat := brv.mat

			if v != a {
				v.checkOverlap(amat)
			}
			if v != b {
				v.checkOverlap(bmat)
			}

			if v.mat.Inc == 1 && amat.Inc == 1 && bmat.Inc == 1 {
				// Fast path for a common case.
				f64.AxpyUnitaryTo(v.mat.Data, 1, bmat.Data, amat.Data)
				return
			}
			f64.AxpyIncTo(v.mat.Data, uintptr(v.mat.Inc), 0,
				1, bmat.Data, amat.Data,
				uintptr(ar), uintptr(bmat.Inc), uintptr(amat.Inc), 0, 0)
			return
		}
	}

	for i := 0; i < ar; i++ {
		v.setVec(i, a.AtVec(i)+b.AtVec(i))
	}
}

// SubVec subtracts the vector b from a, placing the result in the receiver.
func (v *VecDense) SubVec(a, b Vector) {
	ar := a.Len()
	br := b.Len()

	if ar != br {
		panic(ErrShape)
	}

	v.reuseAsNonZeroed(ar)

	aU, _ := untransposeExtract(a)
	bU, _ := untransposeExtract(b)

	if arv, ok := aU.(*VecDense); ok {
		if brv, ok := bU.(*VecDense); ok {
			amat := arv.mat
			bmat := brv.mat

			if v != a {
				v.checkOverlap(amat)
			}
			if v != b {
				v.checkOverlap(bmat)
			}

			if v.mat.Inc == 1 && amat.Inc == 1 && bmat.Inc == 1 {
				// Fast path for a common case.
				f64.AxpyUnitaryTo(v.mat.Data, -1, bmat.Data, amat.Data)
				return
			}
			f64.AxpyIncTo(v.mat.Data, uintptr(v.mat.Inc), 0,
				-1, bmat.Data, amat.Data,
				uintptr(ar), uintptr(bmat.Inc), uintptr(amat.Inc), 0, 0)
			return
		}
	}

	for i := 0; i < ar; i++ {
		v.setVec(i, a.AtVec(i)-b.AtVec(i))
	}
}

// MulElemVec performs element-wise multiplication of a and b, placing the result
// in the receiver.
func (v *VecDense) MulElemVec(a, b Vector) {
	ar := a.Len()
	br := b.Len()

	if ar != br {
		panic(ErrShape)
	}

	v.reuseAsNonZeroed(ar)

	aU, _ := untransposeExtract(a)
	bU, _ := untransposeExtract(b)

	if arv, ok := aU.(*VecDense); ok {
		if brv, ok := bU.(*VecDense); ok {
			amat := arv.mat
			bmat := brv.mat

			if v != a {
				v.checkOverlap(amat)
			}
			if v != b {
				v.checkOverlap(bmat)
			}

			if v.mat.Inc == 1 && amat.Inc == 1 && bmat.Inc == 1 {
				// Fast path for a common case.
				for i, a := range amat.Data {
					v.mat.Data[i] = a * bmat.Data[i]
				}
				return
			}
			var ia, ib int
			for i := 0; i < ar; i++ {
				v.setVec(i, amat.Data[ia]*bmat.Data[ib])
				ia += amat.Inc
				ib += bmat.Inc
			}
			return
		}
	}

	for i := 0; i < ar; i++ {
		v.setVec(i, a.AtVec(i)*b.AtVec(i))
	}
}

// DivElemVec performs element-wise division of a by b, placing the result
// in the receiver.
func (v *VecDense) DivElemVec(a, b Vector) {
	ar := a.Len()
	br := b.Len()

	if ar != br {
		panic(ErrShape)
	}

	v.reuseAsNonZeroed(ar)

	aU, _ := untransposeExtract(a)
	bU, _ := untransposeExtract(b)

	if arv, ok := aU.(*VecDense); ok {
		if brv, ok := bU.(*VecDense); ok {
			amat := arv.mat
			bmat := brv.mat

			if v != a {
				v.checkOverlap(amat)
			}
			if v != b {
				v.checkOverlap(bmat)
			}

			if v.mat.Inc == 1 && amat.Inc == 1 && bmat.Inc == 1 {
				// Fast path for a common case.
				for i, a := range amat.Data {
					v.setVec(i, a/bmat.Data[i])
				}
				return
			}
			var ia, ib int
			for i := 0; i < ar; i++ {
				v.setVec(i, amat.Data[ia]/bmat.Data[ib])
				ia += amat.Inc
				ib += bmat.Inc
			}
		}
	}

	for i := 0; i < ar; i++ {
		v.setVec(i, a.AtVec(i)/b.AtVec(i))
	}
}

// MulVec computes a * b. The result is stored into the receiver.
// MulVec panics if the number of columns in a does not equal the number of rows in b
// or if the number of columns in b does not equal 1.
func (v *VecDense) MulVec(a Matrix, b Vector) {
	r, c := a.Dims()
	br, bc := b.Dims()
	if c != br || bc != 1 {
		panic(ErrShape)
	}

	aU, trans := untransposeExtract(a)
	var bmat blas64.Vector
	fast := true
	bU, _ := untransposeExtract(b)
	if rv, ok := bU.(*VecDense); ok {
		bmat = rv.mat
		if v != b {
			v.checkOverlap(bmat)
		}
	} else {
		fast = false
	}

	v.reuseAsNonZeroed(r)
	var restore func()
	if v == aU {
		v, restore = v.isolatedWorkspace(aU.(*VecDense))
		defer restore()
	} else if v == b {
		v, restore = v.isolatedWorkspace(b)
		defer restore()
	}

	// TODO(kortschak): Improve the non-fast paths.
	switch aU := aU.(type) {
	case Vector:
		if b.Len() == 1 {
			// {n,1} x {1,1}
			v.ScaleVec(b.AtVec(0), aU)
			return
		}

		// {1,n} x {n,1}
		if fast {
			if rv, ok := aU.(*VecDense); ok {
				amat := rv.mat
				if v != aU {
					v.checkOverlap(amat)
				}

				if amat.Inc == 1 && bmat.Inc == 1 {
					// Fast path for a common case.
					v.setVec(0, f64.DotUnitary(amat.Data, bmat.Data))
					return
				}
				v.setVec(0, f64.DotInc(amat.Data, bmat.Data,
					uintptr(c), uintptr(amat.Inc), uintptr(bmat.Inc), 0, 0))
				return
			}
		}
		var sum float64
		for i := 0; i < c; i++ {
			sum += aU.AtVec(i) * b.AtVec(i)
		}
		v.setVec(0, sum)
		return
	case *SymBandDense:
		if fast {
			aU.checkOverlap(v.asGeneral())
			blas64.Sbmv(1, aU.mat, bmat, 0, v.mat)
			return
		}
	case *SymDense:
		if fast {
			aU.checkOverlap(v.asGeneral())
			blas64.Symv(1, aU.mat, bmat, 0, v.mat)
			return
		}
	case *TriDense:
		if fast {
			v.CopyVec(b)
			aU.checkOverlap(v.asGeneral())
			ta := blas.NoTrans
			if trans {
				ta = blas.Trans
			}
			blas64.Trmv(ta, aU.mat, v.mat)
			return
		}
	case *Dense:
		if fast {
			aU.checkOverlap(v.asGeneral())
			t := blas.NoTrans
			if trans {
				t = blas.Trans
			}
			blas64.Gemv(t, 1, aU.mat, bmat, 0, v.mat)
			return
		}
	default:
		if fast {
			for i := 0; i < r; i++ {
				var f float64
				for j := 0; j < c; j++ {
					f += a.At(i, j) * bmat.Data[j*bmat.Inc]
				}
				v.setVec(i, f)
			}
			return
		}
	}

	for i := 0; i < r; i++ {
		var f float64
		for j := 0; j < c; j++ {
			f += a.At(i, j) * b.AtVec(j)
		}
		v.setVec(i, f)
	}
}

// ReuseAsVec changes the receiver if it IsEmpty() to be of size n×1.
//
// ReuseAsVec re-uses the backing data slice if it has sufficient capacity,
// otherwise a new slice is allocated. The backing data is zero on return.
//
// ReuseAsVec panics if the receiver is not empty, and panics if
// the input size is less than one. To empty the receiver for re-use,
// Reset should be used.
func (v *VecDense) ReuseAsVec(n int) {
	if n <= 0 {
		if n == 0 {
			panic(ErrZeroLength)
		}
		panic(ErrNegativeDimension)
	}
	if !v.IsEmpty() {
		panic(ErrReuseNonEmpty)
	}
	v.reuseAsZeroed(n)
}

// reuseAsNonZeroed resizes an empty vector to a r×1 vector,
// or checks that a non-empty matrix is r×1.
func (v *VecDense) reuseAsNonZeroed(r int) {
	// reuseAsNonZeroed must be kept in sync with reuseAsZeroed.
	if r == 0 {
		panic(ErrZeroLength)
	}
	if v.IsEmpty() {
		v.mat = blas64.Vector{
			N:    r,
			Inc:  1,
			Data: use(v.mat.Data, r),
		}
		return
	}
	if r != v.mat.N {
		panic(ErrShape)
	}
}

// reuseAsZeroed resizes an empty vector to a r×1 vector,
// or checks that a non-empty matrix is r×1.
func (v *VecDense) reuseAsZeroed(r int) {
	// reuseAsZeroed must be kept in sync with reuseAsNonZeroed.
	if r == 0 {
		panic(ErrZeroLength)
	}
	if v.IsEmpty() {
		v.mat = blas64.Vector{
			N:    r,
			Inc:  1,
			Data: useZeroed(v.mat.Data, r),
		}
		return
	}
	if r != v.mat.N {
		panic(ErrShape)
	}
	v.Zero()
}

// IsEmpty returns whether the receiver is empty. Empty matrices can be the
// receiver for size-restricted operations. The receiver can be emptied using
// Reset.
func (v *VecDense) IsEmpty() bool {
	// It must be the case that v.Dims() returns
	// zeros in this case. See comment in Reset().
	return v.mat.Inc == 0
}

func (v *VecDense) isolatedWorkspace(a Vector) (n *VecDense, restore func()) {
	l := a.Len()
	if l == 0 {
		panic(ErrZeroLength)
	}
	n = getWorkspaceVec(l, false)
	return n, func() {
		v.CopyVec(n)
		putWorkspaceVec(n)
	}
}

// asDense returns a Dense representation of the receiver with the same
// underlying data.
func (v *VecDense) asDense() *Dense {
	return &Dense{
		mat:     v.asGeneral(),
		capRows: v.mat.N,
		capCols: 1,
	}
}

// asGeneral returns a blas64.General representation of the receiver with the
// same underlying data.
func (v *VecDense) asGeneral() blas64.General {
	return blas64.General{
		Rows:   v.mat.N,
		Cols:   1,
		Stride: v.mat.Inc,
		Data:   v.mat.Data,
	}
}

// ColViewOf reflects the column j of the RawMatrixer m, into the receiver
// backed by the same underlying data. The receiver must either be empty
// have length equal to the number of rows of m.
func (v *VecDense) ColViewOf(m RawMatrixer, j int) {
	rm := m.RawMatrix()

	if j >= rm.Cols || j < 0 {
		panic(ErrColAccess)
	}
	if !v.IsEmpty() && v.mat.N != rm.Rows {
		panic(ErrShape)
	}

	v.mat.Inc = rm.Stride
	v.mat.Data = rm.Data[j : (rm.Rows-1)*rm.Stride+j+1]
	v.mat.N = rm.Rows
}

// RowViewOf reflects the row i of the RawMatrixer m, into the receiver
// backed by the same underlying data. The receiver must either be
// empty or have length equal to the number of columns of m.
func (v *VecDense) RowViewOf(m RawMatrixer, i int) {
	rm := m.RawMatrix()

	if i >= rm.Rows || i < 0 {
		panic(ErrRowAccess)
	}
	if !v.IsEmpty() && v.mat.N != rm.Cols {
		panic(ErrShape)
	}

	v.mat.Inc = 1
	v.mat.Data = rm.Data[i*rm.Stride : i*rm.Stride+rm.Cols]
	v.mat.N = rm.Cols
}
