// Copyright ©2019 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat

import (
	"math/cmplx"

	"gonum.org/v1/gonum/blas/cblas128"
)

var (
	cDense *CDense

	_ CMatrix   = cDense
	_ allMatrix = cDense
)

// CDense is a dense matrix representation with complex data.
type CDense struct {
	mat cblas128.General

	capRows, capCols int
}

// Dims returns the number of rows and columns in the matrix.
func (m *CDense) Dims() (r, c int) {
	return m.mat.Rows, m.mat.Cols
}

// Caps returns the number of rows and columns in the backing matrix.
func (m *CDense) Caps() (r, c int) { return m.capRows, m.capCols }

// H performs an implicit conjugate transpose by returning the receiver inside a
// ConjTranspose.
func (m *CDense) H() CMatrix {
	return ConjTranspose{m}
}

// T performs an implicit transpose by returning the receiver inside a
// CTranspose.
func (m *CDense) T() CMatrix {
	return CTranspose{m}
}

// Conj calculates the element-wise conjugate of a and stores the result in the
// receiver.
// Conj will panic if m and a do not have the same dimension unless m is empty.
func (m *CDense) Conj(a CMatrix) {
	ar, ac := a.Dims()
	aU, aTrans, aConj := untransposeExtractCmplx(a)
	m.reuseAsNonZeroed(ar, ac)

	if arm, ok := a.(*CDense); ok {
		amat := arm.mat
		if m != aU {
			m.checkOverlap(amat)
		}
		for ja, jm := 0, 0; ja < ar*amat.Stride; ja, jm = ja+amat.Stride, jm+m.mat.Stride {
			for i, v := range amat.Data[ja : ja+ac] {
				m.mat.Data[i+jm] = cmplx.Conj(v)
			}
		}
		return
	}

	m.checkOverlapMatrix(aU)
	if aTrans != aConj && m == aU {
		// Only make workspace if the destination is transposed
		// with respect to the source and they are the same
		// matrix.
		var restore func()
		m, restore = m.isolatedWorkspace(aU)
		defer restore()
	}

	for r := 0; r < ar; r++ {
		for c := 0; c < ac; c++ {
			m.set(r, c, cmplx.Conj(a.At(r, c)))
		}
	}
}

// Slice returns a new CMatrix that shares backing data with the receiver.
// The returned matrix starts at {i,j} of the receiver and extends k-i rows
// and l-j columns. The final row in the resulting matrix is k-1 and the
// final column is l-1.
// Slice panics with ErrIndexOutOfRange if the slice is outside the capacity
// of the receiver.
func (m *CDense) Slice(i, k, j, l int) CMatrix {
	return m.slice(i, k, j, l)
}

func (m *CDense) slice(i, k, j, l int) *CDense {
	mr, mc := m.Caps()
	if i < 0 || mr <= i || j < 0 || mc <= j || k < i || mr < k || l < j || mc < l {
		if i == k || j == l {
			panic(ErrZeroLength)
		}
		panic(ErrIndexOutOfRange)
	}
	t := *m
	t.mat.Data = t.mat.Data[i*t.mat.Stride+j : (k-1)*t.mat.Stride+l]
	t.mat.Rows = k - i
	t.mat.Cols = l - j
	t.capRows -= i
	t.capCols -= j
	return &t
}

// NewCDense creates a new complex Dense matrix with r rows and c columns.
// If data == nil, a new slice is allocated for the backing slice.
// If len(data) == r*c, data is used as the backing slice, and changes to the
// elements of the returned CDense will be reflected in data.
// If neither of these is true, NewCDense will panic.
// NewCDense will panic if either r or c is zero.
//
// The data must be arranged in row-major order, i.e. the (i*c + j)-th
// element in the data slice is the {i, j}-th element in the matrix.
func NewCDense(r, c int, data []complex128) *CDense {
	if r <= 0 || c <= 0 {
		if r == 0 || c == 0 {
			panic(ErrZeroLength)
		}
		panic("mat: negative dimension")
	}
	if data != nil && r*c != len(data) {
		panic(ErrShape)
	}
	if data == nil {
		data = make([]complex128, r*c)
	}
	return &CDense{
		mat: cblas128.General{
			Rows:   r,
			Cols:   c,
			Stride: c,
			Data:   data,
		},
		capRows: r,
		capCols: c,
	}
}

// ReuseAs changes the receiver if it IsEmpty() to be of size r×c.
//
// ReuseAs re-uses the backing data slice if it has sufficient capacity,
// otherwise a new slice is allocated. The backing data is zero on return.
//
// ReuseAs panics if the receiver is not empty, and panics if
// the input sizes are less than one. To empty the receiver for re-use,
// Reset should be used.
func (m *CDense) ReuseAs(r, c int) {
	if r <= 0 || c <= 0 {
		if r == 0 || c == 0 {
			panic(ErrZeroLength)
		}
		panic(ErrNegativeDimension)
	}
	if !m.IsEmpty() {
		panic(ErrReuseNonEmpty)
	}
	m.reuseAsZeroed(r, c)
}

// reuseAs resizes an empty matrix to a r×c matrix,
// or checks that a non-empty matrix is r×c.
//
// reuseAs must be kept in sync with reuseAsZeroed.
func (m *CDense) reuseAsNonZeroed(r, c int) {
	if m.mat.Rows > m.capRows || m.mat.Cols > m.capCols {
		// Panic as a string, not a mat.Error.
		panic(badCap)
	}
	if r == 0 || c == 0 {
		panic(ErrZeroLength)
	}
	if m.IsEmpty() {
		m.mat = cblas128.General{
			Rows:   r,
			Cols:   c,
			Stride: c,
			Data:   useC(m.mat.Data, r*c),
		}
		m.capRows = r
		m.capCols = c
		return
	}
	if r != m.mat.Rows || c != m.mat.Cols {
		panic(ErrShape)
	}
}

func (m *CDense) reuseAsZeroed(r, c int) {
	// This must be kept in-sync with reuseAs.
	if m.mat.Rows > m.capRows || m.mat.Cols > m.capCols {
		// Panic as a string, not a mat.Error.
		panic(badCap)
	}
	if r == 0 || c == 0 {
		panic(ErrZeroLength)
	}
	if m.IsEmpty() {
		m.mat = cblas128.General{
			Rows:   r,
			Cols:   c,
			Stride: c,
			Data:   useZeroedC(m.mat.Data, r*c),
		}
		m.capRows = r
		m.capCols = c
		return
	}
	if r != m.mat.Rows || c != m.mat.Cols {
		panic(ErrShape)
	}
	m.Zero()
}

// isolatedWorkspace returns a new dense matrix w with the size of a and
// returns a callback to defer which performs cleanup at the return of the call.
// This should be used when a method receiver is the same pointer as an input argument.
func (m *CDense) isolatedWorkspace(a CMatrix) (w *CDense, restore func()) {
	r, c := a.Dims()
	if r == 0 || c == 0 {
		panic(ErrZeroLength)
	}
	w = getWorkspaceCmplx(r, c, false)
	return w, func() {
		m.Copy(w)
		putWorkspaceCmplx(w)
	}
}

// Reset zeros the dimensions of the matrix so that it can be reused as the
// receiver of a dimensionally restricted operation.
//
// Reset should not be used when the matrix shares backing data.
// See the Reseter interface for more information.
func (m *CDense) Reset() {
	// Row, Cols and Stride must be zeroed in unison.
	m.mat.Rows, m.mat.Cols, m.mat.Stride = 0, 0, 0
	m.capRows, m.capCols = 0, 0
	m.mat.Data = m.mat.Data[:0]
}

// IsEmpty returns whether the receiver is empty. Empty matrices can be the
// receiver for size-restricted operations. The receiver can be zeroed using Reset.
func (m *CDense) IsEmpty() bool {
	// It must be the case that m.Dims() returns
	// zeros in this case. See comment in Reset().
	return m.mat.Stride == 0
}

// Zero sets all of the matrix elements to zero.
func (m *CDense) Zero() {
	r := m.mat.Rows
	c := m.mat.Cols
	for i := 0; i < r; i++ {
		zeroC(m.mat.Data[i*m.mat.Stride : i*m.mat.Stride+c])
	}
}

// Copy makes a copy of elements of a into the receiver. It is similar to the
// built-in copy; it copies as much as the overlap between the two matrices and
// returns the number of rows and columns it copied. If a aliases the receiver
// and is a transposed Dense or VecDense, with a non-unitary increment, Copy will
// panic.
//
// See the Copier interface for more information.
func (m *CDense) Copy(a CMatrix) (r, c int) {
	r, c = a.Dims()
	if a == m {
		return r, c
	}
	r = min(r, m.mat.Rows)
	c = min(c, m.mat.Cols)
	if r == 0 || c == 0 {
		return 0, 0
	}
	// TODO(btracey): Check for overlap when complex version exists.
	// TODO(btracey): Add fast-paths.
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			m.set(i, j, a.At(i, j))
		}
	}
	return r, c
}

// SetRawCMatrix sets the underlying cblas128.General used by the receiver.
// Changes to elements in the receiver following the call will be reflected
// in b.
func (m *CDense) SetRawCMatrix(b cblas128.General) {
	m.capRows, m.capCols = b.Rows, b.Cols
	m.mat = b
}

// RawCMatrix returns the underlying cblas128.General used by the receiver.
// Changes to elements in the receiver following the call will be reflected
// in returned cblas128.General.
func (m *CDense) RawCMatrix() cblas128.General { return m.mat }

// Grow returns the receiver expanded by r rows and c columns. If the dimensions
// of the expanded matrix are outside the capacities of the receiver a new
// allocation is made, otherwise not. Note the receiver itself is not modified
// during the call to Grow.
func (m *CDense) Grow(r, c int) CMatrix {
	if r < 0 || c < 0 {
		panic(ErrIndexOutOfRange)
	}
	if r == 0 && c == 0 {
		return m
	}

	r += m.mat.Rows
	c += m.mat.Cols

	var t CDense
	switch {
	case m.mat.Rows == 0 || m.mat.Cols == 0:
		t.mat = cblas128.General{
			Rows:   r,
			Cols:   c,
			Stride: c,
			// We zero because we don't know how the matrix will be used.
			// In other places, the mat is immediately filled with a result;
			// this is not the case here.
			Data: useZeroedC(m.mat.Data, r*c),
		}
	case r > m.capRows || c > m.capCols:
		cr := max(r, m.capRows)
		cc := max(c, m.capCols)
		t.mat = cblas128.General{
			Rows:   r,
			Cols:   c,
			Stride: cc,
			Data:   make([]complex128, cr*cc),
		}
		t.capRows = cr
		t.capCols = cc
		// Copy the complete matrix over to the new matrix.
		// Including elements not currently visible. Use a temporary structure
		// to avoid modifying the receiver.
		var tmp CDense
		tmp.mat = cblas128.General{
			Rows:   m.mat.Rows,
			Cols:   m.mat.Cols,
			Stride: m.mat.Stride,
			Data:   m.mat.Data,
		}
		tmp.capRows = m.capRows
		tmp.capCols = m.capCols
		t.Copy(&tmp)
		return &t
	default:
		t.mat = cblas128.General{
			Data:   m.mat.Data[:(r-1)*m.mat.Stride+c],
			Rows:   r,
			Cols:   c,
			Stride: m.mat.Stride,
		}
	}
	t.capRows = r
	t.capCols = c
	return &t
}
