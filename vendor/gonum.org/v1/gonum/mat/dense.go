// Copyright ©2013 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat

import (
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
)

var (
	dense *Dense

	_ Matrix  = dense
	_ Mutable = dense

	_ Cloner       = dense
	_ RowViewer    = dense
	_ ColViewer    = dense
	_ RawRowViewer = dense
	_ Grower       = dense

	_ RawMatrixSetter = dense
	_ RawMatrixer     = dense

	_ Reseter = dense
)

// Dense is a dense matrix representation.
type Dense struct {
	mat blas64.General

	capRows, capCols int
}

// NewDense creates a new Dense matrix with r rows and c columns. If data == nil,
// a new slice is allocated for the backing slice. If len(data) == r*c, data is
// used as the backing slice, and changes to the elements of the returned Dense
// will be reflected in data. If neither of these is true, NewDense will panic.
// NewDense will panic if either r or c is zero.
//
// The data must be arranged in row-major order, i.e. the (i*c + j)-th
// element in the data slice is the {i, j}-th element in the matrix.
func NewDense(r, c int, data []float64) *Dense {
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
		data = make([]float64, r*c)
	}
	return &Dense{
		mat: blas64.General{
			Rows:   r,
			Cols:   c,
			Stride: c,
			Data:   data,
		},
		capRows: r,
		capCols: c,
	}
}

// reuseAs resizes an empty matrix to a r×c matrix,
// or checks that a non-empty matrix is r×c.
//
// reuseAs must be kept in sync with reuseAsZeroed.
func (m *Dense) reuseAs(r, c int) {
	if m.mat.Rows > m.capRows || m.mat.Cols > m.capCols {
		// Panic as a string, not a mat.Error.
		panic("mat: caps not correctly set")
	}
	if r == 0 || c == 0 {
		panic(ErrZeroLength)
	}
	if m.IsZero() {
		m.mat = blas64.General{
			Rows:   r,
			Cols:   c,
			Stride: c,
			Data:   use(m.mat.Data, r*c),
		}
		m.capRows = r
		m.capCols = c
		return
	}
	if r != m.mat.Rows || c != m.mat.Cols {
		panic(ErrShape)
	}
}

// reuseAsZeroed resizes an empty matrix to a r×c matrix,
// or checks that a non-empty matrix is r×c. It zeroes
// all the elements of the matrix.
//
// reuseAsZeroed must be kept in sync with reuseAs.
func (m *Dense) reuseAsZeroed(r, c int) {
	if m.mat.Rows > m.capRows || m.mat.Cols > m.capCols {
		// Panic as a string, not a mat.Error.
		panic("mat: caps not correctly set")
	}
	if r == 0 || c == 0 {
		panic(ErrZeroLength)
	}
	if m.IsZero() {
		m.mat = blas64.General{
			Rows:   r,
			Cols:   c,
			Stride: c,
			Data:   useZeroed(m.mat.Data, r*c),
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

// Zero sets all of the matrix elements to zero.
func (m *Dense) Zero() {
	r := m.mat.Rows
	c := m.mat.Cols
	for i := 0; i < r; i++ {
		zero(m.mat.Data[i*m.mat.Stride : i*m.mat.Stride+c])
	}
}

// isolatedWorkspace returns a new dense matrix w with the size of a and
// returns a callback to defer which performs cleanup at the return of the call.
// This should be used when a method receiver is the same pointer as an input argument.
func (m *Dense) isolatedWorkspace(a Matrix) (w *Dense, restore func()) {
	r, c := a.Dims()
	if r == 0 || c == 0 {
		panic(ErrZeroLength)
	}
	w = getWorkspace(r, c, false)
	return w, func() {
		m.Copy(w)
		putWorkspace(w)
	}
}

// Reset zeros the dimensions of the matrix so that it can be reused as the
// receiver of a dimensionally restricted operation.
//
// See the Reseter interface for more information.
func (m *Dense) Reset() {
	// Row, Cols and Stride must be zeroed in unison.
	m.mat.Rows, m.mat.Cols, m.mat.Stride = 0, 0, 0
	m.capRows, m.capCols = 0, 0
	m.mat.Data = m.mat.Data[:0]
}

// IsZero returns whether the receiver is zero-sized. Zero-sized matrices can be the
// receiver for size-restricted operations. Dense matrices can be zeroed using Reset.
func (m *Dense) IsZero() bool {
	// It must be the case that m.Dims() returns
	// zeros in this case. See comment in Reset().
	return m.mat.Stride == 0
}

// asTriDense returns a TriDense with the given size and side. The backing data
// of the TriDense is the same as the receiver.
func (m *Dense) asTriDense(n int, diag blas.Diag, uplo blas.Uplo) *TriDense {
	return &TriDense{
		mat: blas64.Triangular{
			N:      n,
			Stride: m.mat.Stride,
			Data:   m.mat.Data,
			Uplo:   uplo,
			Diag:   diag,
		},
		cap: n,
	}
}

// DenseCopyOf returns a newly allocated copy of the elements of a.
func DenseCopyOf(a Matrix) *Dense {
	d := &Dense{}
	d.Clone(a)
	return d
}

// SetRawMatrix sets the underlying blas64.General used by the receiver.
// Changes to elements in the receiver following the call will be reflected
// in b.
func (m *Dense) SetRawMatrix(b blas64.General) {
	m.capRows, m.capCols = b.Rows, b.Cols
	m.mat = b
}

// RawMatrix returns the underlying blas64.General used by the receiver.
// Changes to elements in the receiver following the call will be reflected
// in returned blas64.General.
func (m *Dense) RawMatrix() blas64.General { return m.mat }

// Dims returns the number of rows and columns in the matrix.
func (m *Dense) Dims() (r, c int) { return m.mat.Rows, m.mat.Cols }

// Caps returns the number of rows and columns in the backing matrix.
func (m *Dense) Caps() (r, c int) { return m.capRows, m.capCols }

// T performs an implicit transpose by returning the receiver inside a Transpose.
func (m *Dense) T() Matrix {
	return Transpose{m}
}

// ColView returns a Vector reflecting the column j, backed by the matrix data.
//
// See ColViewer for more information.
func (m *Dense) ColView(j int) Vector {
	var v VecDense
	v.ColViewOf(m, j)
	return &v
}

// SetCol sets the values in the specified column of the matrix to the values
// in src. len(src) must equal the number of rows in the receiver.
func (m *Dense) SetCol(j int, src []float64) {
	if j >= m.mat.Cols || j < 0 {
		panic(ErrColAccess)
	}
	if len(src) != m.mat.Rows {
		panic(ErrColLength)
	}

	blas64.Copy(
		blas64.Vector{N: m.mat.Rows, Inc: 1, Data: src},
		blas64.Vector{N: m.mat.Rows, Inc: m.mat.Stride, Data: m.mat.Data[j:]},
	)
}

// SetRow sets the values in the specified rows of the matrix to the values
// in src. len(src) must equal the number of columns in the receiver.
func (m *Dense) SetRow(i int, src []float64) {
	if i >= m.mat.Rows || i < 0 {
		panic(ErrRowAccess)
	}
	if len(src) != m.mat.Cols {
		panic(ErrRowLength)
	}

	copy(m.rawRowView(i), src)
}

// RowView returns row i of the matrix data represented as a column vector,
// backed by the matrix data.
//
// See RowViewer for more information.
func (m *Dense) RowView(i int) Vector {
	var v VecDense
	v.RowViewOf(m, i)
	return &v
}

// RawRowView returns a slice backed by the same array as backing the
// receiver.
func (m *Dense) RawRowView(i int) []float64 {
	if i >= m.mat.Rows || i < 0 {
		panic(ErrRowAccess)
	}
	return m.rawRowView(i)
}

func (m *Dense) rawRowView(i int) []float64 {
	return m.mat.Data[i*m.mat.Stride : i*m.mat.Stride+m.mat.Cols]
}

// DiagView returns the diagonal as a matrix backed by the original data.
func (m *Dense) DiagView() Diagonal {
	n := min(m.mat.Rows, m.mat.Cols)
	return &DiagDense{
		mat: blas64.Vector{
			N:    n,
			Inc:  m.mat.Stride + 1,
			Data: m.mat.Data[:(n-1)*m.mat.Stride+n],
		},
	}
}

// Slice returns a new Matrix that shares backing data with the receiver.
// The returned matrix starts at {i,j} of the receiver and extends k-i rows
// and l-j columns. The final row in the resulting matrix is k-1 and the
// final column is l-1.
// Slice panics with ErrIndexOutOfRange if the slice is outside the capacity
// of the receiver.
func (m *Dense) Slice(i, k, j, l int) Matrix {
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

// Grow returns the receiver expanded by r rows and c columns. If the dimensions
// of the expanded matrix are outside the capacities of the receiver a new
// allocation is made, otherwise not. Note the receiver itself is not modified
// during the call to Grow.
func (m *Dense) Grow(r, c int) Matrix {
	if r < 0 || c < 0 {
		panic(ErrIndexOutOfRange)
	}
	if r == 0 && c == 0 {
		return m
	}

	r += m.mat.Rows
	c += m.mat.Cols

	var t Dense
	switch {
	case m.mat.Rows == 0 || m.mat.Cols == 0:
		t.mat = blas64.General{
			Rows:   r,
			Cols:   c,
			Stride: c,
			// We zero because we don't know how the matrix will be used.
			// In other places, the mat is immediately filled with a result;
			// this is not the case here.
			Data: useZeroed(m.mat.Data, r*c),
		}
	case r > m.capRows || c > m.capCols:
		cr := max(r, m.capRows)
		cc := max(c, m.capCols)
		t.mat = blas64.General{
			Rows:   r,
			Cols:   c,
			Stride: cc,
			Data:   make([]float64, cr*cc),
		}
		t.capRows = cr
		t.capCols = cc
		// Copy the complete matrix over to the new matrix.
		// Including elements not currently visible. Use a temporary structure
		// to avoid modifying the receiver.
		var tmp Dense
		tmp.mat = blas64.General{
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
		t.mat = blas64.General{
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

// Clone makes a copy of a into the receiver, overwriting the previous value of
// the receiver. The clone operation does not make any restriction on shape and
// will not cause shadowing.
//
// See the Cloner interface for more information.
func (m *Dense) Clone(a Matrix) {
	r, c := a.Dims()
	mat := blas64.General{
		Rows:   r,
		Cols:   c,
		Stride: c,
	}
	m.capRows, m.capCols = r, c

	aU, trans := untranspose(a)
	switch aU := aU.(type) {
	case RawMatrixer:
		amat := aU.RawMatrix()
		mat.Data = make([]float64, r*c)
		if trans {
			for i := 0; i < r; i++ {
				blas64.Copy(blas64.Vector{N: c, Inc: amat.Stride, Data: amat.Data[i : i+(c-1)*amat.Stride+1]},
					blas64.Vector{N: c, Inc: 1, Data: mat.Data[i*c : (i+1)*c]})
			}
		} else {
			for i := 0; i < r; i++ {
				copy(mat.Data[i*c:(i+1)*c], amat.Data[i*amat.Stride:i*amat.Stride+c])
			}
		}
	case *VecDense:
		amat := aU.mat
		mat.Data = make([]float64, aU.mat.N)
		blas64.Copy(blas64.Vector{N: aU.mat.N, Inc: amat.Inc, Data: amat.Data},
			blas64.Vector{N: aU.mat.N, Inc: 1, Data: mat.Data})
	default:
		mat.Data = make([]float64, r*c)
		w := *m
		w.mat = mat
		for i := 0; i < r; i++ {
			for j := 0; j < c; j++ {
				w.set(i, j, a.At(i, j))
			}
		}
		*m = w
		return
	}
	m.mat = mat
}

// Copy makes a copy of elements of a into the receiver. It is similar to the
// built-in copy; it copies as much as the overlap between the two matrices and
// returns the number of rows and columns it copied. If a aliases the receiver
// and is a transposed Dense or VecDense, with a non-unitary increment, Copy will
// panic.
//
// See the Copier interface for more information.
func (m *Dense) Copy(a Matrix) (r, c int) {
	r, c = a.Dims()
	if a == m {
		return r, c
	}
	r = min(r, m.mat.Rows)
	c = min(c, m.mat.Cols)
	if r == 0 || c == 0 {
		return 0, 0
	}

	aU, trans := untranspose(a)
	switch aU := aU.(type) {
	case RawMatrixer:
		amat := aU.RawMatrix()
		if trans {
			if amat.Stride != 1 {
				m.checkOverlap(amat)
			}
			for i := 0; i < r; i++ {
				blas64.Copy(blas64.Vector{N: c, Inc: amat.Stride, Data: amat.Data[i : i+(c-1)*amat.Stride+1]},
					blas64.Vector{N: c, Inc: 1, Data: m.mat.Data[i*m.mat.Stride : i*m.mat.Stride+c]})
			}
		} else {
			switch o := offset(m.mat.Data, amat.Data); {
			case o < 0:
				for i := r - 1; i >= 0; i-- {
					copy(m.mat.Data[i*m.mat.Stride:i*m.mat.Stride+c], amat.Data[i*amat.Stride:i*amat.Stride+c])
				}
			case o > 0:
				for i := 0; i < r; i++ {
					copy(m.mat.Data[i*m.mat.Stride:i*m.mat.Stride+c], amat.Data[i*amat.Stride:i*amat.Stride+c])
				}
			default:
				// Nothing to do.
			}
		}
	case *VecDense:
		var n, stride int
		amat := aU.mat
		if trans {
			if amat.Inc != 1 {
				m.checkOverlap(aU.asGeneral())
			}
			n = c
			stride = 1
		} else {
			n = r
			stride = m.mat.Stride
		}
		if amat.Inc == 1 && stride == 1 {
			copy(m.mat.Data, amat.Data[:n])
			break
		}
		switch o := offset(m.mat.Data, amat.Data); {
		case o < 0:
			blas64.Copy(blas64.Vector{N: n, Inc: -amat.Inc, Data: amat.Data},
				blas64.Vector{N: n, Inc: -stride, Data: m.mat.Data})
		case o > 0:
			blas64.Copy(blas64.Vector{N: n, Inc: amat.Inc, Data: amat.Data},
				blas64.Vector{N: n, Inc: stride, Data: m.mat.Data})
		default:
			// Nothing to do.
		}
	default:
		m.checkOverlapMatrix(aU)
		for i := 0; i < r; i++ {
			for j := 0; j < c; j++ {
				m.set(i, j, a.At(i, j))
			}
		}
	}

	return r, c
}

// Stack appends the rows of b onto the rows of a, placing the result into the
// receiver with b placed in the greater indexed rows. Stack will panic if the
// two input matrices do not have the same number of columns or the constructed
// stacked matrix is not the same shape as the receiver.
func (m *Dense) Stack(a, b Matrix) {
	ar, ac := a.Dims()
	br, bc := b.Dims()
	if ac != bc || m == a || m == b {
		panic(ErrShape)
	}

	m.reuseAs(ar+br, ac)

	m.Copy(a)
	w := m.Slice(ar, ar+br, 0, bc).(*Dense)
	w.Copy(b)
}

// Augment creates the augmented matrix of a and b, where b is placed in the
// greater indexed columns. Augment will panic if the two input matrices do
// not have the same number of rows or the constructed augmented matrix is
// not the same shape as the receiver.
func (m *Dense) Augment(a, b Matrix) {
	ar, ac := a.Dims()
	br, bc := b.Dims()
	if ar != br || m == a || m == b {
		panic(ErrShape)
	}

	m.reuseAs(ar, ac+bc)

	m.Copy(a)
	w := m.Slice(0, br, ac, ac+bc).(*Dense)
	w.Copy(b)
}

// Trace returns the trace of the matrix. The matrix must be square or Trace
// will panic.
func (m *Dense) Trace() float64 {
	if m.mat.Rows != m.mat.Cols {
		panic(ErrSquare)
	}
	// TODO(btracey): could use internal asm sum routine.
	var v float64
	for i := 0; i < m.mat.Rows; i++ {
		v += m.mat.Data[i*m.mat.Stride+i]
	}
	return v
}
