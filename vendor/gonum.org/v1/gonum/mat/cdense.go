// Copyright ©2019 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat

import "gonum.org/v1/gonum/blas/cblas128"

// Dense is a dense matrix representation with complex data.
type CDense struct {
	mat cblas128.General

	capRows, capCols int
}

// Dims returns the number of rows and columns in the matrix.
func (m *CDense) Dims() (r, c int) {
	return m.mat.Rows, m.mat.Cols
}

// H performs an implicit conjugate transpose by returning the receiver inside a
// Conjugate.
func (m *CDense) H() CMatrix {
	return Conjugate{m}
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

// reuseAs resizes an empty matrix to a r×c matrix,
// or checks that a non-empty matrix is r×c.
//
// reuseAs must be kept in sync with reuseAsZeroed.
func (m *CDense) reuseAs(r, c int) {
	if m.mat.Rows > m.capRows || m.mat.Cols > m.capCols {
		// Panic as a string, not a mat.Error.
		panic("mat: caps not correctly set")
	}
	if r == 0 || c == 0 {
		panic(ErrZeroLength)
	}
	if m.IsZero() {
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
		panic("mat: caps not correctly set")
	}
	if r == 0 || c == 0 {
		panic(ErrZeroLength)
	}
	if m.IsZero() {
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

// Reset zeros the dimensions of the matrix so that it can be reused as the
// receiver of a dimensionally restricted operation.
//
// See the Reseter interface for more information.
func (m *CDense) Reset() {
	// Row, Cols and Stride must be zeroed in unison.
	m.mat.Rows, m.mat.Cols, m.mat.Stride = 0, 0, 0
	m.capRows, m.capCols = 0, 0
	m.mat.Data = m.mat.Data[:0]
}

// IsZero returns whether the receiver is zero-sized. Zero-sized matrices can be the
// receiver for size-restricted operations. CDense matrices can be zeroed using Reset.
func (m *CDense) IsZero() bool {
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
