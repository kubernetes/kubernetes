// Copyright ©2017 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat

import (
	"gonum.org/v1/gonum/blas/blas64"
)

var (
	bandDense *BandDense
	_         Matrix      = bandDense
	_         allMatrix   = bandDense
	_         denseMatrix = bandDense
	_         Banded      = bandDense
	_         RawBander   = bandDense

	_ NonZeroDoer    = bandDense
	_ RowNonZeroDoer = bandDense
	_ ColNonZeroDoer = bandDense
)

// BandDense represents a band matrix in dense storage format.
type BandDense struct {
	mat blas64.Band
}

// Banded is a band matrix representation.
type Banded interface {
	Matrix
	// Bandwidth returns the lower and upper bandwidth values for
	// the matrix. The total bandwidth of the matrix is kl+ku+1.
	Bandwidth() (kl, ku int)

	// TBand is the equivalent of the T() method in the Matrix
	// interface but guarantees the transpose is of banded type.
	TBand() Banded
}

// A RawBander can return a blas64.Band representation of the receiver.
// Changes to the blas64.Band.Data slice will be reflected in the original
// matrix, changes to the Rows, Cols, KL, KU and Stride fields will not.
type RawBander interface {
	RawBand() blas64.Band
}

// A MutableBanded can set elements of a band matrix.
type MutableBanded interface {
	Banded
	SetBand(i, j int, v float64)
}

var (
	_ Matrix            = TransposeBand{}
	_ Banded            = TransposeBand{}
	_ UntransposeBander = TransposeBand{}
)

// TransposeBand is a type for performing an implicit transpose of a band
// matrix. It implements the Banded interface, returning values from the
// transpose of the matrix within.
type TransposeBand struct {
	Banded Banded
}

// At returns the value of the element at row i and column j of the transposed
// matrix, that is, row j and column i of the Banded field.
func (t TransposeBand) At(i, j int) float64 {
	return t.Banded.At(j, i)
}

// Dims returns the dimensions of the transposed matrix.
func (t TransposeBand) Dims() (r, c int) {
	c, r = t.Banded.Dims()
	return r, c
}

// T performs an implicit transpose by returning the Banded field.
func (t TransposeBand) T() Matrix {
	return t.Banded
}

// Bandwidth returns the lower and upper bandwidth values for
// the transposed matrix.
func (t TransposeBand) Bandwidth() (kl, ku int) {
	kl, ku = t.Banded.Bandwidth()
	return ku, kl
}

// TBand performs an implicit transpose by returning the Banded field.
func (t TransposeBand) TBand() Banded {
	return t.Banded
}

// Untranspose returns the Banded field.
func (t TransposeBand) Untranspose() Matrix {
	return t.Banded
}

// UntransposeBand returns the Banded field.
func (t TransposeBand) UntransposeBand() Banded {
	return t.Banded
}

// NewBandDense creates a new Band matrix with r rows and c columns. If data == nil,
// a new slice is allocated for the backing slice. If len(data) == min(r, c+kl)*(kl+ku+1),
// data is used as the backing slice, and changes to the elements of the returned
// BandDense will be reflected in data. If neither of these is true, NewBandDense
// will panic. kl must be at least zero and less r, and ku must be at least zero and
// less than c, otherwise NewBandDense will panic.
// NewBandDense will panic if either r or c is zero.
//
// The data must be arranged in row-major order constructed by removing the zeros
// from the rows outside the band and aligning the diagonals. For example, the matrix
//    1  2  3  0  0  0
//    4  5  6  7  0  0
//    0  8  9 10 11  0
//    0  0 12 13 14 15
//    0  0  0 16 17 18
//    0  0  0  0 19 20
// becomes (* entries are never accessed)
//     *  1  2  3
//     4  5  6  7
//     8  9 10 11
//    12 13 14 15
//    16 17 18  *
//    19 20  *  *
// which is passed to NewBandDense as []float64{*, 1, 2, 3, 4, ...} with kl=1 and ku=2.
// Only the values in the band portion of the matrix are used.
func NewBandDense(r, c, kl, ku int, data []float64) *BandDense {
	if r <= 0 || c <= 0 || kl < 0 || ku < 0 {
		if r == 0 || c == 0 {
			panic(ErrZeroLength)
		}
		panic("mat: negative dimension")
	}
	if kl+1 > r || ku+1 > c {
		panic("mat: band out of range")
	}
	bc := kl + ku + 1
	if data != nil && len(data) != min(r, c+kl)*bc {
		panic(ErrShape)
	}
	if data == nil {
		data = make([]float64, min(r, c+kl)*bc)
	}
	return &BandDense{
		mat: blas64.Band{
			Rows:   r,
			Cols:   c,
			KL:     kl,
			KU:     ku,
			Stride: bc,
			Data:   data,
		},
	}
}

// NewDiagonalRect is a convenience function that returns a diagonal matrix represented by a
// BandDense. The length of data must be min(r, c) otherwise NewDiagonalRect will panic.
func NewDiagonalRect(r, c int, data []float64) *BandDense {
	return NewBandDense(r, c, 0, 0, data)
}

// Dims returns the number of rows and columns in the matrix.
func (b *BandDense) Dims() (r, c int) {
	return b.mat.Rows, b.mat.Cols
}

// Bandwidth returns the upper and lower bandwidths of the matrix.
func (b *BandDense) Bandwidth() (kl, ku int) {
	return b.mat.KL, b.mat.KU
}

// T performs an implicit transpose by returning the receiver inside a Transpose.
func (b *BandDense) T() Matrix {
	return Transpose{b}
}

// TBand performs an implicit transpose by returning the receiver inside a TransposeBand.
func (b *BandDense) TBand() Banded {
	return TransposeBand{b}
}

// RawBand returns the underlying blas64.Band used by the receiver.
// Changes to elements in the receiver following the call will be reflected
// in returned blas64.Band.
func (b *BandDense) RawBand() blas64.Band {
	return b.mat
}

// SetRawBand sets the underlying blas64.Band used by the receiver.
// Changes to elements in the receiver following the call will be reflected
// in the input.
func (b *BandDense) SetRawBand(mat blas64.Band) {
	b.mat = mat
}

// IsEmpty returns whether the receiver is empty. Empty matrices can be the
// receiver for size-restricted operations. The receiver can be zeroed using Reset.
func (b *BandDense) IsEmpty() bool {
	return b.mat.Stride == 0
}

// Reset empties the matrix so that it can be reused as the
// receiver of a dimensionally restricted operation.
//
// Reset should not be used when the matrix shares backing data.
// See the Reseter interface for more information.
func (b *BandDense) Reset() {
	b.mat.Rows = 0
	b.mat.Cols = 0
	b.mat.KL = 0
	b.mat.KU = 0
	b.mat.Stride = 0
	b.mat.Data = b.mat.Data[:0:0]
}

// DiagView returns the diagonal as a matrix backed by the original data.
func (b *BandDense) DiagView() Diagonal {
	n := min(b.mat.Rows, b.mat.Cols)
	return &DiagDense{
		mat: blas64.Vector{
			N:    n,
			Inc:  b.mat.Stride,
			Data: b.mat.Data[b.mat.KL : (n-1)*b.mat.Stride+b.mat.KL+1],
		},
	}
}

// DoNonZero calls the function fn for each of the non-zero elements of b. The function fn
// takes a row/column index and the element value of b at (i, j).
func (b *BandDense) DoNonZero(fn func(i, j int, v float64)) {
	for i := 0; i < min(b.mat.Rows, b.mat.Cols+b.mat.KL); i++ {
		for j := max(0, i-b.mat.KL); j < min(b.mat.Cols, i+b.mat.KU+1); j++ {
			v := b.at(i, j)
			if v != 0 {
				fn(i, j, v)
			}
		}
	}
}

// DoRowNonZero calls the function fn for each of the non-zero elements of row i of b. The function fn
// takes a row/column index and the element value of b at (i, j).
func (b *BandDense) DoRowNonZero(i int, fn func(i, j int, v float64)) {
	if i < 0 || b.mat.Rows <= i {
		panic(ErrRowAccess)
	}
	for j := max(0, i-b.mat.KL); j < min(b.mat.Cols, i+b.mat.KU+1); j++ {
		v := b.at(i, j)
		if v != 0 {
			fn(i, j, v)
		}
	}
}

// DoColNonZero calls the function fn for each of the non-zero elements of column j of b. The function fn
// takes a row/column index and the element value of b at (i, j).
func (b *BandDense) DoColNonZero(j int, fn func(i, j int, v float64)) {
	if j < 0 || b.mat.Cols <= j {
		panic(ErrColAccess)
	}
	for i := 0; i < min(b.mat.Rows, b.mat.Cols+b.mat.KL); i++ {
		if i-b.mat.KL <= j && j < i+b.mat.KU+1 {
			v := b.at(i, j)
			if v != 0 {
				fn(i, j, v)
			}
		}
	}
}

// Zero sets all of the matrix elements to zero.
func (b *BandDense) Zero() {
	m := b.mat.Rows
	kL := b.mat.KL
	nCol := b.mat.KU + 1 + kL
	for i := 0; i < m; i++ {
		l := max(0, kL-i)
		u := min(nCol, m+kL-i)
		zero(b.mat.Data[i*b.mat.Stride+l : i*b.mat.Stride+u])
	}
}

// Trace computes the trace of the matrix.
func (b *BandDense) Trace() float64 {
	r, c := b.Dims()
	if r != c {
		panic(ErrShape)
	}
	rb := b.RawBand()
	var tr float64
	for i := 0; i < r; i++ {
		tr += rb.Data[rb.KL+i*rb.Stride]
	}
	return tr
}
