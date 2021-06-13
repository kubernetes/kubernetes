// Copyright ©2017 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat

import (
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
)

var (
	symBandDense *SymBandDense
	_            Matrix           = symBandDense
	_            allMatrix        = symBandDense
	_            denseMatrix      = symBandDense
	_            Symmetric        = symBandDense
	_            Banded           = symBandDense
	_            SymBanded        = symBandDense
	_            RawSymBander     = symBandDense
	_            MutableSymBanded = symBandDense

	_ NonZeroDoer    = symBandDense
	_ RowNonZeroDoer = symBandDense
	_ ColNonZeroDoer = symBandDense
)

// SymBandDense represents a symmetric band matrix in dense storage format.
type SymBandDense struct {
	mat blas64.SymmetricBand
}

// SymBanded is a symmetric band matrix interface type.
type SymBanded interface {
	Banded

	// Symmetric returns the number of rows/columns in the matrix.
	Symmetric() int

	// SymBand returns the number of rows/columns in the matrix, and the size of
	// the bandwidth.
	SymBand() (n, k int)
}

// MutableSymBanded is a symmetric band matrix interface type that allows elements
// to be altered.
type MutableSymBanded interface {
	SymBanded
	SetSymBand(i, j int, v float64)
}

// A RawSymBander can return a blas64.SymmetricBand representation of the receiver.
// Changes to the blas64.SymmetricBand.Data slice will be reflected in the original
// matrix, changes to the N, K, Stride and Uplo fields will not.
type RawSymBander interface {
	RawSymBand() blas64.SymmetricBand
}

// NewSymBandDense creates a new SymBand matrix with n rows and columns. If data == nil,
// a new slice is allocated for the backing slice. If len(data) == n*(k+1),
// data is used as the backing slice, and changes to the elements of the returned
// SymBandDense will be reflected in data. If neither of these is true, NewSymBandDense
// will panic. k must be at least zero and less than n, otherwise NewSymBandDense will panic.
//
// The data must be arranged in row-major order constructed by removing the zeros
// from the rows outside the band and aligning the diagonals. SymBandDense matrices
// are stored in the upper triangle. For example, the matrix
//    1  2  3  0  0  0
//    2  4  5  6  0  0
//    3  5  7  8  9  0
//    0  6  8 10 11 12
//    0  0  9 11 13 14
//    0  0  0 12 14 15
// becomes (* entries are never accessed)
//     1  2  3
//     4  5  6
//     7  8  9
//    10 11 12
//    13 14  *
//    15  *  *
// which is passed to NewSymBandDense as []float64{1, 2, ..., 15, *, *, *} with k=2.
// Only the values in the band portion of the matrix are used.
func NewSymBandDense(n, k int, data []float64) *SymBandDense {
	if n <= 0 || k < 0 {
		if n == 0 {
			panic(ErrZeroLength)
		}
		panic("mat: negative dimension")
	}
	if k+1 > n {
		panic("mat: band out of range")
	}
	bc := k + 1
	if data != nil && len(data) != n*bc {
		panic(ErrShape)
	}
	if data == nil {
		data = make([]float64, n*bc)
	}
	return &SymBandDense{
		mat: blas64.SymmetricBand{
			N:      n,
			K:      k,
			Stride: bc,
			Uplo:   blas.Upper,
			Data:   data,
		},
	}
}

// Dims returns the number of rows and columns in the matrix.
func (s *SymBandDense) Dims() (r, c int) {
	return s.mat.N, s.mat.N
}

// Symmetric returns the size of the receiver.
func (s *SymBandDense) Symmetric() int {
	return s.mat.N
}

// Bandwidth returns the bandwidths of the matrix.
func (s *SymBandDense) Bandwidth() (kl, ku int) {
	return s.mat.K, s.mat.K
}

// SymBand returns the number of rows/columns in the matrix, and the size of
// the bandwidth.
func (s *SymBandDense) SymBand() (n, k int) {
	return s.mat.N, s.mat.K
}

// T implements the Matrix interface. Symmetric matrices, by definition, are
// equal to their transpose, and this is a no-op.
func (s *SymBandDense) T() Matrix {
	return s
}

// TBand implements the Banded interface.
func (s *SymBandDense) TBand() Banded {
	return s
}

// RawSymBand returns the underlying blas64.SymBand used by the receiver.
// Changes to elements in the receiver following the call will be reflected
// in returned blas64.SymBand.
func (s *SymBandDense) RawSymBand() blas64.SymmetricBand {
	return s.mat
}

// SetRawSymBand sets the underlying blas64.SymmetricBand used by the receiver.
// Changes to elements in the receiver following the call will be reflected
// in the input.
//
// The supplied SymmetricBand must use blas.Upper storage format.
func (s *SymBandDense) SetRawSymBand(mat blas64.SymmetricBand) {
	if mat.Uplo != blas.Upper {
		panic("mat: blas64.SymmetricBand does not have blas.Upper storage")
	}
	s.mat = mat
}

// IsEmpty returns whether the receiver is empty. Empty matrices can be the
// receiver for size-restricted operations. The receiver can be emptied using
// Reset.
func (s *SymBandDense) IsEmpty() bool {
	return s.mat.Stride == 0
}

// Reset empties the matrix so that it can be reused as the
// receiver of a dimensionally restricted operation.
//
// Reset should not be used when the matrix shares backing data.
// See the Reseter interface for more information.
func (s *SymBandDense) Reset() {
	s.mat.N = 0
	s.mat.K = 0
	s.mat.Stride = 0
	s.mat.Uplo = 0
	s.mat.Data = s.mat.Data[:0:0]
}

// Zero sets all of the matrix elements to zero.
func (s *SymBandDense) Zero() {
	for i := 0; i < s.mat.N; i++ {
		u := min(1+s.mat.K, s.mat.N-i)
		zero(s.mat.Data[i*s.mat.Stride : i*s.mat.Stride+u])
	}
}

// DiagView returns the diagonal as a matrix backed by the original data.
func (s *SymBandDense) DiagView() Diagonal {
	n := s.mat.N
	return &DiagDense{
		mat: blas64.Vector{
			N:    n,
			Inc:  s.mat.Stride,
			Data: s.mat.Data[:(n-1)*s.mat.Stride+1],
		},
	}
}

// DoNonZero calls the function fn for each of the non-zero elements of s. The function fn
// takes a row/column index and the element value of s at (i, j).
func (s *SymBandDense) DoNonZero(fn func(i, j int, v float64)) {
	for i := 0; i < s.mat.N; i++ {
		for j := max(0, i-s.mat.K); j < min(s.mat.N, i+s.mat.K+1); j++ {
			v := s.at(i, j)
			if v != 0 {
				fn(i, j, v)
			}
		}
	}
}

// DoRowNonZero calls the function fn for each of the non-zero elements of row i of s. The function fn
// takes a row/column index and the element value of s at (i, j).
func (s *SymBandDense) DoRowNonZero(i int, fn func(i, j int, v float64)) {
	if i < 0 || s.mat.N <= i {
		panic(ErrRowAccess)
	}
	for j := max(0, i-s.mat.K); j < min(s.mat.N, i+s.mat.K+1); j++ {
		v := s.at(i, j)
		if v != 0 {
			fn(i, j, v)
		}
	}
}

// DoColNonZero calls the function fn for each of the non-zero elements of column j of s. The function fn
// takes a row/column index and the element value of s at (i, j).
func (s *SymBandDense) DoColNonZero(j int, fn func(i, j int, v float64)) {
	if j < 0 || s.mat.N <= j {
		panic(ErrColAccess)
	}
	for i := 0; i < s.mat.N; i++ {
		if i-s.mat.K <= j && j < i+s.mat.K+1 {
			v := s.at(i, j)
			if v != 0 {
				fn(i, j, v)
			}
		}
	}
}

// Trace returns the trace.
func (s *SymBandDense) Trace() float64 {
	rb := s.RawSymBand()
	var tr float64
	for i := 0; i < rb.N; i++ {
		tr += rb.Data[i*rb.Stride]
	}
	return tr
}

// MulVecTo computes S⋅x storing the result into dst.
func (s *SymBandDense) MulVecTo(dst *VecDense, _ bool, x Vector) {
	n := s.mat.N
	if x.Len() != n {
		panic(ErrShape)
	}
	dst.reuseAsNonZeroed(n)

	xMat, _ := untransposeExtract(x)
	if xVec, ok := xMat.(*VecDense); ok {
		if dst != xVec {
			dst.checkOverlap(xVec.mat)
			blas64.Sbmv(1, s.mat, xVec.mat, 0, dst.mat)
		} else {
			xCopy := getWorkspaceVec(n, false)
			xCopy.CloneFromVec(xVec)
			blas64.Sbmv(1, s.mat, xCopy.mat, 0, dst.mat)
			putWorkspaceVec(xCopy)
		}
	} else {
		xCopy := getWorkspaceVec(n, false)
		xCopy.CloneFromVec(x)
		blas64.Sbmv(1, s.mat, xCopy.mat, 0, dst.mat)
		putWorkspaceVec(xCopy)
	}
}
