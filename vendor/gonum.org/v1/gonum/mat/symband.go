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
	_            Symmetric        = symBandDense
	_            Banded           = symBandDense
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

// MutableSymBanded is a symmetric band matrix interface type that allows elements
// to be altered.
type MutableSymBanded interface {
	Symmetric
	Bandwidth() (kl, ku int)
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
// will panic. k must be at least zero and less than n, otherwise NewBandDense will panic.
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
// which is passed to NewBandDense as []float64{1, 2, 3, 4, ...} with k=2.
// Only the values in the band portion of the matrix are used.
func NewSymBandDense(n, k int, data []float64) *SymBandDense {
	if n < 0 || k < 0 {
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

// NewDiagonal is a convenience function that returns a diagonal matrix represented by a
// SymBandDense. The length of data must be n or data must be nil, otherwise NewDiagonal
// will panic.
func NewDiagonal(n int, data []float64) *SymBandDense {
	return NewSymBandDense(n, 0, data)
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
