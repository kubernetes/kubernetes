// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package blas64

import "gonum.org/v1/gonum/blas"

// SymmetricCols represents a matrix using the conventional column-major storage scheme.
type SymmetricCols Symmetric

// From fills the receiver with elements from a. The receiver
// must have the same dimensions and uplo as a and have adequate
// backing data storage.
func (t SymmetricCols) From(a Symmetric) {
	if t.N != a.N {
		panic("blas64: mismatched dimension")
	}
	if t.Uplo != a.Uplo {
		panic("blas64: mismatched BLAS uplo")
	}
	switch a.Uplo {
	default:
		panic("blas64: bad BLAS uplo")
	case blas.Upper:
		for i := 0; i < a.N; i++ {
			for j := i; j < a.N; j++ {
				t.Data[i+j*t.Stride] = a.Data[i*a.Stride+j]
			}
		}
	case blas.Lower:
		for i := 0; i < a.N; i++ {
			for j := 0; j <= i; j++ {
				t.Data[i+j*t.Stride] = a.Data[i*a.Stride+j]
			}
		}
	}
}

// From fills the receiver with elements from a. The receiver
// must have the same dimensions and uplo as a and have adequate
// backing data storage.
func (t Symmetric) From(a SymmetricCols) {
	if t.N != a.N {
		panic("blas64: mismatched dimension")
	}
	if t.Uplo != a.Uplo {
		panic("blas64: mismatched BLAS uplo")
	}
	switch a.Uplo {
	default:
		panic("blas64: bad BLAS uplo")
	case blas.Upper:
		for i := 0; i < a.N; i++ {
			for j := i; j < a.N; j++ {
				t.Data[i*t.Stride+j] = a.Data[i+j*a.Stride]
			}
		}
	case blas.Lower:
		for i := 0; i < a.N; i++ {
			for j := 0; j <= i; j++ {
				t.Data[i*t.Stride+j] = a.Data[i+j*a.Stride]
			}
		}
	}
}

// SymmetricBandCols represents a symmetric matrix using the band column-major storage scheme.
type SymmetricBandCols SymmetricBand

// From fills the receiver with elements from a. The receiver
// must have the same dimensions, bandwidth and uplo as a and
// have adequate backing data storage.
func (t SymmetricBandCols) From(a SymmetricBand) {
	if t.N != a.N {
		panic("blas64: mismatched dimension")
	}
	if t.K != a.K {
		panic("blas64: mismatched bandwidth")
	}
	if a.Stride < a.K+1 {
		panic("blas64: short stride for source")
	}
	if t.Stride < t.K+1 {
		panic("blas64: short stride for destination")
	}
	if t.Uplo != a.Uplo {
		panic("blas64: mismatched BLAS uplo")
	}
	dst := BandCols{
		Rows: t.N, Cols: t.N,
		Stride: t.Stride,
		Data:   t.Data,
	}
	src := Band{
		Rows: a.N, Cols: a.N,
		Stride: a.Stride,
		Data:   a.Data,
	}
	switch a.Uplo {
	default:
		panic("blas64: bad BLAS uplo")
	case blas.Upper:
		dst.KU = t.K
		src.KU = a.K
	case blas.Lower:
		dst.KL = t.K
		src.KL = a.K
	}
	dst.From(src)
}

// From fills the receiver with elements from a. The receiver
// must have the same dimensions, bandwidth and uplo as a and
// have adequate backing data storage.
func (t SymmetricBand) From(a SymmetricBandCols) {
	if t.N != a.N {
		panic("blas64: mismatched dimension")
	}
	if t.K != a.K {
		panic("blas64: mismatched bandwidth")
	}
	if a.Stride < a.K+1 {
		panic("blas64: short stride for source")
	}
	if t.Stride < t.K+1 {
		panic("blas64: short stride for destination")
	}
	if t.Uplo != a.Uplo {
		panic("blas64: mismatched BLAS uplo")
	}
	dst := Band{
		Rows: t.N, Cols: t.N,
		Stride: t.Stride,
		Data:   t.Data,
	}
	src := BandCols{
		Rows: a.N, Cols: a.N,
		Stride: a.Stride,
		Data:   a.Data,
	}
	switch a.Uplo {
	default:
		panic("blas64: bad BLAS uplo")
	case blas.Upper:
		dst.KU = t.K
		src.KU = a.K
	case blas.Lower:
		dst.KL = t.K
		src.KL = a.K
	}
	dst.From(src)
}
