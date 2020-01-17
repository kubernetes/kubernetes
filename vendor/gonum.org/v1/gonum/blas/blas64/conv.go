// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package blas64

import "gonum.org/v1/gonum/blas"

// GeneralCols represents a matrix using the conventional column-major storage scheme.
type GeneralCols General

// From fills the receiver with elements from a. The receiver
// must have the same dimensions as a and have adequate backing
// data storage.
func (t GeneralCols) From(a General) {
	if t.Rows != a.Rows || t.Cols != a.Cols {
		panic("blas64: mismatched dimension")
	}
	if len(t.Data) < (t.Cols-1)*t.Stride+t.Rows {
		panic("blas64: short data slice")
	}
	for i := 0; i < a.Rows; i++ {
		for j, v := range a.Data[i*a.Stride : i*a.Stride+a.Cols] {
			t.Data[i+j*t.Stride] = v
		}
	}
}

// From fills the receiver with elements from a. The receiver
// must have the same dimensions as a and have adequate backing
// data storage.
func (t General) From(a GeneralCols) {
	if t.Rows != a.Rows || t.Cols != a.Cols {
		panic("blas64: mismatched dimension")
	}
	if len(t.Data) < (t.Rows-1)*t.Stride+t.Cols {
		panic("blas64: short data slice")
	}
	for j := 0; j < a.Cols; j++ {
		for i, v := range a.Data[j*a.Stride : j*a.Stride+a.Rows] {
			t.Data[i*t.Stride+j] = v
		}
	}
}

// TriangularCols represents a matrix using the conventional column-major storage scheme.
type TriangularCols Triangular

// From fills the receiver with elements from a. The receiver
// must have the same dimensions, uplo and diag as a and have
// adequate backing data storage.
func (t TriangularCols) From(a Triangular) {
	if t.N != a.N {
		panic("blas64: mismatched dimension")
	}
	if t.Uplo != a.Uplo {
		panic("blas64: mismatched BLAS uplo")
	}
	if t.Diag != a.Diag {
		panic("blas64: mismatched BLAS diag")
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
	case blas.All:
		for i := 0; i < a.N; i++ {
			for j := 0; j < a.N; j++ {
				t.Data[i+j*t.Stride] = a.Data[i*a.Stride+j]
			}
		}
	}
}

// From fills the receiver with elements from a. The receiver
// must have the same dimensions, uplo and diag as a and have
// adequate backing data storage.
func (t Triangular) From(a TriangularCols) {
	if t.N != a.N {
		panic("blas64: mismatched dimension")
	}
	if t.Uplo != a.Uplo {
		panic("blas64: mismatched BLAS uplo")
	}
	if t.Diag != a.Diag {
		panic("blas64: mismatched BLAS diag")
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
	case blas.All:
		for i := 0; i < a.N; i++ {
			for j := 0; j < a.N; j++ {
				t.Data[i*t.Stride+j] = a.Data[i+j*a.Stride]
			}
		}
	}
}

// BandCols represents a matrix using the band column-major storage scheme.
type BandCols Band

// From fills the receiver with elements from a. The receiver
// must have the same dimensions and bandwidth as a and have
// adequate backing data storage.
func (t BandCols) From(a Band) {
	if t.Rows != a.Rows || t.Cols != a.Cols {
		panic("blas64: mismatched dimension")
	}
	if t.KL != a.KL || t.KU != a.KU {
		panic("blas64: mismatched bandwidth")
	}
	if a.Stride < a.KL+a.KU+1 {
		panic("blas64: short stride for source")
	}
	if t.Stride < t.KL+t.KU+1 {
		panic("blas64: short stride for destination")
	}
	for i := 0; i < a.Rows; i++ {
		for j := max(0, i-a.KL); j < min(i+a.KU+1, a.Cols); j++ {
			t.Data[i+t.KU-j+j*t.Stride] = a.Data[j+a.KL-i+i*a.Stride]
		}
	}
}

// From fills the receiver with elements from a. The receiver
// must have the same dimensions and bandwidth as a and have
// adequate backing data storage.
func (t Band) From(a BandCols) {
	if t.Rows != a.Rows || t.Cols != a.Cols {
		panic("blas64: mismatched dimension")
	}
	if t.KL != a.KL || t.KU != a.KU {
		panic("blas64: mismatched bandwidth")
	}
	if a.Stride < a.KL+a.KU+1 {
		panic("blas64: short stride for source")
	}
	if t.Stride < t.KL+t.KU+1 {
		panic("blas64: short stride for destination")
	}
	for j := 0; j < a.Cols; j++ {
		for i := max(0, j-a.KU); i < min(j+a.KL+1, a.Rows); i++ {
			t.Data[j+a.KL-i+i*a.Stride] = a.Data[i+t.KU-j+j*t.Stride]
		}
	}
}

// TriangularBandCols represents a triangular matrix using the band column-major storage scheme.
type TriangularBandCols TriangularBand

// From fills the receiver with elements from a. The receiver
// must have the same dimensions, bandwidth and uplo as a and
// have adequate backing data storage.
func (t TriangularBandCols) From(a TriangularBand) {
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
	if t.Diag != a.Diag {
		panic("blas64: mismatched BLAS diag")
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
func (t TriangularBand) From(a TriangularBandCols) {
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
	if t.Diag != a.Diag {
		panic("blas64: mismatched BLAS diag")
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
