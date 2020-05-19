// Copyright ©2018 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat

import (
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
)

var (
	diagDense *DiagDense
	_         Matrix          = diagDense
	_         allMatrix       = diagDense
	_         denseMatrix     = diagDense
	_         Diagonal        = diagDense
	_         MutableDiagonal = diagDense
	_         Triangular      = diagDense
	_         TriBanded       = diagDense
	_         Symmetric       = diagDense
	_         SymBanded       = diagDense
	_         Banded          = diagDense
	_         RawBander       = diagDense
	_         RawSymBander    = diagDense

	diag Diagonal
	_    Matrix     = diag
	_    Diagonal   = diag
	_    Triangular = diag
	_    TriBanded  = diag
	_    Symmetric  = diag
	_    SymBanded  = diag
	_    Banded     = diag
)

// Diagonal represents a diagonal matrix, that is a square matrix that only
// has non-zero terms on the diagonal.
type Diagonal interface {
	Matrix
	// Diag returns the number of rows/columns in the matrix.
	Diag() int

	// Bandwidth and TBand are included in the Diagonal interface
	// to allow the use of Diagonal types in banded functions.
	// Bandwidth will always return (0, 0).
	Bandwidth() (kl, ku int)
	TBand() Banded

	// Triangle and TTri are included in the Diagonal interface
	// to allow the use of Diagonal types in triangular functions.
	Triangle() (int, TriKind)
	TTri() Triangular

	// Symmetric and SymBand are included in the Diagonal interface
	// to allow the use of Diagonal types in symmetric and banded symmetric
	// functions respectively.
	Symmetric() int
	SymBand() (n, k int)

	// TriBand and TTriBand are included in the Diagonal interface
	// to allow the use of Diagonal types in triangular banded functions.
	TriBand() (n, k int, kind TriKind)
	TTriBand() TriBanded
}

// MutableDiagonal is a Diagonal matrix whose elements can be set.
type MutableDiagonal interface {
	Diagonal
	SetDiag(i int, v float64)
}

// DiagDense represents a diagonal matrix in dense storage format.
type DiagDense struct {
	mat blas64.Vector
}

// NewDiagDense creates a new Diagonal matrix with n rows and n columns.
// The length of data must be n or data must be nil, otherwise NewDiagDense
// will panic. NewDiagDense will panic if n is zero.
func NewDiagDense(n int, data []float64) *DiagDense {
	if n <= 0 {
		if n == 0 {
			panic(ErrZeroLength)
		}
		panic("mat: negative dimension")
	}
	if data == nil {
		data = make([]float64, n)
	}
	if len(data) != n {
		panic(ErrShape)
	}
	return &DiagDense{
		mat: blas64.Vector{N: n, Data: data, Inc: 1},
	}
}

// Diag returns the dimension of the receiver.
func (d *DiagDense) Diag() int {
	return d.mat.N
}

// Dims returns the dimensions of the matrix.
func (d *DiagDense) Dims() (r, c int) {
	return d.mat.N, d.mat.N
}

// T returns the transpose of the matrix.
func (d *DiagDense) T() Matrix {
	return d
}

// TTri returns the transpose of the matrix. Note that Diagonal matrices are
// Upper by default.
func (d *DiagDense) TTri() Triangular {
	return TransposeTri{d}
}

// TBand performs an implicit transpose by returning the receiver inside a
// TransposeBand.
func (d *DiagDense) TBand() Banded {
	return TransposeBand{d}
}

// TTriBand performs an implicit transpose by returning the receiver inside a
// TransposeTriBand. Note that Diagonal matrices are Upper by default.
func (d *DiagDense) TTriBand() TriBanded {
	return TransposeTriBand{d}
}

// Bandwidth returns the upper and lower bandwidths of the matrix.
// These values are always zero for diagonal matrices.
func (d *DiagDense) Bandwidth() (kl, ku int) {
	return 0, 0
}

// Symmetric implements the Symmetric interface.
func (d *DiagDense) Symmetric() int {
	return d.mat.N
}

// SymBand returns the number of rows/columns in the matrix, and the size of
// the bandwidth.
func (d *DiagDense) SymBand() (n, k int) {
	return d.mat.N, 0
}

// Triangle implements the Triangular interface.
func (d *DiagDense) Triangle() (int, TriKind) {
	return d.mat.N, Upper
}

// TriBand returns the number of rows/columns in the matrix, the
// size of the bandwidth, and the orientation. Note that Diagonal matrices are
// Upper by default.
func (d *DiagDense) TriBand() (n, k int, kind TriKind) {
	return d.mat.N, 0, Upper
}

// Reset empties the matrix so that it can be reused as the
// receiver of a dimensionally restricted operation.
//
// Reset should not be used when the matrix shares backing data.
// See the Reseter interface for more information.
func (d *DiagDense) Reset() {
	// No change of Inc or n to 0 may be
	// made unless both are set to 0.
	d.mat.Inc = 0
	d.mat.N = 0
	d.mat.Data = d.mat.Data[:0]
}

// Zero sets all of the matrix elements to zero.
func (d *DiagDense) Zero() {
	for i := 0; i < d.mat.N; i++ {
		d.mat.Data[d.mat.Inc*i] = 0
	}
}

// DiagView returns the diagonal as a matrix backed by the original data.
func (d *DiagDense) DiagView() Diagonal {
	return d
}

// DiagFrom copies the diagonal of m into the receiver. The receiver must
// be min(r, c) long or empty, otherwise DiagFrom will panic.
func (d *DiagDense) DiagFrom(m Matrix) {
	n := min(m.Dims())
	d.reuseAsNonZeroed(n)

	var vec blas64.Vector
	switch r := m.(type) {
	case *DiagDense:
		vec = r.mat
	case RawBander:
		mat := r.RawBand()
		vec = blas64.Vector{
			N:    n,
			Inc:  mat.Stride,
			Data: mat.Data[mat.KL : (n-1)*mat.Stride+mat.KL+1],
		}
	case RawMatrixer:
		mat := r.RawMatrix()
		vec = blas64.Vector{
			N:    n,
			Inc:  mat.Stride + 1,
			Data: mat.Data[:(n-1)*mat.Stride+n],
		}
	case RawSymBander:
		mat := r.RawSymBand()
		vec = blas64.Vector{
			N:    n,
			Inc:  mat.Stride,
			Data: mat.Data[:(n-1)*mat.Stride+1],
		}
	case RawSymmetricer:
		mat := r.RawSymmetric()
		vec = blas64.Vector{
			N:    n,
			Inc:  mat.Stride + 1,
			Data: mat.Data[:(n-1)*mat.Stride+n],
		}
	case RawTriBander:
		mat := r.RawTriBand()
		data := mat.Data
		if mat.Uplo == blas.Lower {
			data = data[mat.K:]
		}
		vec = blas64.Vector{
			N:    n,
			Inc:  mat.Stride,
			Data: data[:(n-1)*mat.Stride+1],
		}
	case RawTriangular:
		mat := r.RawTriangular()
		if mat.Diag == blas.Unit {
			for i := 0; i < n; i += d.mat.Inc {
				d.mat.Data[i] = 1
			}
			return
		}
		vec = blas64.Vector{
			N:    n,
			Inc:  mat.Stride + 1,
			Data: mat.Data[:(n-1)*mat.Stride+n],
		}
	case RawVectorer:
		d.mat.Data[0] = r.RawVector().Data[0]
		return
	default:
		for i := 0; i < n; i++ {
			d.setDiag(i, m.At(i, i))
		}
		return
	}
	blas64.Copy(vec, d.mat)
}

// RawBand returns the underlying data used by the receiver represented
// as a blas64.Band.
// Changes to elements in the receiver following the call will be reflected
// in returned blas64.Band.
func (d *DiagDense) RawBand() blas64.Band {
	return blas64.Band{
		Rows:   d.mat.N,
		Cols:   d.mat.N,
		KL:     0,
		KU:     0,
		Stride: d.mat.Inc,
		Data:   d.mat.Data,
	}
}

// RawSymBand returns the underlying data used by the receiver represented
// as a blas64.SymmetricBand.
// Changes to elements in the receiver following the call will be reflected
// in returned blas64.Band.
func (d *DiagDense) RawSymBand() blas64.SymmetricBand {
	return blas64.SymmetricBand{
		N:      d.mat.N,
		K:      0,
		Stride: d.mat.Inc,
		Uplo:   blas.Upper,
		Data:   d.mat.Data,
	}
}

// reuseAsNonZeroed resizes an empty diagonal to a r×r diagonal,
// or checks that a non-empty matrix is r×r.
func (d *DiagDense) reuseAsNonZeroed(r int) {
	if r == 0 {
		panic(ErrZeroLength)
	}
	if d.IsEmpty() {
		d.mat = blas64.Vector{
			Inc:  1,
			Data: use(d.mat.Data, r),
		}
		d.mat.N = r
		return
	}
	if r != d.mat.N {
		panic(ErrShape)
	}
}

// IsEmpty returns whether the receiver is empty. Empty matrices can be the
// receiver for size-restricted operations. The receiver can be emptied using
// Reset.
func (d *DiagDense) IsEmpty() bool {
	// It must be the case that d.Dims() returns
	// zeros in this case. See comment in Reset().
	return d.mat.Inc == 0
}

// Trace returns the trace.
func (d *DiagDense) Trace() float64 {
	rb := d.RawBand()
	var tr float64
	for i := 0; i < rb.Rows; i++ {
		tr += rb.Data[rb.KL+i*rb.Stride]
	}
	return tr

}
