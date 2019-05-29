// Copyright ©2013 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat

import (
	"gonum.org/v1/gonum/lapack"
	"gonum.org/v1/gonum/lapack/lapack64"
)

const (
	badFact   = "mat: use without successful factorization"
	badNoVect = "mat: eigenvectors not computed"
)

// EigenSym is a type for creating and manipulating the Eigen decomposition of
// symmetric matrices.
type EigenSym struct {
	vectorsComputed bool

	values  []float64
	vectors *Dense
}

// Factorize computes the eigenvalue decomposition of the symmetric matrix a.
// The Eigen decomposition is defined as
//  A = P * D * P^-1
// where D is a diagonal matrix containing the eigenvalues of the matrix, and
// P is a matrix of the eigenvectors of A. Factorize computes the eigenvalues
// in ascending order. If the vectors input argument is false, the eigenvectors
// are not computed.
//
// Factorize returns whether the decomposition succeeded. If the decomposition
// failed, methods that require a successful factorization will panic.
func (e *EigenSym) Factorize(a Symmetric, vectors bool) (ok bool) {
	// kill previous decomposition
	e.vectorsComputed = false
	e.values = e.values[:]

	n := a.Symmetric()
	sd := NewSymDense(n, nil)
	sd.CopySym(a)

	jobz := lapack.EVNone
	if vectors {
		jobz = lapack.EVCompute
	}
	w := make([]float64, n)
	work := []float64{0}
	lapack64.Syev(jobz, sd.mat, w, work, -1)

	work = getFloats(int(work[0]), false)
	ok = lapack64.Syev(jobz, sd.mat, w, work, len(work))
	putFloats(work)
	if !ok {
		e.vectorsComputed = false
		e.values = nil
		e.vectors = nil
		return false
	}
	e.vectorsComputed = vectors
	e.values = w
	e.vectors = NewDense(n, n, sd.mat.Data)
	return true
}

// succFact returns whether the receiver contains a successful factorization.
func (e *EigenSym) succFact() bool {
	return len(e.values) != 0
}

// Values extracts the eigenvalues of the factorized matrix. If dst is
// non-nil, the values are stored in-place into dst. In this case
// dst must have length n, otherwise Values will panic. If dst is
// nil, then a new slice will be allocated of the proper length and filled
// with the eigenvalues.
//
// Values panics if the Eigen decomposition was not successful.
func (e *EigenSym) Values(dst []float64) []float64 {
	if !e.succFact() {
		panic(badFact)
	}
	if dst == nil {
		dst = make([]float64, len(e.values))
	}
	if len(dst) != len(e.values) {
		panic(ErrSliceLengthMismatch)
	}
	copy(dst, e.values)
	return dst
}

// VectorsTo returns the eigenvectors of the decomposition. VectorsTo
// will panic if the eigenvectors were not computed during the factorization,
// or if the factorization was not successful.
//
// If dst is not nil, the eigenvectors are stored in-place into dst, and dst
// must have size n×n and panics otherwise. If dst is nil, a new matrix
// is allocated and returned.
func (e *EigenSym) VectorsTo(dst *Dense) *Dense {
	if !e.succFact() {
		panic(badFact)
	}
	if !e.vectorsComputed {
		panic(badNoVect)
	}
	r, c := e.vectors.Dims()
	if dst == nil {
		dst = NewDense(r, c, nil)
	} else {
		dst.reuseAs(r, c)
	}
	dst.Copy(e.vectors)
	return dst
}

// EigenKind specifies the computation of eigenvectors during factorization.
type EigenKind int

const (
	// EigenNone specifies to not compute any eigenvectors.
	EigenNone EigenKind = 0
	// EigenLeft specifies to compute the left eigenvectors.
	EigenLeft EigenKind = 1 << iota
	// EigenRight specifies to compute the right eigenvectors.
	EigenRight
	// EigenBoth is a convenience value for computing both eigenvectors.
	EigenBoth EigenKind = EigenLeft | EigenRight
)

// Eigen is a type for creating and using the eigenvalue decomposition of a dense matrix.
type Eigen struct {
	n int // The size of the factorized matrix.

	kind EigenKind

	values   []complex128
	rVectors *CDense
	lVectors *CDense
}

// succFact returns whether the receiver contains a successful factorization.
func (e *Eigen) succFact() bool {
	return e.n != 0
}

// Factorize computes the eigenvalues of the square matrix a, and optionally
// the eigenvectors.
//
// A right eigenvalue/eigenvector combination is defined by
//  A * x_r = λ * x_r
// where x_r is the column vector called an eigenvector, and λ is the corresponding
// eigenvalue.
//
// Similarly, a left eigenvalue/eigenvector combination is defined by
//  x_l * A = λ * x_l
// The eigenvalues, but not the eigenvectors, are the same for both decompositions.
//
// Typically eigenvectors refer to right eigenvectors.
//
// In all cases, Factorize computes the eigenvalues of the matrix. kind
// specifies which of the eigenvectors, if any, to compute. See the EigenKind
// documentation for more information.
// Eigen panics if the input matrix is not square.
//
// Factorize returns whether the decomposition succeeded. If the decomposition
// failed, methods that require a successful factorization will panic.
func (e *Eigen) Factorize(a Matrix, kind EigenKind) (ok bool) {
	// kill previous factorization.
	e.n = 0
	e.kind = 0
	// Copy a because it is modified during the Lapack call.
	r, c := a.Dims()
	if r != c {
		panic(ErrShape)
	}
	var sd Dense
	sd.Clone(a)

	left := kind&EigenLeft != 0
	right := kind&EigenRight != 0

	var vl, vr Dense
	jobvl := lapack.LeftEVNone
	jobvr := lapack.RightEVNone
	if left {
		vl = *NewDense(r, r, nil)
		jobvl = lapack.LeftEVCompute
	}
	if right {
		vr = *NewDense(c, c, nil)
		jobvr = lapack.RightEVCompute
	}

	wr := getFloats(c, false)
	defer putFloats(wr)
	wi := getFloats(c, false)
	defer putFloats(wi)

	work := []float64{0}
	lapack64.Geev(jobvl, jobvr, sd.mat, wr, wi, vl.mat, vr.mat, work, -1)
	work = getFloats(int(work[0]), false)
	first := lapack64.Geev(jobvl, jobvr, sd.mat, wr, wi, vl.mat, vr.mat, work, len(work))
	putFloats(work)

	if first != 0 {
		e.values = nil
		return false
	}
	e.n = r
	e.kind = kind

	// Construct complex eigenvalues from float64 data.
	values := make([]complex128, r)
	for i, v := range wr {
		values[i] = complex(v, wi[i])
	}
	e.values = values

	// Construct complex eigenvectors from float64 data.
	var cvl, cvr CDense
	if left {
		cvl = *NewCDense(r, r, nil)
		e.complexEigenTo(&cvl, &vl)
		e.lVectors = &cvl
	} else {
		e.lVectors = nil
	}
	if right {
		cvr = *NewCDense(c, c, nil)
		e.complexEigenTo(&cvr, &vr)
		e.rVectors = &cvr
	} else {
		e.rVectors = nil
	}
	return true
}

// Kind returns the EigenKind of the decomposition. If no decomposition has been
// computed, Kind returns -1.
func (e *Eigen) Kind() EigenKind {
	if !e.succFact() {
		return -1
	}
	return e.kind
}

// Values extracts the eigenvalues of the factorized matrix. If dst is
// non-nil, the values are stored in-place into dst. In this case
// dst must have length n, otherwise Values will panic. If dst is
// nil, then a new slice will be allocated of the proper length and
// filed with the eigenvalues.
//
// Values panics if the Eigen decomposition was not successful.
func (e *Eigen) Values(dst []complex128) []complex128 {
	if !e.succFact() {
		panic(badFact)
	}
	if dst == nil {
		dst = make([]complex128, e.n)
	}
	if len(dst) != e.n {
		panic(ErrSliceLengthMismatch)
	}
	copy(dst, e.values)
	return dst
}

// complexEigenTo extracts the complex eigenvectors from the real matrix d
// and stores them into the complex matrix dst.
//
// The columns of the returned n×n dense matrix contain the eigenvectors of the
// decomposition in the same order as the eigenvalues.
// If the j-th eigenvalue is real, then
//  dst[:,j] = d[:,j],
// and if it is not real, then the elements of the j-th and (j+1)-th columns of d
// form complex conjugate pairs and the eigenvectors are recovered as
//  dst[:,j]   = d[:,j] + i*d[:,j+1],
//  dst[:,j+1] = d[:,j] - i*d[:,j+1],
// where i is the imaginary unit.
func (e *Eigen) complexEigenTo(dst *CDense, d *Dense) {
	r, c := d.Dims()
	cr, cc := dst.Dims()
	if r != cr {
		panic("size mismatch")
	}
	if c != cc {
		panic("size mismatch")
	}
	for j := 0; j < c; j++ {
		if imag(e.values[j]) == 0 {
			for i := 0; i < r; i++ {
				dst.set(i, j, complex(d.at(i, j), 0))
			}
			continue
		}
		for i := 0; i < r; i++ {
			real := d.at(i, j)
			imag := d.at(i, j+1)
			dst.set(i, j, complex(real, imag))
			dst.set(i, j+1, complex(real, -imag))
		}
		j++
	}
}

// VectorsTo returns the right eigenvectors of the decomposition. VectorsTo
// will panic if the right eigenvectors were not computed during the factorization,
// or if the factorization was not successful.
//
// The computed eigenvectors are normalized to have Euclidean norm equal to 1
// and largest component real.
func (e *Eigen) VectorsTo(dst *CDense) *CDense {
	if !e.succFact() {
		panic(badFact)
	}
	if e.kind&EigenRight == 0 {
		panic(badNoVect)
	}
	if dst == nil {
		dst = NewCDense(e.n, e.n, nil)
	} else {
		dst.reuseAs(e.n, e.n)
	}
	dst.Copy(e.rVectors)
	return dst
}

// LeftVectorsTo returns the left eigenvectors of the decomposition. LeftVectorsTo
// will panic if the left eigenvectors were not computed during the factorization,
// or if the factorization was not successful.
//
// The computed eigenvectors are normalized to have Euclidean norm equal to 1
// and largest component real.
func (e *Eigen) LeftVectorsTo(dst *CDense) *CDense {
	if !e.succFact() {
		panic(badFact)
	}
	if e.kind&EigenLeft == 0 {
		panic(badNoVect)
	}
	if dst == nil {
		dst = NewCDense(e.n, e.n, nil)
	} else {
		dst.reuseAs(e.n, e.n)
	}
	dst.Copy(e.lVectors)
	return dst
}
