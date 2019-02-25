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
	n := a.Symmetric()
	sd := NewSymDense(n, nil)
	sd.CopySym(a)

	jobz := lapack.EVJob(lapack.None)
	if vectors {
		jobz = lapack.ComputeEV
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

// EigenvectorsSym extracts the eigenvectors of the factorized matrix and stores
// them in the receiver. Each eigenvector is a column corresponding to the
// respective eigenvalue returned by e.Values.
//
// EigenvectorsSym panics if the factorization was not successful or if the
// decomposition did not compute the eigenvectors.
func (m *Dense) EigenvectorsSym(e *EigenSym) {
	if !e.succFact() {
		panic(badFact)
	}
	if !e.vectorsComputed {
		panic(badNoVect)
	}
	m.reuseAs(len(e.values), len(e.values))
	m.Copy(e.vectors)
}

// Eigen is a type for creating and using the eigenvalue decomposition of a dense matrix.
type Eigen struct {
	n int // The size of the factorized matrix.

	right bool // have the right eigenvectors been computed
	left  bool // have the left eigenvectors been computed

	values   []complex128
	rVectors *Dense
	lVectors *Dense
}

// succFact returns whether the receiver contains a successful factorization.
func (e *Eigen) succFact() bool {
	return len(e.values) != 0
}

// Factorize computes the eigenvalues of the square matrix a, and optionally
// the eigenvectors.
//
// A right eigenvalue/eigenvector combination is defined by
//  A * x_r = λ * x_r
// where x_r is the column vector called an eigenvector, and λ is the corresponding
// eigenvector.
//
// Similarly, a left eigenvalue/eigenvector combination is defined by
//  x_l * A = λ * x_l
// The eigenvalues, but not the eigenvectors, are the same for both decompositions.
//
// Typically eigenvectors refer to right eigenvectors.
//
// In all cases, Eigen computes the eigenvalues of the matrix. If right and left
// are true, then the right and left eigenvectors will be computed, respectively.
// Eigen panics if the input matrix is not square.
//
// Factorize returns whether the decomposition succeeded. If the decomposition
// failed, methods that require a successful factorization will panic.
func (e *Eigen) Factorize(a Matrix, left, right bool) (ok bool) {
	// TODO(btracey): Change implementation to store VecDenses as a *CMat when
	// #308 is resolved.

	// Copy a because it is modified during the Lapack call.
	r, c := a.Dims()
	if r != c {
		panic(ErrShape)
	}
	var sd Dense
	sd.Clone(a)

	var vl, vr Dense
	var jobvl lapack.LeftEVJob = lapack.None
	var jobvr lapack.RightEVJob = lapack.None
	if left {
		vl = *NewDense(r, r, nil)
		jobvl = lapack.ComputeLeftEV
	}
	if right {
		vr = *NewDense(c, c, nil)
		jobvr = lapack.ComputeRightEV
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
	e.right = right
	e.left = left
	e.lVectors = &vl
	e.rVectors = &vr
	values := make([]complex128, r)
	for i, v := range wr {
		values[i] = complex(v, wi[i])
	}
	e.values = values
	return true
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

// Vectors returns the right eigenvectors of the decomposition. Vectors
// will panic if the right eigenvectors were not computed during the factorization,
// or if the factorization was not successful.
//
// The returned matrix will contain the right eigenvectors of the decomposition
// in the columns of the n×n matrix in the same order as their eigenvalues.
// If the j-th eigenvalue is real, then
//  u_j = VL[:,j],
//  v_j = VR[:,j],
// and if it is not real, then j and j+1 form a complex conjugate pair and the
// eigenvectors can be recovered as
//  u_j     = VL[:,j] + i*VL[:,j+1],
//  u_{j+1} = VL[:,j] - i*VL[:,j+1],
//  v_j     = VR[:,j] + i*VR[:,j+1],
//  v_{j+1} = VR[:,j] - i*VR[:,j+1],
// where i is the imaginary unit. The computed eigenvectors are normalized to
// have Euclidean norm equal to 1 and largest component real.
//
// BUG: This signature and behavior will change when issue #308 is resolved.
func (e *Eigen) Vectors() *Dense {
	if !e.succFact() {
		panic(badFact)
	}
	if !e.right {
		panic(badNoVect)
	}
	return DenseCopyOf(e.rVectors)
}

// LeftVectors returns the left eigenvectors of the decomposition. LeftVectors
// will panic if the left eigenvectors were not computed during the factorization.
// or if the factorization was not successful.
//
// See the documentation in lapack64.Geev for the format of the vectors.
//
// BUG: This signature and behavior will change when issue #308 is resolved.
func (e *Eigen) LeftVectors() *Dense {
	if !e.succFact() {
		panic(badFact)
	}
	if !e.left {
		panic(badNoVect)
	}
	return DenseCopyOf(e.lVectors)
}
