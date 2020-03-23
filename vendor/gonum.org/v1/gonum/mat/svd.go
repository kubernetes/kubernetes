// Copyright ©2013 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat

import (
	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/gonum/lapack"
	"gonum.org/v1/gonum/lapack/lapack64"
)

// SVD is a type for creating and using the Singular Value Decomposition (SVD)
// of a matrix.
type SVD struct {
	kind SVDKind

	s  []float64
	u  blas64.General
	vt blas64.General
}

// SVDKind specifies the treatment of singular vectors during an SVD
// factorization.
type SVDKind int

const (
	// SVDNone specifies that no singular vectors should be computed during
	// the decomposition.
	SVDNone SVDKind = 0

	// SVDThinU specifies the thin decomposition for U should be computed.
	SVDThinU SVDKind = 1 << (iota - 1)
	// SVDFullU specifies the full decomposition for U should be computed.
	SVDFullU
	// SVDThinV specifies the thin decomposition for V should be computed.
	SVDThinV
	// SVDFullV specifies the full decomposition for V should be computed.
	SVDFullV

	// SVDThin is a convenience value for computing both thin vectors.
	SVDThin SVDKind = SVDThinU | SVDThinV
	// SVDFull is a convenience value for computing both full vectors.
	SVDFull SVDKind = SVDFullU | SVDFullV
)

// succFact returns whether the receiver contains a successful factorization.
func (svd *SVD) succFact() bool {
	return len(svd.s) != 0
}

// Factorize computes the singular value decomposition (SVD) of the input matrix A.
// The singular values of A are computed in all cases, while the singular
// vectors are optionally computed depending on the input kind.
//
// The full singular value decomposition (kind == SVDFull) is a factorization
// of an m×n matrix A of the form
//  A = U * Σ * Vᵀ
// where Σ is an m×n diagonal matrix, U is an m×m orthogonal matrix, and V is an
// n×n orthogonal matrix. The diagonal elements of Σ are the singular values of A.
// The first min(m,n) columns of U and V are, respectively, the left and right
// singular vectors of A.
//
// Significant storage space can be saved by using the thin representation of
// the SVD (kind == SVDThin) instead of the full SVD, especially if
// m >> n or m << n. The thin SVD finds
//  A = U~ * Σ * V~ᵀ
// where U~ is of size m×min(m,n), Σ is a diagonal matrix of size min(m,n)×min(m,n)
// and V~ is of size n×min(m,n).
//
// Factorize returns whether the decomposition succeeded. If the decomposition
// failed, routines that require a successful factorization will panic.
func (svd *SVD) Factorize(a Matrix, kind SVDKind) (ok bool) {
	// kill previous factorization
	svd.s = svd.s[:0]
	svd.kind = kind

	m, n := a.Dims()
	var jobU, jobVT lapack.SVDJob

	// TODO(btracey): This code should be modified to have the smaller
	// matrix written in-place into aCopy when the lapack/native/dgesvd
	// implementation is complete.
	switch {
	case kind&SVDFullU != 0:
		jobU = lapack.SVDAll
		svd.u = blas64.General{
			Rows:   m,
			Cols:   m,
			Stride: m,
			Data:   use(svd.u.Data, m*m),
		}
	case kind&SVDThinU != 0:
		jobU = lapack.SVDStore
		svd.u = blas64.General{
			Rows:   m,
			Cols:   min(m, n),
			Stride: min(m, n),
			Data:   use(svd.u.Data, m*min(m, n)),
		}
	default:
		jobU = lapack.SVDNone
	}
	switch {
	case kind&SVDFullV != 0:
		svd.vt = blas64.General{
			Rows:   n,
			Cols:   n,
			Stride: n,
			Data:   use(svd.vt.Data, n*n),
		}
		jobVT = lapack.SVDAll
	case kind&SVDThinV != 0:
		svd.vt = blas64.General{
			Rows:   min(m, n),
			Cols:   n,
			Stride: n,
			Data:   use(svd.vt.Data, min(m, n)*n),
		}
		jobVT = lapack.SVDStore
	default:
		jobVT = lapack.SVDNone
	}

	// A is destroyed on call, so copy the matrix.
	aCopy := DenseCopyOf(a)
	svd.kind = kind
	svd.s = use(svd.s, min(m, n))

	work := []float64{0}
	lapack64.Gesvd(jobU, jobVT, aCopy.mat, svd.u, svd.vt, svd.s, work, -1)
	work = getFloats(int(work[0]), false)
	ok = lapack64.Gesvd(jobU, jobVT, aCopy.mat, svd.u, svd.vt, svd.s, work, len(work))
	putFloats(work)
	if !ok {
		svd.kind = 0
	}
	return ok
}

// Kind returns the SVDKind of the decomposition. If no decomposition has been
// computed, Kind returns -1.
func (svd *SVD) Kind() SVDKind {
	if !svd.succFact() {
		return -1
	}
	return svd.kind
}

// Cond returns the 2-norm condition number for the factorized matrix. Cond will
// panic if the receiver does not contain a successful factorization.
func (svd *SVD) Cond() float64 {
	if !svd.succFact() {
		panic(badFact)
	}
	return svd.s[0] / svd.s[len(svd.s)-1]
}

// Values returns the singular values of the factorized matrix in descending order.
//
// If the input slice is non-nil, the values will be stored in-place into
// the slice. In this case, the slice must have length min(m,n), and Values will
// panic with ErrSliceLengthMismatch otherwise. If the input slice is nil, a new
// slice of the appropriate length will be allocated and returned.
//
// Values will panic if the receiver does not contain a successful factorization.
func (svd *SVD) Values(s []float64) []float64 {
	if !svd.succFact() {
		panic(badFact)
	}
	if s == nil {
		s = make([]float64, len(svd.s))
	}
	if len(s) != len(svd.s) {
		panic(ErrSliceLengthMismatch)
	}
	copy(s, svd.s)
	return s
}

// UTo extracts the matrix U from the singular value decomposition. The first
// min(m,n) columns are the left singular vectors and correspond to the singular
// values as returned from SVD.Values.
//
// If dst is empty, UTo will resize dst to be m×m if the full U was computed
// and size m×min(m,n) if the thin U was computed. When dst is non-empty, then
// UTo will panic if dst is not the appropriate size. UTo will also panic if
// the receiver does not contain a successful factorization, or if U was
// not computed during factorization.
func (svd *SVD) UTo(dst *Dense) {
	if !svd.succFact() {
		panic(badFact)
	}
	kind := svd.kind
	if kind&SVDThinU == 0 && kind&SVDFullU == 0 {
		panic("svd: u not computed during factorization")
	}
	r := svd.u.Rows
	c := svd.u.Cols
	if dst.IsEmpty() {
		dst.ReuseAs(r, c)
	} else {
		r2, c2 := dst.Dims()
		if r != r2 || c != c2 {
			panic(ErrShape)
		}
	}

	tmp := &Dense{
		mat:     svd.u,
		capRows: r,
		capCols: c,
	}
	dst.Copy(tmp)
}

// VTo extracts the matrix V from the singular value decomposition. The first
// min(m,n) columns are the right singular vectors and correspond to the singular
// values as returned from SVD.Values.
//
// If dst is empty, VTo will resize dst to be n×n if the full V was computed
// and size n×min(m,n) if the thin V was computed. When dst is non-empty, then
// VTo will panic if dst is not the appropriate size. VTo will also panic if
// the receiver does not contain a successful factorization, or if V was
// not computed during factorization.
func (svd *SVD) VTo(dst *Dense) {
	if !svd.succFact() {
		panic(badFact)
	}
	kind := svd.kind
	if kind&SVDThinU == 0 && kind&SVDFullV == 0 {
		panic("svd: v not computed during factorization")
	}
	r := svd.vt.Rows
	c := svd.vt.Cols
	if dst.IsEmpty() {
		dst.ReuseAs(c, r)
	} else {
		r2, c2 := dst.Dims()
		if c != r2 || r != c2 {
			panic(ErrShape)
		}
	}

	tmp := &Dense{
		mat:     svd.vt,
		capRows: r,
		capCols: c,
	}
	dst.Copy(tmp.T())
}
