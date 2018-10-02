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

// Factorize computes the singular value decomposition (SVD) of the input matrix
// A. The singular values of A are computed in all cases, while the singular
// vectors are optionally computed depending on the input kind.
//
// The full singular value decomposition (kind == SVDFull) deconstructs A as
//  A = U * Σ * V^T
// where Σ is an m×n diagonal matrix of singular vectors, U is an m×m unitary
// matrix of left singular vectors, and V is an n×n matrix of right singular vectors.
//
// It is frequently not necessary to compute the full SVD. Computation time and
// storage costs can be reduced using the appropriate kind. Only the singular
// values can be computed (kind == SVDNone), or a "thin" representation of the
// singular vectors (kind = SVDThin). The thin representation can save a significant
// amount of memory if m >> n. See the documentation for the lapack.SVDKind values
// for more information.
//
// Factorize returns whether the decomposition succeeded. If the decomposition
// failed, routines that require a successful factorization will panic.
func (svd *SVD) Factorize(a Matrix, kind SVDKind) (ok bool) {
	m, n := a.Dims()
	var jobU, jobVT lapack.SVDJob
	switch kind {
	default:
		panic("svd: bad input kind")
	case SVDNone:
		jobU = lapack.SVDNone
		jobVT = lapack.SVDNone
	case SVDFull:
		// TODO(btracey): This code should be modified to have the smaller
		// matrix written in-place into aCopy when the lapack/native/dgesvd
		// implementation is complete.
		svd.u = blas64.General{
			Rows:   m,
			Cols:   m,
			Stride: m,
			Data:   use(svd.u.Data, m*m),
		}
		svd.vt = blas64.General{
			Rows:   n,
			Cols:   n,
			Stride: n,
			Data:   use(svd.vt.Data, n*n),
		}
		jobU = lapack.SVDAll
		jobVT = lapack.SVDAll
	case SVDThin:
		// TODO(btracey): This code should be modified to have the larger
		// matrix written in-place into aCopy when the lapack/native/dgesvd
		// implementation is complete.
		svd.u = blas64.General{
			Rows:   m,
			Cols:   min(m, n),
			Stride: min(m, n),
			Data:   use(svd.u.Data, m*min(m, n)),
		}
		svd.vt = blas64.General{
			Rows:   min(m, n),
			Cols:   n,
			Stride: n,
			Data:   use(svd.vt.Data, min(m, n)*n),
		}
		jobU = lapack.SVDInPlace
		jobVT = lapack.SVDInPlace
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

// Kind returns the matrix.SVDKind of the decomposition. If no decomposition has been
// computed, Kind returns 0.
func (svd *SVD) Kind() SVDKind {
	return svd.kind
}

// Cond returns the 2-norm condition number for the factorized matrix. Cond will
// panic if the receiver does not contain a successful factorization.
func (svd *SVD) Cond() float64 {
	if svd.kind == 0 {
		panic("svd: no decomposition computed")
	}
	return svd.s[0] / svd.s[len(svd.s)-1]
}

// Values returns the singular values of the factorized matrix in decreasing order.
// If the input slice is non-nil, the values will be stored in-place into the slice.
// In this case, the slice must have length min(m,n), and Values will panic with
// matrix.ErrSliceLengthMismatch otherwise. If the input slice is nil,
// a new slice of the appropriate length will be allocated and returned.
//
// Values will panic if the receiver does not contain a successful factorization.
func (svd *SVD) Values(s []float64) []float64 {
	if svd.kind == 0 {
		panic("svd: no decomposition computed")
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

// UTo extracts the matrix U from the singular value decomposition, storing
// the result in-place into dst. U is size m×m if svd.Kind() == SVDFull,
// of size m×min(m,n) if svd.Kind() == SVDThin, and UTo panics otherwise.
func (svd *SVD) UTo(dst *Dense) *Dense {
	kind := svd.kind
	if kind != SVDFull && kind != SVDThin {
		panic("mat: improper SVD kind")
	}
	r := svd.u.Rows
	c := svd.u.Cols
	if dst == nil {
		dst = NewDense(r, c, nil)
	} else {
		dst.reuseAs(r, c)
	}

	tmp := &Dense{
		mat:     svd.u,
		capRows: r,
		capCols: c,
	}
	dst.Copy(tmp)

	return dst
}

// VTo extracts the matrix V from the singular value decomposition, storing
// the result in-place into dst. V is size n×n if svd.Kind() == SVDFull,
// of size n×min(m,n) if svd.Kind() == SVDThin, and VTo panics otherwise.
func (svd *SVD) VTo(dst *Dense) *Dense {
	kind := svd.kind
	if kind != SVDFull && kind != SVDThin {
		panic("mat: improper SVD kind")
	}
	r := svd.vt.Rows
	c := svd.vt.Cols
	if dst == nil {
		dst = NewDense(c, r, nil)
	} else {
		dst.reuseAs(c, r)
	}

	tmp := &Dense{
		mat:     svd.vt,
		capRows: r,
		capCols: c,
	}
	dst.Copy(tmp.T())

	return dst
}
