// Copyright ©2013 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat

import (
	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/gonum/lapack"
	"gonum.org/v1/gonum/lapack/lapack64"
)

const badRcond = "mat: invalid rcond value"

// SVD is a type for creating and using the Singular Value Decomposition
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

// Rank returns the rank of A based on the count of singular values greater than
// rcond scaled by the largest singular value.
// Rank will panic if the receiver does not contain a successful factorization or
// rcond is negative.
func (svd *SVD) Rank(rcond float64) int {
	if rcond < 0 {
		panic(badRcond)
	}
	if !svd.succFact() {
		panic(badFact)
	}
	s0 := svd.s[0]
	for i, v := range svd.s {
		if v <= rcond*s0 {
			return i
		}
	}
	return len(svd.s)
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
	if kind&SVDThinV == 0 && kind&SVDFullV == 0 {
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

// SolveTo calculates the minimum-norm solution to a linear least squares problem
//  minimize over n-element vectors x: |b - A*x|_2 and |x|_2
// where b is a given m-element vector, using the SVD of m×n matrix A stored in
// the receiver. A may be rank-deficient, that is, the given effective rank can be
//  rank ≤ min(m,n)
// The rank can be computed using SVD.Rank.
//
// Several right-hand side vectors b and solution vectors x can be handled in a
// single call. Vectors b are stored in the columns of the m×k matrix B and the
// resulting vectors x will be stored in the columns of dst. dst must be either
// empty or have the size equal to n×k.
//
// The decomposition must have been factorized computing both the U and V
// singular vectors.
//
// SolveTo returns the residuals calculated from the complete SVD. For this
// value to be valid the factorization must have been performed with at least
// SVDFullU.
func (svd *SVD) SolveTo(dst *Dense, b Matrix, rank int) []float64 {
	if !svd.succFact() {
		panic(badFact)
	}
	if rank < 1 || len(svd.s) < rank {
		panic("svd: rank out of range")
	}
	kind := svd.kind
	if kind&SVDThinU == 0 && kind&SVDFullU == 0 {
		panic("svd: u not computed during factorization")
	}
	if kind&SVDThinV == 0 && kind&SVDFullV == 0 {
		panic("svd: v not computed during factorization")
	}

	u := Dense{
		mat:     svd.u,
		capRows: svd.u.Rows,
		capCols: svd.u.Cols,
	}
	vt := Dense{
		mat:     svd.vt,
		capRows: svd.vt.Rows,
		capCols: svd.vt.Cols,
	}
	s := svd.s[:rank]

	_, bc := b.Dims()
	c := getWorkspace(svd.u.Cols, bc, false)
	defer putWorkspace(c)
	c.Mul(u.T(), b)

	y := getWorkspace(rank, bc, false)
	defer putWorkspace(y)
	y.DivElem(c.slice(0, rank, 0, bc), repVector{vec: s, cols: bc})
	dst.Mul(vt.slice(0, rank, 0, svd.vt.Cols).T(), y)

	res := make([]float64, bc)
	if rank < svd.u.Cols {
		c = c.slice(len(s), svd.u.Cols, 0, bc)
		for j := range res {
			col := c.ColView(j)
			res[j] = Dot(col, col)
		}
	}
	return res
}

type repVector struct {
	vec  []float64
	cols int
}

func (m repVector) Dims() (r, c int) { return len(m.vec), m.cols }
func (m repVector) At(i, j int) float64 {
	if i < 0 || len(m.vec) <= i || j < 0 || m.cols <= j {
		panic(ErrIndexOutOfRange.string) // Panic with string to prevent mat.Error recovery.
	}
	return m.vec[i]
}
func (m repVector) T() Matrix { return Transpose{m} }

// SolveVecTo calculates the minimum-norm solution to a linear least squares problem
//  minimize over n-element vectors x: |b - A*x|_2 and |x|_2
// where b is a given m-element vector, using the SVD of m×n matrix A stored in
// the receiver. A may be rank-deficient, that is, the given effective rank can be
//  rank ≤ min(m,n)
// The rank can be computed using SVD.Rank.
//
// The resulting vector x will be stored in dst. dst must be either empty or
// have length equal to n.
//
// The decomposition must have been factorized computing both the U and V
// singular vectors.
//
// SolveVecTo returns the residuals calculated from the complete SVD. For this
// value to be valid the factorization must have been performed with at least
// SVDFullU.
func (svd *SVD) SolveVecTo(dst *VecDense, b Vector, rank int) float64 {
	if !svd.succFact() {
		panic(badFact)
	}
	if rank < 1 || len(svd.s) < rank {
		panic("svd: rank out of range")
	}
	kind := svd.kind
	if kind&SVDThinU == 0 && kind&SVDFullU == 0 {
		panic("svd: u not computed during factorization")
	}
	if kind&SVDThinV == 0 && kind&SVDFullV == 0 {
		panic("svd: v not computed during factorization")
	}

	u := Dense{
		mat:     svd.u,
		capRows: svd.u.Rows,
		capCols: svd.u.Cols,
	}
	vt := Dense{
		mat:     svd.vt,
		capRows: svd.vt.Rows,
		capCols: svd.vt.Cols,
	}
	s := svd.s[:rank]

	c := getWorkspaceVec(svd.u.Cols, false)
	defer putWorkspaceVec(c)
	c.MulVec(u.T(), b)

	y := getWorkspaceVec(rank, false)
	defer putWorkspaceVec(y)
	y.DivElemVec(c.sliceVec(0, rank), NewVecDense(rank, s))
	dst.MulVec(vt.slice(0, rank, 0, svd.vt.Cols).T(), y)

	var res float64
	if rank < c.Len() {
		c = c.sliceVec(rank, c.Len())
		res = Dot(c, c)
	}
	return res
}
