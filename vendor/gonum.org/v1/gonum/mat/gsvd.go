// Copyright ©2017 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat

import (
	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/lapack"
	"gonum.org/v1/gonum/lapack/lapack64"
)

// GSVDKind specifies the treatment of singular vectors during a GSVD
// factorization.
type GSVDKind int

const (
	// GSVDNone specifies that no singular vectors should be computed during
	// the decomposition.
	GSVDNone GSVDKind = 0

	// GSVDU specifies that the U singular vectors should be computed during
	// the decomposition.
	GSVDU GSVDKind = 1 << iota
	// GSVDV specifies that the V singular vectors should be computed during
	// the decomposition.
	GSVDV
	// GSVDQ specifies that the Q singular vectors should be computed during
	// the decomposition.
	GSVDQ

	// GSVDAll is a convenience value for computing all of the singular vectors.
	GSVDAll = GSVDU | GSVDV | GSVDQ
)

// GSVD is a type for creating and using the Generalized Singular Value Decomposition
// (GSVD) of a matrix.
//
// The factorization is a linear transformation of the data sets from the given
// variable×sample spaces to reduced and diagonalized "eigenvariable"×"eigensample"
// spaces.
type GSVD struct {
	kind GSVDKind

	r, p, c, k, l int
	s1, s2        []float64
	a, b, u, v, q blas64.General

	work  []float64
	iwork []int
}

// succFact returns whether the receiver contains a successful factorization.
func (gsvd *GSVD) succFact() bool {
	return gsvd.r != 0
}

// Factorize computes the generalized singular value decomposition (GSVD) of the input
// the r×c matrix A and the p×c matrix B. The singular values of A and B are computed
// in all cases, while the singular vectors are optionally computed depending on the
// input kind.
//
// The full singular value decomposition (kind == GSVDAll) deconstructs A and B as
//  A = U * Σ₁ * [ 0 R ] * Qᵀ
//
//  B = V * Σ₂ * [ 0 R ] * Qᵀ
// where Σ₁ and Σ₂ are r×(k+l) and p×(k+l) diagonal matrices of singular values, and
// U, V and Q are r×r, p×p and c×c orthogonal matrices of singular vectors. k+l is the
// effective numerical rank of the matrix [ Aᵀ Bᵀ ]ᵀ.
//
// It is frequently not necessary to compute the full GSVD. Computation time and
// storage costs can be reduced using the appropriate kind. Either only the singular
// values can be computed (kind == SVDNone), or in conjunction with specific singular
// vectors (kind bit set according to matrix.GSVDU, matrix.GSVDV and matrix.GSVDQ).
//
// Factorize returns whether the decomposition succeeded. If the decomposition
// failed, routines that require a successful factorization will panic.
func (gsvd *GSVD) Factorize(a, b Matrix, kind GSVDKind) (ok bool) {
	// kill the previous decomposition
	gsvd.r = 0
	gsvd.kind = 0

	r, c := a.Dims()
	gsvd.r, gsvd.c = r, c
	p, c := b.Dims()
	gsvd.p = p
	if gsvd.c != c {
		panic(ErrShape)
	}
	var jobU, jobV, jobQ lapack.GSVDJob
	switch {
	default:
		panic("gsvd: bad input kind")
	case kind == GSVDNone:
		jobU = lapack.GSVDNone
		jobV = lapack.GSVDNone
		jobQ = lapack.GSVDNone
	case GSVDAll&kind != 0:
		if GSVDU&kind != 0 {
			jobU = lapack.GSVDU
			gsvd.u = blas64.General{
				Rows:   r,
				Cols:   r,
				Stride: r,
				Data:   use(gsvd.u.Data, r*r),
			}
		}
		if GSVDV&kind != 0 {
			jobV = lapack.GSVDV
			gsvd.v = blas64.General{
				Rows:   p,
				Cols:   p,
				Stride: p,
				Data:   use(gsvd.v.Data, p*p),
			}
		}
		if GSVDQ&kind != 0 {
			jobQ = lapack.GSVDQ
			gsvd.q = blas64.General{
				Rows:   c,
				Cols:   c,
				Stride: c,
				Data:   use(gsvd.q.Data, c*c),
			}
		}
	}

	// A and B are destroyed on call, so copy the matrices.
	aCopy := DenseCopyOf(a)
	bCopy := DenseCopyOf(b)

	gsvd.s1 = use(gsvd.s1, c)
	gsvd.s2 = use(gsvd.s2, c)

	gsvd.iwork = useInt(gsvd.iwork, c)

	gsvd.work = use(gsvd.work, 1)
	lapack64.Ggsvd3(jobU, jobV, jobQ, aCopy.mat, bCopy.mat, gsvd.s1, gsvd.s2, gsvd.u, gsvd.v, gsvd.q, gsvd.work, -1, gsvd.iwork)
	gsvd.work = use(gsvd.work, int(gsvd.work[0]))
	gsvd.k, gsvd.l, ok = lapack64.Ggsvd3(jobU, jobV, jobQ, aCopy.mat, bCopy.mat, gsvd.s1, gsvd.s2, gsvd.u, gsvd.v, gsvd.q, gsvd.work, len(gsvd.work), gsvd.iwork)
	if ok {
		gsvd.a = aCopy.mat
		gsvd.b = bCopy.mat
		gsvd.kind = kind
	}
	return ok
}

// Kind returns the GSVDKind of the decomposition. If no decomposition has been
// computed, Kind returns -1.
func (gsvd *GSVD) Kind() GSVDKind {
	if !gsvd.succFact() {
		return -1
	}
	return gsvd.kind
}

// Rank returns the k and l terms of the rank of [ Aᵀ Bᵀ ]ᵀ.
func (gsvd *GSVD) Rank() (k, l int) {
	return gsvd.k, gsvd.l
}

// GeneralizedValues returns the generalized singular values of the factorized matrices.
// If the input slice is non-nil, the values will be stored in-place into the slice.
// In this case, the slice must have length min(r,c)-k, and GeneralizedValues will
// panic with matrix.ErrSliceLengthMismatch otherwise. If the input slice is nil,
// a new slice of the appropriate length will be allocated and returned.
//
// GeneralizedValues will panic if the receiver does not contain a successful factorization.
func (gsvd *GSVD) GeneralizedValues(v []float64) []float64 {
	if !gsvd.succFact() {
		panic(badFact)
	}
	r := gsvd.r
	c := gsvd.c
	k := gsvd.k
	d := min(r, c)
	if v == nil {
		v = make([]float64, d-k)
	}
	if len(v) != d-k {
		panic(ErrSliceLengthMismatch)
	}
	floats.DivTo(v, gsvd.s1[k:d], gsvd.s2[k:d])
	return v
}

// ValuesA returns the singular values of the factorized A matrix.
// If the input slice is non-nil, the values will be stored in-place into the slice.
// In this case, the slice must have length min(r,c)-k, and ValuesA will panic with
// matrix.ErrSliceLengthMismatch otherwise. If the input slice is nil,
// a new slice of the appropriate length will be allocated and returned.
//
// ValuesA will panic if the receiver does not contain a successful factorization.
func (gsvd *GSVD) ValuesA(s []float64) []float64 {
	if !gsvd.succFact() {
		panic(badFact)
	}
	r := gsvd.r
	c := gsvd.c
	k := gsvd.k
	d := min(r, c)
	if s == nil {
		s = make([]float64, d-k)
	}
	if len(s) != d-k {
		panic(ErrSliceLengthMismatch)
	}
	copy(s, gsvd.s1[k:min(r, c)])
	return s
}

// ValuesB returns the singular values of the factorized B matrix.
// If the input slice is non-nil, the values will be stored in-place into the slice.
// In this case, the slice must have length min(r,c)-k, and ValuesB will panic with
// matrix.ErrSliceLengthMismatch otherwise. If the input slice is nil,
// a new slice of the appropriate length will be allocated and returned.
//
// ValuesB will panic if the receiver does not contain a successful factorization.
func (gsvd *GSVD) ValuesB(s []float64) []float64 {
	if !gsvd.succFact() {
		panic(badFact)
	}
	r := gsvd.r
	c := gsvd.c
	k := gsvd.k
	d := min(r, c)
	if s == nil {
		s = make([]float64, d-k)
	}
	if len(s) != d-k {
		panic(ErrSliceLengthMismatch)
	}
	copy(s, gsvd.s2[k:d])
	return s
}

// ZeroRTo extracts the matrix [ 0 R ] from the singular value decomposition,
// storing the result into dst. [ 0 R ] is of size (k+l)×c.
//
// If dst is empty, ZeroRTo will resize dst to be (k+l)×c. When dst is
// non-empty, ZeroRTo will panic if dst is not (k+l)×c. ZeroRTo will also panic
// if the receiver does not contain a successful factorization.
func (gsvd *GSVD) ZeroRTo(dst *Dense) {
	if !gsvd.succFact() {
		panic(badFact)
	}
	r := gsvd.r
	c := gsvd.c
	k := gsvd.k
	l := gsvd.l
	h := min(k+l, r)
	if dst.IsEmpty() {
		dst.ReuseAs(k+l, c)
	} else {
		r2, c2 := dst.Dims()
		if r2 != k+l || c != c2 {
			panic(ErrShape)
		}
		dst.Zero()
	}
	a := Dense{
		mat:     gsvd.a,
		capRows: r,
		capCols: c,
	}
	dst.slice(0, h, c-k-l, c).Copy(a.Slice(0, h, c-k-l, c))
	if r < k+l {
		b := Dense{
			mat:     gsvd.b,
			capRows: gsvd.p,
			capCols: c,
		}
		dst.slice(r, k+l, c+r-k-l, c).Copy(b.Slice(r-k, l, c+r-k-l, c))
	}
}

// SigmaATo extracts the matrix Σ₁ from the singular value decomposition, storing
// the result into dst. Σ₁ is size r×(k+l).
//
// If dst is empty, SigmaATo will resize dst to be r×(k+l). When dst is
// non-empty, SigmATo will panic if dst is not r×(k+l). SigmaATo will also
// panic if the receiver does not contain a successful factorization.
func (gsvd *GSVD) SigmaATo(dst *Dense) {
	if !gsvd.succFact() {
		panic(badFact)
	}
	r := gsvd.r
	k := gsvd.k
	l := gsvd.l
	if dst.IsEmpty() {
		dst.ReuseAs(r, k+l)
	} else {
		r2, c := dst.Dims()
		if r2 != r || c != k+l {
			panic(ErrShape)
		}
		dst.Zero()
	}
	for i := 0; i < k; i++ {
		dst.set(i, i, 1)
	}
	for i := k; i < min(r, k+l); i++ {
		dst.set(i, i, gsvd.s1[i])
	}
}

// SigmaBTo extracts the matrix Σ₂ from the singular value decomposition, storing
// the result into dst. Σ₂ is size p×(k+l).
//
// If dst is empty, SigmaBTo will resize dst to be p×(k+l). When dst is
// non-empty, SigmBTo will panic if dst is not p×(k+l). SigmaBTo will also
// panic if the receiver does not contain a successful factorization.
func (gsvd *GSVD) SigmaBTo(dst *Dense) {
	if !gsvd.succFact() {
		panic(badFact)
	}
	r := gsvd.r
	p := gsvd.p
	k := gsvd.k
	l := gsvd.l
	if dst.IsEmpty() {
		dst.ReuseAs(p, k+l)
	} else {
		r, c := dst.Dims()
		if r != p || c != k+l {
			panic(ErrShape)
		}
		dst.Zero()
	}
	for i := 0; i < min(l, r-k); i++ {
		dst.set(i, i+k, gsvd.s2[k+i])
	}
	for i := r - k; i < l; i++ {
		dst.set(i, i+k, 1)
	}
}

// UTo extracts the matrix U from the singular value decomposition, storing
// the result into dst. U is size r×r.
//
// If dst is empty, UTo will resize dst to be r×r. When dst is
// non-empty, UTo will panic if dst is not r×r. UTo will also
// panic if the receiver does not contain a successful factorization.
func (gsvd *GSVD) UTo(dst *Dense) {
	if !gsvd.succFact() {
		panic(badFact)
	}
	if gsvd.kind&GSVDU == 0 {
		panic("mat: improper GSVD kind")
	}
	r := gsvd.u.Rows
	c := gsvd.u.Cols
	if dst.IsEmpty() {
		dst.ReuseAs(r, c)
	} else {
		r2, c2 := dst.Dims()
		if r != r2 || c != c2 {
			panic(ErrShape)
		}
	}

	tmp := &Dense{
		mat:     gsvd.u,
		capRows: r,
		capCols: c,
	}
	dst.Copy(tmp)
}

// VTo extracts the matrix V from the singular value decomposition, storing
// the result into dst. V is size p×p.
//
// If dst is empty, VTo will resize dst to be p×p. When dst is
// non-empty, VTo will panic if dst is not p×p. VTo will also
// panic if the receiver does not contain a successful factorization.
func (gsvd *GSVD) VTo(dst *Dense) {
	if !gsvd.succFact() {
		panic(badFact)
	}
	if gsvd.kind&GSVDV == 0 {
		panic("mat: improper GSVD kind")
	}
	r := gsvd.v.Rows
	c := gsvd.v.Cols
	if dst.IsEmpty() {
		dst.ReuseAs(r, c)
	} else {
		r2, c2 := dst.Dims()
		if r != r2 || c != c2 {
			panic(ErrShape)
		}
	}

	tmp := &Dense{
		mat:     gsvd.v,
		capRows: r,
		capCols: c,
	}
	dst.Copy(tmp)
}

// QTo extracts the matrix Q from the singular value decomposition, storing
// the result into dst. Q is size c×c.
//
// If dst is empty, QTo will resize dst to be c×c. When dst is
// non-empty, QTo will panic if dst is not c×c. QTo will also
// panic if the receiver does not contain a successful factorization.
func (gsvd *GSVD) QTo(dst *Dense) {
	if !gsvd.succFact() {
		panic(badFact)
	}
	if gsvd.kind&GSVDQ == 0 {
		panic("mat: improper GSVD kind")
	}
	r := gsvd.q.Rows
	c := gsvd.q.Cols
	if dst.IsEmpty() {
		dst.ReuseAs(r, c)
	} else {
		r2, c2 := dst.Dims()
		if r != r2 || c != c2 {
			panic(ErrShape)
		}
	}

	tmp := &Dense{
		mat:     gsvd.q,
		capRows: r,
		capCols: c,
	}
	dst.Copy(tmp)
}
