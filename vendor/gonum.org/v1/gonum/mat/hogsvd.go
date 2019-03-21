// Copyright ©2017 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat

import (
	"errors"

	"gonum.org/v1/gonum/blas/blas64"
)

// HOGSVD is a type for creating and using the Higher Order Generalized Singular Value
// Decomposition (HOGSVD) of a set of matrices.
//
// The factorization is a linear transformation of the data sets from the given
// variable×sample spaces to reduced and diagonalized "eigenvariable"×"eigensample"
// spaces.
type HOGSVD struct {
	n int
	v *Dense
	b []Dense

	err error
}

// Factorize computes the higher order generalized singular value decomposition (HOGSVD)
// of the n input r_i×c column tall matrices in m. HOGSV extends the GSVD case from 2 to n
// input matrices.
//
//  M_0 = U_0 * Σ_0 * V^T
//  M_1 = U_1 * Σ_1 * V^T
//  .
//  .
//  .
//  M_{n-1} = U_{n-1} * Σ_{n-1} * V^T
//
// where U_i are r_i×c matrices of singular vectors, Σ are c×c matrices singular values, and V
// is a c×c matrix of singular vectors.
//
// Factorize returns whether the decomposition succeeded. If the decomposition
// failed, routines that require a successful factorization will panic.
func (gsvd *HOGSVD) Factorize(m ...Matrix) (ok bool) {
	// Factorize performs the HOGSVD factorisation
	// essentially as described by Ponnapalli et al.
	// https://doi.org/10.1371/journal.pone.0028072

	if len(m) < 2 {
		panic("hogsvd: too few matrices")
	}
	gsvd.n = 0

	r, c := m[0].Dims()
	a := make([]Cholesky, len(m))
	var ts SymDense
	for i, d := range m {
		rd, cd := d.Dims()
		if rd < cd {
			gsvd.err = ErrShape
			return false
		}
		if rd > r {
			r = rd
		}
		if cd != c {
			panic(ErrShape)
		}
		ts.Reset()
		ts.SymOuterK(1, d.T())
		ok = a[i].Factorize(&ts)
		if !ok {
			gsvd.err = errors.New("hogsvd: cholesky decomposition failed")
			return false
		}
	}

	s := getWorkspace(c, c, true)
	defer putWorkspace(s)
	sij := getWorkspace(c, c, false)
	defer putWorkspace(sij)
	for i, ai := range a {
		for _, aj := range a[i+1:] {
			gsvd.err = ai.SolveChol(sij, &aj)
			if gsvd.err != nil {
				return false
			}
			s.Add(s, sij)

			gsvd.err = aj.SolveChol(sij, &ai)
			if gsvd.err != nil {
				return false
			}
			s.Add(s, sij)
		}
	}
	s.Scale(1/float64(len(m)*(len(m)-1)), s)

	var eig Eigen
	ok = eig.Factorize(s.T(), false, true)
	if !ok {
		gsvd.err = errors.New("hogsvd: eigen decomposition failed")
		return false
	}
	v := eig.Vectors()
	var cv VecDense
	for j := 0; j < c; j++ {
		cv.ColViewOf(v, j)
		cv.ScaleVec(1/blas64.Nrm2(c, cv.mat), &cv)
	}

	b := make([]Dense, len(m))
	biT := getWorkspace(c, r, false)
	defer putWorkspace(biT)
	for i, d := range m {
		// All calls to reset will leave a zeroed
		// matrix with capacity to store the result
		// without additional allocation.
		biT.Reset()
		gsvd.err = biT.Solve(v, d.T())
		if gsvd.err != nil {
			return false
		}
		b[i].Clone(biT.T())
	}

	gsvd.n = len(m)
	gsvd.v = v
	gsvd.b = b
	return true
}

// Err returns the reason for a factorization failure.
func (gsvd *HOGSVD) Err() error {
	return gsvd.err
}

// Len returns the number of matrices that have been factorized. If Len returns
// zero, the factorization was not successful.
func (gsvd *HOGSVD) Len() int {
	return gsvd.n
}

// UTo extracts the matrix U_n from the singular value decomposition, storing
// the result in-place into dst. U_n is size r×c.
// If dst is nil, a new matrix is allocated. The resulting U matrix is returned.
//
// UTo will panic if the receiver does not contain a successful factorization.
func (gsvd *HOGSVD) UTo(dst *Dense, n int) *Dense {
	if gsvd.n == 0 {
		panic("hogsvd: unsuccessful factorization")
	}
	if n < 0 || gsvd.n <= n {
		panic("hogsvd: invalid index")
	}

	if dst == nil {
		r, c := gsvd.b[n].Dims()
		dst = NewDense(r, c, nil)
	} else {
		dst.reuseAs(gsvd.b[n].Dims())
	}
	dst.Copy(&gsvd.b[n])
	var v VecDense
	for j, f := range gsvd.Values(nil, n) {
		v.ColViewOf(dst, j)
		v.ScaleVec(1/f, &v)
	}
	return dst
}

// Values returns the nth set of singular values of the factorized system.
// If the input slice is non-nil, the values will be stored in-place into the slice.
// In this case, the slice must have length c, and Values will panic with
// matrix.ErrSliceLengthMismatch otherwise. If the input slice is nil,
// a new slice of the appropriate length will be allocated and returned.
//
// Values will panic if the receiver does not contain a successful factorization.
func (gsvd *HOGSVD) Values(s []float64, n int) []float64 {
	if gsvd.n == 0 {
		panic("hogsvd: unsuccessful factorization")
	}
	if n < 0 || gsvd.n <= n {
		panic("hogsvd: invalid index")
	}

	r, c := gsvd.b[n].Dims()
	if s == nil {
		s = make([]float64, c)
	} else if len(s) != c {
		panic(ErrSliceLengthMismatch)
	}
	var v VecDense
	for j := 0; j < c; j++ {
		v.ColViewOf(&gsvd.b[n], j)
		s[j] = blas64.Nrm2(r, v.mat)
	}
	return s
}

// VTo extracts the matrix V from the singular value decomposition, storing
// the result in-place into dst. V is size c×c.
// If dst is nil, a new matrix is allocated. The resulting V matrix is returned.
//
// VTo will panic if the receiver does not contain a successful factorization.
func (gsvd *HOGSVD) VTo(dst *Dense) *Dense {
	if gsvd.n == 0 {
		panic("hogsvd: unsuccessful factorization")
	}
	if dst == nil {
		r, c := gsvd.v.Dims()
		dst = NewDense(r, c, nil)
	} else {
		dst.reuseAs(gsvd.v.Dims())
	}
	dst.Copy(gsvd.v)
	return dst
}
