// Copyright ©2013 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat

import (
	"math"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/lapack"
	"gonum.org/v1/gonum/lapack/lapack64"
)

const (
	badSliceLength = "mat: improper slice length"
	badLU          = "mat: invalid LU factorization"
)

// LU is a type for creating and using the LU factorization of a matrix.
type LU struct {
	lu    *Dense
	pivot []int
	cond  float64
}

// updateCond updates the stored condition number of the matrix. anorm is the
// norm of the original matrix. If anorm is negative it will be estimated.
func (lu *LU) updateCond(anorm float64, norm lapack.MatrixNorm) {
	n := lu.lu.mat.Cols
	work := getFloats(4*n, false)
	defer putFloats(work)
	iwork := getInts(n, false)
	defer putInts(iwork)
	if anorm < 0 {
		// This is an approximation. By the definition of a norm,
		//  |AB| <= |A| |B|.
		// Since A = L*U, we get for the condition number κ that
		//  κ(A) := |A| |A^-1| = |L*U| |A^-1| <= |L| |U| |A^-1|,
		// so this will overestimate the condition number somewhat.
		// The norm of the original factorized matrix cannot be stored
		// because of update possibilities.
		u := lu.lu.asTriDense(n, blas.NonUnit, blas.Upper)
		l := lu.lu.asTriDense(n, blas.Unit, blas.Lower)
		unorm := lapack64.Lantr(norm, u.mat, work)
		lnorm := lapack64.Lantr(norm, l.mat, work)
		anorm = unorm * lnorm
	}
	v := lapack64.Gecon(norm, lu.lu.mat, anorm, work, iwork)
	lu.cond = 1 / v
}

// Factorize computes the LU factorization of the square matrix a and stores the
// result. The LU decomposition will complete regardless of the singularity of a.
//
// The LU factorization is computed with pivoting, and so really the decomposition
// is a PLU decomposition where P is a permutation matrix. The individual matrix
// factors can be extracted from the factorization using the Permutation method
// on Dense, and the LU.LTo and LU.UTo methods.
func (lu *LU) Factorize(a Matrix) {
	lu.factorize(a, CondNorm)
}

func (lu *LU) factorize(a Matrix, norm lapack.MatrixNorm) {
	r, c := a.Dims()
	if r != c {
		panic(ErrSquare)
	}
	if lu.lu == nil {
		lu.lu = NewDense(r, r, nil)
	} else {
		lu.lu.Reset()
		lu.lu.reuseAsNonZeroed(r, r)
	}
	lu.lu.Copy(a)
	if cap(lu.pivot) < r {
		lu.pivot = make([]int, r)
	}
	lu.pivot = lu.pivot[:r]
	work := getFloats(r, false)
	anorm := lapack64.Lange(norm, lu.lu.mat, work)
	putFloats(work)
	lapack64.Getrf(lu.lu.mat, lu.pivot)
	lu.updateCond(anorm, norm)
}

// isValid returns whether the receiver contains a factorization.
func (lu *LU) isValid() bool {
	return lu.lu != nil && !lu.lu.IsEmpty()
}

// Cond returns the condition number for the factorized matrix.
// Cond will panic if the receiver does not contain a factorization.
func (lu *LU) Cond() float64 {
	if !lu.isValid() {
		panic(badLU)
	}
	return lu.cond
}

// Reset resets the factorization so that it can be reused as the receiver of a
// dimensionally restricted operation.
func (lu *LU) Reset() {
	if lu.lu != nil {
		lu.lu.Reset()
	}
	lu.pivot = lu.pivot[:0]
}

func (lu *LU) isZero() bool {
	return len(lu.pivot) == 0
}

// Det returns the determinant of the matrix that has been factorized. In many
// expressions, using LogDet will be more numerically stable.
// Det will panic if the receiver does not contain a factorization.
func (lu *LU) Det() float64 {
	det, sign := lu.LogDet()
	return math.Exp(det) * sign
}

// LogDet returns the log of the determinant and the sign of the determinant
// for the matrix that has been factorized. Numerical stability in product and
// division expressions is generally improved by working in log space.
// LogDet will panic if the receiver does not contain a factorization.
func (lu *LU) LogDet() (det float64, sign float64) {
	if !lu.isValid() {
		panic(badLU)
	}

	_, n := lu.lu.Dims()
	logDiag := getFloats(n, false)
	defer putFloats(logDiag)
	sign = 1.0
	for i := 0; i < n; i++ {
		v := lu.lu.at(i, i)
		if v < 0 {
			sign *= -1
		}
		if lu.pivot[i] != i {
			sign *= -1
		}
		logDiag[i] = math.Log(math.Abs(v))
	}
	return floats.Sum(logDiag), sign
}

// Pivot returns pivot indices that enable the construction of the permutation
// matrix P (see Dense.Permutation). If swaps == nil, then new memory will be
// allocated, otherwise the length of the input must be equal to the size of the
// factorized matrix.
// Pivot will panic if the receiver does not contain a factorization.
func (lu *LU) Pivot(swaps []int) []int {
	if !lu.isValid() {
		panic(badLU)
	}

	_, n := lu.lu.Dims()
	if swaps == nil {
		swaps = make([]int, n)
	}
	if len(swaps) != n {
		panic(badSliceLength)
	}
	// Perform the inverse of the row swaps in order to find the final
	// row swap position.
	for i := range swaps {
		swaps[i] = i
	}
	for i := n - 1; i >= 0; i-- {
		v := lu.pivot[i]
		swaps[i], swaps[v] = swaps[v], swaps[i]
	}
	return swaps
}

// RankOne updates an LU factorization as if a rank-one update had been applied to
// the original matrix A, storing the result into the receiver. That is, if in
// the original LU decomposition P * L * U = A, in the updated decomposition
// P * L * U = A + alpha * x * yᵀ.
// RankOne will panic if orig does not contain a factorization.
func (lu *LU) RankOne(orig *LU, alpha float64, x, y Vector) {
	if !orig.isValid() {
		panic(badLU)
	}

	// RankOne uses algorithm a1 on page 28 of "Multiple-Rank Updates to Matrix
	// Factorizations for Nonlinear Analysis and Circuit Design" by Linzhong Deng.
	// http://web.stanford.edu/group/SOL/dissertations/Linzhong-Deng-thesis.pdf
	_, n := orig.lu.Dims()
	if r, c := x.Dims(); r != n || c != 1 {
		panic(ErrShape)
	}
	if r, c := y.Dims(); r != n || c != 1 {
		panic(ErrShape)
	}
	if orig != lu {
		if lu.isZero() {
			if cap(lu.pivot) < n {
				lu.pivot = make([]int, n)
			}
			lu.pivot = lu.pivot[:n]
			if lu.lu == nil {
				lu.lu = NewDense(n, n, nil)
			} else {
				lu.lu.reuseAsNonZeroed(n, n)
			}
		} else if len(lu.pivot) != n {
			panic(ErrShape)
		}
		copy(lu.pivot, orig.pivot)
		lu.lu.Copy(orig.lu)
	}

	xs := getFloats(n, false)
	defer putFloats(xs)
	ys := getFloats(n, false)
	defer putFloats(ys)
	for i := 0; i < n; i++ {
		xs[i] = x.AtVec(i)
		ys[i] = y.AtVec(i)
	}

	// Adjust for the pivoting in the LU factorization
	for i, v := range lu.pivot {
		xs[i], xs[v] = xs[v], xs[i]
	}

	lum := lu.lu.mat
	omega := alpha
	for j := 0; j < n; j++ {
		ujj := lum.Data[j*lum.Stride+j]
		ys[j] /= ujj
		theta := 1 + xs[j]*ys[j]*omega
		beta := omega * ys[j] / theta
		gamma := omega * xs[j]
		omega -= beta * gamma
		lum.Data[j*lum.Stride+j] *= theta
		for i := j + 1; i < n; i++ {
			xs[i] -= lum.Data[i*lum.Stride+j] * xs[j]
			tmp := ys[i]
			ys[i] -= lum.Data[j*lum.Stride+i] * ys[j]
			lum.Data[i*lum.Stride+j] += beta * xs[i]
			lum.Data[j*lum.Stride+i] += gamma * tmp
		}
	}
	lu.updateCond(-1, CondNorm)
}

// LTo extracts the lower triangular matrix from an LU factorization.
//
// If dst is empty, LTo will resize dst to be a lower-triangular n×n matrix.
// When dst is non-empty, LTo will panic if dst is not n×n or not Lower.
// LTo will also panic if the receiver does not contain a successful
// factorization.
func (lu *LU) LTo(dst *TriDense) *TriDense {
	if !lu.isValid() {
		panic(badLU)
	}

	_, n := lu.lu.Dims()
	if dst.IsEmpty() {
		dst.ReuseAsTri(n, Lower)
	} else {
		n2, kind := dst.Triangle()
		if n != n2 {
			panic(ErrShape)
		}
		if kind != Lower {
			panic(ErrTriangle)
		}
	}
	// Extract the lower triangular elements.
	for i := 0; i < n; i++ {
		for j := 0; j < i; j++ {
			dst.mat.Data[i*dst.mat.Stride+j] = lu.lu.mat.Data[i*lu.lu.mat.Stride+j]
		}
	}
	// Set ones on the diagonal.
	for i := 0; i < n; i++ {
		dst.mat.Data[i*dst.mat.Stride+i] = 1
	}
	return dst
}

// UTo extracts the upper triangular matrix from an LU factorization.
//
// If dst is empty, UTo will resize dst to be an upper-triangular n×n matrix.
// When dst is non-empty, UTo will panic if dst is not n×n or not Upper.
// UTo will also panic if the receiver does not contain a successful
// factorization.
func (lu *LU) UTo(dst *TriDense) {
	if !lu.isValid() {
		panic(badLU)
	}

	_, n := lu.lu.Dims()
	if dst.IsEmpty() {
		dst.ReuseAsTri(n, Upper)
	} else {
		n2, kind := dst.Triangle()
		if n != n2 {
			panic(ErrShape)
		}
		if kind != Upper {
			panic(ErrTriangle)
		}
	}
	// Extract the upper triangular elements.
	for i := 0; i < n; i++ {
		for j := i; j < n; j++ {
			dst.mat.Data[i*dst.mat.Stride+j] = lu.lu.mat.Data[i*lu.lu.mat.Stride+j]
		}
	}
}

// Permutation constructs an r×r permutation matrix with the given row swaps.
// A permutation matrix has exactly one element equal to one in each row and column
// and all other elements equal to zero. swaps[i] specifies the row with which
// i will be swapped, which is equivalent to the non-zero column of row i.
func (m *Dense) Permutation(r int, swaps []int) {
	m.reuseAsNonZeroed(r, r)
	for i := 0; i < r; i++ {
		zero(m.mat.Data[i*m.mat.Stride : i*m.mat.Stride+r])
		v := swaps[i]
		if v < 0 || v >= r {
			panic(ErrRowAccess)
		}
		m.mat.Data[i*m.mat.Stride+v] = 1
	}
}

// SolveTo solves a system of linear equations using the LU decomposition of a matrix.
// It computes
//  A * X = B if trans == false
//  Aᵀ * X = B if trans == true
// In both cases, A is represented in LU factorized form, and the matrix X is
// stored into dst.
//
// If A is singular or near-singular a Condition error is returned. See
// the documentation for Condition for more information.
// SolveTo will panic if the receiver does not contain a factorization.
func (lu *LU) SolveTo(dst *Dense, trans bool, b Matrix) error {
	if !lu.isValid() {
		panic(badLU)
	}

	_, n := lu.lu.Dims()
	br, bc := b.Dims()
	if br != n {
		panic(ErrShape)
	}
	// TODO(btracey): Should test the condition number instead of testing that
	// the determinant is exactly zero.
	if lu.Det() == 0 {
		return Condition(math.Inf(1))
	}

	dst.reuseAsNonZeroed(n, bc)
	bU, _ := untranspose(b)
	var restore func()
	if dst == bU {
		dst, restore = dst.isolatedWorkspace(bU)
		defer restore()
	} else if rm, ok := bU.(RawMatrixer); ok {
		dst.checkOverlap(rm.RawMatrix())
	}

	dst.Copy(b)
	t := blas.NoTrans
	if trans {
		t = blas.Trans
	}
	lapack64.Getrs(t, lu.lu.mat, dst.mat, lu.pivot)
	if lu.cond > ConditionTolerance {
		return Condition(lu.cond)
	}
	return nil
}

// SolveVecTo solves a system of linear equations using the LU decomposition of a matrix.
// It computes
//  A * x = b if trans == false
//  Aᵀ * x = b if trans == true
// In both cases, A is represented in LU factorized form, and the vector x is
// stored into dst.
//
// If A is singular or near-singular a Condition error is returned. See
// the documentation for Condition for more information.
// SolveVecTo will panic if the receiver does not contain a factorization.
func (lu *LU) SolveVecTo(dst *VecDense, trans bool, b Vector) error {
	if !lu.isValid() {
		panic(badLU)
	}

	_, n := lu.lu.Dims()
	if br, bc := b.Dims(); br != n || bc != 1 {
		panic(ErrShape)
	}
	switch rv := b.(type) {
	default:
		dst.reuseAsNonZeroed(n)
		return lu.SolveTo(dst.asDense(), trans, b)
	case RawVectorer:
		if dst != b {
			dst.checkOverlap(rv.RawVector())
		}
		// TODO(btracey): Should test the condition number instead of testing that
		// the determinant is exactly zero.
		if lu.Det() == 0 {
			return Condition(math.Inf(1))
		}

		dst.reuseAsNonZeroed(n)
		var restore func()
		if dst == b {
			dst, restore = dst.isolatedWorkspace(b)
			defer restore()
		}
		dst.CopyVec(b)
		vMat := blas64.General{
			Rows:   n,
			Cols:   1,
			Stride: dst.mat.Inc,
			Data:   dst.mat.Data,
		}
		t := blas.NoTrans
		if trans {
			t = blas.Trans
		}
		lapack64.Getrs(t, lu.lu.mat, vMat, lu.pivot)
		if lu.cond > ConditionTolerance {
			return Condition(lu.cond)
		}
		return nil
	}
}
