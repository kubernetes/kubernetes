// Copyright ©2013 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat

import (
	"math"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/gonum/lapack/lapack64"
)

const (
	badTriangle = "mat: invalid triangle"
	badCholesky = "mat: invalid Cholesky factorization"
)

var (
	_ Matrix    = (*Cholesky)(nil)
	_ Symmetric = (*Cholesky)(nil)

	_ Matrix    = (*BandCholesky)(nil)
	_ Symmetric = (*BandCholesky)(nil)
	_ Banded    = (*BandCholesky)(nil)
	_ SymBanded = (*BandCholesky)(nil)
)

// Cholesky is a symmetric positive definite matrix represented by its
// Cholesky decomposition.
//
// The decomposition can be constructed using the Factorize method. The
// factorization itself can be extracted using the UTo or LTo methods, and the
// original symmetric matrix can be recovered with ToSym.
//
// Note that this matrix representation is useful for certain operations, in
// particular finding solutions to linear equations. It is very inefficient
// at other operations, in particular At is slow.
//
// Cholesky methods may only be called on a value that has been successfully
// initialized by a call to Factorize that has returned true. Calls to methods
// of an unsuccessful Cholesky factorization will panic.
type Cholesky struct {
	// The chol pointer must never be retained as a pointer outside the Cholesky
	// struct, either by returning chol outside the struct or by setting it to
	// a pointer coming from outside. The same prohibition applies to the data
	// slice within chol.
	chol *TriDense
	cond float64
}

// updateCond updates the condition number of the Cholesky decomposition. If
// norm > 0, then that norm is used as the norm of the original matrix A, otherwise
// the norm is estimated from the decomposition.
func (c *Cholesky) updateCond(norm float64) {
	n := c.chol.mat.N
	work := getFloats(3*n, false)
	defer putFloats(work)
	if norm < 0 {
		// This is an approximation. By the definition of a norm,
		//  |AB| <= |A| |B|.
		// Since A = Uᵀ*U, we get for the condition number κ that
		//  κ(A) := |A| |A^-1| = |Uᵀ*U| |A^-1| <= |Uᵀ| |U| |A^-1|,
		// so this will overestimate the condition number somewhat.
		// The norm of the original factorized matrix cannot be stored
		// because of update possibilities.
		unorm := lapack64.Lantr(CondNorm, c.chol.mat, work)
		lnorm := lapack64.Lantr(CondNormTrans, c.chol.mat, work)
		norm = unorm * lnorm
	}
	sym := c.chol.asSymBlas()
	iwork := getInts(n, false)
	v := lapack64.Pocon(sym, norm, work, iwork)
	putInts(iwork)
	c.cond = 1 / v
}

// Dims returns the dimensions of the matrix.
func (ch *Cholesky) Dims() (r, c int) {
	if !ch.valid() {
		panic(badCholesky)
	}
	r, c = ch.chol.Dims()
	return r, c
}

// At returns the element at row i, column j.
func (c *Cholesky) At(i, j int) float64 {
	if !c.valid() {
		panic(badCholesky)
	}
	n := c.Symmetric()
	if uint(i) >= uint(n) {
		panic(ErrRowAccess)
	}
	if uint(j) >= uint(n) {
		panic(ErrColAccess)
	}

	var val float64
	for k := 0; k <= min(i, j); k++ {
		val += c.chol.at(k, i) * c.chol.at(k, j)
	}
	return val
}

// T returns the receiver, the transpose of a symmetric matrix.
func (c *Cholesky) T() Matrix {
	return c
}

// Symmetric implements the Symmetric interface and returns the number of rows
// in the matrix (this is also the number of columns).
func (c *Cholesky) Symmetric() int {
	r, _ := c.chol.Dims()
	return r
}

// Cond returns the condition number of the factorized matrix.
func (c *Cholesky) Cond() float64 {
	if !c.valid() {
		panic(badCholesky)
	}
	return c.cond
}

// Factorize calculates the Cholesky decomposition of the matrix A and returns
// whether the matrix is positive definite. If Factorize returns false, the
// factorization must not be used.
func (c *Cholesky) Factorize(a Symmetric) (ok bool) {
	n := a.Symmetric()
	if c.chol == nil {
		c.chol = NewTriDense(n, Upper, nil)
	} else {
		c.chol.Reset()
		c.chol.reuseAsNonZeroed(n, Upper)
	}
	copySymIntoTriangle(c.chol, a)

	sym := c.chol.asSymBlas()
	work := getFloats(c.chol.mat.N, false)
	norm := lapack64.Lansy(CondNorm, sym, work)
	putFloats(work)
	_, ok = lapack64.Potrf(sym)
	if ok {
		c.updateCond(norm)
	} else {
		c.Reset()
	}
	return ok
}

// Reset resets the factorization so that it can be reused as the receiver of a
// dimensionally restricted operation.
func (c *Cholesky) Reset() {
	if c.chol != nil {
		c.chol.Reset()
	}
	c.cond = math.Inf(1)
}

// IsEmpty returns whether the receiver is empty. Empty matrices can be the
// receiver for size-restricted operations. The receiver can be emptied using
// Reset.
func (c *Cholesky) IsEmpty() bool {
	return c.chol == nil || c.chol.IsEmpty()
}

// SetFromU sets the Cholesky decomposition from the given triangular matrix.
// SetFromU panics if t is not upper triangular. If the receiver is empty it
// is resized to be n×n, the size of t. If dst is non-empty, SetFromU panics
// if c is not of size n×n. Note that t is copied into, not stored inside, the
// receiver.
func (c *Cholesky) SetFromU(t Triangular) {
	n, kind := t.Triangle()
	if kind != Upper {
		panic("cholesky: matrix must be upper triangular")
	}
	if c.chol == nil {
		c.chol = NewTriDense(n, Upper, nil)
	} else {
		c.chol.reuseAsNonZeroed(n, Upper)
	}
	c.chol.Copy(t)
	c.updateCond(-1)
}

// Clone makes a copy of the input Cholesky into the receiver, overwriting the
// previous value of the receiver. Clone does not place any restrictions on receiver
// shape. Clone panics if the input Cholesky is not the result of a valid decomposition.
func (c *Cholesky) Clone(chol *Cholesky) {
	if !chol.valid() {
		panic(badCholesky)
	}
	n := chol.Symmetric()
	if c.chol == nil {
		c.chol = NewTriDense(n, Upper, nil)
	} else {
		c.chol = NewTriDense(n, Upper, use(c.chol.mat.Data, n*n))
	}
	c.chol.Copy(chol.chol)
	c.cond = chol.cond
}

// Det returns the determinant of the matrix that has been factorized.
func (c *Cholesky) Det() float64 {
	if !c.valid() {
		panic(badCholesky)
	}
	return math.Exp(c.LogDet())
}

// LogDet returns the log of the determinant of the matrix that has been factorized.
func (c *Cholesky) LogDet() float64 {
	if !c.valid() {
		panic(badCholesky)
	}
	var det float64
	for i := 0; i < c.chol.mat.N; i++ {
		det += 2 * math.Log(c.chol.mat.Data[i*c.chol.mat.Stride+i])
	}
	return det
}

// SolveTo finds the matrix X that solves A * X = B where A is represented
// by the Cholesky decomposition. The result is stored in-place into dst.
// If the Cholesky decomposition is singular or near-singular a Condition error
// is returned. See the documentation for Condition for more information.
func (c *Cholesky) SolveTo(dst *Dense, b Matrix) error {
	if !c.valid() {
		panic(badCholesky)
	}
	n := c.chol.mat.N
	bm, bn := b.Dims()
	if n != bm {
		panic(ErrShape)
	}

	dst.reuseAsNonZeroed(bm, bn)
	if b != dst {
		dst.Copy(b)
	}
	lapack64.Potrs(c.chol.mat, dst.mat)
	if c.cond > ConditionTolerance {
		return Condition(c.cond)
	}
	return nil
}

// SolveCholTo finds the matrix X that solves A * X = B where A and B are represented
// by their Cholesky decompositions a and b. The result is stored in-place into
// dst.
// If the Cholesky decomposition is singular or near-singular a Condition error
// is returned. See the documentation for Condition for more information.
func (a *Cholesky) SolveCholTo(dst *Dense, b *Cholesky) error {
	if !a.valid() || !b.valid() {
		panic(badCholesky)
	}
	bn := b.chol.mat.N
	if a.chol.mat.N != bn {
		panic(ErrShape)
	}

	dst.reuseAsZeroed(bn, bn)
	dst.Copy(b.chol.T())
	blas64.Trsm(blas.Left, blas.Trans, 1, a.chol.mat, dst.mat)
	blas64.Trsm(blas.Left, blas.NoTrans, 1, a.chol.mat, dst.mat)
	blas64.Trmm(blas.Right, blas.NoTrans, 1, b.chol.mat, dst.mat)
	if a.cond > ConditionTolerance {
		return Condition(a.cond)
	}
	return nil
}

// SolveVecTo finds the vector x that solves A * x = b where A is represented
// by the Cholesky decomposition. The result is stored in-place into
// dst.
// If the Cholesky decomposition is singular or near-singular a Condition error
// is returned. See the documentation for Condition for more information.
func (c *Cholesky) SolveVecTo(dst *VecDense, b Vector) error {
	if !c.valid() {
		panic(badCholesky)
	}
	n := c.chol.mat.N
	if br, bc := b.Dims(); br != n || bc != 1 {
		panic(ErrShape)
	}
	switch rv := b.(type) {
	default:
		dst.reuseAsNonZeroed(n)
		return c.SolveTo(dst.asDense(), b)
	case RawVectorer:
		bmat := rv.RawVector()
		if dst != b {
			dst.checkOverlap(bmat)
		}
		dst.reuseAsNonZeroed(n)
		if dst != b {
			dst.CopyVec(b)
		}
		lapack64.Potrs(c.chol.mat, dst.asGeneral())
		if c.cond > ConditionTolerance {
			return Condition(c.cond)
		}
		return nil
	}
}

// RawU returns the Triangular matrix used to store the Cholesky decomposition of
// the original matrix A. The returned matrix should not be modified. If it is
// modified, the decomposition is invalid and should not be used.
func (c *Cholesky) RawU() Triangular {
	return c.chol
}

// UTo stores into dst the n×n upper triangular matrix U from a Cholesky
// decomposition
//  A = Uᵀ * U.
// If dst is empty, it is resized to be an n×n upper triangular matrix. When dst
// is non-empty, UTo panics if dst is not n×n or not Upper. UTo will also panic
// if the receiver does not contain a successful factorization.
func (c *Cholesky) UTo(dst *TriDense) {
	if !c.valid() {
		panic(badCholesky)
	}
	n := c.chol.mat.N
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
	dst.Copy(c.chol)
}

// LTo stores into dst the n×n lower triangular matrix L from a Cholesky
// decomposition
//  A = L * Lᵀ.
// If dst is empty, it is resized to be an n×n lower triangular matrix. When dst
// is non-empty, LTo panics if dst is not n×n or not Lower. LTo will also panic
// if the receiver does not contain a successful factorization.
func (c *Cholesky) LTo(dst *TriDense) {
	if !c.valid() {
		panic(badCholesky)
	}
	n := c.chol.mat.N
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
	dst.Copy(c.chol.TTri())
}

// ToSym reconstructs the original positive definite matrix from its
// Cholesky decomposition, storing the result into dst. If dst is
// empty it is resized to be n×n. If dst is non-empty, ToSym panics
// if dst is not of size n×n. ToSym will also panic if the receiver
// does not contain a successful factorization.
func (c *Cholesky) ToSym(dst *SymDense) {
	if !c.valid() {
		panic(badCholesky)
	}
	n := c.chol.mat.N
	if dst.IsEmpty() {
		dst.ReuseAsSym(n)
	} else {
		n2 := dst.Symmetric()
		if n != n2 {
			panic(ErrShape)
		}
	}
	// Create a TriDense representing the Cholesky factor U with dst's
	// backing slice.
	// Operations on u are reflected in s.
	u := &TriDense{
		mat: blas64.Triangular{
			Uplo:   blas.Upper,
			Diag:   blas.NonUnit,
			N:      n,
			Data:   dst.mat.Data,
			Stride: dst.mat.Stride,
		},
		cap: n,
	}
	u.Copy(c.chol)
	// Compute the product Uᵀ*U using the algorithm from LAPACK/TESTING/LIN/dpot01.f
	a := u.mat.Data
	lda := u.mat.Stride
	bi := blas64.Implementation()
	for k := n - 1; k >= 0; k-- {
		a[k*lda+k] = bi.Ddot(k+1, a[k:], lda, a[k:], lda)
		if k > 0 {
			bi.Dtrmv(blas.Upper, blas.Trans, blas.NonUnit, k, a, lda, a[k:], lda)
		}
	}
}

// InverseTo computes the inverse of the matrix represented by its Cholesky
// factorization and stores the result into s. If the factorized
// matrix is ill-conditioned, a Condition error will be returned.
// Note that matrix inversion is numerically unstable, and should generally be
// avoided where possible, for example by using the Solve routines.
func (c *Cholesky) InverseTo(dst *SymDense) error {
	if !c.valid() {
		panic(badCholesky)
	}
	dst.reuseAsNonZeroed(c.chol.mat.N)
	// Create a TriDense representing the Cholesky factor U with the backing
	// slice from dst.
	// Operations on u are reflected in dst.
	u := &TriDense{
		mat: blas64.Triangular{
			Uplo:   blas.Upper,
			Diag:   blas.NonUnit,
			N:      dst.mat.N,
			Data:   dst.mat.Data,
			Stride: dst.mat.Stride,
		},
		cap: dst.mat.N,
	}
	u.Copy(c.chol)

	_, ok := lapack64.Potri(u.mat)
	if !ok {
		return Condition(math.Inf(1))
	}
	if c.cond > ConditionTolerance {
		return Condition(c.cond)
	}
	return nil
}

// Scale multiplies the original matrix A by a positive constant using
// its Cholesky decomposition, storing the result in-place into the receiver.
// That is, if the original Cholesky factorization is
//  Uᵀ * U = A
// the updated factorization is
//  U'ᵀ * U' = f A = A'
// Scale panics if the constant is non-positive, or if the receiver is non-empty
// and is of a different size from the input.
func (c *Cholesky) Scale(f float64, orig *Cholesky) {
	if !orig.valid() {
		panic(badCholesky)
	}
	if f <= 0 {
		panic("cholesky: scaling by a non-positive constant")
	}
	n := orig.Symmetric()
	if c.chol == nil {
		c.chol = NewTriDense(n, Upper, nil)
	} else if c.chol.mat.N != n {
		panic(ErrShape)
	}
	c.chol.ScaleTri(math.Sqrt(f), orig.chol)
	c.cond = orig.cond // Scaling by a positive constant does not change the condition number.
}

// ExtendVecSym computes the Cholesky decomposition of the original matrix A,
// whose Cholesky decomposition is in a, extended by a the n×1 vector v according to
//  [A  w]
//  [w' k]
// where k = v[n-1] and w = v[:n-1]. The result is stored into the receiver.
// In order for the updated matrix to be positive definite, it must be the case
// that k > w' A^-1 w. If this condition does not hold then ExtendVecSym will
// return false and the receiver will not be updated.
//
// ExtendVecSym will panic if v.Len() != a.Symmetric()+1 or if a does not contain
// a valid decomposition.
func (c *Cholesky) ExtendVecSym(a *Cholesky, v Vector) (ok bool) {
	n := a.Symmetric()

	if v.Len() != n+1 {
		panic(badSliceLength)
	}
	if !a.valid() {
		panic(badCholesky)
	}

	// The algorithm is commented here, but see also
	//  https://math.stackexchange.com/questions/955874/cholesky-factor-when-adding-a-row-and-column-to-already-factorized-matrix
	// We have A and want to compute the Cholesky of
	//  [A  w]
	//  [w' k]
	// We want
	//  [U c]
	//  [0 d]
	// to be the updated Cholesky, and so it must be that
	//  [A  w] = [U' 0] [U c]
	//  [w' k]   [c' d] [0 d]
	// Thus, we need
	//  1) A = U'U (true by the original decomposition being valid),
	//  2) U' * c = w  =>  c = U'^-1 w
	//  3) c'*c + d'*d = k  =>  d = sqrt(k-c'*c)

	// First, compute c = U'^-1 a
	w := NewVecDense(n, nil)
	w.CopyVec(v)
	k := v.At(n, 0)

	var t VecDense
	_ = t.SolveVec(a.chol.T(), w)

	dot := Dot(&t, &t)
	if dot >= k {
		return false
	}
	d := math.Sqrt(k - dot)

	newU := NewTriDense(n+1, Upper, nil)
	newU.Copy(a.chol)
	for i := 0; i < n; i++ {
		newU.SetTri(i, n, t.At(i, 0))
	}
	newU.SetTri(n, n, d)
	c.chol = newU
	c.updateCond(-1)
	return true
}

// SymRankOne performs a rank-1 update of the original matrix A and refactorizes
// its Cholesky factorization, storing the result into the receiver. That is, if
// in the original Cholesky factorization
//  Uᵀ * U = A,
// in the updated factorization
//  U'ᵀ * U' = A + alpha * x * xᵀ = A'.
//
// Note that when alpha is negative, the updating problem may be ill-conditioned
// and the results may be inaccurate, or the updated matrix A' may not be
// positive definite and not have a Cholesky factorization. SymRankOne returns
// whether the updated matrix A' is positive definite. If the update fails
// the receiver is left unchanged.
//
// SymRankOne updates a Cholesky factorization in O(n²) time. The Cholesky
// factorization computation from scratch is O(n³).
func (c *Cholesky) SymRankOne(orig *Cholesky, alpha float64, x Vector) (ok bool) {
	if !orig.valid() {
		panic(badCholesky)
	}
	n := orig.Symmetric()
	if r, c := x.Dims(); r != n || c != 1 {
		panic(ErrShape)
	}
	if orig != c {
		if c.chol == nil {
			c.chol = NewTriDense(n, Upper, nil)
		} else if c.chol.mat.N != n {
			panic(ErrShape)
		}
		c.chol.Copy(orig.chol)
	}

	if alpha == 0 {
		return true
	}

	// Algorithms for updating and downdating the Cholesky factorization are
	// described, for example, in
	// - J. J. Dongarra, J. R. Bunch, C. B. Moler, G. W. Stewart: LINPACK
	//   Users' Guide. SIAM (1979), pages 10.10--10.14
	// or
	// - P. E. Gill, G. H. Golub, W. Murray, and M. A. Saunders: Methods for
	//   modifying matrix factorizations. Mathematics of Computation 28(126)
	//   (1974), Method C3 on page 521
	//
	// The implementation is based on LINPACK code
	// http://www.netlib.org/linpack/dchud.f
	// http://www.netlib.org/linpack/dchdd.f
	// and
	// https://icl.cs.utk.edu/lapack-forum/viewtopic.php?f=2&t=2646
	//
	// According to http://icl.cs.utk.edu/lapack-forum/archives/lapack/msg00301.html
	// LINPACK is released under BSD license.
	//
	// See also:
	// - M. A. Saunders: Large-scale Linear Programming Using the Cholesky
	//   Factorization. Technical Report Stanford University (1972)
	//   http://i.stanford.edu/pub/cstr/reports/cs/tr/72/252/CS-TR-72-252.pdf
	// - Matthias Seeger: Low rank updates for the Cholesky decomposition.
	//   EPFL Technical Report 161468 (2004)
	//   http://infoscience.epfl.ch/record/161468

	work := getFloats(n, false)
	defer putFloats(work)
	var xmat blas64.Vector
	if rv, ok := x.(RawVectorer); ok {
		xmat = rv.RawVector()
	} else {
		var tmp *VecDense
		tmp.CopyVec(x)
		xmat = tmp.RawVector()
	}
	blas64.Copy(xmat, blas64.Vector{N: n, Data: work, Inc: 1})

	if alpha > 0 {
		// Compute rank-1 update.
		if alpha != 1 {
			blas64.Scal(math.Sqrt(alpha), blas64.Vector{N: n, Data: work, Inc: 1})
		}
		umat := c.chol.mat
		stride := umat.Stride
		for i := 0; i < n; i++ {
			// Compute parameters of the Givens matrix that zeroes
			// the i-th element of x.
			c, s, r, _ := blas64.Rotg(umat.Data[i*stride+i], work[i])
			if r < 0 {
				// Multiply by -1 to have positive diagonal
				// elemnts.
				r *= -1
				c *= -1
				s *= -1
			}
			umat.Data[i*stride+i] = r
			if i < n-1 {
				// Multiply the extended factorization matrix by
				// the Givens matrix from the left. Only
				// the i-th row and x are modified.
				blas64.Rot(
					blas64.Vector{N: n - i - 1, Data: umat.Data[i*stride+i+1 : i*stride+n], Inc: 1},
					blas64.Vector{N: n - i - 1, Data: work[i+1 : n], Inc: 1},
					c, s)
			}
		}
		c.updateCond(-1)
		return true
	}

	// Compute rank-1 downdate.
	alpha = math.Sqrt(-alpha)
	if alpha != 1 {
		blas64.Scal(alpha, blas64.Vector{N: n, Data: work, Inc: 1})
	}
	// Solve Uᵀ * p = x storing the result into work.
	ok = lapack64.Trtrs(blas.Trans, c.chol.RawTriangular(), blas64.General{
		Rows:   n,
		Cols:   1,
		Stride: 1,
		Data:   work,
	})
	if !ok {
		// The original matrix is singular. Should not happen, because
		// the factorization is valid.
		panic(badCholesky)
	}
	norm := blas64.Nrm2(blas64.Vector{N: n, Data: work, Inc: 1})
	if norm >= 1 {
		// The updated matrix is not positive definite.
		return false
	}
	norm = math.Sqrt((1 + norm) * (1 - norm))
	cos := getFloats(n, false)
	defer putFloats(cos)
	sin := getFloats(n, false)
	defer putFloats(sin)
	for i := n - 1; i >= 0; i-- {
		// Compute parameters of Givens matrices that zero elements of p
		// backwards.
		cos[i], sin[i], norm, _ = blas64.Rotg(norm, work[i])
		if norm < 0 {
			norm *= -1
			cos[i] *= -1
			sin[i] *= -1
		}
	}
	workMat := getWorkspaceTri(c.chol.mat.N, c.chol.triKind(), false)
	defer putWorkspaceTri(workMat)
	workMat.Copy(c.chol)
	umat := workMat.mat
	stride := workMat.mat.Stride
	for i := n - 1; i >= 0; i-- {
		work[i] = 0
		// Apply Givens matrices to U.
		blas64.Rot(
			blas64.Vector{N: n - i, Data: work[i:n], Inc: 1},
			blas64.Vector{N: n - i, Data: umat.Data[i*stride+i : i*stride+n], Inc: 1},
			cos[i], sin[i])
		if umat.Data[i*stride+i] == 0 {
			// The matrix is singular (may rarely happen due to
			// floating-point effects?).
			ok = false
		} else if umat.Data[i*stride+i] < 0 {
			// Diagonal elements should be positive. If it happens
			// that on the i-th row the diagonal is negative,
			// multiply U from the left by an identity matrix that
			// has -1 on the i-th row.
			blas64.Scal(-1, blas64.Vector{N: n - i, Data: umat.Data[i*stride+i : i*stride+n], Inc: 1})
		}
	}
	if ok {
		c.chol.Copy(workMat)
		c.updateCond(-1)
	}
	return ok
}

func (c *Cholesky) valid() bool {
	return c.chol != nil && !c.chol.IsEmpty()
}

// BandCholesky is a symmetric positive-definite band matrix represented by its
// Cholesky decomposition.
//
// Note that this matrix representation is useful for certain operations, in
// particular finding solutions to linear equations. It is very inefficient at
// other operations, in particular At is slow.
//
// BandCholesky methods may only be called on a value that has been successfully
// initialized by a call to Factorize that has returned true. Calls to methods
// of an unsuccessful Cholesky factorization will panic.
type BandCholesky struct {
	// The chol pointer must never be retained as a pointer outside the Cholesky
	// struct, either by returning chol outside the struct or by setting it to
	// a pointer coming from outside. The same prohibition applies to the data
	// slice within chol.
	chol *TriBandDense
	cond float64
}

// Factorize calculates the Cholesky decomposition of the matrix A and returns
// whether the matrix is positive definite. If Factorize returns false, the
// factorization must not be used.
func (ch *BandCholesky) Factorize(a SymBanded) (ok bool) {
	n, k := a.SymBand()
	if ch.chol == nil {
		ch.chol = NewTriBandDense(n, k, Upper, nil)
	} else {
		ch.chol.Reset()
		ch.chol.ReuseAsTriBand(n, k, Upper)
	}
	copySymBandIntoTriBand(ch.chol, a)
	cSym := blas64.SymmetricBand{
		Uplo:   blas.Upper,
		N:      n,
		K:      k,
		Data:   ch.chol.RawTriBand().Data,
		Stride: ch.chol.RawTriBand().Stride,
	}
	_, ok = lapack64.Pbtrf(cSym)
	if !ok {
		ch.Reset()
		return false
	}
	work := getFloats(3*n, false)
	iwork := getInts(n, false)
	aNorm := lapack64.Lansb(CondNorm, cSym, work)
	ch.cond = 1 / lapack64.Pbcon(cSym, aNorm, work, iwork)
	putInts(iwork)
	putFloats(work)
	return true
}

// SolveTo finds the matrix X that solves A * X = B where A is represented by
// the Cholesky decomposition. The result is stored in-place into dst.
// If the Cholesky decomposition is singular or near-singular a Condition error
// is returned. See the documentation for Condition for more information.
func (ch *BandCholesky) SolveTo(dst *Dense, b Matrix) error {
	if !ch.valid() {
		panic(badCholesky)
	}
	br, bc := b.Dims()
	if br != ch.chol.mat.N {
		panic(ErrShape)
	}
	dst.reuseAsNonZeroed(br, bc)
	if b != dst {
		dst.Copy(b)
	}
	lapack64.Pbtrs(ch.chol.mat, dst.mat)
	if ch.cond > ConditionTolerance {
		return Condition(ch.cond)
	}
	return nil
}

// SolveVecTo finds the vector x that solves A * x = b where A is represented by
// the Cholesky decomposition. The result is stored in-place into dst.
// If the Cholesky decomposition is singular or near-singular a Condition error
// is returned. See the documentation for Condition for more information.
func (ch *BandCholesky) SolveVecTo(dst *VecDense, b Vector) error {
	if !ch.valid() {
		panic(badCholesky)
	}
	n := ch.chol.mat.N
	if br, bc := b.Dims(); br != n || bc != 1 {
		panic(ErrShape)
	}
	if b, ok := b.(RawVectorer); ok && dst != b {
		dst.checkOverlap(b.RawVector())
	}
	dst.reuseAsNonZeroed(n)
	if dst != b {
		dst.CopyVec(b)
	}
	lapack64.Pbtrs(ch.chol.mat, dst.asGeneral())
	if ch.cond > ConditionTolerance {
		return Condition(ch.cond)
	}
	return nil
}

// Cond returns the condition number of the factorized matrix.
func (ch *BandCholesky) Cond() float64 {
	if !ch.valid() {
		panic(badCholesky)
	}
	return ch.cond
}

// Reset resets the factorization so that it can be reused as the receiver of
// a dimensionally restricted operation.
func (ch *BandCholesky) Reset() {
	if ch.chol != nil {
		ch.chol.Reset()
	}
	ch.cond = math.Inf(1)
}

// Dims returns the dimensions of the matrix.
func (ch *BandCholesky) Dims() (r, c int) {
	if !ch.valid() {
		panic(badCholesky)
	}
	r, c = ch.chol.Dims()
	return r, c
}

// At returns the element at row i, column j.
func (ch *BandCholesky) At(i, j int) float64 {
	if !ch.valid() {
		panic(badCholesky)
	}
	n, k, _ := ch.chol.TriBand()
	if uint(i) >= uint(n) {
		panic(ErrRowAccess)
	}
	if uint(j) >= uint(n) {
		panic(ErrColAccess)
	}

	if i > j {
		i, j = j, i
	}
	if j-i > k {
		return 0
	}
	var aij float64
	for k := max(0, j-k); k <= i; k++ {
		aij += ch.chol.at(k, i) * ch.chol.at(k, j)
	}
	return aij
}

// T returns the receiver, the transpose of a symmetric matrix.
func (ch *BandCholesky) T() Matrix {
	return ch
}

// TBand returns the receiver, the transpose of a symmetric band matrix.
func (ch *BandCholesky) TBand() Banded {
	return ch
}

// Symmetric implements the Symmetric interface and returns the number of rows
// in the matrix (this is also the number of columns).
func (ch *BandCholesky) Symmetric() int {
	n, _ := ch.chol.Triangle()
	return n
}

// Bandwidth returns the lower and upper bandwidth values for the matrix.
// The total bandwidth of the matrix is kl+ku+1.
func (ch *BandCholesky) Bandwidth() (kl, ku int) {
	_, k, _ := ch.chol.TriBand()
	return k, k
}

// SymBand returns the number of rows/columns in the matrix, and the size of the
// bandwidth. The total bandwidth of the matrix is 2*k+1.
func (ch *BandCholesky) SymBand() (n, k int) {
	n, k, _ = ch.chol.TriBand()
	return n, k
}

// IsEmpty returns whether the receiver is empty. Empty matrices can be the
// receiver for dimensionally restricted operations. The receiver can be emptied
// using Reset.
func (ch *BandCholesky) IsEmpty() bool {
	return ch == nil || ch.chol.IsEmpty()
}

// Det returns the determinant of the matrix that has been factorized.
func (ch *BandCholesky) Det() float64 {
	if !ch.valid() {
		panic(badCholesky)
	}
	return math.Exp(ch.LogDet())
}

// LogDet returns the log of the determinant of the matrix that has been factorized.
func (ch *BandCholesky) LogDet() float64 {
	if !ch.valid() {
		panic(badCholesky)
	}
	var det float64
	for i := 0; i < ch.chol.mat.N; i++ {
		det += 2 * math.Log(ch.chol.mat.Data[i*ch.chol.mat.Stride])
	}
	return det
}

func (ch *BandCholesky) valid() bool {
	return ch.chol != nil && !ch.chol.IsEmpty()
}
