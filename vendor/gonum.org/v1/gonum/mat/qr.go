// Copyright ©2013 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat

import (
	"math"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/gonum/lapack"
	"gonum.org/v1/gonum/lapack/lapack64"
)

const badQR = "mat: invalid QR factorization"

// QR is a type for creating and using the QR factorization of a matrix.
type QR struct {
	qr   *Dense
	tau  []float64
	cond float64
}

func (qr *QR) updateCond(norm lapack.MatrixNorm) {
	// Since A = Q*R, and Q is orthogonal, we get for the condition number κ
	//  κ(A) := |A| |A^-1| = |Q*R| |(Q*R)^-1| = |R| |R^-1 * Q^T|
	//        = |R| |R^-1| = κ(R),
	// where we used that fact that Q^-1 = Q^T. However, this assumes that
	// the matrix norm is invariant under orthogonal transformations which
	// is not the case for CondNorm. Hopefully the error is negligible: κ
	// is only a qualitative measure anyway.
	n := qr.qr.mat.Cols
	work := getFloats(3*n, false)
	iwork := getInts(n, false)
	r := qr.qr.asTriDense(n, blas.NonUnit, blas.Upper)
	v := lapack64.Trcon(norm, r.mat, work, iwork)
	putFloats(work)
	putInts(iwork)
	qr.cond = 1 / v
}

// Factorize computes the QR factorization of an m×n matrix a where m >= n. The QR
// factorization always exists even if A is singular.
//
// The QR decomposition is a factorization of the matrix A such that A = Q * R.
// The matrix Q is an orthonormal m×m matrix, and R is an m×n upper triangular matrix.
// Q and R can be extracted using the QTo and RTo methods.
func (qr *QR) Factorize(a Matrix) {
	qr.factorize(a, CondNorm)
}

func (qr *QR) factorize(a Matrix, norm lapack.MatrixNorm) {
	m, n := a.Dims()
	if m < n {
		panic(ErrShape)
	}
	k := min(m, n)
	if qr.qr == nil {
		qr.qr = &Dense{}
	}
	qr.qr.Clone(a)
	work := []float64{0}
	qr.tau = make([]float64, k)
	lapack64.Geqrf(qr.qr.mat, qr.tau, work, -1)

	work = getFloats(int(work[0]), false)
	lapack64.Geqrf(qr.qr.mat, qr.tau, work, len(work))
	putFloats(work)
	qr.updateCond(norm)
}

// isValid returns whether the receiver contains a factorization.
func (qr *QR) isValid() bool {
	return qr.qr != nil && !qr.qr.IsZero()
}

// Cond returns the condition number for the factorized matrix.
// Cond will panic if the receiver does not contain a factorization.
func (qr *QR) Cond() float64 {
	if !qr.isValid() {
		panic(badQR)
	}
	return qr.cond
}

// TODO(btracey): Add in the "Reduced" forms for extracting the n×n orthogonal
// and upper triangular matrices.

// RTo extracts the m×n upper trapezoidal matrix from a QR decomposition.
// If dst is nil, a new matrix is allocated. The resulting dst matrix is returned.
// RTo will panic if the receiver does not contain a factorization.
func (qr *QR) RTo(dst *Dense) *Dense {
	if !qr.isValid() {
		panic(badQR)
	}

	r, c := qr.qr.Dims()
	if dst == nil {
		dst = NewDense(r, c, nil)
	} else {
		dst.reuseAs(r, c)
	}

	// Disguise the QR as an upper triangular
	t := &TriDense{
		mat: blas64.Triangular{
			N:      c,
			Stride: qr.qr.mat.Stride,
			Data:   qr.qr.mat.Data,
			Uplo:   blas.Upper,
			Diag:   blas.NonUnit,
		},
		cap: qr.qr.capCols,
	}
	dst.Copy(t)

	// Zero below the triangular.
	for i := r; i < c; i++ {
		zero(dst.mat.Data[i*dst.mat.Stride : i*dst.mat.Stride+c])
	}

	return dst
}

// QTo extracts the m×m orthonormal matrix Q from a QR decomposition.
// If dst is nil, a new matrix is allocated. The resulting Q matrix is returned.
// QTo will panic if the receiver does not contain a factorization.
func (qr *QR) QTo(dst *Dense) *Dense {
	if !qr.isValid() {
		panic(badQR)
	}

	r, _ := qr.qr.Dims()
	if dst == nil {
		dst = NewDense(r, r, nil)
	} else {
		dst.reuseAsZeroed(r, r)
	}

	// Set Q = I.
	for i := 0; i < r*r; i += r + 1 {
		dst.mat.Data[i] = 1
	}

	// Construct Q from the elementary reflectors.
	work := []float64{0}
	lapack64.Ormqr(blas.Left, blas.NoTrans, qr.qr.mat, qr.tau, dst.mat, work, -1)
	work = getFloats(int(work[0]), false)
	lapack64.Ormqr(blas.Left, blas.NoTrans, qr.qr.mat, qr.tau, dst.mat, work, len(work))
	putFloats(work)

	return dst
}

// SolveTo finds a minimum-norm solution to a system of linear equations defined
// by the matrices A and b, where A is an m×n matrix represented in its QR factorized
// form. If A is singular or near-singular a Condition error is returned.
// See the documentation for Condition for more information.
//
// The minimization problem solved depends on the input parameters.
//  If trans == false, find X such that ||A*X - B||_2 is minimized.
//  If trans == true, find the minimum norm solution of A^T * X = B.
// The solution matrix, X, is stored in place into dst.
// SolveTo will panic if the receiver does not contain a factorization.
func (qr *QR) SolveTo(dst *Dense, trans bool, b Matrix) error {
	if !qr.isValid() {
		panic(badQR)
	}

	r, c := qr.qr.Dims()
	br, bc := b.Dims()

	// The QR solve algorithm stores the result in-place into the right hand side.
	// The storage for the answer must be large enough to hold both b and x.
	// However, this method's receiver must be the size of x. Copy b, and then
	// copy the result into m at the end.
	if trans {
		if c != br {
			panic(ErrShape)
		}
		dst.reuseAs(r, bc)
	} else {
		if r != br {
			panic(ErrShape)
		}
		dst.reuseAs(c, bc)
	}
	// Do not need to worry about overlap between m and b because x has its own
	// independent storage.
	w := getWorkspace(max(r, c), bc, false)
	w.Copy(b)
	t := qr.qr.asTriDense(qr.qr.mat.Cols, blas.NonUnit, blas.Upper).mat
	if trans {
		ok := lapack64.Trtrs(blas.Trans, t, w.mat)
		if !ok {
			return Condition(math.Inf(1))
		}
		for i := c; i < r; i++ {
			zero(w.mat.Data[i*w.mat.Stride : i*w.mat.Stride+bc])
		}
		work := []float64{0}
		lapack64.Ormqr(blas.Left, blas.NoTrans, qr.qr.mat, qr.tau, w.mat, work, -1)
		work = getFloats(int(work[0]), false)
		lapack64.Ormqr(blas.Left, blas.NoTrans, qr.qr.mat, qr.tau, w.mat, work, len(work))
		putFloats(work)
	} else {
		work := []float64{0}
		lapack64.Ormqr(blas.Left, blas.Trans, qr.qr.mat, qr.tau, w.mat, work, -1)
		work = getFloats(int(work[0]), false)
		lapack64.Ormqr(blas.Left, blas.Trans, qr.qr.mat, qr.tau, w.mat, work, len(work))
		putFloats(work)

		ok := lapack64.Trtrs(blas.NoTrans, t, w.mat)
		if !ok {
			return Condition(math.Inf(1))
		}
	}
	// X was set above to be the correct size for the result.
	dst.Copy(w)
	putWorkspace(w)
	if qr.cond > ConditionTolerance {
		return Condition(qr.cond)
	}
	return nil
}

// SolveVecTo finds a minimum-norm solution to a system of linear equations,
//  Ax = b.
// See QR.SolveTo for the full documentation.
// SolveVecTo will panic if the receiver does not contain a factorization.
func (qr *QR) SolveVecTo(dst *VecDense, trans bool, b Vector) error {
	if !qr.isValid() {
		panic(badQR)
	}

	r, c := qr.qr.Dims()
	if _, bc := b.Dims(); bc != 1 {
		panic(ErrShape)
	}

	// The Solve implementation is non-trivial, so rather than duplicate the code,
	// instead recast the VecDenses as Dense and call the matrix code.
	bm := Matrix(b)
	if rv, ok := b.(RawVectorer); ok {
		bmat := rv.RawVector()
		if dst != b {
			dst.checkOverlap(bmat)
		}
		b := VecDense{mat: bmat}
		bm = b.asDense()
	}
	if trans {
		dst.reuseAs(r)
	} else {
		dst.reuseAs(c)
	}
	return qr.SolveTo(dst.asDense(), trans, bm)

}
