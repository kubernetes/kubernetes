// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat

import (
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/gonum/lapack/lapack64"
)

// Solve solves the linear least squares problem
//  minimize over x |b - A*x|_2
// where A is an m×n matrix A, b is a given m element vector and x is n element
// solution vector. Solve assumes that A has full rank, that is
//  rank(A) = min(m,n)
//
// If m >= n, Solve finds the unique least squares solution of an overdetermined
// system.
//
// If m < n, there is an infinite number of solutions that satisfy b-A*x=0. In
// this case Solve finds the unique solution of an underdetermined system that
// minimizes |x|_2.
//
// Several right-hand side vectors b and solution vectors x can be handled in a
// single call. Vectors b are stored in the columns of the m×k matrix B. Vectors
// x will be stored in-place into the n×k receiver.
//
// If A does not have full rank, a Condition error is returned. See the
// documentation for Condition for more information.
func (m *Dense) Solve(a, b Matrix) error {
	ar, ac := a.Dims()
	br, bc := b.Dims()
	if ar != br {
		panic(ErrShape)
	}
	m.reuseAsNonZeroed(ac, bc)

	// TODO(btracey): Add special cases for SymDense, etc.
	aU, aTrans := untranspose(a)
	bU, bTrans := untranspose(b)
	switch rma := aU.(type) {
	case RawTriangular:
		side := blas.Left
		tA := blas.NoTrans
		if aTrans {
			tA = blas.Trans
		}

		switch rm := bU.(type) {
		case RawMatrixer:
			if m != bU || bTrans {
				if m == bU || m.checkOverlap(rm.RawMatrix()) {
					tmp := getWorkspace(br, bc, false)
					tmp.Copy(b)
					m.Copy(tmp)
					putWorkspace(tmp)
					break
				}
				m.Copy(b)
			}
		default:
			if m != bU {
				m.Copy(b)
			} else if bTrans {
				// m and b share data so Copy cannot be used directly.
				tmp := getWorkspace(br, bc, false)
				tmp.Copy(b)
				m.Copy(tmp)
				putWorkspace(tmp)
			}
		}

		rm := rma.RawTriangular()
		blas64.Trsm(side, tA, 1, rm, m.mat)
		work := getFloats(3*rm.N, false)
		iwork := getInts(rm.N, false)
		cond := lapack64.Trcon(CondNorm, rm, work, iwork)
		putFloats(work)
		putInts(iwork)
		if cond > ConditionTolerance {
			return Condition(cond)
		}
		return nil
	}

	switch {
	case ar == ac:
		if a == b {
			// x = I.
			if ar == 1 {
				m.mat.Data[0] = 1
				return nil
			}
			for i := 0; i < ar; i++ {
				v := m.mat.Data[i*m.mat.Stride : i*m.mat.Stride+ac]
				zero(v)
				v[i] = 1
			}
			return nil
		}
		var lu LU
		lu.Factorize(a)
		return lu.SolveTo(m, false, b)
	case ar > ac:
		var qr QR
		qr.Factorize(a)
		return qr.SolveTo(m, false, b)
	default:
		var lq LQ
		lq.Factorize(a)
		return lq.SolveTo(m, false, b)
	}
}

// SolveVec solves the linear least squares problem
//  minimize over x |b - A*x|_2
// where A is an m×n matrix A, b is a given m element vector and x is n element
// solution vector. Solve assumes that A has full rank, that is
//  rank(A) = min(m,n)
//
// If m >= n, Solve finds the unique least squares solution of an overdetermined
// system.
//
// If m < n, there is an infinite number of solutions that satisfy b-A*x=0. In
// this case Solve finds the unique solution of an underdetermined system that
// minimizes |x|_2.
//
// The solution vector x will be stored in-place into the receiver.
//
// If A does not have full rank, a Condition error is returned. See the
// documentation for Condition for more information.
func (v *VecDense) SolveVec(a Matrix, b Vector) error {
	if _, bc := b.Dims(); bc != 1 {
		panic(ErrShape)
	}
	_, c := a.Dims()

	// The Solve implementation is non-trivial, so rather than duplicate the code,
	// instead recast the VecDenses as Dense and call the matrix code.

	if rv, ok := b.(RawVectorer); ok {
		bmat := rv.RawVector()
		if v != b {
			v.checkOverlap(bmat)
		}
		v.reuseAsNonZeroed(c)
		m := v.asDense()
		// We conditionally create bm as m when b and v are identical
		// to prevent the overlap detection code from identifying m
		// and bm as overlapping but not identical.
		bm := m
		if v != b {
			b := VecDense{mat: bmat}
			bm = b.asDense()
		}
		return m.Solve(a, bm)
	}

	v.reuseAsNonZeroed(c)
	m := v.asDense()
	return m.Solve(a, b)
}
