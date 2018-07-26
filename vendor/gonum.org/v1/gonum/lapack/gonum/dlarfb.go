// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/gonum/lapack"
)

// Dlarfb applies a block reflector to a matrix.
//
// In the call to Dlarfb, the mxn c is multiplied by the implicitly defined matrix h as follows:
//  c = h * c if side == Left and trans == NoTrans
//  c = c * h if side == Right and trans == NoTrans
//  c = h^T * c if side == Left and trans == Trans
//  c = c * h^T if side == Right and trans == Trans
// h is a product of elementary reflectors. direct sets the direction of multiplication
//  h = h_1 * h_2 * ... * h_k if direct == Forward
//  h = h_k * h_k-1 * ... * h_1 if direct == Backward
// The combination of direct and store defines the orientation of the elementary
// reflectors. In all cases the ones on the diagonal are implicitly represented.
//
// If direct == lapack.Forward and store == lapack.ColumnWise
//  V = [ 1        ]
//      [v1   1    ]
//      [v1  v2   1]
//      [v1  v2  v3]
//      [v1  v2  v3]
// If direct == lapack.Forward and store == lapack.RowWise
//  V = [ 1  v1  v1  v1  v1]
//      [     1  v2  v2  v2]
//      [         1  v3  v3]
// If direct == lapack.Backward and store == lapack.ColumnWise
//  V = [v1  v2  v3]
//      [v1  v2  v3]
//      [ 1  v2  v3]
//      [     1  v3]
//      [         1]
// If direct == lapack.Backward and store == lapack.RowWise
//  V = [v1  v1   1        ]
//      [v2  v2  v2   1    ]
//      [v3  v3  v3  v3   1]
// An elementary reflector can be explicitly constructed by extracting the
// corresponding elements of v, placing a 1 where the diagonal would be, and
// placing zeros in the remaining elements.
//
// t is a k×k matrix containing the block reflector, and this function will panic
// if t is not of sufficient size. See Dlarft for more information.
//
// work is a temporary storage matrix with stride ldwork.
// work must be of size at least n×k side == Left and m×k if side == Right, and
// this function will panic if this size is not met.
//
// Dlarfb is an internal routine. It is exported for testing purposes.
func (Implementation) Dlarfb(side blas.Side, trans blas.Transpose, direct lapack.Direct, store lapack.StoreV, m, n, k int, v []float64, ldv int, t []float64, ldt int, c []float64, ldc int, work []float64, ldwork int) {
	if side != blas.Left && side != blas.Right {
		panic(badSide)
	}
	if trans != blas.Trans && trans != blas.NoTrans {
		panic(badTrans)
	}
	if direct != lapack.Forward && direct != lapack.Backward {
		panic(badDirect)
	}
	if store != lapack.ColumnWise && store != lapack.RowWise {
		panic(badStore)
	}
	checkMatrix(m, n, c, ldc)
	if k < 0 {
		panic(kLT0)
	}
	checkMatrix(k, k, t, ldt)
	nv := m
	nw := n
	if side == blas.Right {
		nv = n
		nw = m
	}
	if store == lapack.ColumnWise {
		checkMatrix(nv, k, v, ldv)
	} else {
		checkMatrix(k, nv, v, ldv)
	}
	checkMatrix(nw, k, work, ldwork)

	if m == 0 || n == 0 {
		return
	}

	bi := blas64.Implementation()

	transt := blas.Trans
	if trans == blas.Trans {
		transt = blas.NoTrans
	}
	// TODO(btracey): This follows the original Lapack code where the
	// elements are copied into the columns of the working array. The
	// loops should go in the other direction so the data is written
	// into the rows of work so the copy is not strided. A bigger change
	// would be to replace work with work^T, but benchmarks would be
	// needed to see if the change is merited.
	if store == lapack.ColumnWise {
		if direct == lapack.Forward {
			// V1 is the first k rows of C. V2 is the remaining rows.
			if side == blas.Left {
				// W = C^T V = C1^T V1 + C2^T V2 (stored in work).

				// W = C1.
				for j := 0; j < k; j++ {
					bi.Dcopy(n, c[j*ldc:], 1, work[j:], ldwork)
				}
				// W = W * V1.
				bi.Dtrmm(blas.Right, blas.Lower, blas.NoTrans, blas.Unit,
					n, k, 1,
					v, ldv,
					work, ldwork)
				if m > k {
					// W = W + C2^T V2.
					bi.Dgemm(blas.Trans, blas.NoTrans, n, k, m-k,
						1, c[k*ldc:], ldc, v[k*ldv:], ldv,
						1, work, ldwork)
				}
				// W = W * T^T or W * T.
				bi.Dtrmm(blas.Right, blas.Upper, transt, blas.NonUnit, n, k,
					1, t, ldt,
					work, ldwork)
				// C -= V * W^T.
				if m > k {
					// C2 -= V2 * W^T.
					bi.Dgemm(blas.NoTrans, blas.Trans, m-k, n, k,
						-1, v[k*ldv:], ldv, work, ldwork,
						1, c[k*ldc:], ldc)
				}
				// W *= V1^T.
				bi.Dtrmm(blas.Right, blas.Lower, blas.Trans, blas.Unit, n, k,
					1, v, ldv,
					work, ldwork)
				// C1 -= W^T.
				// TODO(btracey): This should use blas.Axpy.
				for i := 0; i < n; i++ {
					for j := 0; j < k; j++ {
						c[j*ldc+i] -= work[i*ldwork+j]
					}
				}
				return
			}
			// Form C = C * H or C * H^T, where C = (C1 C2).

			// W = C1.
			for i := 0; i < k; i++ {
				bi.Dcopy(m, c[i:], ldc, work[i:], ldwork)
			}
			// W *= V1.
			bi.Dtrmm(blas.Right, blas.Lower, blas.NoTrans, blas.Unit, m, k,
				1, v, ldv,
				work, ldwork)
			if n > k {
				bi.Dgemm(blas.NoTrans, blas.NoTrans, m, k, n-k,
					1, c[k:], ldc, v[k*ldv:], ldv,
					1, work, ldwork)
			}
			// W *= T or T^T.
			bi.Dtrmm(blas.Right, blas.Upper, trans, blas.NonUnit, m, k,
				1, t, ldt,
				work, ldwork)
			if n > k {
				bi.Dgemm(blas.NoTrans, blas.Trans, m, n-k, k,
					-1, work, ldwork, v[k*ldv:], ldv,
					1, c[k:], ldc)
			}
			// C -= W * V^T.
			bi.Dtrmm(blas.Right, blas.Lower, blas.Trans, blas.Unit, m, k,
				1, v, ldv,
				work, ldwork)
			// C -= W.
			// TODO(btracey): This should use blas.Axpy.
			for i := 0; i < m; i++ {
				for j := 0; j < k; j++ {
					c[i*ldc+j] -= work[i*ldwork+j]
				}
			}
			return
		}
		// V = (V1)
		//   = (V2) (last k rows)
		// Where V2 is unit upper triangular.
		if side == blas.Left {
			// Form H * C or
			// W = C^T V.

			// W = C2^T.
			for j := 0; j < k; j++ {
				bi.Dcopy(n, c[(m-k+j)*ldc:], 1, work[j:], ldwork)
			}
			// W *= V2.
			bi.Dtrmm(blas.Right, blas.Upper, blas.NoTrans, blas.Unit, n, k,
				1, v[(m-k)*ldv:], ldv,
				work, ldwork)
			if m > k {
				// W += C1^T * V1.
				bi.Dgemm(blas.Trans, blas.NoTrans, n, k, m-k,
					1, c, ldc, v, ldv,
					1, work, ldwork)
			}
			// W *= T or T^T.
			bi.Dtrmm(blas.Right, blas.Lower, transt, blas.NonUnit, n, k,
				1, t, ldt,
				work, ldwork)
			// C -= V * W^T.
			if m > k {
				bi.Dgemm(blas.NoTrans, blas.Trans, m-k, n, k,
					-1, v, ldv, work, ldwork,
					1, c, ldc)
			}
			// W *= V2^T.
			bi.Dtrmm(blas.Right, blas.Upper, blas.Trans, blas.Unit, n, k,
				1, v[(m-k)*ldv:], ldv,
				work, ldwork)
			// C2 -= W^T.
			// TODO(btracey): This should use blas.Axpy.
			for i := 0; i < n; i++ {
				for j := 0; j < k; j++ {
					c[(m-k+j)*ldc+i] -= work[i*ldwork+j]
				}
			}
			return
		}
		// Form C * H or C * H^T where C = (C1 C2).
		// W = C * V.

		// W = C2.
		for j := 0; j < k; j++ {
			bi.Dcopy(m, c[n-k+j:], ldc, work[j:], ldwork)
		}

		// W = W * V2.
		bi.Dtrmm(blas.Right, blas.Upper, blas.NoTrans, blas.Unit, m, k,
			1, v[(n-k)*ldv:], ldv,
			work, ldwork)
		if n > k {
			bi.Dgemm(blas.NoTrans, blas.NoTrans, m, k, n-k,
				1, c, ldc, v, ldv,
				1, work, ldwork)
		}
		// W *= T or T^T.
		bi.Dtrmm(blas.Right, blas.Lower, trans, blas.NonUnit, m, k,
			1, t, ldt,
			work, ldwork)
		// C -= W * V^T.
		if n > k {
			// C1 -= W * V1^T.
			bi.Dgemm(blas.NoTrans, blas.Trans, m, n-k, k,
				-1, work, ldwork, v, ldv,
				1, c, ldc)
		}
		// W *= V2^T.
		bi.Dtrmm(blas.Right, blas.Upper, blas.Trans, blas.Unit, m, k,
			1, v[(n-k)*ldv:], ldv,
			work, ldwork)
		// C2 -= W.
		// TODO(btracey): This should use blas.Axpy.
		for i := 0; i < m; i++ {
			for j := 0; j < k; j++ {
				c[i*ldc+n-k+j] -= work[i*ldwork+j]
			}
		}
		return
	}
	// Store = Rowwise.
	if direct == lapack.Forward {
		// V = (V1 V2) where v1 is unit upper triangular.
		if side == blas.Left {
			// Form H * C or H^T * C where C = (C1; C2).
			// W = C^T * V^T.

			// W = C1^T.
			for j := 0; j < k; j++ {
				bi.Dcopy(n, c[j*ldc:], 1, work[j:], ldwork)
			}
			// W *= V1^T.
			bi.Dtrmm(blas.Right, blas.Upper, blas.Trans, blas.Unit, n, k,
				1, v, ldv,
				work, ldwork)
			if m > k {
				bi.Dgemm(blas.Trans, blas.Trans, n, k, m-k,
					1, c[k*ldc:], ldc, v[k:], ldv,
					1, work, ldwork)
			}
			// W *= T or T^T.
			bi.Dtrmm(blas.Right, blas.Upper, transt, blas.NonUnit, n, k,
				1, t, ldt,
				work, ldwork)
			// C -= V^T * W^T.
			if m > k {
				bi.Dgemm(blas.Trans, blas.Trans, m-k, n, k,
					-1, v[k:], ldv, work, ldwork,
					1, c[k*ldc:], ldc)
			}
			// W *= V1.
			bi.Dtrmm(blas.Right, blas.Upper, blas.NoTrans, blas.Unit, n, k,
				1, v, ldv,
				work, ldwork)
			// C1 -= W^T.
			// TODO(btracey): This should use blas.Axpy.
			for i := 0; i < n; i++ {
				for j := 0; j < k; j++ {
					c[j*ldc+i] -= work[i*ldwork+j]
				}
			}
			return
		}
		// Form C * H or C * H^T where C = (C1 C2).
		// W = C * V^T.

		// W = C1.
		for j := 0; j < k; j++ {
			bi.Dcopy(m, c[j:], ldc, work[j:], ldwork)
		}
		// W *= V1^T.
		bi.Dtrmm(blas.Right, blas.Upper, blas.Trans, blas.Unit, m, k,
			1, v, ldv,
			work, ldwork)
		if n > k {
			bi.Dgemm(blas.NoTrans, blas.Trans, m, k, n-k,
				1, c[k:], ldc, v[k:], ldv,
				1, work, ldwork)
		}
		// W *= T or T^T.
		bi.Dtrmm(blas.Right, blas.Upper, trans, blas.NonUnit, m, k,
			1, t, ldt,
			work, ldwork)
		// C -= W * V.
		if n > k {
			bi.Dgemm(blas.NoTrans, blas.NoTrans, m, n-k, k,
				-1, work, ldwork, v[k:], ldv,
				1, c[k:], ldc)
		}
		// W *= V1.
		bi.Dtrmm(blas.Right, blas.Upper, blas.NoTrans, blas.Unit, m, k,
			1, v, ldv,
			work, ldwork)
		// C1 -= W.
		// TODO(btracey): This should use blas.Axpy.
		for i := 0; i < m; i++ {
			for j := 0; j < k; j++ {
				c[i*ldc+j] -= work[i*ldwork+j]
			}
		}
		return
	}
	// V = (V1 V2) where V2 is the last k columns and is lower unit triangular.
	if side == blas.Left {
		// Form H * C or H^T C where C = (C1 ; C2).
		// W = C^T * V^T.

		// W = C2^T.
		for j := 0; j < k; j++ {
			bi.Dcopy(n, c[(m-k+j)*ldc:], 1, work[j:], ldwork)
		}
		// W *= V2^T.
		bi.Dtrmm(blas.Right, blas.Lower, blas.Trans, blas.Unit, n, k,
			1, v[m-k:], ldv,
			work, ldwork)
		if m > k {
			bi.Dgemm(blas.Trans, blas.Trans, n, k, m-k,
				1, c, ldc, v, ldv,
				1, work, ldwork)
		}
		// W *= T or T^T.
		bi.Dtrmm(blas.Right, blas.Lower, transt, blas.NonUnit, n, k,
			1, t, ldt,
			work, ldwork)
		// C -= V^T * W^T.
		if m > k {
			bi.Dgemm(blas.Trans, blas.Trans, m-k, n, k,
				-1, v, ldv, work, ldwork,
				1, c, ldc)
		}
		// W *= V2.
		bi.Dtrmm(blas.Right, blas.Lower, blas.NoTrans, blas.Unit, n, k,
			1, v[m-k:], ldv,
			work, ldwork)
		// C2 -= W^T.
		// TODO(btracey): This should use blas.Axpy.
		for i := 0; i < n; i++ {
			for j := 0; j < k; j++ {
				c[(m-k+j)*ldc+i] -= work[i*ldwork+j]
			}
		}
		return
	}
	// Form C * H or C * H^T where C = (C1 C2).
	// W = C * V^T.
	// W = C2.
	for j := 0; j < k; j++ {
		bi.Dcopy(m, c[n-k+j:], ldc, work[j:], ldwork)
	}
	// W *= V2^T.
	bi.Dtrmm(blas.Right, blas.Lower, blas.Trans, blas.Unit, m, k,
		1, v[n-k:], ldv,
		work, ldwork)
	if n > k {
		bi.Dgemm(blas.NoTrans, blas.Trans, m, k, n-k,
			1, c, ldc, v, ldv,
			1, work, ldwork)
	}
	// W *= T or T^T.
	bi.Dtrmm(blas.Right, blas.Lower, trans, blas.NonUnit, m, k,
		1, t, ldt,
		work, ldwork)
	// C -= W * V.
	if n > k {
		bi.Dgemm(blas.NoTrans, blas.NoTrans, m, n-k, k,
			-1, work, ldwork, v, ldv,
			1, c, ldc)
	}
	// W *= V2.
	bi.Dtrmm(blas.Right, blas.Lower, blas.NoTrans, blas.Unit, m, k,
		1, v[n-k:], ldv,
		work, ldwork)
	// C1 -= W.
	// TODO(btracey): This should use blas.Axpy.
	for i := 0; i < m; i++ {
		for j := 0; j < k; j++ {
			c[i*ldc+n-k+j] -= work[i*ldwork+j]
		}
	}
}
