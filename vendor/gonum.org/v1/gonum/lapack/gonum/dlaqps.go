// Copyright ©2017 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"math"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
)

// Dlaqps computes a step of QR factorization with column pivoting
// of an m×n matrix A by using Blas-3. It tries to factorize nb
// columns from A starting from the row offset, and updates all
// of the matrix with Dgemm.
//
// In some cases, due to catastrophic cancellations, it cannot
// factorize nb columns. Hence, the actual number of factorized
// columns is returned in kb.
//
// Dlaqps computes a QR factorization with column pivoting of the
// block A[offset:m, 0:nb] of the m×n matrix A. The block
// A[0:offset, 0:n] is accordingly pivoted, but not factorized.
//
// On exit, the upper triangle of block A[offset:m, 0:kb] is the
// triangular factor obtained. The elements in block A[offset:m, 0:n]
// below the diagonal, together with tau, represent the orthogonal
// matrix Q as a product of elementary reflectors.
//
// offset is number of rows of the matrix A that must be pivoted but
// not factorized. offset must not be negative otherwise Dlaqps will panic.
//
// On exit, jpvt holds the permutation that was applied; the jth column
// of A*P was the jpvt[j] column of A. jpvt must have length n,
// otherwise Dlapqs will panic.
//
// On exit tau holds the scalar factors of the elementary reflectors.
// It must have length nb, otherwise Dlapqs will panic.
//
// vn1 and vn2 hold the partial and complete column norms respectively.
// They must have length n, otherwise Dlapqs will panic.
//
// auxv must have length nb, otherwise Dlaqps will panic.
//
// f and ldf represent an n×nb matrix F that is overwritten during the
// call.
//
// Dlaqps is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dlaqps(m, n, offset, nb int, a []float64, lda int, jpvt []int, tau, vn1, vn2, auxv, f []float64, ldf int) (kb int) {
	checkMatrix(m, n, a, lda)
	checkMatrix(n, nb, f, ldf)
	if offset > m {
		panic(offsetGTM)
	}
	if n < 0 || nb > n {
		panic(badNb)
	}
	if len(jpvt) != n {
		panic(badIpiv)
	}
	if len(tau) < nb {
		panic(badTau)
	}
	if len(vn1) < n {
		panic(badVn1)
	}
	if len(vn2) < n {
		panic(badVn2)
	}
	if len(auxv) < nb {
		panic(badAuxv)
	}

	lastrk := min(m, n+offset)
	lsticc := -1
	tol3z := math.Sqrt(dlamchE)

	bi := blas64.Implementation()

	var k, rk int
	for ; k < nb && lsticc == -1; k++ {
		rk = offset + k

		// Determine kth pivot column and swap if necessary.
		p := k + bi.Idamax(n-k, vn1[k:], 1)
		if p != k {
			bi.Dswap(m, a[p:], lda, a[k:], lda)
			bi.Dswap(k, f[p*ldf:], 1, f[k*ldf:], 1)
			jpvt[p], jpvt[k] = jpvt[k], jpvt[p]
			vn1[p] = vn1[k]
			vn2[p] = vn2[k]
		}

		// Apply previous Householder reflectors to column K:
		//
		// A[rk:m, k] = A[rk:m, k] - A[rk:m, 0:k-1]*F[k, 0:k-1]^T.
		if k > 0 {
			bi.Dgemv(blas.NoTrans, m-rk, k, -1,
				a[rk*lda:], lda,
				f[k*ldf:], 1,
				1,
				a[rk*lda+k:], lda)
		}

		// Generate elementary reflector H_k.
		if rk < m-1 {
			a[rk*lda+k], tau[k] = impl.Dlarfg(m-rk, a[rk*lda+k], a[(rk+1)*lda+k:], lda)
		} else {
			tau[k] = 0
		}

		akk := a[rk*lda+k]
		a[rk*lda+k] = 1

		// Compute kth column of F:
		//
		// Compute F[k+1:n, k] = tau[k]*A[rk:m, k+1:n]^T*A[rk:m, k].
		if k < n-1 {
			bi.Dgemv(blas.Trans, m-rk, n-k-1, tau[k],
				a[rk*lda+k+1:], lda,
				a[rk*lda+k:], lda,
				0,
				f[(k+1)*ldf+k:], ldf)
		}

		// Padding F[0:k, k] with zeros.
		for j := 0; j < k; j++ {
			f[j*ldf+k] = 0
		}

		// Incremental updating of F:
		//
		// F[0:n, k] := F[0:n, k] - tau[k]*F[0:n, 0:k-1]*A[rk:m, 0:k-1]^T*A[rk:m,k].
		if k > 0 {
			bi.Dgemv(blas.Trans, m-rk, k, -tau[k],
				a[rk*lda:], lda,
				a[rk*lda+k:], lda,
				0,
				auxv, 1)
			bi.Dgemv(blas.NoTrans, n, k, 1,
				f, ldf,
				auxv, 1,
				1,
				f[k:], ldf)
		}

		// Update the current row of A:
		//
		// A[rk, k+1:n] = A[rk, k+1:n] - A[rk, 0:k]*F[k+1:n, 0:k]^T.
		if k < n-1 {
			bi.Dgemv(blas.NoTrans, n-k-1, k+1, -1,
				f[(k+1)*ldf:], ldf,
				a[rk*lda:], 1,
				1,
				a[rk*lda+k+1:], 1)
		}

		// Update partial column norms.
		if rk < lastrk-1 {
			for j := k + 1; j < n; j++ {
				if vn1[j] == 0 {
					continue
				}

				// The following marked lines follow from the
				// analysis in Lapack Working Note 176.
				r := math.Abs(a[rk*lda+j]) / vn1[j] // *
				temp := math.Max(0, 1-r*r)          // *
				r = vn1[j] / vn2[j]                 // *
				temp2 := temp * r * r               // *
				if temp2 < tol3z {
					// vn2 is used here as a collection of
					// indices into vn2 and also a collection
					// of column norms.
					vn2[j] = float64(lsticc)
					lsticc = j
				} else {
					vn1[j] *= math.Sqrt(temp) // *
				}
			}
		}

		a[rk*lda+k] = akk
	}
	kb = k
	rk = offset + kb

	// Apply the block reflector to the rest of the matrix:
	//
	// A[offset+kb+1:m, kb+1:n] := A[offset+kb+1:m, kb+1:n] - A[offset+kb+1:m, 1:kb]*F[kb+1:n, 1:kb]^T.
	if kb < min(n, m-offset) {
		bi.Dgemm(blas.NoTrans, blas.Trans,
			m-rk, n-kb, kb, -1,
			a[rk*lda:], lda,
			f[kb*ldf:], ldf,
			1,
			a[rk*lda+kb:], lda)
	}

	// Recomputation of difficult columns.
	for lsticc >= 0 {
		itemp := int(vn2[lsticc])

		// NOTE: The computation of vn1[lsticc] relies on the fact that
		// Dnrm2 does not fail on vectors with norm below the value of
		// sqrt(dlamchS)
		v := bi.Dnrm2(m-rk, a[rk*lda+lsticc:], lda)
		vn1[lsticc] = v
		vn2[lsticc] = v

		lsticc = itemp
	}

	return kb
}
