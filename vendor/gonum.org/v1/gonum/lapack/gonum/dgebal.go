// Copyright ©2016 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"math"

	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/gonum/lapack"
)

// Dgebal balances an n×n matrix A. Balancing consists of two stages, permuting
// and scaling. Both steps are optional and depend on the value of job.
//
// Permuting consists of applying a permutation matrix P such that the matrix
// that results from P^T*A*P takes the upper block triangular form
//            [ T1  X  Y  ]
//  P^T A P = [  0  B  Z  ],
//            [  0  0  T2 ]
// where T1 and T2 are upper triangular matrices and B contains at least one
// nonzero off-diagonal element in each row and column. The indices ilo and ihi
// mark the starting and ending columns of the submatrix B. The eigenvalues of A
// isolated in the first 0 to ilo-1 and last ihi+1 to n-1 elements on the
// diagonal can be read off without any roundoff error.
//
// Scaling consists of applying a diagonal similarity transformation D such that
// D^{-1}*B*D has the 1-norm of each row and its corresponding column nearly
// equal. The output matrix is
//  [ T1     X*D          Y    ]
//  [  0  inv(D)*B*D  inv(D)*Z ].
//  [  0      0           T2   ]
// Scaling may reduce the 1-norm of the matrix, and improve the accuracy of
// the computed eigenvalues and/or eigenvectors.
//
// job specifies the operations that will be performed on A.
// If job is lapack.BalanceNone, Dgebal sets scale[i] = 1 for all i and returns ilo=0, ihi=n-1.
// If job is lapack.Permute, only permuting will be done.
// If job is lapack.Scale, only scaling will be done.
// If job is lapack.PermuteScale, both permuting and scaling will be done.
//
// On return, if job is lapack.Permute or lapack.PermuteScale, it will hold that
//  A[i,j] == 0,   for i > j and j ∈ {0, ..., ilo-1, ihi+1, ..., n-1}.
// If job is lapack.BalanceNone or lapack.Scale, or if n == 0, it will hold that
//  ilo == 0 and ihi == n-1.
//
// On return, scale will contain information about the permutations and scaling
// factors applied to A. If π(j) denotes the index of the column interchanged
// with column j, and D[j,j] denotes the scaling factor applied to column j,
// then
//  scale[j] == π(j),     for j ∈ {0, ..., ilo-1, ihi+1, ..., n-1},
//           == D[j,j],   for j ∈ {ilo, ..., ihi}.
// scale must have length equal to n, otherwise Dgebal will panic.
//
// Dgebal is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dgebal(job lapack.BalanceJob, n int, a []float64, lda int, scale []float64) (ilo, ihi int) {
	switch {
	case job != lapack.BalanceNone && job != lapack.Permute && job != lapack.Scale && job != lapack.PermuteScale:
		panic(badBalanceJob)
	case n < 0:
		panic(nLT0)
	case lda < max(1, n):
		panic(badLdA)
	}

	ilo = 0
	ihi = n - 1

	if n == 0 {
		return ilo, ihi
	}

	if len(scale) != n {
		panic(shortScale)
	}

	if job == lapack.BalanceNone {
		for i := range scale {
			scale[i] = 1
		}
		return ilo, ihi
	}

	if len(a) < (n-1)*lda+n {
		panic(shortA)
	}

	bi := blas64.Implementation()
	swapped := true

	if job == lapack.Scale {
		goto scaling
	}

	// Permutation to isolate eigenvalues if possible.
	//
	// Search for rows isolating an eigenvalue and push them down.
	for swapped {
		swapped = false
	rows:
		for i := ihi; i >= 0; i-- {
			for j := 0; j <= ihi; j++ {
				if i == j {
					continue
				}
				if a[i*lda+j] != 0 {
					continue rows
				}
			}
			// Row i has only zero off-diagonal elements in the
			// block A[ilo:ihi+1,ilo:ihi+1].
			scale[ihi] = float64(i)
			if i != ihi {
				bi.Dswap(ihi+1, a[i:], lda, a[ihi:], lda)
				bi.Dswap(n, a[i*lda:], 1, a[ihi*lda:], 1)
			}
			if ihi == 0 {
				scale[0] = 1
				return ilo, ihi
			}
			ihi--
			swapped = true
			break
		}
	}
	// Search for columns isolating an eigenvalue and push them left.
	swapped = true
	for swapped {
		swapped = false
	columns:
		for j := ilo; j <= ihi; j++ {
			for i := ilo; i <= ihi; i++ {
				if i == j {
					continue
				}
				if a[i*lda+j] != 0 {
					continue columns
				}
			}
			// Column j has only zero off-diagonal elements in the
			// block A[ilo:ihi+1,ilo:ihi+1].
			scale[ilo] = float64(j)
			if j != ilo {
				bi.Dswap(ihi+1, a[j:], lda, a[ilo:], lda)
				bi.Dswap(n-ilo, a[j*lda+ilo:], 1, a[ilo*lda+ilo:], 1)
			}
			swapped = true
			ilo++
			break
		}
	}

scaling:
	for i := ilo; i <= ihi; i++ {
		scale[i] = 1
	}

	if job == lapack.Permute {
		return ilo, ihi
	}

	// Balance the submatrix in rows ilo to ihi.

	const (
		// sclfac should be a power of 2 to avoid roundoff errors.
		// Elements of scale are restricted to powers of sclfac,
		// therefore the matrix will be only nearly balanced.
		sclfac = 2
		// factor determines the minimum reduction of the row and column
		// norms that is considered non-negligible. It must be less than 1.
		factor = 0.95
	)
	sfmin1 := dlamchS / dlamchP
	sfmax1 := 1 / sfmin1
	sfmin2 := sfmin1 * sclfac
	sfmax2 := 1 / sfmin2

	// Iterative loop for norm reduction.
	var conv bool
	for !conv {
		conv = true
		for i := ilo; i <= ihi; i++ {
			c := bi.Dnrm2(ihi-ilo+1, a[ilo*lda+i:], lda)
			r := bi.Dnrm2(ihi-ilo+1, a[i*lda+ilo:], 1)
			ica := bi.Idamax(ihi+1, a[i:], lda)
			ca := math.Abs(a[ica*lda+i])
			ira := bi.Idamax(n-ilo, a[i*lda+ilo:], 1)
			ra := math.Abs(a[i*lda+ilo+ira])

			// Guard against zero c or r due to underflow.
			if c == 0 || r == 0 {
				continue
			}
			g := r / sclfac
			f := 1.0
			s := c + r
			for c < g && math.Max(f, math.Max(c, ca)) < sfmax2 && math.Min(r, math.Min(g, ra)) > sfmin2 {
				if math.IsNaN(c + f + ca + r + g + ra) {
					// Panic if NaN to avoid infinite loop.
					panic("lapack: NaN")
				}
				f *= sclfac
				c *= sclfac
				ca *= sclfac
				g /= sclfac
				r /= sclfac
				ra /= sclfac
			}
			g = c / sclfac
			for r <= g && math.Max(r, ra) < sfmax2 && math.Min(math.Min(f, c), math.Min(g, ca)) > sfmin2 {
				f /= sclfac
				c /= sclfac
				ca /= sclfac
				g /= sclfac
				r *= sclfac
				ra *= sclfac
			}

			if c+r >= factor*s {
				// Reduction would be negligible.
				continue
			}
			if f < 1 && scale[i] < 1 && f*scale[i] <= sfmin1 {
				continue
			}
			if f > 1 && scale[i] > 1 && scale[i] >= sfmax1/f {
				continue
			}

			// Now balance.
			scale[i] *= f
			bi.Dscal(n-ilo, 1/f, a[i*lda+ilo:], 1)
			bi.Dscal(ihi+1, f, a[i:], lda)
			conv = false
		}
	}
	return ilo, ihi
}
