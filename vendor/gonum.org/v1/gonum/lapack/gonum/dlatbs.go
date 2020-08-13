// Copyright ©2019 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"math"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
)

// Dlatbs solves a triangular banded system of equations
//  A * x = s*b    if trans == blas.NoTrans
//  Aᵀ * x = s*b  if trans == blas.Trans or blas.ConjTrans
// where A is an upper or lower triangular band matrix, x and b are n-element
// vectors, and s is a scaling factor chosen so that the components of x will be
// less than the overflow threshold.
//
// On entry, x contains the right-hand side b of the triangular system.
// On return, x is overwritten by the solution vector x.
//
// normin specifies whether the cnorm parameter contains the column norms of A on
// entry. If it is true, cnorm[j] contains the norm of the off-diagonal part of
// the j-th column of A. If it is false, the norms will be computed and stored
// in cnorm.
//
// Dlatbs returns the scaling factor s for the triangular system. If the matrix
// A is singular (A[j,j]==0 for some j), then scale is set to 0 and a
// non-trivial solution to A*x = 0 is returned.
//
// Dlatbs is an internal routine. It is exported for testing purposes.
func (Implementation) Dlatbs(uplo blas.Uplo, trans blas.Transpose, diag blas.Diag, normin bool, n, kd int, ab []float64, ldab int, x, cnorm []float64) (scale float64) {
	noTran := trans == blas.NoTrans
	switch {
	case uplo != blas.Upper && uplo != blas.Lower:
		panic(badUplo)
	case !noTran && trans != blas.Trans && trans != blas.ConjTrans:
		panic(badTrans)
	case diag != blas.NonUnit && diag != blas.Unit:
		panic(badDiag)
	case n < 0:
		panic(nLT0)
	case kd < 0:
		panic(kdLT0)
	case ldab < kd+1:
		panic(badLdA)
	}

	// Quick return if possible.
	if n == 0 {
		return 0
	}

	switch {
	case len(ab) < (n-1)*ldab+kd+1:
		panic(shortAB)
	case len(x) < n:
		panic(shortX)
	case len(cnorm) < n:
		panic(shortCNorm)
	}

	// Parameters to control overflow.
	smlnum := dlamchS / dlamchP
	bignum := 1 / smlnum

	bi := blas64.Implementation()
	kld := max(1, ldab-1)
	if !normin {
		// Compute the 1-norm of each column, not including the diagonal.
		if uplo == blas.Upper {
			for j := 0; j < n; j++ {
				jlen := min(j, kd)
				if jlen > 0 {
					cnorm[j] = bi.Dasum(jlen, ab[(j-jlen)*ldab+jlen:], kld)
				} else {
					cnorm[j] = 0
				}
			}
		} else {
			for j := 0; j < n; j++ {
				jlen := min(n-j-1, kd)
				if jlen > 0 {
					cnorm[j] = bi.Dasum(jlen, ab[(j+1)*ldab+kd-1:], kld)
				} else {
					cnorm[j] = 0
				}
			}
		}
	}

	// Set up indices and increments for loops below.
	var (
		jFirst, jLast, jInc int
		maind               int
	)
	if noTran {
		if uplo == blas.Upper {
			jFirst = n - 1
			jLast = -1
			jInc = -1
			maind = 0
		} else {
			jFirst = 0
			jLast = n
			jInc = 1
			maind = kd
		}
	} else {
		if uplo == blas.Upper {
			jFirst = 0
			jLast = n
			jInc = 1
			maind = 0
		} else {
			jFirst = n - 1
			jLast = -1
			jInc = -1
			maind = kd
		}
	}

	// Scale the column norms by tscal if the maximum element in cnorm is
	// greater than bignum.
	tmax := cnorm[bi.Idamax(n, cnorm, 1)]
	tscal := 1.0
	if tmax > bignum {
		tscal = 1 / (smlnum * tmax)
		bi.Dscal(n, tscal, cnorm, 1)
	}

	// Compute a bound on the computed solution vector to see if the Level 2
	// BLAS routine Dtbsv can be used.

	xMax := math.Abs(x[bi.Idamax(n, x, 1)])
	xBnd := xMax
	grow := 0.0
	// Compute the growth only if the maximum element in cnorm is NOT greater
	// than bignum.
	if tscal != 1 {
		goto skipComputeGrow
	}
	if noTran {
		// Compute the growth in A * x = b.
		if diag == blas.NonUnit {
			// A is non-unit triangular.
			//
			// Compute grow = 1/G_j and xBnd = 1/M_j.
			// Initially, G_0 = max{x(i), i=1,...,n}.
			grow = 1 / math.Max(xBnd, smlnum)
			xBnd = grow
			for j := jFirst; j != jLast; j += jInc {
				if grow <= smlnum {
					// Exit the loop because the growth factor is too small.
					goto skipComputeGrow
				}
				// M_j = G_{j-1} / abs(A[j,j])
				tjj := math.Abs(ab[j*ldab+maind])
				xBnd = math.Min(xBnd, math.Min(1, tjj)*grow)
				if tjj+cnorm[j] >= smlnum {
					// G_j = G_{j-1}*( 1 + cnorm[j] / abs(A[j,j]) )
					grow *= tjj / (tjj + cnorm[j])
				} else {
					// G_j could overflow, set grow to 0.
					grow = 0
				}
			}
			grow = xBnd
		} else {
			// A is unit triangular.
			//
			// Compute grow = 1/G_j, where G_0 = max{x(i), i=1,...,n}.
			grow = math.Min(1, 1/math.Max(xBnd, smlnum))
			for j := jFirst; j != jLast; j += jInc {
				if grow <= smlnum {
					// Exit the loop because the growth factor is too small.
					goto skipComputeGrow
				}
				// G_j = G_{j-1}*( 1 + cnorm[j] )
				grow /= 1 + cnorm[j]
			}
		}
	} else {
		// Compute the growth in Aᵀ * x = b.
		if diag == blas.NonUnit {
			// A is non-unit triangular.
			//
			// Compute grow = 1/G_j and xBnd = 1/M_j.
			// Initially, G_0 = max{x(i), i=1,...,n}.
			grow = 1 / math.Max(xBnd, smlnum)
			xBnd = grow
			for j := jFirst; j != jLast; j += jInc {
				if grow <= smlnum {
					// Exit the loop because the growth factor is too small.
					goto skipComputeGrow
				}
				// G_j = max( G_{j-1}, M_{j-1}*( 1 + cnorm[j] ) )
				xj := 1 + cnorm[j]
				grow = math.Min(grow, xBnd/xj)
				// M_j = M_{j-1}*( 1 + cnorm[j] ) / abs(A[j,j])
				tjj := math.Abs(ab[j*ldab+maind])
				if xj > tjj {
					xBnd *= tjj / xj
				}
			}
			grow = math.Min(grow, xBnd)
		} else {
			// A is unit triangular.
			//
			// Compute grow = 1/G_j, where G_0 = max{x(i), i=1,...,n}.
			grow = math.Min(1, 1/math.Max(xBnd, smlnum))
			for j := jFirst; j != jLast; j += jInc {
				if grow <= smlnum {
					// Exit the loop because the growth factor is too small.
					goto skipComputeGrow
				}
				// G_j = G_{j-1}*( 1 + cnorm[j] )
				grow /= 1 + cnorm[j]
			}
		}
	}
skipComputeGrow:

	if grow*tscal > smlnum {
		// The reciprocal of the bound on elements of X is not too small, use
		// the Level 2 BLAS solve.
		bi.Dtbsv(uplo, trans, diag, n, kd, ab, ldab, x, 1)
		// Scale the column norms by 1/tscal for return.
		if tscal != 1 {
			bi.Dscal(n, 1/tscal, cnorm, 1)
		}
		return 1
	}

	// Use a Level 1 BLAS solve, scaling intermediate results.

	scale = 1
	if xMax > bignum {
		// Scale x so that its components are less than or equal to bignum in
		// absolute value.
		scale = bignum / xMax
		bi.Dscal(n, scale, x, 1)
		xMax = bignum
	}

	if noTran {
		// Solve A * x = b.
		for j := jFirst; j != jLast; j += jInc {
			// Compute x[j] = b[j] / A[j,j], scaling x if necessary.
			xj := math.Abs(x[j])
			tjjs := tscal
			if diag == blas.NonUnit {
				tjjs *= ab[j*ldab+maind]
			}
			tjj := math.Abs(tjjs)
			switch {
			case tjj > smlnum:
				// smlnum < abs(A[j,j])
				if tjj < 1 && xj > tjj*bignum {
					// Scale x by 1/b[j].
					rec := 1 / xj
					bi.Dscal(n, rec, x, 1)
					scale *= rec
					xMax *= rec
				}
				x[j] /= tjjs
				xj = math.Abs(x[j])
			case tjj > 0:
				// 0 < abs(A[j,j]) <= smlnum
				if xj > tjj*bignum {
					// Scale x by (1/abs(x[j]))*abs(A[j,j])*bignum to avoid
					// overflow when dividing by A[j,j].
					rec := tjj * bignum / xj
					if cnorm[j] > 1 {
						// Scale by 1/cnorm[j] to avoid overflow when
						// multiplying x[j] times column j.
						rec /= cnorm[j]
					}
					bi.Dscal(n, rec, x, 1)
					scale *= rec
					xMax *= rec
				}
				x[j] /= tjjs
				xj = math.Abs(x[j])
			default:
				// A[j,j] == 0: Set x[0:n] = 0, x[j] = 1, and scale = 0, and
				// compute a solution to A*x = 0.
				for i := range x[:n] {
					x[i] = 0
				}
				x[j] = 1
				xj = 1
				scale = 0
				xMax = 0
			}

			// Scale x if necessary to avoid overflow when adding a multiple of
			// column j of A.
			switch {
			case xj > 1:
				rec := 1 / xj
				if cnorm[j] > (bignum-xMax)*rec {
					// Scale x by 1/(2*abs(x[j])).
					rec *= 0.5
					bi.Dscal(n, rec, x, 1)
					scale *= rec
				}
			case xj*cnorm[j] > bignum-xMax:
				// Scale x by 1/2.
				bi.Dscal(n, 0.5, x, 1)
				scale *= 0.5
			}

			if uplo == blas.Upper {
				if j > 0 {
					// Compute the update
					//  x[max(0,j-kd):j] := x[max(0,j-kd):j] - x[j] * A[max(0,j-kd):j,j]
					jlen := min(j, kd)
					if jlen > 0 {
						bi.Daxpy(jlen, -x[j]*tscal, ab[(j-jlen)*ldab+jlen:], kld, x[j-jlen:], 1)
					}
					i := bi.Idamax(j, x, 1)
					xMax = math.Abs(x[i])
				}
			} else if j < n-1 {
				// Compute the update
				//  x[j+1:min(j+kd,n)] := x[j+1:min(j+kd,n)] - x[j] * A[j+1:min(j+kd,n),j]
				jlen := min(kd, n-j-1)
				if jlen > 0 {
					bi.Daxpy(jlen, -x[j]*tscal, ab[(j+1)*ldab+kd-1:], kld, x[j+1:], 1)
				}
				i := j + 1 + bi.Idamax(n-j-1, x[j+1:], 1)
				xMax = math.Abs(x[i])
			}
		}
	} else {
		// Solve Aᵀ * x = b.
		for j := jFirst; j != jLast; j += jInc {
			// Compute x[j] = b[j] - sum A[k,j]*x[k].
			//                       k!=j
			xj := math.Abs(x[j])
			tjjs := tscal
			if diag == blas.NonUnit {
				tjjs *= ab[j*ldab+maind]
			}
			tjj := math.Abs(tjjs)
			rec := 1 / math.Max(1, xMax)
			uscal := tscal
			if cnorm[j] > (bignum-xj)*rec {
				// If x[j] could overflow, scale x by 1/(2*xMax).
				rec *= 0.5
				if tjj > 1 {
					// Divide by A[j,j] when scaling x if A[j,j] > 1.
					rec = math.Min(1, rec*tjj)
					uscal /= tjjs
				}
				if rec < 1 {
					bi.Dscal(n, rec, x, 1)
					scale *= rec
					xMax *= rec
				}
			}

			var sumj float64
			if uscal == 1 {
				// If the scaling needed for A in the dot product is 1, call
				// Ddot to perform the dot product...
				if uplo == blas.Upper {
					jlen := min(j, kd)
					if jlen > 0 {
						sumj = bi.Ddot(jlen, ab[(j-jlen)*ldab+jlen:], kld, x[j-jlen:], 1)
					}
				} else {
					jlen := min(n-j-1, kd)
					if jlen > 0 {
						sumj = bi.Ddot(jlen, ab[(j+1)*ldab+kd-1:], kld, x[j+1:], 1)
					}
				}
			} else {
				// ...otherwise, use in-line code for the dot product.
				if uplo == blas.Upper {
					jlen := min(j, kd)
					for i := 0; i < jlen; i++ {
						sumj += (ab[(j-jlen+i)*ldab+jlen-i] * uscal) * x[j-jlen+i]
					}
				} else {
					jlen := min(n-j-1, kd)
					for i := 0; i < jlen; i++ {
						sumj += (ab[(j+1+i)*ldab+kd-1-i] * uscal) * x[j+i+1]
					}
				}
			}

			if uscal == tscal {
				// Compute x[j] := ( x[j] - sumj ) / A[j,j]
				// if 1/A[j,j] was not used to scale the dot product.
				x[j] -= sumj
				xj = math.Abs(x[j])
				// Compute x[j] = x[j] / A[j,j], scaling if necessary.
				// Note: the reference implementation skips this step for blas.Unit matrices
				// when tscal is equal to 1 but it complicates the logic and only saves
				// the comparison and division in the first switch-case. Not skipping it
				// is also consistent with the NoTrans case above.
				switch {
				case tjj > smlnum:
					// smlnum < abs(A[j,j]):
					if tjj < 1 && xj > tjj*bignum {
						// Scale x by 1/abs(x[j]).
						rec := 1 / xj
						bi.Dscal(n, rec, x, 1)
						scale *= rec
						xMax *= rec
					}
					x[j] /= tjjs
				case tjj > 0:
					// 0 < abs(A[j,j]) <= smlnum:
					if xj > tjj*bignum {
						// Scale x by (1/abs(x[j]))*abs(A[j,j])*bignum.
						rec := (tjj * bignum) / xj
						bi.Dscal(n, rec, x, 1)
						scale *= rec
						xMax *= rec
					}
					x[j] /= tjjs
				default:
					// A[j,j] == 0: Set x[0:n] = 0, x[j] = 1, and scale = 0, and
					// compute a solution Aᵀ * x = 0.
					for i := range x[:n] {
						x[i] = 0
					}
					x[j] = 1
					scale = 0
					xMax = 0
				}
			} else {
				// Compute x[j] := x[j] / A[j,j] - sumj
				// if the dot product has already been divided by 1/A[j,j].
				x[j] = x[j]/tjjs - sumj
			}
			xMax = math.Max(xMax, math.Abs(x[j]))
		}
		scale /= tscal
	}

	// Scale the column norms by 1/tscal for return.
	if tscal != 1 {
		bi.Dscal(n, 1/tscal, cnorm, 1)
	}
	return scale
}
