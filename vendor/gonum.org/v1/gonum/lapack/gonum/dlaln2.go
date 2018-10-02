// Copyright ©2016 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import "math"

// Dlaln2 solves a linear equation or a system of 2 linear equations of the form
//  (ca A   - w D) X = scale B,  if trans == false,
//  (ca A^T - w D) X = scale B,  if trans == true,
// where A is a na×na real matrix, ca is a real scalar, D is a na×na diagonal
// real matrix, w is a scalar, real if nw == 1, complex if nw == 2, and X and B
// are na×1 matrices, real if w is real, complex if w is complex.
//
// If w is complex, X and B are represented as na×2 matrices, the first column
// of each being the real part and the second being the imaginary part.
//
// na and nw must be 1 or 2, otherwise Dlaln2 will panic.
//
// d1 and d2 are the diagonal elements of D. d2 is not used if na == 1.
//
// wr and wi represent the real and imaginary part, respectively, of the scalar
// w. wi is not used if nw == 1.
//
// smin is the desired lower bound on the singular values of A. This should be
// a safe distance away from underflow or overflow, say, between
// (underflow/machine precision) and (overflow*machine precision).
//
// If both singular values of (ca A - w D) are less than smin, smin*identity
// will be used instead of (ca A - w D). If only one singular value is less than
// smin, one element of (ca A - w D) will be perturbed enough to make the
// smallest singular value roughly smin. If both singular values are at least
// smin, (ca A - w D) will not be perturbed. In any case, the perturbation will
// be at most some small multiple of max(smin, ulp*norm(ca A - w D)). The
// singular values are computed by infinity-norm approximations, and thus will
// only be correct to a factor of 2 or so.
//
// All input quantities are assumed to be smaller than overflow by a reasonable
// factor.
//
// scale is a scaling factor less than or equal to 1 which is chosen so that X
// can be computed without overflow. X is further scaled if necessary to assure
// that norm(ca A - w D)*norm(X) is less than overflow.
//
// xnorm contains the infinity-norm of X when X is regarded as a na×nw real
// matrix.
//
// ok will be false if (ca A - w D) had to be perturbed to make its smallest
// singular value greater than smin, otherwise ok will be true.
//
// Dlaln2 is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dlaln2(trans bool, na, nw int, smin, ca float64, a []float64, lda int, d1, d2 float64, b []float64, ldb int, wr, wi float64, x []float64, ldx int) (scale, xnorm float64, ok bool) {
	// TODO(vladimir-ch): Consider splitting this function into two, one
	// handling the real case (nw == 1) and the other handling the complex
	// case (nw == 2). Given that Go has complex types, their signatures
	// would be simpler and more natural, and the implementation not as
	// convoluted.

	if na != 1 && na != 2 {
		panic("lapack: invalid value of na")
	}
	if nw != 1 && nw != 2 {
		panic("lapack: invalid value of nw")
	}
	checkMatrix(na, na, a, lda)
	checkMatrix(na, nw, b, ldb)
	checkMatrix(na, nw, x, ldx)

	smlnum := 2 * dlamchS
	bignum := 1 / smlnum
	smini := math.Max(smin, smlnum)

	ok = true
	scale = 1

	if na == 1 {
		// 1×1 (i.e., scalar) system C X = B.

		if nw == 1 {
			// Real 1×1 system.

			// C = ca A - w D.
			csr := ca*a[0] - wr*d1
			cnorm := math.Abs(csr)

			// If |C| < smini, use C = smini.
			if cnorm < smini {
				csr = smini
				cnorm = smini
				ok = false
			}

			// Check scaling for X = B / C.
			bnorm := math.Abs(b[0])
			if cnorm < 1 && bnorm > math.Max(1, bignum*cnorm) {
				scale = 1 / bnorm
			}

			// Compute X.
			x[0] = b[0] * scale / csr
			xnorm = math.Abs(x[0])

			return scale, xnorm, ok
		}

		// Complex 1×1 system (w is complex).

		// C = ca A - w D.
		csr := ca*a[0] - wr*d1
		csi := -wi * d1
		cnorm := math.Abs(csr) + math.Abs(csi)

		// If |C| < smini, use C = smini.
		if cnorm < smini {
			csr = smini
			csi = 0
			cnorm = smini
			ok = false
		}

		// Check scaling for X = B / C.
		bnorm := math.Abs(b[0]) + math.Abs(b[1])
		if cnorm < 1 && bnorm > math.Max(1, bignum*cnorm) {
			scale = 1 / bnorm
		}

		// Compute X.
		cx := complex(scale*b[0], scale*b[1]) / complex(csr, csi)
		x[0], x[1] = real(cx), imag(cx)
		xnorm = math.Abs(x[0]) + math.Abs(x[1])

		return scale, xnorm, ok
	}

	// 2×2 system.

	// Compute the real part of
	//  C = ca A   - w D
	// or
	//  C = ca A^T - w D.
	crv := [4]float64{
		ca*a[0] - wr*d1,
		ca * a[1],
		ca * a[lda],
		ca*a[lda+1] - wr*d2,
	}
	if trans {
		crv[1] = ca * a[lda]
		crv[2] = ca * a[1]
	}

	pivot := [4][4]int{
		{0, 1, 2, 3},
		{1, 0, 3, 2},
		{2, 3, 0, 1},
		{3, 2, 1, 0},
	}

	if nw == 1 {
		// Real 2×2 system (w is real).

		// Find the largest element in C.
		var cmax float64
		var icmax int
		for j, v := range crv {
			v = math.Abs(v)
			if v > cmax {
				cmax = v
				icmax = j
			}
		}

		// If norm(C) < smini, use smini*identity.
		if cmax < smini {
			bnorm := math.Max(math.Abs(b[0]), math.Abs(b[ldb]))
			if smini < 1 && bnorm > math.Max(1, bignum*smini) {
				scale = 1 / bnorm
			}
			temp := scale / smini
			x[0] = temp * b[0]
			x[ldx] = temp * b[ldb]
			xnorm = temp * bnorm
			ok = false

			return scale, xnorm, ok
		}

		// Gaussian elimination with complete pivoting.
		// Form upper triangular matrix
		//  [ur11 ur12]
		//  [   0 ur22]
		ur11 := crv[icmax]
		ur12 := crv[pivot[icmax][1]]
		cr21 := crv[pivot[icmax][2]]
		cr22 := crv[pivot[icmax][3]]
		ur11r := 1 / ur11
		lr21 := ur11r * cr21
		ur22 := cr22 - ur12*lr21

		// If smaller pivot < smini, use smini.
		if math.Abs(ur22) < smini {
			ur22 = smini
			ok = false
		}

		var br1, br2 float64
		if icmax > 1 {
			// If the pivot lies in the second row, swap the rows.
			br1 = b[ldb]
			br2 = b[0]
		} else {
			br1 = b[0]
			br2 = b[ldb]
		}
		br2 -= lr21 * br1 // Apply the Gaussian elimination step to the right-hand side.

		bbnd := math.Max(math.Abs(ur22*ur11r*br1), math.Abs(br2))
		if bbnd > 1 && math.Abs(ur22) < 1 && bbnd >= bignum*math.Abs(ur22) {
			scale = 1 / bbnd
		}

		// Solve the linear system ur*xr=br.
		xr2 := br2 * scale / ur22
		xr1 := scale*br1*ur11r - ur11r*ur12*xr2
		if icmax&0x1 != 0 {
			// If the pivot lies in the second column, swap the components of the solution.
			x[0] = xr2
			x[ldx] = xr1
		} else {
			x[0] = xr1
			x[ldx] = xr2
		}
		xnorm = math.Max(math.Abs(xr1), math.Abs(xr2))

		// Further scaling if norm(A)*norm(X) > overflow.
		if xnorm > 1 && cmax > 1 && xnorm > bignum/cmax {
			temp := cmax / bignum
			x[0] *= temp
			x[ldx] *= temp
			xnorm *= temp
			scale *= temp
		}

		return scale, xnorm, ok
	}

	// Complex 2×2 system (w is complex).

	// Find the largest element in C.
	civ := [4]float64{
		-wi * d1,
		0,
		0,
		-wi * d2,
	}
	var cmax float64
	var icmax int
	for j, v := range crv {
		v := math.Abs(v)
		if v+math.Abs(civ[j]) > cmax {
			cmax = v + math.Abs(civ[j])
			icmax = j
		}
	}

	// If norm(C) < smini, use smini*identity.
	if cmax < smini {
		br1 := math.Abs(b[0]) + math.Abs(b[1])
		br2 := math.Abs(b[ldb]) + math.Abs(b[ldb+1])
		bnorm := math.Max(br1, br2)
		if smini < 1 && bnorm > 1 && bnorm > bignum*smini {
			scale = 1 / bnorm
		}
		temp := scale / smini
		x[0] = temp * b[0]
		x[1] = temp * b[1]
		x[ldb] = temp * b[ldb]
		x[ldb+1] = temp * b[ldb+1]
		xnorm = temp * bnorm
		ok = false

		return scale, xnorm, ok
	}

	// Gaussian elimination with complete pivoting.
	ur11 := crv[icmax]
	ui11 := civ[icmax]
	ur12 := crv[pivot[icmax][1]]
	ui12 := civ[pivot[icmax][1]]
	cr21 := crv[pivot[icmax][2]]
	ci21 := civ[pivot[icmax][2]]
	cr22 := crv[pivot[icmax][3]]
	ci22 := civ[pivot[icmax][3]]
	var (
		ur11r, ui11r float64
		lr21, li21   float64
		ur12s, ui12s float64
		ur22, ui22   float64
	)
	if icmax == 0 || icmax == 3 {
		// Off-diagonals of pivoted C are real.
		if math.Abs(ur11) > math.Abs(ui11) {
			temp := ui11 / ur11
			ur11r = 1 / (ur11 * (1 + temp*temp))
			ui11r = -temp * ur11r
		} else {
			temp := ur11 / ui11
			ui11r = -1 / (ui11 * (1 + temp*temp))
			ur11r = -temp * ui11r
		}
		lr21 = cr21 * ur11r
		li21 = cr21 * ui11r
		ur12s = ur12 * ur11r
		ui12s = ur12 * ui11r
		ur22 = cr22 - ur12*lr21
		ui22 = ci22 - ur12*li21
	} else {
		// Diagonals of pivoted C are real.
		ur11r = 1 / ur11
		// ui11r is already 0.
		lr21 = cr21 * ur11r
		li21 = ci21 * ur11r
		ur12s = ur12 * ur11r
		ui12s = ui12 * ur11r
		ur22 = cr22 - ur12*lr21 + ui12*li21
		ui22 = -ur12*li21 - ui12*lr21
	}
	u22abs := math.Abs(ur22) + math.Abs(ui22)

	// If smaller pivot < smini, use smini.
	if u22abs < smini {
		ur22 = smini
		ui22 = 0
		ok = false
	}

	var br1, bi1 float64
	var br2, bi2 float64
	if icmax > 1 {
		// If the pivot lies in the second row, swap the rows.
		br1 = b[ldb]
		bi1 = b[ldb+1]
		br2 = b[0]
		bi2 = b[1]
	} else {
		br1 = b[0]
		bi1 = b[1]
		br2 = b[ldb]
		bi2 = b[ldb+1]
	}
	br2 += -lr21*br1 + li21*bi1
	bi2 += -li21*br1 - lr21*bi1

	bbnd1 := u22abs * (math.Abs(ur11r) + math.Abs(ui11r)) * (math.Abs(br1) + math.Abs(bi1))
	bbnd2 := math.Abs(br2) + math.Abs(bi2)
	bbnd := math.Max(bbnd1, bbnd2)
	if bbnd > 1 && u22abs < 1 && bbnd >= bignum*u22abs {
		scale = 1 / bbnd
		br1 *= scale
		bi1 *= scale
		br2 *= scale
		bi2 *= scale
	}

	cx2 := complex(br2, bi2) / complex(ur22, ui22)
	xr2, xi2 := real(cx2), imag(cx2)
	xr1 := ur11r*br1 - ui11r*bi1 - ur12s*xr2 + ui12s*xi2
	xi1 := ui11r*br1 + ur11r*bi1 - ui12s*xr2 - ur12s*xi2
	if icmax&0x1 != 0 {
		// If the pivot lies in the second column, swap the components of the solution.
		x[0] = xr2
		x[1] = xi2
		x[ldx] = xr1
		x[ldx+1] = xi1
	} else {
		x[0] = xr1
		x[1] = xi1
		x[ldx] = xr2
		x[ldx+1] = xi2
	}
	xnorm = math.Max(math.Abs(xr1)+math.Abs(xi1), math.Abs(xr2)+math.Abs(xi2))

	// Further scaling if norm(A)*norm(X) > overflow.
	if xnorm > 1 && cmax > 1 && xnorm > bignum/cmax {
		temp := cmax / bignum
		x[0] *= temp
		x[1] *= temp
		x[ldx] *= temp
		x[ldx+1] *= temp
		xnorm *= temp
		scale *= temp
	}

	return scale, xnorm, ok
}
