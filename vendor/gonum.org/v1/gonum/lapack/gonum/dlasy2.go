// Copyright ©2016 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"math"

	"gonum.org/v1/gonum/blas/blas64"
)

// Dlasy2 solves the Sylvester matrix equation where the matrices are of order 1
// or 2. It computes the unknown n1×n2 matrix X so that
//  TL*X   + sgn*X*TR   = scale*B,  if tranl == false and tranr == false,
//  TL^T*X + sgn*X*TR   = scale*B,  if tranl == true  and tranr == false,
//  TL*X   + sgn*X*TR^T = scale*B,  if tranl == false and tranr == true,
//  TL^T*X + sgn*X*TR^T = scale*B,  if tranl == true  and tranr == true,
// where TL is n1×n1, TR is n2×n2, B is n1×n2, and 1 <= n1,n2 <= 2.
//
// isgn must be 1 or -1, and n1 and n2 must be 0, 1, or 2, but these conditions
// are not checked.
//
// Dlasy2 returns three values, a scale factor that is chosen less than or equal
// to 1 to prevent the solution overflowing, the infinity norm of the solution,
// and an indicator of success. If ok is false, TL and TR have eigenvalues that
// are too close, so TL or TR is perturbed to get a non-singular equation.
//
// Dlasy2 is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dlasy2(tranl, tranr bool, isgn, n1, n2 int, tl []float64, ldtl int, tr []float64, ldtr int, b []float64, ldb int, x []float64, ldx int) (scale, xnorm float64, ok bool) {
	// TODO(vladimir-ch): Add input validation checks conditionally skipped
	// using the build tag mechanism.

	ok = true
	// Quick return if possible.
	if n1 == 0 || n2 == 0 {
		return scale, xnorm, ok
	}

	// Set constants to control overflow.
	eps := dlamchP
	smlnum := dlamchS / eps
	sgn := float64(isgn)

	if n1 == 1 && n2 == 1 {
		// 1×1 case: TL11*X + sgn*X*TR11 = B11.
		tau1 := tl[0] + sgn*tr[0]
		bet := math.Abs(tau1)
		if bet <= smlnum {
			tau1 = smlnum
			bet = smlnum
			ok = false
		}
		scale = 1
		gam := math.Abs(b[0])
		if smlnum*gam > bet {
			scale = 1 / gam
		}
		x[0] = b[0] * scale / tau1
		xnorm = math.Abs(x[0])
		return scale, xnorm, ok
	}

	if n1+n2 == 3 {
		// 1×2 or 2×1 case.
		var (
			smin float64
			tmp  [4]float64 // tmp is used as a 2×2 row-major matrix.
			btmp [2]float64
		)
		if n1 == 1 && n2 == 2 {
			// 1×2 case: TL11*[X11 X12] + sgn*[X11 X12]*op[TR11 TR12] = [B11 B12].
			//                                            [TR21 TR22]
			smin = math.Abs(tl[0])
			smin = math.Max(smin, math.Max(math.Abs(tr[0]), math.Abs(tr[1])))
			smin = math.Max(smin, math.Max(math.Abs(tr[ldtr]), math.Abs(tr[ldtr+1])))
			smin = math.Max(eps*smin, smlnum)
			tmp[0] = tl[0] + sgn*tr[0]
			tmp[3] = tl[0] + sgn*tr[ldtr+1]
			if tranr {
				tmp[1] = sgn * tr[1]
				tmp[2] = sgn * tr[ldtr]
			} else {
				tmp[1] = sgn * tr[ldtr]
				tmp[2] = sgn * tr[1]
			}
			btmp[0] = b[0]
			btmp[1] = b[1]
		} else {
			// 2×1 case: op[TL11 TL12]*[X11] + sgn*[X11]*TR11 = [B11].
			//             [TL21 TL22]*[X21]       [X21]        [B21]
			smin = math.Abs(tr[0])
			smin = math.Max(smin, math.Max(math.Abs(tl[0]), math.Abs(tl[1])))
			smin = math.Max(smin, math.Max(math.Abs(tl[ldtl]), math.Abs(tl[ldtl+1])))
			smin = math.Max(eps*smin, smlnum)
			tmp[0] = tl[0] + sgn*tr[0]
			tmp[3] = tl[ldtl+1] + sgn*tr[0]
			if tranl {
				tmp[1] = tl[ldtl]
				tmp[2] = tl[1]
			} else {
				tmp[1] = tl[1]
				tmp[2] = tl[ldtl]
			}
			btmp[0] = b[0]
			btmp[1] = b[ldb]
		}

		// Solve 2×2 system using complete pivoting.
		// Set pivots less than smin to smin.

		bi := blas64.Implementation()
		ipiv := bi.Idamax(len(tmp), tmp[:], 1)
		// Compute the upper triangular matrix [u11 u12].
		//                                     [  0 u22]
		u11 := tmp[ipiv]
		if math.Abs(u11) <= smin {
			ok = false
			u11 = smin
		}
		locu12 := [4]int{1, 0, 3, 2} // Index in tmp of the element on the same row as the pivot.
		u12 := tmp[locu12[ipiv]]
		locl21 := [4]int{2, 3, 0, 1} // Index in tmp of the element on the same column as the pivot.
		l21 := tmp[locl21[ipiv]] / u11
		locu22 := [4]int{3, 2, 1, 0} // Index in tmp of the remaining element.
		u22 := tmp[locu22[ipiv]] - l21*u12
		if math.Abs(u22) <= smin {
			ok = false
			u22 = smin
		}
		if ipiv&0x2 != 0 { // true for ipiv equal to 2 and 3.
			// The pivot was in the second row, swap the elements of
			// the right-hand side.
			btmp[0], btmp[1] = btmp[1], btmp[0]-l21*btmp[1]
		} else {
			btmp[1] -= l21 * btmp[0]
		}
		scale = 1
		if 2*smlnum*math.Abs(btmp[1]) > math.Abs(u22) || 2*smlnum*math.Abs(btmp[0]) > math.Abs(u11) {
			scale = 0.5 / math.Max(math.Abs(btmp[0]), math.Abs(btmp[1]))
			btmp[0] *= scale
			btmp[1] *= scale
		}
		// Solve the system [u11 u12] [x21] = [ btmp[0] ].
		//                  [  0 u22] [x22]   [ btmp[1] ]
		x22 := btmp[1] / u22
		x21 := btmp[0]/u11 - (u12/u11)*x22
		if ipiv&0x1 != 0 { // true for ipiv equal to 1 and 3.
			// The pivot was in the second column, swap the elements
			// of the solution.
			x21, x22 = x22, x21
		}
		x[0] = x21
		if n1 == 1 {
			x[1] = x22
			xnorm = math.Abs(x[0]) + math.Abs(x[1])
		} else {
			x[ldx] = x22
			xnorm = math.Max(math.Abs(x[0]), math.Abs(x[ldx]))
		}
		return scale, xnorm, ok
	}

	// 2×2 case: op[TL11 TL12]*[X11 X12] + SGN*[X11 X12]*op[TR11 TR12] = [B11 B12].
	//             [TL21 TL22] [X21 X22]       [X21 X22]   [TR21 TR22]   [B21 B22]
	//
	// Solve equivalent 4×4 system using complete pivoting.
	// Set pivots less than smin to smin.

	smin := math.Max(math.Abs(tr[0]), math.Abs(tr[1]))
	smin = math.Max(smin, math.Max(math.Abs(tr[ldtr]), math.Abs(tr[ldtr+1])))
	smin = math.Max(smin, math.Max(math.Abs(tl[0]), math.Abs(tl[1])))
	smin = math.Max(smin, math.Max(math.Abs(tl[ldtl]), math.Abs(tl[ldtl+1])))
	smin = math.Max(eps*smin, smlnum)

	var t [4][4]float64
	t[0][0] = tl[0] + sgn*tr[0]
	t[1][1] = tl[0] + sgn*tr[ldtr+1]
	t[2][2] = tl[ldtl+1] + sgn*tr[0]
	t[3][3] = tl[ldtl+1] + sgn*tr[ldtr+1]
	if tranl {
		t[0][2] = tl[ldtl]
		t[1][3] = tl[ldtl]
		t[2][0] = tl[1]
		t[3][1] = tl[1]
	} else {
		t[0][2] = tl[1]
		t[1][3] = tl[1]
		t[2][0] = tl[ldtl]
		t[3][1] = tl[ldtl]
	}
	if tranr {
		t[0][1] = sgn * tr[1]
		t[1][0] = sgn * tr[ldtr]
		t[2][3] = sgn * tr[1]
		t[3][2] = sgn * tr[ldtr]
	} else {
		t[0][1] = sgn * tr[ldtr]
		t[1][0] = sgn * tr[1]
		t[2][3] = sgn * tr[ldtr]
		t[3][2] = sgn * tr[1]
	}

	var btmp [4]float64
	btmp[0] = b[0]
	btmp[1] = b[1]
	btmp[2] = b[ldb]
	btmp[3] = b[ldb+1]

	// Perform elimination.
	var jpiv [4]int // jpiv records any column swaps for pivoting.
	for i := 0; i < 3; i++ {
		var (
			xmax       float64
			ipsv, jpsv int
		)
		for ip := i; ip < 4; ip++ {
			for jp := i; jp < 4; jp++ {
				if math.Abs(t[ip][jp]) >= xmax {
					xmax = math.Abs(t[ip][jp])
					ipsv = ip
					jpsv = jp
				}
			}
		}
		if ipsv != i {
			// The pivot is not in the top row of the unprocessed
			// block, swap rows ipsv and i of t and btmp.
			t[ipsv], t[i] = t[i], t[ipsv]
			btmp[ipsv], btmp[i] = btmp[i], btmp[ipsv]
		}
		if jpsv != i {
			// The pivot is not in the left column of the
			// unprocessed block, swap columns jpsv and i of t.
			for k := 0; k < 4; k++ {
				t[k][jpsv], t[k][i] = t[k][i], t[k][jpsv]
			}
		}
		jpiv[i] = jpsv
		if math.Abs(t[i][i]) < smin {
			ok = false
			t[i][i] = smin
		}
		for k := i + 1; k < 4; k++ {
			t[k][i] /= t[i][i]
			btmp[k] -= t[k][i] * btmp[i]
			for j := i + 1; j < 4; j++ {
				t[k][j] -= t[k][i] * t[i][j]
			}
		}
	}
	if math.Abs(t[3][3]) < smin {
		ok = false
		t[3][3] = smin
	}
	scale = 1
	if 8*smlnum*math.Abs(btmp[0]) > math.Abs(t[0][0]) ||
		8*smlnum*math.Abs(btmp[1]) > math.Abs(t[1][1]) ||
		8*smlnum*math.Abs(btmp[2]) > math.Abs(t[2][2]) ||
		8*smlnum*math.Abs(btmp[3]) > math.Abs(t[3][3]) {

		maxbtmp := math.Max(math.Abs(btmp[0]), math.Abs(btmp[1]))
		maxbtmp = math.Max(maxbtmp, math.Max(math.Abs(btmp[2]), math.Abs(btmp[3])))
		scale = 1 / 8 / maxbtmp
		btmp[0] *= scale
		btmp[1] *= scale
		btmp[2] *= scale
		btmp[3] *= scale
	}
	// Compute the solution of the upper triangular system t * tmp = btmp.
	var tmp [4]float64
	for i := 3; i >= 0; i-- {
		temp := 1 / t[i][i]
		tmp[i] = btmp[i] * temp
		for j := i + 1; j < 4; j++ {
			tmp[i] -= temp * t[i][j] * tmp[j]
		}
	}
	for i := 2; i >= 0; i-- {
		if jpiv[i] != i {
			tmp[i], tmp[jpiv[i]] = tmp[jpiv[i]], tmp[i]
		}
	}
	x[0] = tmp[0]
	x[1] = tmp[1]
	x[ldx] = tmp[2]
	x[ldx+1] = tmp[3]
	xnorm = math.Max(math.Abs(tmp[0])+math.Abs(tmp[1]), math.Abs(tmp[2])+math.Abs(tmp[3]))
	return scale, xnorm, ok
}
