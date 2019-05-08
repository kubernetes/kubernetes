// Copyright ©2016 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"math"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/gonum/lapack"
)

// Dlaexc swaps two adjacent diagonal blocks of order 1 or 2 in an n×n upper
// quasi-triangular matrix T by an orthogonal similarity transformation.
//
// T must be in Schur canonical form, that is, block upper triangular with 1×1
// and 2×2 diagonal blocks; each 2×2 diagonal block has its diagonal elements
// equal and its off-diagonal elements of opposite sign. On return, T will
// contain the updated matrix again in Schur canonical form.
//
// If wantq is true, the transformation is accumulated in the n×n matrix Q,
// otherwise Q is not referenced.
//
// j1 is the index of the first row of the first block. n1 and n2 are the order
// of the first and second block, respectively.
//
// work must have length at least n, otherwise Dlaexc will panic.
//
// If ok is false, the transformed matrix T would be too far from Schur form.
// The blocks are not swapped, and T and Q are not modified.
//
// If n1 and n2 are both equal to 1, Dlaexc will always return true.
//
// Dlaexc is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dlaexc(wantq bool, n int, t []float64, ldt int, q []float64, ldq int, j1, n1, n2 int, work []float64) (ok bool) {
	switch {
	case n < 0:
		panic(nLT0)
	case ldt < max(1, n):
		panic(badLdT)
	case wantq && ldt < max(1, n):
		panic(badLdQ)
	case j1 < 0 || n <= j1:
		panic(badJ1)
	case len(work) < n:
		panic(shortWork)
	case n1 < 0 || 2 < n1:
		panic(badN1)
	case n2 < 0 || 2 < n2:
		panic(badN2)
	}

	if n == 0 || n1 == 0 || n2 == 0 {
		return true
	}

	switch {
	case len(t) < (n-1)*ldt+n:
		panic(shortT)
	case wantq && len(q) < (n-1)*ldq+n:
		panic(shortQ)
	}

	if j1+n1 >= n {
		// TODO(vladimir-ch): Reference LAPACK does this check whether
		// the start of the second block is in the matrix T. It returns
		// true if it is not and moreover it does not check whether the
		// whole second block fits into T. This does not feel
		// satisfactory. The only caller of Dlaexc is Dtrexc, so if the
		// caller makes sure that this does not happen, we could be
		// stricter here.
		return true
	}

	j2 := j1 + 1
	j3 := j1 + 2

	bi := blas64.Implementation()

	if n1 == 1 && n2 == 1 {
		// Swap two 1×1 blocks.
		t11 := t[j1*ldt+j1]
		t22 := t[j2*ldt+j2]

		// Determine the transformation to perform the interchange.
		cs, sn, _ := impl.Dlartg(t[j1*ldt+j2], t22-t11)

		// Apply transformation to the matrix T.
		if n-j3 > 0 {
			bi.Drot(n-j3, t[j1*ldt+j3:], 1, t[j2*ldt+j3:], 1, cs, sn)
		}
		if j1 > 0 {
			bi.Drot(j1, t[j1:], ldt, t[j2:], ldt, cs, sn)
		}

		t[j1*ldt+j1] = t22
		t[j2*ldt+j2] = t11

		if wantq {
			// Accumulate transformation in the matrix Q.
			bi.Drot(n, q[j1:], ldq, q[j2:], ldq, cs, sn)
		}

		return true
	}

	// Swapping involves at least one 2×2 block.
	//
	// Copy the diagonal block of order n1+n2 to the local array d and
	// compute its norm.
	nd := n1 + n2
	var d [16]float64
	const ldd = 4
	impl.Dlacpy(blas.All, nd, nd, t[j1*ldt+j1:], ldt, d[:], ldd)
	dnorm := impl.Dlange(lapack.MaxAbs, nd, nd, d[:], ldd, work)

	// Compute machine-dependent threshold for test for accepting swap.
	eps := dlamchP
	thresh := math.Max(10*eps*dnorm, dlamchS/eps)

	// Solve T11*X - X*T22 = scale*T12 for X.
	var x [4]float64
	const ldx = 2
	scale, _, _ := impl.Dlasy2(false, false, -1, n1, n2, d[:], ldd, d[n1*ldd+n1:], ldd, d[n1:], ldd, x[:], ldx)

	// Swap the adjacent diagonal blocks.
	switch {
	case n1 == 1 && n2 == 2:
		// Generate elementary reflector H so that
		//  ( scale, X11, X12 ) H = ( 0, 0, * )
		u := [3]float64{scale, x[0], 1}
		_, tau := impl.Dlarfg(3, x[1], u[:2], 1)
		t11 := t[j1*ldt+j1]

		// Perform swap provisionally on diagonal block in d.
		impl.Dlarfx(blas.Left, 3, 3, u[:], tau, d[:], ldd, work)
		impl.Dlarfx(blas.Right, 3, 3, u[:], tau, d[:], ldd, work)

		// Test whether to reject swap.
		if math.Max(math.Abs(d[2*ldd]), math.Max(math.Abs(d[2*ldd+1]), math.Abs(d[2*ldd+2]-t11))) > thresh {
			return false
		}

		// Accept swap: apply transformation to the entire matrix T.
		impl.Dlarfx(blas.Left, 3, n-j1, u[:], tau, t[j1*ldt+j1:], ldt, work)
		impl.Dlarfx(blas.Right, j2+1, 3, u[:], tau, t[j1:], ldt, work)

		t[j3*ldt+j1] = 0
		t[j3*ldt+j2] = 0
		t[j3*ldt+j3] = t11

		if wantq {
			// Accumulate transformation in the matrix Q.
			impl.Dlarfx(blas.Right, n, 3, u[:], tau, q[j1:], ldq, work)
		}

	case n1 == 2 && n2 == 1:
		//  Generate elementary reflector H so that:
		//   H (  -X11 ) = ( * )
		//     (  -X21 ) = ( 0 )
		//     ( scale ) = ( 0 )
		u := [3]float64{1, -x[ldx], scale}
		_, tau := impl.Dlarfg(3, -x[0], u[1:], 1)
		t33 := t[j3*ldt+j3]

		// Perform swap provisionally on diagonal block in D.
		impl.Dlarfx(blas.Left, 3, 3, u[:], tau, d[:], ldd, work)
		impl.Dlarfx(blas.Right, 3, 3, u[:], tau, d[:], ldd, work)

		// Test whether to reject swap.
		if math.Max(math.Abs(d[ldd]), math.Max(math.Abs(d[2*ldd]), math.Abs(d[0]-t33))) > thresh {
			return false
		}

		// Accept swap: apply transformation to the entire matrix T.
		impl.Dlarfx(blas.Right, j3+1, 3, u[:], tau, t[j1:], ldt, work)
		impl.Dlarfx(blas.Left, 3, n-j1-1, u[:], tau, t[j1*ldt+j2:], ldt, work)

		t[j1*ldt+j1] = t33
		t[j2*ldt+j1] = 0
		t[j3*ldt+j1] = 0

		if wantq {
			// Accumulate transformation in the matrix Q.
			impl.Dlarfx(blas.Right, n, 3, u[:], tau, q[j1:], ldq, work)
		}

	default: // n1 == 2 && n2 == 2
		// Generate elementary reflectors H_1 and H_2 so that:
		//  H_2 H_1 (  -X11  -X12 ) = (  *  * )
		//          (  -X21  -X22 )   (  0  * )
		//          ( scale    0  )   (  0  0 )
		//          (    0  scale )   (  0  0 )
		u1 := [3]float64{1, -x[ldx], scale}
		_, tau1 := impl.Dlarfg(3, -x[0], u1[1:], 1)

		temp := -tau1 * (x[1] + u1[1]*x[ldx+1])
		u2 := [3]float64{1, -temp * u1[2], scale}
		_, tau2 := impl.Dlarfg(3, -temp*u1[1]-x[ldx+1], u2[1:], 1)

		// Perform swap provisionally on diagonal block in D.
		impl.Dlarfx(blas.Left, 3, 4, u1[:], tau1, d[:], ldd, work)
		impl.Dlarfx(blas.Right, 4, 3, u1[:], tau1, d[:], ldd, work)
		impl.Dlarfx(blas.Left, 3, 4, u2[:], tau2, d[ldd:], ldd, work)
		impl.Dlarfx(blas.Right, 4, 3, u2[:], tau2, d[1:], ldd, work)

		// Test whether to reject swap.
		m1 := math.Max(math.Abs(d[2*ldd]), math.Abs(d[2*ldd+1]))
		m2 := math.Max(math.Abs(d[3*ldd]), math.Abs(d[3*ldd+1]))
		if math.Max(m1, m2) > thresh {
			return false
		}

		// Accept swap: apply transformation to the entire matrix T.
		j4 := j1 + 3
		impl.Dlarfx(blas.Left, 3, n-j1, u1[:], tau1, t[j1*ldt+j1:], ldt, work)
		impl.Dlarfx(blas.Right, j4+1, 3, u1[:], tau1, t[j1:], ldt, work)
		impl.Dlarfx(blas.Left, 3, n-j1, u2[:], tau2, t[j2*ldt+j1:], ldt, work)
		impl.Dlarfx(blas.Right, j4+1, 3, u2[:], tau2, t[j2:], ldt, work)

		t[j3*ldt+j1] = 0
		t[j3*ldt+j2] = 0
		t[j4*ldt+j1] = 0
		t[j4*ldt+j2] = 0

		if wantq {
			// Accumulate transformation in the matrix Q.
			impl.Dlarfx(blas.Right, n, 3, u1[:], tau1, q[j1:], ldq, work)
			impl.Dlarfx(blas.Right, n, 3, u2[:], tau2, q[j2:], ldq, work)
		}
	}

	if n2 == 2 {
		// Standardize new 2×2 block T11.
		a, b := t[j1*ldt+j1], t[j1*ldt+j2]
		c, d := t[j2*ldt+j1], t[j2*ldt+j2]
		var cs, sn float64
		t[j1*ldt+j1], t[j1*ldt+j2], t[j2*ldt+j1], t[j2*ldt+j2], _, _, _, _, cs, sn = impl.Dlanv2(a, b, c, d)
		if n-j1-2 > 0 {
			bi.Drot(n-j1-2, t[j1*ldt+j1+2:], 1, t[j2*ldt+j1+2:], 1, cs, sn)
		}
		if j1 > 0 {
			bi.Drot(j1, t[j1:], ldt, t[j2:], ldt, cs, sn)
		}
		if wantq {
			bi.Drot(n, q[j1:], ldq, q[j2:], ldq, cs, sn)
		}
	}
	if n1 == 2 {
		// Standardize new 2×2 block T22.
		j3 := j1 + n2
		j4 := j3 + 1
		a, b := t[j3*ldt+j3], t[j3*ldt+j4]
		c, d := t[j4*ldt+j3], t[j4*ldt+j4]
		var cs, sn float64
		t[j3*ldt+j3], t[j3*ldt+j4], t[j4*ldt+j3], t[j4*ldt+j4], _, _, _, _, cs, sn = impl.Dlanv2(a, b, c, d)
		if n-j3-2 > 0 {
			bi.Drot(n-j3-2, t[j3*ldt+j3+2:], 1, t[j4*ldt+j3+2:], 1, cs, sn)
		}
		bi.Drot(j3, t[j3:], ldt, t[j4:], ldt, cs, sn)
		if wantq {
			bi.Drot(n, q[j3:], ldq, q[j4:], ldq, cs, sn)
		}
	}

	return true
}
