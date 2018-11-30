// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"math"

	"gonum.org/v1/gonum/lapack"
)

// Dlasq2 computes all the eigenvalues of the symmetric positive
// definite tridiagonal matrix associated with the qd array Z. Eigevalues
// are computed to high relative accuracy avoiding denormalization, underflow
// and overflow.
//
// To see the relation of Z to the tridiagonal matrix, let L be a
// unit lower bidiagonal matrix with sub-diagonals Z(2,4,6,,..) and
// let U be an upper bidiagonal matrix with 1's above and diagonal
// Z(1,3,5,,..). The tridiagonal is L*U or, if you prefer, the
// symmetric tridiagonal to which it is similar.
//
// info returns a status error. The return codes mean as follows:
//  0: The algorithm completed successfully.
//  1: A split was marked by a positive value in e.
//  2: Current block of Z not diagonalized after 100*n iterations (in inner
//     while loop). On exit Z holds a qd array with the same eigenvalues as
//     the given Z.
//  3: Termination criterion of outer while loop not met (program created more
//     than N unreduced blocks).
//
// z must have length at least 4*n, and must not contain any negative elements.
// Dlasq2 will panic otherwise.
//
// Dlasq2 is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dlasq2(n int, z []float64) (info int) {
	// TODO(btracey): make info an error.
	if len(z) < 4*n {
		panic(badZ)
	}
	const cbias = 1.5

	eps := dlamchP
	safmin := dlamchS
	tol := eps * 100
	tol2 := tol * tol
	if n < 0 {
		panic(nLT0)
	}
	if n == 0 {
		return info
	}
	if n == 1 {
		if z[0] < 0 {
			panic(negZ)
		}
		return info
	}
	if n == 2 {
		if z[1] < 0 || z[2] < 0 {
			panic("lapack: bad z value")
		} else if z[2] > z[0] {
			z[0], z[2] = z[2], z[0]
		}
		z[4] = z[0] + z[1] + z[2]
		if z[1] > z[2]*tol2 {
			t := 0.5 * (z[0] - z[2] + z[1])
			s := z[2] * (z[1] / t)
			if s <= t {
				s = z[2] * (z[1] / (t * (1 + math.Sqrt(1+s/t))))
			} else {
				s = z[2] * (z[1] / (t + math.Sqrt(t)*math.Sqrt(t+s)))
			}
			t = z[0] + s + z[1]
			z[2] *= z[0] / t
			z[0] = t
		}
		z[1] = z[2]
		z[5] = z[1] + z[0]
		return info
	}
	// Check for negative data and compute sums of q's and e's.
	z[2*n-1] = 0
	emin := z[1]
	var d, e, qmax float64
	var i1, n1 int
	for k := 0; k < 2*(n-1); k += 2 {
		if z[k] < 0 || z[k+1] < 0 {
			panic("lapack: bad z value")
		}
		d += z[k]
		e += z[k+1]
		qmax = math.Max(qmax, z[k])
		emin = math.Min(emin, z[k+1])
	}
	if z[2*(n-1)] < 0 {
		panic("lapack: bad z value")
	}
	d += z[2*(n-1)]
	qmax = math.Max(qmax, z[2*(n-1)])
	// Check for diagonality.
	if e == 0 {
		for k := 1; k < n; k++ {
			z[k] = z[2*k]
		}
		impl.Dlasrt(lapack.SortDecreasing, n, z)
		z[2*(n-1)] = d
		return info
	}
	trace := d + e
	// Check for zero data.
	if trace == 0 {
		z[2*(n-1)] = 0
		return info
	}
	// Rearrange data for locality: Z=(q1,qq1,e1,ee1,q2,qq2,e2,ee2,...).
	for k := 2 * n; k >= 2; k -= 2 {
		z[2*k-1] = 0
		z[2*k-2] = z[k-1]
		z[2*k-3] = 0
		z[2*k-4] = z[k-2]
	}
	i0 := 0
	n0 := n - 1

	// Reverse the qd-array, if warranted.
	// z[4*i0-3] --> z[4*(i0+1)-3-1] --> z[4*i0]
	if cbias*z[4*i0] < z[4*n0] {
		ipn4Out := 4 * (i0 + n0 + 2)
		for i4loop := 4 * (i0 + 1); i4loop <= 2*(i0+n0+1); i4loop += 4 {
			i4 := i4loop - 1
			ipn4 := ipn4Out - 1
			z[i4-3], z[ipn4-i4-4] = z[ipn4-i4-4], z[i4-3]
			z[i4-1], z[ipn4-i4-6] = z[ipn4-i4-6], z[i4-1]
		}
	}

	// Initial split checking via dqd and Li's test.
	pp := 0
	for k := 0; k < 2; k++ {
		d = z[4*n0+pp]
		for i4loop := 4*n0 + pp; i4loop >= 4*(i0+1)+pp; i4loop -= 4 {
			i4 := i4loop - 1
			if z[i4-1] <= tol2*d {
				z[i4-1] = math.Copysign(0, -1)
				d = z[i4-3]
			} else {
				d = z[i4-3] * (d / (d + z[i4-1]))
			}
		}
		// dqd maps Z to ZZ plus Li's test.
		emin = z[4*(i0+1)+pp]
		d = z[4*i0+pp]
		for i4loop := 4*(i0+1) + pp; i4loop <= 4*n0+pp; i4loop += 4 {
			i4 := i4loop - 1
			z[i4-2*pp-2] = d + z[i4-1]
			if z[i4-1] <= tol2*d {
				z[i4-1] = math.Copysign(0, -1)
				z[i4-2*pp-2] = d
				z[i4-2*pp] = 0
				d = z[i4+1]
			} else if safmin*z[i4+1] < z[i4-2*pp-2] && safmin*z[i4-2*pp-2] < z[i4+1] {
				tmp := z[i4+1] / z[i4-2*pp-2]
				z[i4-2*pp] = z[i4-1] * tmp
				d *= tmp
			} else {
				z[i4-2*pp] = z[i4+1] * (z[i4-1] / z[i4-2*pp-2])
				d = z[i4+1] * (d / z[i4-2*pp-2])
			}
			emin = math.Min(emin, z[i4-2*pp])
		}
		z[4*(n0+1)-pp-3] = d

		// Now find qmax.
		qmax = z[4*(i0+1)-pp-3]
		for i4loop := 4*(i0+1) - pp + 2; i4loop <= 4*(n0+1)+pp-2; i4loop += 4 {
			i4 := i4loop - 1
			qmax = math.Max(qmax, z[i4])
		}
		// Prepare for the next iteration on K.
		pp = 1 - pp
	}

	// Initialise variables to pass to DLASQ3.
	var ttype int
	var dmin1, dmin2, dn, dn1, dn2, g, tau float64
	var tempq float64
	iter := 2
	var nFail int
	nDiv := 2 * (n0 - i0)
	var i4 int
outer:
	for iwhila := 1; iwhila <= n+1; iwhila++ {
		// Test for completion.
		if n0 < 0 {
			// Move q's to the front.
			for k := 1; k < n; k++ {
				z[k] = z[4*k]
			}
			// Sort and compute sum of eigenvalues.
			impl.Dlasrt(lapack.SortDecreasing, n, z)
			e = 0
			for k := n - 1; k >= 0; k-- {
				e += z[k]
			}
			// Store trace, sum(eigenvalues) and information on performance.
			z[2*n] = trace
			z[2*n+1] = e
			z[2*n+2] = float64(iter)
			z[2*n+3] = float64(nDiv) / float64(n*n)
			z[2*n+4] = 100 * float64(nFail) / float64(iter)
			return info
		}

		// While array unfinished do
		// e[n0] holds the value of sigma when submatrix in i0:n0
		// splits from the rest of the array, but is negated.
		var desig float64
		var sigma float64
		if n0 != n-1 {
			sigma = -z[4*(n0+1)-2]
		}
		if sigma < 0 {
			info = 1
			return info
		}
		// Find last unreduced submatrix's top index i0, find qmax and
		// emin. Find Gershgorin-type bound if Q's much greater than E's.
		var emax float64
		if n0 > i0 {
			emin = math.Abs(z[4*(n0+1)-6])
		} else {
			emin = 0
		}
		qmin := z[4*(n0+1)-4]
		qmax = qmin
		zSmall := false
		for i4loop := 4 * (n0 + 1); i4loop >= 8; i4loop -= 4 {
			i4 = i4loop - 1
			if z[i4-5] <= 0 {
				zSmall = true
				break
			}
			if qmin >= 4*emax {
				qmin = math.Min(qmin, z[i4-3])
				emax = math.Max(emax, z[i4-5])
			}
			qmax = math.Max(qmax, z[i4-7]+z[i4-5])
			emin = math.Min(emin, z[i4-5])
		}
		if !zSmall {
			i4 = 3
		}
		i0 = (i4+1)/4 - 1
		pp = 0
		if n0-i0 > 1 {
			dee := z[4*i0]
			deemin := dee
			kmin := i0
			for i4loop := 4*(i0+1) + 1; i4loop <= 4*(n0+1)-3; i4loop += 4 {
				i4 := i4loop - 1
				dee = z[i4] * (dee / (dee + z[i4-2]))
				if dee <= deemin {
					deemin = dee
					kmin = (i4+4)/4 - 1
				}
			}
			if (kmin-i0)*2 < n0-kmin && deemin <= 0.5*z[4*n0] {
				ipn4Out := 4 * (i0 + n0 + 2)
				pp = 2
				for i4loop := 4 * (i0 + 1); i4loop <= 2*(i0+n0+1); i4loop += 4 {
					i4 := i4loop - 1
					ipn4 := ipn4Out - 1
					z[i4-3], z[ipn4-i4-4] = z[ipn4-i4-4], z[i4-3]
					z[i4-2], z[ipn4-i4-3] = z[ipn4-i4-3], z[i4-2]
					z[i4-1], z[ipn4-i4-6] = z[ipn4-i4-6], z[i4-1]
					z[i4], z[ipn4-i4-5] = z[ipn4-i4-5], z[i4]
				}
			}
		}
		// Put -(initial shift) into DMIN.
		dmin := -math.Max(0, qmin-2*math.Sqrt(qmin)*math.Sqrt(emax))

		// Now i0:n0 is unreduced.
		// PP = 0 for ping, PP = 1 for pong.
		// PP = 2 indicates that flipping was applied to the Z array and
		// 		and that the tests for deflation upon entry in Dlasq3
		// 		should not be performed.
		nbig := 100 * (n0 - i0 + 1)
		for iwhilb := 0; iwhilb < nbig; iwhilb++ {
			if i0 > n0 {
				continue outer
			}

			// While submatrix unfinished take a good dqds step.
			i0, n0, pp, dmin, sigma, desig, qmax, nFail, iter, nDiv, ttype, dmin1, dmin2, dn, dn1, dn2, g, tau =
				impl.Dlasq3(i0, n0, z, pp, dmin, sigma, desig, qmax, nFail, iter, nDiv, ttype, dmin1, dmin2, dn, dn1, dn2, g, tau)

			pp = 1 - pp
			// When emin is very small check for splits.
			if pp == 0 && n0-i0 >= 3 {
				if z[4*(n0+1)-1] <= tol2*qmax || z[4*(n0+1)-2] <= tol2*sigma {
					splt := i0 - 1
					qmax = z[4*i0]
					emin = z[4*(i0+1)-2]
					oldemn := z[4*(i0+1)-1]
					for i4loop := 4 * (i0 + 1); i4loop <= 4*(n0-2); i4loop += 4 {
						i4 := i4loop - 1
						if z[i4] <= tol2*z[i4-3] || z[i4-1] <= tol2*sigma {
							z[i4-1] = -sigma
							splt = i4 / 4
							qmax = 0
							emin = z[i4+3]
							oldemn = z[i4+4]
						} else {
							qmax = math.Max(qmax, z[i4+1])
							emin = math.Min(emin, z[i4-1])
							oldemn = math.Min(oldemn, z[i4])
						}
					}
					z[4*(n0+1)-2] = emin
					z[4*(n0+1)-1] = oldemn
					i0 = splt + 1
				}
			}
		}
		// Maximum number of iterations exceeded, restore the shift
		// sigma and place the new d's and e's in a qd array.
		// This might need to be done for several blocks.
		info = 2
		i1 = i0
		n1 = n0
		for {
			tempq = z[4*i0]
			z[4*i0] += sigma
			for k := i0 + 1; k <= n0; k++ {
				tempe := z[4*(k+1)-6]
				z[4*(k+1)-6] *= tempq / z[4*(k+1)-8]
				tempq = z[4*k]
				z[4*k] += sigma + tempe - z[4*(k+1)-6]
			}
			// Prepare to do this on the previous block if there is one.
			if i1 <= 0 {
				break
			}
			n1 = i1 - 1
			for i1 >= 1 && z[4*(i1+1)-6] >= 0 {
				i1 -= 1
			}
			sigma = -z[4*(n1+1)-2]
		}
		for k := 0; k < n; k++ {
			z[2*k] = z[4*k]
			// Only the block 1..N0 is unfinished.  The rest of the e's
			// must be essentially zero, although sometimes other data
			// has been stored in them.
			if k < n0 {
				z[2*(k+1)-1] = z[4*(k+1)-1]
			} else {
				z[2*(k+1)] = 0
			}
		}
		return info
	}
	info = 3
	return info
}
