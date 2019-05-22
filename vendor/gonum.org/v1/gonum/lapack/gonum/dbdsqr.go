// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"math"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/gonum/lapack"
)

// Dbdsqr performs a singular value decomposition of a real n×n bidiagonal matrix.
//
// The SVD of the bidiagonal matrix B is
//  B = Q * S * P^T
// where S is a diagonal matrix of singular values, Q is an orthogonal matrix of
// left singular vectors, and P is an orthogonal matrix of right singular vectors.
//
// Q and P are only computed if requested. If left singular vectors are requested,
// this routine returns U * Q instead of Q, and if right singular vectors are
// requested P^T * VT is returned instead of P^T.
//
// Frequently Dbdsqr is used in conjunction with Dgebrd which reduces a general
// matrix A into bidiagonal form. In this case, the SVD of A is
//  A = (U * Q) * S * (P^T * VT)
//
// This routine may also compute Q^T * C.
//
// d and e contain the elements of the bidiagonal matrix b. d must have length at
// least n, and e must have length at least n-1. Dbdsqr will panic if there is
// insufficient length. On exit, D contains the singular values of B in decreasing
// order.
//
// VT is a matrix of size n×ncvt whose elements are stored in vt. The elements
// of vt are modified to contain P^T * VT on exit. VT is not used if ncvt == 0.
//
// U is a matrix of size nru×n whose elements are stored in u. The elements
// of u are modified to contain U * Q on exit. U is not used if nru == 0.
//
// C is a matrix of size n×ncc whose elements are stored in c. The elements
// of c are modified to contain Q^T * C on exit. C is not used if ncc == 0.
//
// work contains temporary storage and must have length at least 4*(n-1). Dbdsqr
// will panic if there is insufficient working memory.
//
// Dbdsqr returns whether the decomposition was successful.
//
// Dbdsqr is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dbdsqr(uplo blas.Uplo, n, ncvt, nru, ncc int, d, e, vt []float64, ldvt int, u []float64, ldu int, c []float64, ldc int, work []float64) (ok bool) {
	switch {
	case uplo != blas.Upper && uplo != blas.Lower:
		panic(badUplo)
	case n < 0:
		panic(nLT0)
	case ncvt < 0:
		panic(ncvtLT0)
	case nru < 0:
		panic(nruLT0)
	case ncc < 0:
		panic(nccLT0)
	case ldvt < max(1, ncvt):
		panic(badLdVT)
	case (ldu < max(1, n) && nru > 0) || (ldu < 1 && nru == 0):
		panic(badLdU)
	case ldc < max(1, ncc):
		panic(badLdC)
	}

	// Quick return if possible.
	if n == 0 {
		return true
	}

	if len(vt) < (n-1)*ldvt+ncvt && ncvt != 0 {
		panic(shortVT)
	}
	if len(u) < (nru-1)*ldu+n && nru != 0 {
		panic(shortU)
	}
	if len(c) < (n-1)*ldc+ncc && ncc != 0 {
		panic(shortC)
	}
	if len(d) < n {
		panic(shortD)
	}
	if len(e) < n-1 {
		panic(shortE)
	}
	if len(work) < 4*(n-1) {
		panic(shortWork)
	}

	var info int
	bi := blas64.Implementation()
	const maxIter = 6

	if n != 1 {
		// If the singular vectors do not need to be computed, use qd algorithm.
		if !(ncvt > 0 || nru > 0 || ncc > 0) {
			info = impl.Dlasq1(n, d, e, work)
			// If info is 2 dqds didn't finish, and so try to.
			if info != 2 {
				return info == 0
			}
		}
		nm1 := n - 1
		nm12 := nm1 + nm1
		nm13 := nm12 + nm1
		idir := 0

		eps := dlamchE
		unfl := dlamchS
		lower := uplo == blas.Lower
		var cs, sn, r float64
		if lower {
			for i := 0; i < n-1; i++ {
				cs, sn, r = impl.Dlartg(d[i], e[i])
				d[i] = r
				e[i] = sn * d[i+1]
				d[i+1] *= cs
				work[i] = cs
				work[nm1+i] = sn
			}
			if nru > 0 {
				impl.Dlasr(blas.Right, lapack.Variable, lapack.Forward, nru, n, work, work[n-1:], u, ldu)
			}
			if ncc > 0 {
				impl.Dlasr(blas.Left, lapack.Variable, lapack.Forward, n, ncc, work, work[n-1:], c, ldc)
			}
		}
		// Compute singular values to a relative accuracy of tol. If tol is negative
		// the values will be computed to an absolute accuracy of math.Abs(tol) * norm(b)
		tolmul := math.Max(10, math.Min(100, math.Pow(eps, -1.0/8)))
		tol := tolmul * eps
		var smax float64
		for i := 0; i < n; i++ {
			smax = math.Max(smax, math.Abs(d[i]))
		}
		for i := 0; i < n-1; i++ {
			smax = math.Max(smax, math.Abs(e[i]))
		}

		var sminl float64
		var thresh float64
		if tol >= 0 {
			sminoa := math.Abs(d[0])
			if sminoa != 0 {
				mu := sminoa
				for i := 1; i < n; i++ {
					mu = math.Abs(d[i]) * (mu / (mu + math.Abs(e[i-1])))
					sminoa = math.Min(sminoa, mu)
					if sminoa == 0 {
						break
					}
				}
			}
			sminoa = sminoa / math.Sqrt(float64(n))
			thresh = math.Max(tol*sminoa, float64(maxIter*n*n)*unfl)
		} else {
			thresh = math.Max(math.Abs(tol)*smax, float64(maxIter*n*n)*unfl)
		}
		// Prepare for the main iteration loop for the singular values.
		maxIt := maxIter * n * n
		iter := 0
		oldl2 := -1
		oldm := -1
		// m points to the last element of unconverged part of matrix.
		m := n

	Outer:
		for m > 1 {
			if iter > maxIt {
				info = 0
				for i := 0; i < n-1; i++ {
					if e[i] != 0 {
						info++
					}
				}
				return info == 0
			}
			// Find diagonal block of matrix to work on.
			if tol < 0 && math.Abs(d[m-1]) <= thresh {
				d[m-1] = 0
			}
			smax = math.Abs(d[m-1])
			smin := smax
			var l2 int
			var broke bool
			for l3 := 0; l3 < m-1; l3++ {
				l2 = m - l3 - 2
				abss := math.Abs(d[l2])
				abse := math.Abs(e[l2])
				if tol < 0 && abss <= thresh {
					d[l2] = 0
				}
				if abse <= thresh {
					broke = true
					break
				}
				smin = math.Min(smin, abss)
				smax = math.Max(math.Max(smax, abss), abse)
			}
			if broke {
				e[l2] = 0
				if l2 == m-2 {
					// Convergence of bottom singular value, return to top.
					m--
					continue
				}
				l2++
			} else {
				l2 = 0
			}
			// e[ll] through e[m-2] are nonzero, e[ll-1] is zero
			if l2 == m-2 {
				// Handle 2×2 block separately.
				var sinr, cosr, sinl, cosl float64
				d[m-1], d[m-2], sinr, cosr, sinl, cosl = impl.Dlasv2(d[m-2], e[m-2], d[m-1])
				e[m-2] = 0
				if ncvt > 0 {
					bi.Drot(ncvt, vt[(m-2)*ldvt:], 1, vt[(m-1)*ldvt:], 1, cosr, sinr)
				}
				if nru > 0 {
					bi.Drot(nru, u[m-2:], ldu, u[m-1:], ldu, cosl, sinl)
				}
				if ncc > 0 {
					bi.Drot(ncc, c[(m-2)*ldc:], 1, c[(m-1)*ldc:], 1, cosl, sinl)
				}
				m -= 2
				continue
			}
			// If working on a new submatrix, choose shift direction from larger end
			// diagonal element toward smaller.
			if l2 > oldm-1 || m-1 < oldl2 {
				if math.Abs(d[l2]) >= math.Abs(d[m-1]) {
					idir = 1
				} else {
					idir = 2
				}
			}
			// Apply convergence tests.
			// TODO(btracey): There is a lot of similar looking code here. See
			// if there is a better way to de-duplicate.
			if idir == 1 {
				// Run convergence test in forward direction.
				// First apply standard test to bottom of matrix.
				if math.Abs(e[m-2]) <= math.Abs(tol)*math.Abs(d[m-1]) || (tol < 0 && math.Abs(e[m-2]) <= thresh) {
					e[m-2] = 0
					continue
				}
				if tol >= 0 {
					// If relative accuracy desired, apply convergence criterion forward.
					mu := math.Abs(d[l2])
					sminl = mu
					for l3 := l2; l3 < m-1; l3++ {
						if math.Abs(e[l3]) <= tol*mu {
							e[l3] = 0
							continue Outer
						}
						mu = math.Abs(d[l3+1]) * (mu / (mu + math.Abs(e[l3])))
						sminl = math.Min(sminl, mu)
					}
				}
			} else {
				// Run convergence test in backward direction.
				// First apply standard test to top of matrix.
				if math.Abs(e[l2]) <= math.Abs(tol)*math.Abs(d[l2]) || (tol < 0 && math.Abs(e[l2]) <= thresh) {
					e[l2] = 0
					continue
				}
				if tol >= 0 {
					// If relative accuracy desired, apply convergence criterion backward.
					mu := math.Abs(d[m-1])
					sminl = mu
					for l3 := m - 2; l3 >= l2; l3-- {
						if math.Abs(e[l3]) <= tol*mu {
							e[l3] = 0
							continue Outer
						}
						mu = math.Abs(d[l3]) * (mu / (mu + math.Abs(e[l3])))
						sminl = math.Min(sminl, mu)
					}
				}
			}
			oldl2 = l2
			oldm = m
			// Compute shift. First, test if shifting would ruin relative accuracy,
			// and if so set the shift to zero.
			var shift float64
			if tol >= 0 && float64(n)*tol*(sminl/smax) <= math.Max(eps, (1.0/100)*tol) {
				shift = 0
			} else {
				var sl2 float64
				if idir == 1 {
					sl2 = math.Abs(d[l2])
					shift, _ = impl.Dlas2(d[m-2], e[m-2], d[m-1])
				} else {
					sl2 = math.Abs(d[m-1])
					shift, _ = impl.Dlas2(d[l2], e[l2], d[l2+1])
				}
				// Test if shift is negligible
				if sl2 > 0 {
					if (shift/sl2)*(shift/sl2) < eps {
						shift = 0
					}
				}
			}
			iter += m - l2 + 1
			// If no shift, do simplified QR iteration.
			if shift == 0 {
				if idir == 1 {
					cs := 1.0
					oldcs := 1.0
					var sn, r, oldsn float64
					for i := l2; i < m-1; i++ {
						cs, sn, r = impl.Dlartg(d[i]*cs, e[i])
						if i > l2 {
							e[i-1] = oldsn * r
						}
						oldcs, oldsn, d[i] = impl.Dlartg(oldcs*r, d[i+1]*sn)
						work[i-l2] = cs
						work[i-l2+nm1] = sn
						work[i-l2+nm12] = oldcs
						work[i-l2+nm13] = oldsn
					}
					h := d[m-1] * cs
					d[m-1] = h * oldcs
					e[m-2] = h * oldsn
					if ncvt > 0 {
						impl.Dlasr(blas.Left, lapack.Variable, lapack.Forward, m-l2, ncvt, work, work[n-1:], vt[l2*ldvt:], ldvt)
					}
					if nru > 0 {
						impl.Dlasr(blas.Right, lapack.Variable, lapack.Forward, nru, m-l2, work[nm12:], work[nm13:], u[l2:], ldu)
					}
					if ncc > 0 {
						impl.Dlasr(blas.Left, lapack.Variable, lapack.Forward, m-l2, ncc, work[nm12:], work[nm13:], c[l2*ldc:], ldc)
					}
					if math.Abs(e[m-2]) < thresh {
						e[m-2] = 0
					}
				} else {
					cs := 1.0
					oldcs := 1.0
					var sn, r, oldsn float64
					for i := m - 1; i >= l2+1; i-- {
						cs, sn, r = impl.Dlartg(d[i]*cs, e[i-1])
						if i < m-1 {
							e[i] = oldsn * r
						}
						oldcs, oldsn, d[i] = impl.Dlartg(oldcs*r, d[i-1]*sn)
						work[i-l2-1] = cs
						work[i-l2+nm1-1] = -sn
						work[i-l2+nm12-1] = oldcs
						work[i-l2+nm13-1] = -oldsn
					}
					h := d[l2] * cs
					d[l2] = h * oldcs
					e[l2] = h * oldsn
					if ncvt > 0 {
						impl.Dlasr(blas.Left, lapack.Variable, lapack.Backward, m-l2, ncvt, work[nm12:], work[nm13:], vt[l2*ldvt:], ldvt)
					}
					if nru > 0 {
						impl.Dlasr(blas.Right, lapack.Variable, lapack.Backward, nru, m-l2, work, work[n-1:], u[l2:], ldu)
					}
					if ncc > 0 {
						impl.Dlasr(blas.Left, lapack.Variable, lapack.Backward, m-l2, ncc, work, work[n-1:], c[l2*ldc:], ldc)
					}
					if math.Abs(e[l2]) <= thresh {
						e[l2] = 0
					}
				}
			} else {
				// Use nonzero shift.
				if idir == 1 {
					// Chase bulge from top to bottom. Save cosines and sines for
					// later singular vector updates.
					f := (math.Abs(d[l2]) - shift) * (math.Copysign(1, d[l2]) + shift/d[l2])
					g := e[l2]
					var cosl, sinl float64
					for i := l2; i < m-1; i++ {
						cosr, sinr, r := impl.Dlartg(f, g)
						if i > l2 {
							e[i-1] = r
						}
						f = cosr*d[i] + sinr*e[i]
						e[i] = cosr*e[i] - sinr*d[i]
						g = sinr * d[i+1]
						d[i+1] *= cosr
						cosl, sinl, r = impl.Dlartg(f, g)
						d[i] = r
						f = cosl*e[i] + sinl*d[i+1]
						d[i+1] = cosl*d[i+1] - sinl*e[i]
						if i < m-2 {
							g = sinl * e[i+1]
							e[i+1] = cosl * e[i+1]
						}
						work[i-l2] = cosr
						work[i-l2+nm1] = sinr
						work[i-l2+nm12] = cosl
						work[i-l2+nm13] = sinl
					}
					e[m-2] = f
					if ncvt > 0 {
						impl.Dlasr(blas.Left, lapack.Variable, lapack.Forward, m-l2, ncvt, work, work[n-1:], vt[l2*ldvt:], ldvt)
					}
					if nru > 0 {
						impl.Dlasr(blas.Right, lapack.Variable, lapack.Forward, nru, m-l2, work[nm12:], work[nm13:], u[l2:], ldu)
					}
					if ncc > 0 {
						impl.Dlasr(blas.Left, lapack.Variable, lapack.Forward, m-l2, ncc, work[nm12:], work[nm13:], c[l2*ldc:], ldc)
					}
					if math.Abs(e[m-2]) <= thresh {
						e[m-2] = 0
					}
				} else {
					// Chase bulge from top to bottom. Save cosines and sines for
					// later singular vector updates.
					f := (math.Abs(d[m-1]) - shift) * (math.Copysign(1, d[m-1]) + shift/d[m-1])
					g := e[m-2]
					for i := m - 1; i > l2; i-- {
						cosr, sinr, r := impl.Dlartg(f, g)
						if i < m-1 {
							e[i] = r
						}
						f = cosr*d[i] + sinr*e[i-1]
						e[i-1] = cosr*e[i-1] - sinr*d[i]
						g = sinr * d[i-1]
						d[i-1] *= cosr
						cosl, sinl, r := impl.Dlartg(f, g)
						d[i] = r
						f = cosl*e[i-1] + sinl*d[i-1]
						d[i-1] = cosl*d[i-1] - sinl*e[i-1]
						if i > l2+1 {
							g = sinl * e[i-2]
							e[i-2] *= cosl
						}
						work[i-l2-1] = cosr
						work[i-l2+nm1-1] = -sinr
						work[i-l2+nm12-1] = cosl
						work[i-l2+nm13-1] = -sinl
					}
					e[l2] = f
					if math.Abs(e[l2]) <= thresh {
						e[l2] = 0
					}
					if ncvt > 0 {
						impl.Dlasr(blas.Left, lapack.Variable, lapack.Backward, m-l2, ncvt, work[nm12:], work[nm13:], vt[l2*ldvt:], ldvt)
					}
					if nru > 0 {
						impl.Dlasr(blas.Right, lapack.Variable, lapack.Backward, nru, m-l2, work, work[n-1:], u[l2:], ldu)
					}
					if ncc > 0 {
						impl.Dlasr(blas.Left, lapack.Variable, lapack.Backward, m-l2, ncc, work, work[n-1:], c[l2*ldc:], ldc)
					}
				}
			}
		}
	}

	// All singular values converged, make them positive.
	for i := 0; i < n; i++ {
		if d[i] < 0 {
			d[i] *= -1
			if ncvt > 0 {
				bi.Dscal(ncvt, -1, vt[i*ldvt:], 1)
			}
		}
	}

	// Sort the singular values in decreasing order.
	for i := 0; i < n-1; i++ {
		isub := 0
		smin := d[0]
		for j := 1; j < n-i; j++ {
			if d[j] <= smin {
				isub = j
				smin = d[j]
			}
		}
		if isub != n-i {
			// Swap singular values and vectors.
			d[isub] = d[n-i-1]
			d[n-i-1] = smin
			if ncvt > 0 {
				bi.Dswap(ncvt, vt[isub*ldvt:], 1, vt[(n-i-1)*ldvt:], 1)
			}
			if nru > 0 {
				bi.Dswap(nru, u[isub:], ldu, u[n-i-1:], ldu)
			}
			if ncc > 0 {
				bi.Dswap(ncc, c[isub*ldc:], 1, c[(n-i-1)*ldc:], 1)
			}
		}
	}
	info = 0
	for i := 0; i < n-1; i++ {
		if e[i] != 0 {
			info++
		}
	}
	return info == 0
}
