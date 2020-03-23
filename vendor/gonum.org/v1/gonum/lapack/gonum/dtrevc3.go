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

// Dtrevc3 computes some or all of the right and/or left eigenvectors of an n×n
// upper quasi-triangular matrix T in Schur canonical form. Matrices of this
// type are produced by the Schur factorization of a real general matrix A
//  A = Q T Qᵀ,
// as computed by Dhseqr.
//
// The right eigenvector x of T corresponding to an
// eigenvalue λ is defined by
//  T x = λ x,
// and the left eigenvector y is defined by
//  yᵀ T = λ yᵀ.
//
// The eigenvalues are read directly from the diagonal blocks of T.
//
// This routine returns the matrices X and/or Y of right and left eigenvectors
// of T, or the products Q*X and/or Q*Y, where Q is an input matrix. If Q is the
// orthogonal factor that reduces a matrix A to Schur form T, then Q*X and Q*Y
// are the matrices of right and left eigenvectors of A.
//
// If side == lapack.EVRight, only right eigenvectors will be computed.
// If side == lapack.EVLeft, only left eigenvectors will be computed.
// If side == lapack.EVBoth, both right and left eigenvectors will be computed.
// For other values of side, Dtrevc3 will panic.
//
// If howmny == lapack.EVAll, all right and/or left eigenvectors will be
// computed.
// If howmny == lapack.EVAllMulQ, all right and/or left eigenvectors will be
// computed and multiplied from left by the matrices in VR and/or VL.
// If howmny == lapack.EVSelected, right and/or left eigenvectors will be
// computed as indicated by selected.
// For other values of howmny, Dtrevc3 will panic.
//
// selected specifies which eigenvectors will be computed. It must have length n
// if howmny == lapack.EVSelected, and it is not referenced otherwise.
// If w_j is a real eigenvalue, the corresponding real eigenvector will be
// computed if selected[j] is true.
// If w_j and w_{j+1} are the real and imaginary parts of a complex eigenvalue,
// the corresponding complex eigenvector is computed if either selected[j] or
// selected[j+1] is true, and on return selected[j] will be set to true and
// selected[j+1] will be set to false.
//
// VL and VR are n×mm matrices. If howmny is lapack.EVAll or
// lapack.AllEVMulQ, mm must be at least n. If howmny is
// lapack.EVSelected, mm must be large enough to store the selected
// eigenvectors. Each selected real eigenvector occupies one column and each
// selected complex eigenvector occupies two columns. If mm is not sufficiently
// large, Dtrevc3 will panic.
//
// On entry, if howmny is lapack.EVAllMulQ, it is assumed that VL (if side
// is lapack.EVLeft or lapack.EVBoth) contains an n×n matrix QL,
// and that VR (if side is lapack.EVLeft or lapack.EVBoth) contains
// an n×n matrix QR. QL and QR are typically the orthogonal matrix Q of Schur
// vectors returned by Dhseqr.
//
// On return, if side is lapack.EVLeft or lapack.EVBoth,
// VL will contain:
//  if howmny == lapack.EVAll,      the matrix Y of left eigenvectors of T,
//  if howmny == lapack.EVAllMulQ,  the matrix Q*Y,
//  if howmny == lapack.EVSelected, the left eigenvectors of T specified by
//                                  selected, stored consecutively in the
//                                  columns of VL, in the same order as their
//                                  eigenvalues.
// VL is not referenced if side == lapack.EVRight.
//
// On return, if side is lapack.EVRight or lapack.EVBoth,
// VR will contain:
//  if howmny == lapack.EVAll,      the matrix X of right eigenvectors of T,
//  if howmny == lapack.EVAllMulQ,  the matrix Q*X,
//  if howmny == lapack.EVSelected, the left eigenvectors of T specified by
//                                  selected, stored consecutively in the
//                                  columns of VR, in the same order as their
//                                  eigenvalues.
// VR is not referenced if side == lapack.EVLeft.
//
// Complex eigenvectors corresponding to a complex eigenvalue are stored in VL
// and VR in two consecutive columns, the first holding the real part, and the
// second the imaginary part.
//
// Each eigenvector will be normalized so that the element of largest magnitude
// has magnitude 1. Here the magnitude of a complex number (x,y) is taken to be
// |x| + |y|.
//
// work must have length at least lwork and lwork must be at least max(1,3*n),
// otherwise Dtrevc3 will panic. For optimum performance, lwork should be at
// least n+2*n*nb, where nb is the optimal blocksize.
//
// If lwork == -1, instead of performing Dtrevc3, the function only estimates
// the optimal workspace size based on n and stores it into work[0].
//
// Dtrevc3 returns the number of columns in VL and/or VR actually used to store
// the eigenvectors.
//
// Dtrevc3 is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dtrevc3(side lapack.EVSide, howmny lapack.EVHowMany, selected []bool, n int, t []float64, ldt int, vl []float64, ldvl int, vr []float64, ldvr int, mm int, work []float64, lwork int) (m int) {
	bothv := side == lapack.EVBoth
	rightv := side == lapack.EVRight || bothv
	leftv := side == lapack.EVLeft || bothv
	switch {
	case !rightv && !leftv:
		panic(badEVSide)
	case howmny != lapack.EVAll && howmny != lapack.EVAllMulQ && howmny != lapack.EVSelected:
		panic(badEVHowMany)
	case n < 0:
		panic(nLT0)
	case ldt < max(1, n):
		panic(badLdT)
	case mm < 0:
		panic(mmLT0)
	case ldvl < 1:
		// ldvl and ldvr are also checked below after the computation of
		// m (number of columns of VL and VR) in case of howmny == EVSelected.
		panic(badLdVL)
	case ldvr < 1:
		panic(badLdVR)
	case lwork < max(1, 3*n) && lwork != -1:
		panic(badLWork)
	case len(work) < max(1, lwork):
		panic(shortWork)
	}

	// Quick return if possible.
	if n == 0 {
		work[0] = 1
		return 0
	}

	// Normally we don't check slice lengths until after the workspace
	// query. However, even in case of the workspace query we need to
	// compute and return the value of m, and since the computation accesses t,
	// we put the length check of t here.
	if len(t) < (n-1)*ldt+n {
		panic(shortT)
	}

	if howmny == lapack.EVSelected {
		if len(selected) != n {
			panic(badLenSelected)
		}
		// Set m to the number of columns required to store the selected
		// eigenvectors, and standardize the slice selected.
		// Each selected real eigenvector occupies one column and each
		// selected complex eigenvector occupies two columns.
		for j := 0; j < n; {
			if j == n-1 || t[(j+1)*ldt+j] == 0 {
				// Diagonal 1×1 block corresponding to a
				// real eigenvalue.
				if selected[j] {
					m++
				}
				j++
			} else {
				// Diagonal 2×2 block corresponding to a
				// complex eigenvalue.
				if selected[j] || selected[j+1] {
					selected[j] = true
					selected[j+1] = false
					m += 2
				}
				j += 2
			}
		}
	} else {
		m = n
	}
	if mm < m {
		panic(badMm)
	}

	// Quick return in case of a workspace query.
	nb := impl.Ilaenv(1, "DTREVC", string(side)+string(howmny), n, -1, -1, -1)
	if lwork == -1 {
		work[0] = float64(n + 2*n*nb)
		return m
	}

	// Quick return if no eigenvectors were selected.
	if m == 0 {
		return 0
	}

	switch {
	case leftv && ldvl < mm:
		panic(badLdVL)
	case leftv && len(vl) < (n-1)*ldvl+mm:
		panic(shortVL)

	case rightv && ldvr < mm:
		panic(badLdVR)
	case rightv && len(vr) < (n-1)*ldvr+mm:
		panic(shortVR)
	}

	// Use blocked version of back-transformation if sufficient workspace.
	// Zero-out the workspace to avoid potential NaN propagation.
	const (
		nbmin = 8
		nbmax = 128
	)
	if howmny == lapack.EVAllMulQ && lwork >= n+2*n*nbmin {
		nb = min((lwork-n)/(2*n), nbmax)
		impl.Dlaset(blas.All, n, 1+2*nb, 0, 0, work[:n+2*nb*n], 1+2*nb)
	} else {
		nb = 1
	}

	// Set the constants to control overflow.
	ulp := dlamchP
	smlnum := float64(n) / ulp * dlamchS
	bignum := (1 - ulp) / smlnum

	// Split work into a vector of column norms and an n×2*nb matrix b.
	norms := work[:n]
	ldb := 2 * nb
	b := work[n : n+n*ldb]

	// Compute 1-norm of each column of strictly upper triangular part of T
	// to control overflow in triangular solver.
	norms[0] = 0
	for j := 1; j < n; j++ {
		var cn float64
		for i := 0; i < j; i++ {
			cn += math.Abs(t[i*ldt+j])
		}
		norms[j] = cn
	}

	bi := blas64.Implementation()

	var (
		x [4]float64

		iv int // Index of column in current block.
		is int

		// ip is used below to specify the real or complex eigenvalue:
		//  ip == 0, real eigenvalue,
		//        1, first  of conjugate complex pair (wr,wi),
		//       -1, second of conjugate complex pair (wr,wi).
		ip        int
		iscomplex [nbmax]int // Stores ip for each column in current block.
	)

	if side == lapack.EVLeft {
		goto leftev
	}

	// Compute right eigenvectors.

	// For complex right vector, iv-1 is for real part and iv for complex
	// part. Non-blocked version always uses iv=1, blocked version starts
	// with iv=nb-1 and goes down to 0 or 1.
	iv = max(2, nb) - 1
	ip = 0
	is = m - 1
	for ki := n - 1; ki >= 0; ki-- {
		if ip == -1 {
			// Previous iteration (ki+1) was second of
			// conjugate pair, so this ki is first of
			// conjugate pair.
			ip = 1
			continue
		}

		if ki == 0 || t[ki*ldt+ki-1] == 0 {
			// Last column or zero on sub-diagonal, so this
			// ki must be real eigenvalue.
			ip = 0
		} else {
			// Non-zero on sub-diagonal, so this ki is
			// second of conjugate pair.
			ip = -1
		}

		if howmny == lapack.EVSelected {
			if ip == 0 {
				if !selected[ki] {
					continue
				}
			} else if !selected[ki-1] {
				continue
			}
		}

		// Compute the ki-th eigenvalue (wr,wi).
		wr := t[ki*ldt+ki]
		var wi float64
		if ip != 0 {
			wi = math.Sqrt(math.Abs(t[ki*ldt+ki-1])) * math.Sqrt(math.Abs(t[(ki-1)*ldt+ki]))
		}
		smin := math.Max(ulp*(math.Abs(wr)+math.Abs(wi)), smlnum)

		if ip == 0 {
			// Real right eigenvector.

			b[ki*ldb+iv] = 1
			// Form right-hand side.
			for k := 0; k < ki; k++ {
				b[k*ldb+iv] = -t[k*ldt+ki]
			}
			// Solve upper quasi-triangular system:
			//  [ T[0:ki,0:ki] - wr ]*X = scale*b.
			for j := ki - 1; j >= 0; {
				if j == 0 || t[j*ldt+j-1] == 0 {
					// 1×1 diagonal block.
					scale, xnorm, _ := impl.Dlaln2(false, 1, 1, smin, 1, t[j*ldt+j:], ldt,
						1, 1, b[j*ldb+iv:], ldb, wr, 0, x[:1], 2)
					// Scale X[0,0] to avoid overflow when updating the
					// right-hand side.
					if xnorm > 1 && norms[j] > bignum/xnorm {
						x[0] /= xnorm
						scale /= xnorm
					}
					// Scale if necessary.
					if scale != 1 {
						bi.Dscal(ki+1, scale, b[iv:], ldb)
					}
					b[j*ldb+iv] = x[0]
					// Update right-hand side.
					bi.Daxpy(j, -x[0], t[j:], ldt, b[iv:], ldb)
					j--
				} else {
					// 2×2 diagonal block.
					scale, xnorm, _ := impl.Dlaln2(false, 2, 1, smin, 1, t[(j-1)*ldt+j-1:], ldt,
						1, 1, b[(j-1)*ldb+iv:], ldb, wr, 0, x[:3], 2)
					// Scale X[0,0] and X[1,0] to avoid overflow
					// when updating the right-hand side.
					if xnorm > 1 {
						beta := math.Max(norms[j-1], norms[j])
						if beta > bignum/xnorm {
							x[0] /= xnorm
							x[2] /= xnorm
							scale /= xnorm
						}
					}
					// Scale if necessary.
					if scale != 1 {
						bi.Dscal(ki+1, scale, b[iv:], ldb)
					}
					b[(j-1)*ldb+iv] = x[0]
					b[j*ldb+iv] = x[2]
					// Update right-hand side.
					bi.Daxpy(j-1, -x[0], t[j-1:], ldt, b[iv:], ldb)
					bi.Daxpy(j-1, -x[2], t[j:], ldt, b[iv:], ldb)
					j -= 2
				}
			}
			// Copy the vector x or Q*x to VR and normalize.
			switch {
			case howmny != lapack.EVAllMulQ:
				// No back-transform: copy x to VR and normalize.
				bi.Dcopy(ki+1, b[iv:], ldb, vr[is:], ldvr)
				ii := bi.Idamax(ki+1, vr[is:], ldvr)
				remax := 1 / math.Abs(vr[ii*ldvr+is])
				bi.Dscal(ki+1, remax, vr[is:], ldvr)
				for k := ki + 1; k < n; k++ {
					vr[k*ldvr+is] = 0
				}
			case nb == 1:
				// Version 1: back-transform each vector with GEMV, Q*x.
				if ki > 0 {
					bi.Dgemv(blas.NoTrans, n, ki, 1, vr, ldvr, b[iv:], ldb,
						b[ki*ldb+iv], vr[ki:], ldvr)
				}
				ii := bi.Idamax(n, vr[ki:], ldvr)
				remax := 1 / math.Abs(vr[ii*ldvr+ki])
				bi.Dscal(n, remax, vr[ki:], ldvr)
			default:
				// Version 2: back-transform block of vectors with GEMM.
				// Zero out below vector.
				for k := ki + 1; k < n; k++ {
					b[k*ldb+iv] = 0
				}
				iscomplex[iv] = ip
				// Back-transform and normalization is done below.
			}
		} else {
			// Complex right eigenvector.

			// Initial solve
			//  [ ( T[ki-1,ki-1] T[ki-1,ki] ) - (wr + i*wi) ]*X = 0.
			//  [ ( T[ki,  ki-1] T[ki,  ki] )               ]
			if math.Abs(t[(ki-1)*ldt+ki]) >= math.Abs(t[ki*ldt+ki-1]) {
				b[(ki-1)*ldb+iv-1] = 1
				b[ki*ldb+iv] = wi / t[(ki-1)*ldt+ki]
			} else {
				b[(ki-1)*ldb+iv-1] = -wi / t[ki*ldt+ki-1]
				b[ki*ldb+iv] = 1
			}
			b[ki*ldb+iv-1] = 0
			b[(ki-1)*ldb+iv] = 0
			// Form right-hand side.
			for k := 0; k < ki-1; k++ {
				b[k*ldb+iv-1] = -b[(ki-1)*ldb+iv-1] * t[k*ldt+ki-1]
				b[k*ldb+iv] = -b[ki*ldb+iv] * t[k*ldt+ki]
			}
			// Solve upper quasi-triangular system:
			//  [ T[0:ki-1,0:ki-1] - (wr+i*wi) ]*X = scale*(b1+i*b2)
			for j := ki - 2; j >= 0; {
				if j == 0 || t[j*ldt+j-1] == 0 {
					// 1×1 diagonal block.

					scale, xnorm, _ := impl.Dlaln2(false, 1, 2, smin, 1, t[j*ldt+j:], ldt,
						1, 1, b[j*ldb+iv-1:], ldb, wr, wi, x[:2], 2)
					// Scale X[0,0] and X[0,1] to avoid
					// overflow when updating the right-hand side.
					if xnorm > 1 && norms[j] > bignum/xnorm {
						x[0] /= xnorm
						x[1] /= xnorm
						scale /= xnorm
					}
					// Scale if necessary.
					if scale != 1 {
						bi.Dscal(ki+1, scale, b[iv-1:], ldb)
						bi.Dscal(ki+1, scale, b[iv:], ldb)
					}
					b[j*ldb+iv-1] = x[0]
					b[j*ldb+iv] = x[1]
					// Update the right-hand side.
					bi.Daxpy(j, -x[0], t[j:], ldt, b[iv-1:], ldb)
					bi.Daxpy(j, -x[1], t[j:], ldt, b[iv:], ldb)
					j--
				} else {
					// 2×2 diagonal block.

					scale, xnorm, _ := impl.Dlaln2(false, 2, 2, smin, 1, t[(j-1)*ldt+j-1:], ldt,
						1, 1, b[(j-1)*ldb+iv-1:], ldb, wr, wi, x[:], 2)
					// Scale X to avoid overflow when updating
					// the right-hand side.
					if xnorm > 1 {
						beta := math.Max(norms[j-1], norms[j])
						if beta > bignum/xnorm {
							rec := 1 / xnorm
							x[0] *= rec
							x[1] *= rec
							x[2] *= rec
							x[3] *= rec
							scale *= rec
						}
					}
					// Scale if necessary.
					if scale != 1 {
						bi.Dscal(ki+1, scale, b[iv-1:], ldb)
						bi.Dscal(ki+1, scale, b[iv:], ldb)
					}
					b[(j-1)*ldb+iv-1] = x[0]
					b[(j-1)*ldb+iv] = x[1]
					b[j*ldb+iv-1] = x[2]
					b[j*ldb+iv] = x[3]
					// Update the right-hand side.
					bi.Daxpy(j-1, -x[0], t[j-1:], ldt, b[iv-1:], ldb)
					bi.Daxpy(j-1, -x[1], t[j-1:], ldt, b[iv:], ldb)
					bi.Daxpy(j-1, -x[2], t[j:], ldt, b[iv-1:], ldb)
					bi.Daxpy(j-1, -x[3], t[j:], ldt, b[iv:], ldb)
					j -= 2
				}
			}

			// Copy the vector x or Q*x to VR and normalize.
			switch {
			case howmny != lapack.EVAllMulQ:
				// No back-transform: copy x to VR and normalize.
				bi.Dcopy(ki+1, b[iv-1:], ldb, vr[is-1:], ldvr)
				bi.Dcopy(ki+1, b[iv:], ldb, vr[is:], ldvr)
				emax := 0.0
				for k := 0; k <= ki; k++ {
					emax = math.Max(emax, math.Abs(vr[k*ldvr+is-1])+math.Abs(vr[k*ldvr+is]))
				}
				remax := 1 / emax
				bi.Dscal(ki+1, remax, vr[is-1:], ldvr)
				bi.Dscal(ki+1, remax, vr[is:], ldvr)
				for k := ki + 1; k < n; k++ {
					vr[k*ldvr+is-1] = 0
					vr[k*ldvr+is] = 0
				}
			case nb == 1:
				// Version 1: back-transform each vector with GEMV, Q*x.
				if ki-1 > 0 {
					bi.Dgemv(blas.NoTrans, n, ki-1, 1, vr, ldvr, b[iv-1:], ldb,
						b[(ki-1)*ldb+iv-1], vr[ki-1:], ldvr)
					bi.Dgemv(blas.NoTrans, n, ki-1, 1, vr, ldvr, b[iv:], ldb,
						b[ki*ldb+iv], vr[ki:], ldvr)
				} else {
					bi.Dscal(n, b[(ki-1)*ldb+iv-1], vr[ki-1:], ldvr)
					bi.Dscal(n, b[ki*ldb+iv], vr[ki:], ldvr)
				}
				emax := 0.0
				for k := 0; k < n; k++ {
					emax = math.Max(emax, math.Abs(vr[k*ldvr+ki-1])+math.Abs(vr[k*ldvr+ki]))
				}
				remax := 1 / emax
				bi.Dscal(n, remax, vr[ki-1:], ldvr)
				bi.Dscal(n, remax, vr[ki:], ldvr)
			default:
				// Version 2: back-transform block of vectors with GEMM.
				// Zero out below vector.
				for k := ki + 1; k < n; k++ {
					b[k*ldb+iv-1] = 0
					b[k*ldb+iv] = 0
				}
				iscomplex[iv-1] = -ip
				iscomplex[iv] = ip
				iv--
				// Back-transform and normalization is done below.
			}
		}
		if nb > 1 {
			// Blocked version of back-transform.

			// For complex case, ki2 includes both vectors (ki-1 and ki).
			ki2 := ki
			if ip != 0 {
				ki2--
			}
			// Columns iv:nb of b are valid vectors.
			// When the number of vectors stored reaches nb-1 or nb,
			// or if this was last vector, do the Gemm.
			if iv < 2 || ki2 == 0 {
				bi.Dgemm(blas.NoTrans, blas.NoTrans, n, nb-iv, ki2+nb-iv,
					1, vr, ldvr, b[iv:], ldb,
					0, b[nb+iv:], ldb)
				// Normalize vectors.
				var remax float64
				for k := iv; k < nb; k++ {
					if iscomplex[k] == 0 {
						// Real eigenvector.
						ii := bi.Idamax(n, b[nb+k:], ldb)
						remax = 1 / math.Abs(b[ii*ldb+nb+k])
					} else if iscomplex[k] == 1 {
						// First eigenvector of conjugate pair.
						emax := 0.0
						for ii := 0; ii < n; ii++ {
							emax = math.Max(emax, math.Abs(b[ii*ldb+nb+k])+math.Abs(b[ii*ldb+nb+k+1]))
						}
						remax = 1 / emax
						// Second eigenvector of conjugate pair
						// will reuse this value of remax.
					}
					bi.Dscal(n, remax, b[nb+k:], ldb)
				}
				impl.Dlacpy(blas.All, n, nb-iv, b[nb+iv:], ldb, vr[ki2:], ldvr)
				iv = nb - 1
			} else {
				iv--
			}
		}
		is--
		if ip != 0 {
			is--
		}
	}

	if side == lapack.EVRight {
		return m
	}

leftev:
	// Compute left eigenvectors.

	// For complex left vector, iv is for real part and iv+1 for complex
	// part. Non-blocked version always uses iv=0. Blocked version starts
	// with iv=0, goes up to nb-2 or nb-1.
	iv = 0
	ip = 0
	is = 0
	for ki := 0; ki < n; ki++ {
		if ip == 1 {
			// Previous iteration ki-1 was first of conjugate pair,
			// so this ki is second of conjugate pair.
			ip = -1
			continue
		}

		if ki == n-1 || t[(ki+1)*ldt+ki] == 0 {
			// Last column or zero on sub-diagonal, so this ki must
			// be real eigenvalue.
			ip = 0
		} else {
			// Non-zero on sub-diagonal, so this ki is first of
			// conjugate pair.
			ip = 1
		}
		if howmny == lapack.EVSelected && !selected[ki] {
			continue
		}

		// Compute the ki-th eigenvalue (wr,wi).
		wr := t[ki*ldt+ki]
		var wi float64
		if ip != 0 {
			wi = math.Sqrt(math.Abs(t[ki*ldt+ki+1])) * math.Sqrt(math.Abs(t[(ki+1)*ldt+ki]))
		}
		smin := math.Max(ulp*(math.Abs(wr)+math.Abs(wi)), smlnum)

		if ip == 0 {
			// Real left eigenvector.

			b[ki*ldb+iv] = 1
			// Form right-hand side.
			for k := ki + 1; k < n; k++ {
				b[k*ldb+iv] = -t[ki*ldt+k]
			}
			// Solve transposed quasi-triangular system:
			//  [ T[ki+1:n,ki+1:n] - wr ]ᵀ * X = scale*b
			vmax := 1.0
			vcrit := bignum
			for j := ki + 1; j < n; {
				if j == n-1 || t[(j+1)*ldt+j] == 0 {
					// 1×1 diagonal block.

					// Scale if necessary to avoid overflow
					// when forming the right-hand side.
					if norms[j] > vcrit {
						rec := 1 / vmax
						bi.Dscal(n-ki, rec, b[ki*ldb+iv:], ldb)
						vmax = 1
					}
					b[j*ldb+iv] -= bi.Ddot(j-ki-1, t[(ki+1)*ldt+j:], ldt, b[(ki+1)*ldb+iv:], ldb)
					// Solve [ T[j,j] - wr ]ᵀ * X = b.
					scale, _, _ := impl.Dlaln2(false, 1, 1, smin, 1, t[j*ldt+j:], ldt,
						1, 1, b[j*ldb+iv:], ldb, wr, 0, x[:1], 2)
					// Scale if necessary.
					if scale != 1 {
						bi.Dscal(n-ki, scale, b[ki*ldb+iv:], ldb)
					}
					b[j*ldb+iv] = x[0]
					vmax = math.Max(math.Abs(b[j*ldb+iv]), vmax)
					vcrit = bignum / vmax
					j++
				} else {
					// 2×2 diagonal block.

					// Scale if necessary to avoid overflow
					// when forming the right-hand side.
					beta := math.Max(norms[j], norms[j+1])
					if beta > vcrit {
						bi.Dscal(n-ki+1, 1/vmax, b[ki*ldb+iv:], 1)
						vmax = 1
					}
					b[j*ldb+iv] -= bi.Ddot(j-ki-1, t[(ki+1)*ldt+j:], ldt, b[(ki+1)*ldb+iv:], ldb)
					b[(j+1)*ldb+iv] -= bi.Ddot(j-ki-1, t[(ki+1)*ldt+j+1:], ldt, b[(ki+1)*ldb+iv:], ldb)
					// Solve
					//  [ T[j,j]-wr  T[j,j+1]      ]ᵀ * X = scale*[ b1 ]
					//  [ T[j+1,j]   T[j+1,j+1]-wr ]              [ b2 ]
					scale, _, _ := impl.Dlaln2(true, 2, 1, smin, 1, t[j*ldt+j:], ldt,
						1, 1, b[j*ldb+iv:], ldb, wr, 0, x[:3], 2)
					// Scale if necessary.
					if scale != 1 {
						bi.Dscal(n-ki, scale, b[ki*ldb+iv:], ldb)
					}
					b[j*ldb+iv] = x[0]
					b[(j+1)*ldb+iv] = x[2]
					vmax = math.Max(vmax, math.Max(math.Abs(b[j*ldb+iv]), math.Abs(b[(j+1)*ldb+iv])))
					vcrit = bignum / vmax
					j += 2
				}
			}
			// Copy the vector x or Q*x to VL and normalize.
			switch {
			case howmny != lapack.EVAllMulQ:
				// No back-transform: copy x to VL and normalize.
				bi.Dcopy(n-ki, b[ki*ldb+iv:], ldb, vl[ki*ldvl+is:], ldvl)
				ii := bi.Idamax(n-ki, vl[ki*ldvl+is:], ldvl) + ki
				remax := 1 / math.Abs(vl[ii*ldvl+is])
				bi.Dscal(n-ki, remax, vl[ki*ldvl+is:], ldvl)
				for k := 0; k < ki; k++ {
					vl[k*ldvl+is] = 0
				}
			case nb == 1:
				// Version 1: back-transform each vector with Gemv, Q*x.
				if n-ki-1 > 0 {
					bi.Dgemv(blas.NoTrans, n, n-ki-1,
						1, vl[ki+1:], ldvl, b[(ki+1)*ldb+iv:], ldb,
						b[ki*ldb+iv], vl[ki:], ldvl)
				}
				ii := bi.Idamax(n, vl[ki:], ldvl)
				remax := 1 / math.Abs(vl[ii*ldvl+ki])
				bi.Dscal(n, remax, vl[ki:], ldvl)
			default:
				// Version 2: back-transform block of vectors with Gemm
				// zero out above vector.
				for k := 0; k < ki; k++ {
					b[k*ldb+iv] = 0
				}
				iscomplex[iv] = ip
				// Back-transform and normalization is done below.
			}
		} else {
			// Complex left eigenvector.

			// Initial solve:
			// [ [ T[ki,ki]   T[ki,ki+1]   ]ᵀ - (wr - i* wi) ]*X = 0.
			// [ [ T[ki+1,ki] T[ki+1,ki+1] ]                 ]
			if math.Abs(t[ki*ldt+ki+1]) >= math.Abs(t[(ki+1)*ldt+ki]) {
				b[ki*ldb+iv] = wi / t[ki*ldt+ki+1]
				b[(ki+1)*ldb+iv+1] = 1
			} else {
				b[ki*ldb+iv] = 1
				b[(ki+1)*ldb+iv+1] = -wi / t[(ki+1)*ldt+ki]
			}
			b[(ki+1)*ldb+iv] = 0
			b[ki*ldb+iv+1] = 0
			// Form right-hand side.
			for k := ki + 2; k < n; k++ {
				b[k*ldb+iv] = -b[ki*ldb+iv] * t[ki*ldt+k]
				b[k*ldb+iv+1] = -b[(ki+1)*ldb+iv+1] * t[(ki+1)*ldt+k]
			}
			// Solve transposed quasi-triangular system:
			// [ T[ki+2:n,ki+2:n]ᵀ - (wr-i*wi) ]*X = b1+i*b2
			vmax := 1.0
			vcrit := bignum
			for j := ki + 2; j < n; {
				if j == n-1 || t[(j+1)*ldt+j] == 0 {
					// 1×1 diagonal block.

					// Scale if necessary to avoid overflow
					// when forming the right-hand side elements.
					if norms[j] > vcrit {
						rec := 1 / vmax
						bi.Dscal(n-ki, rec, b[ki*ldb+iv:], ldb)
						bi.Dscal(n-ki, rec, b[ki*ldb+iv+1:], ldb)
						vmax = 1
					}
					b[j*ldb+iv] -= bi.Ddot(j-ki-2, t[(ki+2)*ldt+j:], ldt, b[(ki+2)*ldb+iv:], ldb)
					b[j*ldb+iv+1] -= bi.Ddot(j-ki-2, t[(ki+2)*ldt+j:], ldt, b[(ki+2)*ldb+iv+1:], ldb)
					// Solve [ T[j,j]-(wr-i*wi) ]*(X11+i*X12) = b1+i*b2.
					scale, _, _ := impl.Dlaln2(false, 1, 2, smin, 1, t[j*ldt+j:], ldt,
						1, 1, b[j*ldb+iv:], ldb, wr, -wi, x[:2], 2)
					// Scale if necessary.
					if scale != 1 {
						bi.Dscal(n-ki, scale, b[ki*ldb+iv:], ldb)
						bi.Dscal(n-ki, scale, b[ki*ldb+iv+1:], ldb)
					}
					b[j*ldb+iv] = x[0]
					b[j*ldb+iv+1] = x[1]
					vmax = math.Max(vmax, math.Max(math.Abs(b[j*ldb+iv]), math.Abs(b[j*ldb+iv+1])))
					vcrit = bignum / vmax
					j++
				} else {
					// 2×2 diagonal block.

					// Scale if necessary to avoid overflow
					// when forming the right-hand side elements.
					if math.Max(norms[j], norms[j+1]) > vcrit {
						rec := 1 / vmax
						bi.Dscal(n-ki, rec, b[ki*ldb+iv:], ldb)
						bi.Dscal(n-ki, rec, b[ki*ldb+iv+1:], ldb)
						vmax = 1
					}
					b[j*ldb+iv] -= bi.Ddot(j-ki-2, t[(ki+2)*ldt+j:], ldt, b[(ki+2)*ldb+iv:], ldb)
					b[j*ldb+iv+1] -= bi.Ddot(j-ki-2, t[(ki+2)*ldt+j:], ldt, b[(ki+2)*ldb+iv+1:], ldb)
					b[(j+1)*ldb+iv] -= bi.Ddot(j-ki-2, t[(ki+2)*ldt+j+1:], ldt, b[(ki+2)*ldb+iv:], ldb)
					b[(j+1)*ldb+iv+1] -= bi.Ddot(j-ki-2, t[(ki+2)*ldt+j+1:], ldt, b[(ki+2)*ldb+iv+1:], ldb)
					// Solve 2×2 complex linear equation
					//  [ [T[j,j]   T[j,j+1]  ]ᵀ - (wr-i*wi)*I ]*X = scale*b
					//  [ [T[j+1,j] T[j+1,j+1]]                ]
					scale, _, _ := impl.Dlaln2(true, 2, 2, smin, 1, t[j*ldt+j:], ldt,
						1, 1, b[j*ldb+iv:], ldb, wr, -wi, x[:], 2)
					// Scale if necessary.
					if scale != 1 {
						bi.Dscal(n-ki, scale, b[ki*ldb+iv:], ldb)
						bi.Dscal(n-ki, scale, b[ki*ldb+iv+1:], ldb)
					}
					b[j*ldb+iv] = x[0]
					b[j*ldb+iv+1] = x[1]
					b[(j+1)*ldb+iv] = x[2]
					b[(j+1)*ldb+iv+1] = x[3]
					vmax01 := math.Max(math.Abs(x[0]), math.Abs(x[1]))
					vmax23 := math.Max(math.Abs(x[2]), math.Abs(x[3]))
					vmax = math.Max(vmax, math.Max(vmax01, vmax23))
					vcrit = bignum / vmax
					j += 2
				}
			}
			// Copy the vector x or Q*x to VL and normalize.
			switch {
			case howmny != lapack.EVAllMulQ:
				// No back-transform: copy x to VL and normalize.
				bi.Dcopy(n-ki, b[ki*ldb+iv:], ldb, vl[ki*ldvl+is:], ldvl)
				bi.Dcopy(n-ki, b[ki*ldb+iv+1:], ldb, vl[ki*ldvl+is+1:], ldvl)
				emax := 0.0
				for k := ki; k < n; k++ {
					emax = math.Max(emax, math.Abs(vl[k*ldvl+is])+math.Abs(vl[k*ldvl+is+1]))
				}
				remax := 1 / emax
				bi.Dscal(n-ki, remax, vl[ki*ldvl+is:], ldvl)
				bi.Dscal(n-ki, remax, vl[ki*ldvl+is+1:], ldvl)
				for k := 0; k < ki; k++ {
					vl[k*ldvl+is] = 0
					vl[k*ldvl+is+1] = 0
				}
			case nb == 1:
				// Version 1: back-transform each vector with GEMV, Q*x.
				if n-ki-2 > 0 {
					bi.Dgemv(blas.NoTrans, n, n-ki-2,
						1, vl[ki+2:], ldvl, b[(ki+2)*ldb+iv:], ldb,
						b[ki*ldb+iv], vl[ki:], ldvl)
					bi.Dgemv(blas.NoTrans, n, n-ki-2,
						1, vl[ki+2:], ldvl, b[(ki+2)*ldb+iv+1:], ldb,
						b[(ki+1)*ldb+iv+1], vl[ki+1:], ldvl)
				} else {
					bi.Dscal(n, b[ki*ldb+iv], vl[ki:], ldvl)
					bi.Dscal(n, b[(ki+1)*ldb+iv+1], vl[ki+1:], ldvl)
				}
				emax := 0.0
				for k := 0; k < n; k++ {
					emax = math.Max(emax, math.Abs(vl[k*ldvl+ki])+math.Abs(vl[k*ldvl+ki+1]))
				}
				remax := 1 / emax
				bi.Dscal(n, remax, vl[ki:], ldvl)
				bi.Dscal(n, remax, vl[ki+1:], ldvl)
			default:
				// Version 2: back-transform block of vectors with GEMM.
				// Zero out above vector.
				// Could go from ki-nv+1 to ki-1.
				for k := 0; k < ki; k++ {
					b[k*ldb+iv] = 0
					b[k*ldb+iv+1] = 0
				}
				iscomplex[iv] = ip
				iscomplex[iv+1] = -ip
				iv++
				// Back-transform and normalization is done below.
			}
		}
		if nb > 1 {
			// Blocked version of back-transform.
			// For complex case, ki2 includes both vectors ki and ki+1.
			ki2 := ki
			if ip != 0 {
				ki2++
			}
			// Columns [0:iv] of work are valid vectors. When the
			// number of vectors stored reaches nb-1 or nb, or if
			// this was last vector, do the Gemm.
			if iv >= nb-2 || ki2 == n-1 {
				bi.Dgemm(blas.NoTrans, blas.NoTrans, n, iv+1, n-ki2+iv,
					1, vl[ki2-iv:], ldvl, b[(ki2-iv)*ldb:], ldb,
					0, b[nb:], ldb)
				// Normalize vectors.
				var remax float64
				for k := 0; k <= iv; k++ {
					if iscomplex[k] == 0 {
						// Real eigenvector.
						ii := bi.Idamax(n, b[nb+k:], ldb)
						remax = 1 / math.Abs(b[ii*ldb+nb+k])
					} else if iscomplex[k] == 1 {
						// First eigenvector of conjugate pair.
						emax := 0.0
						for ii := 0; ii < n; ii++ {
							emax = math.Max(emax, math.Abs(b[ii*ldb+nb+k])+math.Abs(b[ii*ldb+nb+k+1]))
						}
						remax = 1 / emax
						// Second eigenvector of conjugate pair
						// will reuse this value of remax.
					}
					bi.Dscal(n, remax, b[nb+k:], ldb)
				}
				impl.Dlacpy(blas.All, n, iv+1, b[nb:], ldb, vl[ki2-iv:], ldvl)
				iv = 0
			} else {
				iv++
			}
		}
		is++
		if ip != 0 {
			is++
		}
	}

	return m
}
