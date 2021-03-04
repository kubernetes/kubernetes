// Copyright ©2019 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
)

// Dpbtrf computes the Cholesky factorization of an n×n symmetric positive
// definite band matrix
//  A = Uᵀ * U  if uplo == blas.Upper
//  A = L * Lᵀ  if uplo == blas.Lower
// where U is an upper triangular band matrix and L is lower triangular. kd is
// the number of super- or sub-diagonals of A.
//
// The band storage scheme is illustrated below when n = 6 and kd = 2. Elements
// marked * are not used by the function.
//
//  uplo == blas.Upper
//  On entry:         On return:
//   a00  a01  a02     u00  u01  u02
//   a11  a12  a13     u11  u12  u13
//   a22  a23  a24     u22  u23  u24
//   a33  a34  a35     u33  u34  u35
//   a44  a45   *      u44  u45   *
//   a55   *    *      u55   *    *
//
//  uplo == blas.Lower
//  On entry:         On return:
//    *    *   a00       *    *   l00
//    *   a10  a11       *   l10  l11
//   a20  a21  a22      l20  l21  l22
//   a31  a32  a33      l31  l32  l33
//   a42  a43  a44      l42  l43  l44
//   a53  a54  a55      l53  l54  l55
func (impl Implementation) Dpbtrf(uplo blas.Uplo, n, kd int, ab []float64, ldab int) (ok bool) {
	const nbmax = 32

	switch {
	case uplo != blas.Upper && uplo != blas.Lower:
		panic(badUplo)
	case n < 0:
		panic(nLT0)
	case kd < 0:
		panic(kdLT0)
	case ldab < kd+1:
		panic(badLdA)
	}

	// Quick return if possible.
	if n == 0 {
		return true
	}

	if len(ab) < (n-1)*ldab+kd+1 {
		panic(shortAB)
	}

	opts := string(blas.Upper)
	if uplo == blas.Lower {
		opts = string(blas.Lower)
	}
	nb := impl.Ilaenv(1, "DPBTRF", opts, n, kd, -1, -1)
	// The block size must not exceed the semi-bandwidth kd, and must not
	// exceed the limit set by the size of the local array work.
	nb = min(nb, nbmax)

	if nb <= 1 || kd < nb {
		// Use unblocked code.
		return impl.Dpbtf2(uplo, n, kd, ab, ldab)
	}

	// Use blocked code.
	ldwork := nb
	work := make([]float64, nb*ldwork)
	bi := blas64.Implementation()
	if uplo == blas.Upper {
		// Compute the Cholesky factorization of a symmetric band
		// matrix, given the upper triangle of the matrix in band
		// storage.

		// Process the band matrix one diagonal block at a time.
		for i := 0; i < n; i += nb {
			ib := min(nb, n-i)
			// Factorize the diagonal block.
			ok := impl.Dpotf2(uplo, ib, ab[i*ldab:], ldab-1)
			if !ok {
				return false
			}
			if i+ib >= n {
				continue
			}
			// Update the relevant part of the trailing submatrix.
			// If A11 denotes the diagonal block which has just been
			// factorized, then we need to update the remaining
			// blocks in the diagram:
			//
			//  A11   A12   A13
			//        A22   A23
			//              A33
			//
			// The numbers of rows and columns in the partitioning
			// are ib, i2, i3 respectively. The blocks A12, A22 and
			// A23 are empty if ib = kd. The upper triangle of A13
			// lies outside the band.
			i2 := min(kd-ib, n-i-ib)
			if i2 > 0 {
				// Update A12.
				bi.Dtrsm(blas.Left, blas.Upper, blas.Trans, blas.NonUnit, ib, i2,
					1, ab[i*ldab:], ldab-1, ab[i*ldab+ib:], ldab-1)
				// Update A22.
				bi.Dsyrk(blas.Upper, blas.Trans, i2, ib,
					-1, ab[i*ldab+ib:], ldab-1, 1, ab[(i+ib)*ldab:], ldab-1)
			}
			i3 := min(ib, n-i-kd)
			if i3 > 0 {
				// Copy the lower triangle of A13 into the work array.
				for ii := 0; ii < ib; ii++ {
					for jj := 0; jj <= min(ii, i3-1); jj++ {
						work[ii*ldwork+jj] = ab[(i+ii)*ldab+kd-ii+jj]
					}
				}
				// Update A13 (in the work array).
				bi.Dtrsm(blas.Left, blas.Upper, blas.Trans, blas.NonUnit, ib, i3,
					1, ab[i*ldab:], ldab-1, work, ldwork)
				// Update A23.
				if i2 > 0 {
					bi.Dgemm(blas.Trans, blas.NoTrans, i2, i3, ib,
						-1, ab[i*ldab+ib:], ldab-1, work, ldwork,
						1, ab[(i+ib)*ldab+kd-ib:], ldab-1)
				}
				// Update A33.
				bi.Dsyrk(blas.Upper, blas.Trans, i3, ib,
					-1, work, ldwork, 1, ab[(i+kd)*ldab:], ldab-1)
				// Copy the lower triangle of A13 back into place.
				for ii := 0; ii < ib; ii++ {
					for jj := 0; jj <= min(ii, i3-1); jj++ {
						ab[(i+ii)*ldab+kd-ii+jj] = work[ii*ldwork+jj]
					}
				}
			}
		}
	} else {
		// Compute the Cholesky factorization of a symmetric band
		// matrix, given the lower triangle of the matrix in band
		// storage.

		// Process the band matrix one diagonal block at a time.
		for i := 0; i < n; i += nb {
			ib := min(nb, n-i)
			// Factorize the diagonal block.
			ok := impl.Dpotf2(uplo, ib, ab[i*ldab+kd:], ldab-1)
			if !ok {
				return false
			}
			if i+ib >= n {
				continue
			}
			// Update the relevant part of the trailing submatrix.
			// If A11 denotes the diagonal block which has just been
			// factorized, then we need to update the remaining
			// blocks in the diagram:
			//
			//  A11
			//  A21   A22
			//  A31   A32   A33
			//
			// The numbers of rows and columns in the partitioning
			// are ib, i2, i3 respectively. The blocks A21, A22 and
			// A32 are empty if ib = kd. The lowr triangle of A31
			// lies outside the band.
			i2 := min(kd-ib, n-i-ib)
			if i2 > 0 {
				// Update A21.
				bi.Dtrsm(blas.Right, blas.Lower, blas.Trans, blas.NonUnit, i2, ib,
					1, ab[i*ldab+kd:], ldab-1, ab[(i+ib)*ldab+kd-ib:], ldab-1)
				// Update A22.
				bi.Dsyrk(blas.Lower, blas.NoTrans, i2, ib,
					-1, ab[(i+ib)*ldab+kd-ib:], ldab-1, 1, ab[(i+ib)*ldab+kd:], ldab-1)
			}
			i3 := min(ib, n-i-kd)
			if i3 > 0 {
				// Copy the upper triangle of A31 into the work array.
				for ii := 0; ii < i3; ii++ {
					for jj := ii; jj < ib; jj++ {
						work[ii*ldwork+jj] = ab[(ii+i+kd)*ldab+jj-ii]
					}
				}
				// Update A31 (in the work array).
				bi.Dtrsm(blas.Right, blas.Lower, blas.Trans, blas.NonUnit, i3, ib,
					1, ab[i*ldab+kd:], ldab-1, work, ldwork)
				// Update A32.
				if i2 > 0 {
					bi.Dgemm(blas.NoTrans, blas.Trans, i3, i2, ib,
						-1, work, ldwork, ab[(i+ib)*ldab+kd-ib:], ldab-1,
						1, ab[(i+kd)*ldab+ib:], ldab-1)
				}
				// Update A33.
				bi.Dsyrk(blas.Lower, blas.NoTrans, i3, ib,
					-1, work, ldwork, 1, ab[(i+kd)*ldab+kd:], ldab-1)
				// Copy the upper triangle of A31 back into place.
				for ii := 0; ii < i3; ii++ {
					for jj := ii; jj < ib; jj++ {
						ab[(ii+i+kd)*ldab+jj-ii] = work[ii*ldwork+jj]
					}
				}
			}
		}
	}
	return true
}
