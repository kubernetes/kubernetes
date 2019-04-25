// Copyright ©2016 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import "gonum.org/v1/gonum/lapack"

// Dtrexc reorders the real Schur factorization of a n×n real matrix
//  A = Q*T*Q^T
// so that the diagonal block of T with row index ifst is moved to row ilst.
//
// On entry, T must be in Schur canonical form, that is, block upper triangular
// with 1×1 and 2×2 diagonal blocks; each 2×2 diagonal block has its diagonal
// elements equal and its off-diagonal elements of opposite sign.
//
// On return, T will be reordered by an orthogonal similarity transformation Z
// as Z^T*T*Z, and will be again in Schur canonical form.
//
// If compq is lapack.UpdateSchur, on return the matrix Q of Schur vectors will be
// updated by post-multiplying it with Z.
// If compq is lapack.UpdateSchurNone, the matrix Q is not referenced and will not be
// updated.
// For other values of compq Dtrexc will panic.
//
// ifst and ilst specify the reordering of the diagonal blocks of T. The block
// with row index ifst is moved to row ilst, by a sequence of transpositions
// between adjacent blocks.
//
// If ifst points to the second row of a 2×2 block, ifstOut will point to the
// first row, otherwise it will be equal to ifst.
//
// ilstOut will point to the first row of the block in its final position. If ok
// is true, ilstOut may differ from ilst by +1 or -1.
//
// It must hold that
//  0 <= ifst < n, and  0 <= ilst < n,
// otherwise Dtrexc will panic.
//
// If ok is false, two adjacent blocks were too close to swap because the
// problem is very ill-conditioned. T may have been partially reordered, and
// ilstOut will point to the first row of the block at the position to which it
// has been moved.
//
// work must have length at least n, otherwise Dtrexc will panic.
//
// Dtrexc is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dtrexc(compq lapack.UpdateSchurComp, n int, t []float64, ldt int, q []float64, ldq int, ifst, ilst int, work []float64) (ifstOut, ilstOut int, ok bool) {
	switch {
	case compq != lapack.UpdateSchur && compq != lapack.UpdateSchurNone:
		panic(badUpdateSchurComp)
	case n < 0:
		panic(nLT0)
	case ldt < max(1, n):
		panic(badLdT)
	case ldq < 1, compq == lapack.UpdateSchur && ldq < n:
		panic(badLdQ)
	case (ifst < 0 || n <= ifst) && n > 0:
		panic(badIfst)
	case (ilst < 0 || n <= ilst) && n > 0:
		panic(badIlst)
	}

	// Quick return if possible.
	if n == 0 {
		return ifst, ilst, true
	}

	switch {
	case len(t) < (n-1)*ldt+n:
		panic(shortT)
	case compq == lapack.UpdateSchur && len(q) < (n-1)*ldq+n:
		panic(shortQ)
	case len(work) < n:
		panic(shortWork)
	}

	// Quick return if possible.
	if n == 1 {
		return ifst, ilst, true
	}

	// Determine the first row of specified block
	// and find out it is 1×1 or 2×2.
	if ifst > 0 && t[ifst*ldt+ifst-1] != 0 {
		ifst--
	}
	nbf := 1 // Size of the first block.
	if ifst+1 < n && t[(ifst+1)*ldt+ifst] != 0 {
		nbf = 2
	}
	// Determine the first row of the final block
	// and find out it is 1×1 or 2×2.
	if ilst > 0 && t[ilst*ldt+ilst-1] != 0 {
		ilst--
	}
	nbl := 1 // Size of the last block.
	if ilst+1 < n && t[(ilst+1)*ldt+ilst] != 0 {
		nbl = 2
	}

	ok = true
	wantq := compq == lapack.UpdateSchur

	switch {
	case ifst == ilst:
		return ifst, ilst, true

	case ifst < ilst:
		// Update ilst.
		switch {
		case nbf == 2 && nbl == 1:
			ilst--
		case nbf == 1 && nbl == 2:
			ilst++
		}
		here := ifst
		for here < ilst {
			// Swap block with next one below.
			if nbf == 1 || nbf == 2 {
				// Current block either 1×1 or 2×2.
				nbnext := 1 // Size of the next block.
				if here+nbf+1 < n && t[(here+nbf+1)*ldt+here+nbf] != 0 {
					nbnext = 2
				}
				ok = impl.Dlaexc(wantq, n, t, ldt, q, ldq, here, nbf, nbnext, work)
				if !ok {
					return ifst, here, false
				}
				here += nbnext
				// Test if 2×2 block breaks into two 1×1 blocks.
				if nbf == 2 && t[(here+1)*ldt+here] == 0 {
					nbf = 3
				}
				continue
			}

			// Current block consists of two 1×1 blocks each of
			// which must be swapped individually.
			nbnext := 1 // Size of the next block.
			if here+3 < n && t[(here+3)*ldt+here+2] != 0 {
				nbnext = 2
			}
			ok = impl.Dlaexc(wantq, n, t, ldt, q, ldq, here+1, 1, nbnext, work)
			if !ok {
				return ifst, here, false
			}
			if nbnext == 1 {
				// Swap two 1×1 blocks, no problems possible.
				impl.Dlaexc(wantq, n, t, ldt, q, ldq, here, 1, nbnext, work)
				here++
				continue
			}
			// Recompute nbnext in case 2×2 split.
			if t[(here+2)*ldt+here+1] == 0 {
				nbnext = 1
			}
			if nbnext == 2 {
				// 2×2 block did not split.
				ok = impl.Dlaexc(wantq, n, t, ldt, q, ldq, here, 1, nbnext, work)
				if !ok {
					return ifst, here, false
				}
			} else {
				// 2×2 block did split.
				impl.Dlaexc(wantq, n, t, ldt, q, ldq, here, 1, 1, work)
				impl.Dlaexc(wantq, n, t, ldt, q, ldq, here+1, 1, 1, work)
			}
			here += 2
		}
		return ifst, here, true

	default: // ifst > ilst
		here := ifst
		for here > ilst {
			// Swap block with next one above.
			if nbf == 1 || nbf == 2 {
				// Current block either 1×1 or 2×2.
				nbnext := 1
				if here-2 >= 0 && t[(here-1)*ldt+here-2] != 0 {
					nbnext = 2
				}
				ok = impl.Dlaexc(wantq, n, t, ldt, q, ldq, here-nbnext, nbnext, nbf, work)
				if !ok {
					return ifst, here, false
				}
				here -= nbnext
				// Test if 2×2 block breaks into two 1×1 blocks.
				if nbf == 2 && t[(here+1)*ldt+here] == 0 {
					nbf = 3
				}
				continue
			}

			// Current block consists of two 1×1 blocks each of
			// which must be swapped individually.
			nbnext := 1
			if here-2 >= 0 && t[(here-1)*ldt+here-2] != 0 {
				nbnext = 2
			}
			ok = impl.Dlaexc(wantq, n, t, ldt, q, ldq, here-nbnext, nbnext, 1, work)
			if !ok {
				return ifst, here, false
			}
			if nbnext == 1 {
				// Swap two 1×1 blocks, no problems possible.
				impl.Dlaexc(wantq, n, t, ldt, q, ldq, here, nbnext, 1, work)
				here--
				continue
			}
			// Recompute nbnext in case 2×2 split.
			if t[here*ldt+here-1] == 0 {
				nbnext = 1
			}
			if nbnext == 2 {
				// 2×2 block did not split.
				ok = impl.Dlaexc(wantq, n, t, ldt, q, ldq, here-1, 2, 1, work)
				if !ok {
					return ifst, here, false
				}
			} else {
				// 2×2 block did split.
				impl.Dlaexc(wantq, n, t, ldt, q, ldq, here, 1, 1, work)
				impl.Dlaexc(wantq, n, t, ldt, q, ldq, here-1, 1, 1, work)
			}
			here -= 2
		}
		return ifst, here, true
	}
}
