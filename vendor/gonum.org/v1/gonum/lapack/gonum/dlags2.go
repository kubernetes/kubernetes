// Copyright ©2017 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import "math"

// Dlags2 computes 2-by-2 orthogonal matrices U, V and Q with the
// triangles of A and B specified by upper.
//
// If upper is true
//
//  U^T*A*Q = U^T*[ a1 a2 ]*Q = [ x  0 ]
//                [ 0  a3 ]     [ x  x ]
// and
//  V^T*B*Q = V^T*[ b1 b2 ]*Q = [ x  0 ]
//                [ 0  b3 ]     [ x  x ]
//
// otherwise
//
//  U^T*A*Q = U^T*[ a1 0  ]*Q = [ x  x ]
//                [ a2 a3 ]     [ 0  x ]
// and
//  V^T*B*Q = V^T*[ b1 0  ]*Q = [ x  x ]
//                [ b2 b3 ]     [ 0  x ].
//
// The rows of the transformed A and B are parallel, where
//
//  U = [  csu  snu ], V = [  csv snv ], Q = [  csq   snq ]
//      [ -snu  csu ]      [ -snv csv ]      [ -snq   csq ]
//
// Dlags2 is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dlags2(upper bool, a1, a2, a3, b1, b2, b3 float64) (csu, snu, csv, snv, csq, snq float64) {
	if upper {
		// Input matrices A and B are upper triangular matrices.
		//
		// Form matrix C = A*adj(B) = [ a b ]
		//                            [ 0 d ]
		a := a1 * b3
		d := a3 * b1
		b := a2*b1 - a1*b2

		// The SVD of real 2-by-2 triangular C.
		//
		//  [ csl -snl ]*[ a b ]*[  csr  snr ] = [ r 0 ]
		//  [ snl  csl ] [ 0 d ] [ -snr  csr ]   [ 0 t ]
		_, _, snr, csr, snl, csl := impl.Dlasv2(a, b, d)

		if math.Abs(csl) >= math.Abs(snl) || math.Abs(csr) >= math.Abs(snr) {
			// Compute the [0, 0] and [0, 1] elements of U^T*A and V^T*B,
			// and [0, 1] element of |U|^T*|A| and |V|^T*|B|.

			ua11r := csl * a1
			ua12 := csl*a2 + snl*a3

			vb11r := csr * b1
			vb12 := csr*b2 + snr*b3

			aua12 := math.Abs(csl)*math.Abs(a2) + math.Abs(snl)*math.Abs(a3)
			avb12 := math.Abs(csr)*math.Abs(b2) + math.Abs(snr)*math.Abs(b3)

			// Zero [0, 1] elements of U^T*A and V^T*B.
			if math.Abs(ua11r)+math.Abs(ua12) != 0 {
				if aua12/(math.Abs(ua11r)+math.Abs(ua12)) <= avb12/(math.Abs(vb11r)+math.Abs(vb12)) {
					csq, snq, _ = impl.Dlartg(-ua11r, ua12)
				} else {
					csq, snq, _ = impl.Dlartg(-vb11r, vb12)
				}
			} else {
				csq, snq, _ = impl.Dlartg(-vb11r, vb12)
			}

			csu = csl
			snu = -snl
			csv = csr
			snv = -snr
		} else {
			// Compute the [1, 0] and [1, 1] elements of U^T*A and V^T*B,
			// and [1, 1] element of |U|^T*|A| and |V|^T*|B|.

			ua21 := -snl * a1
			ua22 := -snl*a2 + csl*a3

			vb21 := -snr * b1
			vb22 := -snr*b2 + csr*b3

			aua22 := math.Abs(snl)*math.Abs(a2) + math.Abs(csl)*math.Abs(a3)
			avb22 := math.Abs(snr)*math.Abs(b2) + math.Abs(csr)*math.Abs(b3)

			// Zero [1, 1] elements of U^T*A and V^T*B, and then swap.
			if math.Abs(ua21)+math.Abs(ua22) != 0 {
				if aua22/(math.Abs(ua21)+math.Abs(ua22)) <= avb22/(math.Abs(vb21)+math.Abs(vb22)) {
					csq, snq, _ = impl.Dlartg(-ua21, ua22)
				} else {
					csq, snq, _ = impl.Dlartg(-vb21, vb22)
				}
			} else {
				csq, snq, _ = impl.Dlartg(-vb21, vb22)
			}

			csu = snl
			snu = csl
			csv = snr
			snv = csr
		}
	} else {
		// Input matrices A and B are lower triangular matrices
		//
		// Form matrix C = A*adj(B) = [ a 0 ]
		//                            [ c d ]
		a := a1 * b3
		d := a3 * b1
		c := a2*b3 - a3*b2

		// The SVD of real 2-by-2 triangular C
		//
		// [ csl -snl ]*[ a 0 ]*[  csr  snr ] = [ r 0 ]
		// [ snl  csl ] [ c d ] [ -snr  csr ]   [ 0 t ]
		_, _, snr, csr, snl, csl := impl.Dlasv2(a, c, d)

		if math.Abs(csr) >= math.Abs(snr) || math.Abs(csl) >= math.Abs(snl) {
			// Compute the [1, 0] and [1, 1] elements of U^T*A and V^T*B,
			// and [1, 0] element of |U|^T*|A| and |V|^T*|B|.

			ua21 := -snr*a1 + csr*a2
			ua22r := csr * a3

			vb21 := -snl*b1 + csl*b2
			vb22r := csl * b3

			aua21 := math.Abs(snr)*math.Abs(a1) + math.Abs(csr)*math.Abs(a2)
			avb21 := math.Abs(snl)*math.Abs(b1) + math.Abs(csl)*math.Abs(b2)

			// Zero [1, 0] elements of U^T*A and V^T*B.
			if (math.Abs(ua21) + math.Abs(ua22r)) != 0 {
				if aua21/(math.Abs(ua21)+math.Abs(ua22r)) <= avb21/(math.Abs(vb21)+math.Abs(vb22r)) {
					csq, snq, _ = impl.Dlartg(ua22r, ua21)
				} else {
					csq, snq, _ = impl.Dlartg(vb22r, vb21)
				}
			} else {
				csq, snq, _ = impl.Dlartg(vb22r, vb21)
			}

			csu = csr
			snu = -snr
			csv = csl
			snv = -snl
		} else {
			// Compute the [0, 0] and [0, 1] elements of U^T *A and V^T *B,
			// and [0, 0] element of |U|^T*|A| and |V|^T*|B|.

			ua11 := csr*a1 + snr*a2
			ua12 := snr * a3

			vb11 := csl*b1 + snl*b2
			vb12 := snl * b3

			aua11 := math.Abs(csr)*math.Abs(a1) + math.Abs(snr)*math.Abs(a2)
			avb11 := math.Abs(csl)*math.Abs(b1) + math.Abs(snl)*math.Abs(b2)

			// Zero [0, 0] elements of U^T*A and V^T*B, and then swap.
			if (math.Abs(ua11) + math.Abs(ua12)) != 0 {
				if aua11/(math.Abs(ua11)+math.Abs(ua12)) <= avb11/(math.Abs(vb11)+math.Abs(vb12)) {
					csq, snq, _ = impl.Dlartg(ua12, ua11)
				} else {
					csq, snq, _ = impl.Dlartg(vb12, vb11)
				}
			} else {
				csq, snq, _ = impl.Dlartg(vb12, vb11)
			}

			csu = snr
			snu = csr
			csv = snl
			snv = csl
		}
	}

	return csu, snu, csv, snv, csq, snq
}
