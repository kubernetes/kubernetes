// Copyright ©2016 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import "math"

// Dlaqr1 sets v to a scalar multiple of the first column of the product
//  (H - (sr1 + i*si1)*I)*(H - (sr2 + i*si2)*I)
// where H is a 2×2 or 3×3 matrix, I is the identity matrix of the same size,
// and i is the imaginary unit. Scaling is done to avoid overflows and most
// underflows.
//
// n is the order of H and must be either 2 or 3. It must hold that either sr1 =
// sr2 and si1 = -si2, or si1 = si2 = 0. The length of v must be equal to n. If
// any of these conditions is not met, Dlaqr1 will panic.
//
// Dlaqr1 is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dlaqr1(n int, h []float64, ldh int, sr1, si1, sr2, si2 float64, v []float64) {
	if n != 2 && n != 3 {
		panic(badDims)
	}
	checkMatrix(n, n, h, ldh)
	if len(v) != n {
		panic(badSlice)
	}
	if !((sr1 == sr2 && si1 == -si2) || (si1 == 0 && si2 == 0)) {
		panic(badShifts)
	}

	if n == 2 {
		s := math.Abs(h[0]-sr2) + math.Abs(si2) + math.Abs(h[ldh])
		if s == 0 {
			v[0] = 0
			v[1] = 0
		} else {
			h21s := h[ldh] / s
			v[0] = h21s*h[1] + (h[0]-sr1)*((h[0]-sr2)/s) - si1*(si2/s)
			v[1] = h21s * (h[0] + h[ldh+1] - sr1 - sr2)
		}
		return
	}

	s := math.Abs(h[0]-sr2) + math.Abs(si2) + math.Abs(h[ldh]) + math.Abs(h[2*ldh])
	if s == 0 {
		v[0] = 0
		v[1] = 0
		v[2] = 0
	} else {
		h21s := h[ldh] / s
		h31s := h[2*ldh] / s
		v[0] = (h[0]-sr1)*((h[0]-sr2)/s) - si1*(si2/s) + h[1]*h21s + h[2]*h31s
		v[1] = h21s*(h[0]+h[ldh+1]-sr1-sr2) + h[ldh+2]*h31s
		v[2] = h31s*(h[0]+h[2*ldh+2]-sr1-sr2) + h21s*h[2*ldh+1]
	}
}
