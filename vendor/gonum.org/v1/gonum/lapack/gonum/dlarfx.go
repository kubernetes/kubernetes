// Copyright ©2016 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import "gonum.org/v1/gonum/blas"

// Dlarfx applies an elementary reflector H to a real m×n matrix C, from either
// the left or the right, with loop unrolling when the reflector has order less
// than 11.
//
// H is represented in the form
//  H = I - tau * v * vᵀ,
// where tau is a real scalar and v is a real vector. If tau = 0, then H is
// taken to be the identity matrix.
//
// v must have length equal to m if side == blas.Left, and equal to n if side ==
// blas.Right, otherwise Dlarfx will panic.
//
// c and ldc represent the m×n matrix C. On return, C is overwritten by the
// matrix H * C if side == blas.Left, or C * H if side == blas.Right.
//
// work must have length at least n if side == blas.Left, and at least m if side
// == blas.Right, otherwise Dlarfx will panic. work is not referenced if H has
// order < 11.
//
// Dlarfx is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dlarfx(side blas.Side, m, n int, v []float64, tau float64, c []float64, ldc int, work []float64) {
	switch {
	case side != blas.Left && side != blas.Right:
		panic(badSide)
	case m < 0:
		panic(mLT0)
	case n < 0:
		panic(nLT0)
	case ldc < max(1, n):
		panic(badLdC)
	}

	// Quick return if possible.
	if m == 0 || n == 0 {
		return
	}

	nh := m
	lwork := n
	if side == blas.Right {
		nh = n
		lwork = m
	}
	switch {
	case len(v) < nh:
		panic(shortV)
	case len(c) < (m-1)*ldc+n:
		panic(shortC)
	case nh > 10 && len(work) < lwork:
		panic(shortWork)
	}

	if tau == 0 {
		return
	}

	if side == blas.Left {
		// Form H * C, where H has order m.
		switch m {
		default: // Code for general m.
			impl.Dlarf(side, m, n, v, 1, tau, c, ldc, work)
			return

		case 0: // No-op for zero size matrix.
			return

		case 1: // Special code for 1×1 Householder matrix.
			t0 := 1 - tau*v[0]*v[0]
			for j := 0; j < n; j++ {
				c[j] *= t0
			}
			return

		case 2: // Special code for 2×2 Householder matrix.
			v0 := v[0]
			t0 := tau * v0
			v1 := v[1]
			t1 := tau * v1
			for j := 0; j < n; j++ {
				sum := v0*c[j] + v1*c[ldc+j]
				c[j] -= sum * t0
				c[ldc+j] -= sum * t1
			}
			return

		case 3: // Special code for 3×3 Householder matrix.
			v0 := v[0]
			t0 := tau * v0
			v1 := v[1]
			t1 := tau * v1
			v2 := v[2]
			t2 := tau * v2
			for j := 0; j < n; j++ {
				sum := v0*c[j] + v1*c[ldc+j] + v2*c[2*ldc+j]
				c[j] -= sum * t0
				c[ldc+j] -= sum * t1
				c[2*ldc+j] -= sum * t2
			}
			return

		case 4: // Special code for 4×4 Householder matrix.
			v0 := v[0]
			t0 := tau * v0
			v1 := v[1]
			t1 := tau * v1
			v2 := v[2]
			t2 := tau * v2
			v3 := v[3]
			t3 := tau * v3
			for j := 0; j < n; j++ {
				sum := v0*c[j] + v1*c[ldc+j] + v2*c[2*ldc+j] + v3*c[3*ldc+j]
				c[j] -= sum * t0
				c[ldc+j] -= sum * t1
				c[2*ldc+j] -= sum * t2
				c[3*ldc+j] -= sum * t3
			}
			return

		case 5: // Special code for 5×5 Householder matrix.
			v0 := v[0]
			t0 := tau * v0
			v1 := v[1]
			t1 := tau * v1
			v2 := v[2]
			t2 := tau * v2
			v3 := v[3]
			t3 := tau * v3
			v4 := v[4]
			t4 := tau * v4
			for j := 0; j < n; j++ {
				sum := v0*c[j] + v1*c[ldc+j] + v2*c[2*ldc+j] + v3*c[3*ldc+j] + v4*c[4*ldc+j]
				c[j] -= sum * t0
				c[ldc+j] -= sum * t1
				c[2*ldc+j] -= sum * t2
				c[3*ldc+j] -= sum * t3
				c[4*ldc+j] -= sum * t4
			}
			return

		case 6: // Special code for 6×6 Householder matrix.
			v0 := v[0]
			t0 := tau * v0
			v1 := v[1]
			t1 := tau * v1
			v2 := v[2]
			t2 := tau * v2
			v3 := v[3]
			t3 := tau * v3
			v4 := v[4]
			t4 := tau * v4
			v5 := v[5]
			t5 := tau * v5
			for j := 0; j < n; j++ {
				sum := v0*c[j] + v1*c[ldc+j] + v2*c[2*ldc+j] + v3*c[3*ldc+j] + v4*c[4*ldc+j] +
					v5*c[5*ldc+j]
				c[j] -= sum * t0
				c[ldc+j] -= sum * t1
				c[2*ldc+j] -= sum * t2
				c[3*ldc+j] -= sum * t3
				c[4*ldc+j] -= sum * t4
				c[5*ldc+j] -= sum * t5
			}
			return

		case 7: // Special code for 7×7 Householder matrix.
			v0 := v[0]
			t0 := tau * v0
			v1 := v[1]
			t1 := tau * v1
			v2 := v[2]
			t2 := tau * v2
			v3 := v[3]
			t3 := tau * v3
			v4 := v[4]
			t4 := tau * v4
			v5 := v[5]
			t5 := tau * v5
			v6 := v[6]
			t6 := tau * v6
			for j := 0; j < n; j++ {
				sum := v0*c[j] + v1*c[ldc+j] + v2*c[2*ldc+j] + v3*c[3*ldc+j] + v4*c[4*ldc+j] +
					v5*c[5*ldc+j] + v6*c[6*ldc+j]
				c[j] -= sum * t0
				c[ldc+j] -= sum * t1
				c[2*ldc+j] -= sum * t2
				c[3*ldc+j] -= sum * t3
				c[4*ldc+j] -= sum * t4
				c[5*ldc+j] -= sum * t5
				c[6*ldc+j] -= sum * t6
			}
			return

		case 8: // Special code for 8×8 Householder matrix.
			v0 := v[0]
			t0 := tau * v0
			v1 := v[1]
			t1 := tau * v1
			v2 := v[2]
			t2 := tau * v2
			v3 := v[3]
			t3 := tau * v3
			v4 := v[4]
			t4 := tau * v4
			v5 := v[5]
			t5 := tau * v5
			v6 := v[6]
			t6 := tau * v6
			v7 := v[7]
			t7 := tau * v7
			for j := 0; j < n; j++ {
				sum := v0*c[j] + v1*c[ldc+j] + v2*c[2*ldc+j] + v3*c[3*ldc+j] + v4*c[4*ldc+j] +
					v5*c[5*ldc+j] + v6*c[6*ldc+j] + v7*c[7*ldc+j]
				c[j] -= sum * t0
				c[ldc+j] -= sum * t1
				c[2*ldc+j] -= sum * t2
				c[3*ldc+j] -= sum * t3
				c[4*ldc+j] -= sum * t4
				c[5*ldc+j] -= sum * t5
				c[6*ldc+j] -= sum * t6
				c[7*ldc+j] -= sum * t7
			}
			return

		case 9: // Special code for 9×9 Householder matrix.
			v0 := v[0]
			t0 := tau * v0
			v1 := v[1]
			t1 := tau * v1
			v2 := v[2]
			t2 := tau * v2
			v3 := v[3]
			t3 := tau * v3
			v4 := v[4]
			t4 := tau * v4
			v5 := v[5]
			t5 := tau * v5
			v6 := v[6]
			t6 := tau * v6
			v7 := v[7]
			t7 := tau * v7
			v8 := v[8]
			t8 := tau * v8
			for j := 0; j < n; j++ {
				sum := v0*c[j] + v1*c[ldc+j] + v2*c[2*ldc+j] + v3*c[3*ldc+j] + v4*c[4*ldc+j] +
					v5*c[5*ldc+j] + v6*c[6*ldc+j] + v7*c[7*ldc+j] + v8*c[8*ldc+j]
				c[j] -= sum * t0
				c[ldc+j] -= sum * t1
				c[2*ldc+j] -= sum * t2
				c[3*ldc+j] -= sum * t3
				c[4*ldc+j] -= sum * t4
				c[5*ldc+j] -= sum * t5
				c[6*ldc+j] -= sum * t6
				c[7*ldc+j] -= sum * t7
				c[8*ldc+j] -= sum * t8
			}
			return

		case 10: // Special code for 10×10 Householder matrix.
			v0 := v[0]
			t0 := tau * v0
			v1 := v[1]
			t1 := tau * v1
			v2 := v[2]
			t2 := tau * v2
			v3 := v[3]
			t3 := tau * v3
			v4 := v[4]
			t4 := tau * v4
			v5 := v[5]
			t5 := tau * v5
			v6 := v[6]
			t6 := tau * v6
			v7 := v[7]
			t7 := tau * v7
			v8 := v[8]
			t8 := tau * v8
			v9 := v[9]
			t9 := tau * v9
			for j := 0; j < n; j++ {
				sum := v0*c[j] + v1*c[ldc+j] + v2*c[2*ldc+j] + v3*c[3*ldc+j] + v4*c[4*ldc+j] +
					v5*c[5*ldc+j] + v6*c[6*ldc+j] + v7*c[7*ldc+j] + v8*c[8*ldc+j] + v9*c[9*ldc+j]
				c[j] -= sum * t0
				c[ldc+j] -= sum * t1
				c[2*ldc+j] -= sum * t2
				c[3*ldc+j] -= sum * t3
				c[4*ldc+j] -= sum * t4
				c[5*ldc+j] -= sum * t5
				c[6*ldc+j] -= sum * t6
				c[7*ldc+j] -= sum * t7
				c[8*ldc+j] -= sum * t8
				c[9*ldc+j] -= sum * t9
			}
			return
		}
	}

	// Form C * H, where H has order n.
	switch n {
	default: // Code for general n.
		impl.Dlarf(side, m, n, v, 1, tau, c, ldc, work)
		return

	case 0: // No-op for zero size matrix.
		return

	case 1: // Special code for 1×1 Householder matrix.
		t0 := 1 - tau*v[0]*v[0]
		for j := 0; j < m; j++ {
			c[j*ldc] *= t0
		}
		return

	case 2: // Special code for 2×2 Householder matrix.
		v0 := v[0]
		t0 := tau * v0
		v1 := v[1]
		t1 := tau * v1
		for j := 0; j < m; j++ {
			cs := c[j*ldc:]
			sum := v0*cs[0] + v1*cs[1]
			cs[0] -= sum * t0
			cs[1] -= sum * t1
		}
		return

	case 3: // Special code for 3×3 Householder matrix.
		v0 := v[0]
		t0 := tau * v0
		v1 := v[1]
		t1 := tau * v1
		v2 := v[2]
		t2 := tau * v2
		for j := 0; j < m; j++ {
			cs := c[j*ldc:]
			sum := v0*cs[0] + v1*cs[1] + v2*cs[2]
			cs[0] -= sum * t0
			cs[1] -= sum * t1
			cs[2] -= sum * t2
		}
		return

	case 4: // Special code for 4×4 Householder matrix.
		v0 := v[0]
		t0 := tau * v0
		v1 := v[1]
		t1 := tau * v1
		v2 := v[2]
		t2 := tau * v2
		v3 := v[3]
		t3 := tau * v3
		for j := 0; j < m; j++ {
			cs := c[j*ldc:]
			sum := v0*cs[0] + v1*cs[1] + v2*cs[2] + v3*cs[3]
			cs[0] -= sum * t0
			cs[1] -= sum * t1
			cs[2] -= sum * t2
			cs[3] -= sum * t3
		}
		return

	case 5: // Special code for 5×5 Householder matrix.
		v0 := v[0]
		t0 := tau * v0
		v1 := v[1]
		t1 := tau * v1
		v2 := v[2]
		t2 := tau * v2
		v3 := v[3]
		t3 := tau * v3
		v4 := v[4]
		t4 := tau * v4
		for j := 0; j < m; j++ {
			cs := c[j*ldc:]
			sum := v0*cs[0] + v1*cs[1] + v2*cs[2] + v3*cs[3] + v4*cs[4]
			cs[0] -= sum * t0
			cs[1] -= sum * t1
			cs[2] -= sum * t2
			cs[3] -= sum * t3
			cs[4] -= sum * t4
		}
		return

	case 6: // Special code for 6×6 Householder matrix.
		v0 := v[0]
		t0 := tau * v0
		v1 := v[1]
		t1 := tau * v1
		v2 := v[2]
		t2 := tau * v2
		v3 := v[3]
		t3 := tau * v3
		v4 := v[4]
		t4 := tau * v4
		v5 := v[5]
		t5 := tau * v5
		for j := 0; j < m; j++ {
			cs := c[j*ldc:]
			sum := v0*cs[0] + v1*cs[1] + v2*cs[2] + v3*cs[3] + v4*cs[4] + v5*cs[5]
			cs[0] -= sum * t0
			cs[1] -= sum * t1
			cs[2] -= sum * t2
			cs[3] -= sum * t3
			cs[4] -= sum * t4
			cs[5] -= sum * t5
		}
		return

	case 7: // Special code for 7×7 Householder matrix.
		v0 := v[0]
		t0 := tau * v0
		v1 := v[1]
		t1 := tau * v1
		v2 := v[2]
		t2 := tau * v2
		v3 := v[3]
		t3 := tau * v3
		v4 := v[4]
		t4 := tau * v4
		v5 := v[5]
		t5 := tau * v5
		v6 := v[6]
		t6 := tau * v6
		for j := 0; j < m; j++ {
			cs := c[j*ldc:]
			sum := v0*cs[0] + v1*cs[1] + v2*cs[2] + v3*cs[3] + v4*cs[4] +
				v5*cs[5] + v6*cs[6]
			cs[0] -= sum * t0
			cs[1] -= sum * t1
			cs[2] -= sum * t2
			cs[3] -= sum * t3
			cs[4] -= sum * t4
			cs[5] -= sum * t5
			cs[6] -= sum * t6
		}
		return

	case 8: // Special code for 8×8 Householder matrix.
		v0 := v[0]
		t0 := tau * v0
		v1 := v[1]
		t1 := tau * v1
		v2 := v[2]
		t2 := tau * v2
		v3 := v[3]
		t3 := tau * v3
		v4 := v[4]
		t4 := tau * v4
		v5 := v[5]
		t5 := tau * v5
		v6 := v[6]
		t6 := tau * v6
		v7 := v[7]
		t7 := tau * v7
		for j := 0; j < m; j++ {
			cs := c[j*ldc:]
			sum := v0*cs[0] + v1*cs[1] + v2*cs[2] + v3*cs[3] + v4*cs[4] +
				v5*cs[5] + v6*cs[6] + v7*cs[7]
			cs[0] -= sum * t0
			cs[1] -= sum * t1
			cs[2] -= sum * t2
			cs[3] -= sum * t3
			cs[4] -= sum * t4
			cs[5] -= sum * t5
			cs[6] -= sum * t6
			cs[7] -= sum * t7
		}
		return

	case 9: // Special code for 9×9 Householder matrix.
		v0 := v[0]
		t0 := tau * v0
		v1 := v[1]
		t1 := tau * v1
		v2 := v[2]
		t2 := tau * v2
		v3 := v[3]
		t3 := tau * v3
		v4 := v[4]
		t4 := tau * v4
		v5 := v[5]
		t5 := tau * v5
		v6 := v[6]
		t6 := tau * v6
		v7 := v[7]
		t7 := tau * v7
		v8 := v[8]
		t8 := tau * v8
		for j := 0; j < m; j++ {
			cs := c[j*ldc:]
			sum := v0*cs[0] + v1*cs[1] + v2*cs[2] + v3*cs[3] + v4*cs[4] +
				v5*cs[5] + v6*cs[6] + v7*cs[7] + v8*cs[8]
			cs[0] -= sum * t0
			cs[1] -= sum * t1
			cs[2] -= sum * t2
			cs[3] -= sum * t3
			cs[4] -= sum * t4
			cs[5] -= sum * t5
			cs[6] -= sum * t6
			cs[7] -= sum * t7
			cs[8] -= sum * t8
		}
		return

	case 10: // Special code for 10×10 Householder matrix.
		v0 := v[0]
		t0 := tau * v0
		v1 := v[1]
		t1 := tau * v1
		v2 := v[2]
		t2 := tau * v2
		v3 := v[3]
		t3 := tau * v3
		v4 := v[4]
		t4 := tau * v4
		v5 := v[5]
		t5 := tau * v5
		v6 := v[6]
		t6 := tau * v6
		v7 := v[7]
		t7 := tau * v7
		v8 := v[8]
		t8 := tau * v8
		v9 := v[9]
		t9 := tau * v9
		for j := 0; j < m; j++ {
			cs := c[j*ldc:]
			sum := v0*cs[0] + v1*cs[1] + v2*cs[2] + v3*cs[3] + v4*cs[4] +
				v5*cs[5] + v6*cs[6] + v7*cs[7] + v8*cs[8] + v9*cs[9]
			cs[0] -= sum * t0
			cs[1] -= sum * t1
			cs[2] -= sum * t2
			cs[3] -= sum * t3
			cs[4] -= sum * t4
			cs[5] -= sum * t5
			cs[6] -= sum * t6
			cs[7] -= sum * t7
			cs[8] -= sum * t8
			cs[9] -= sum * t9
		}
		return
	}
}
