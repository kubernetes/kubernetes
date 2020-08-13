// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import "gonum.org/v1/gonum/lapack"

// Implementation is the native Go implementation of LAPACK routines. It
// is built on top of calls to the return of blas64.Implementation(), so while
// this code is in pure Go, the underlying BLAS implementation may not be.
type Implementation struct{}

var _ lapack.Float64 = Implementation{}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func abs(a int) int {
	if a < 0 {
		return -a
	}
	return a
}

const (
	// dlamchE is the machine epsilon. For IEEE this is 2^{-53}.
	dlamchE = 1.0 / (1 << 53)
	// TODO(kortschak) Replace this with 0x1p-53 when go1.12 is no
	// longer supported.

	// dlamchB is the radix of the machine (the base of the number system).
	dlamchB = 2

	// dlamchP is base * eps.
	dlamchP = dlamchB * dlamchE

	// dlamchS is the "safe minimum", that is, the lowest number such that
	// 1/dlamchS does not overflow, or also the smallest normal number.
	// For IEEE this is 2^{-1022}.
	dlamchS = 1.0 / (1 << 256) / (1 << 256) / (1 << 256) / (1 << 254)
	// TODO(kortschak) Replace this with 0x1p-1022 when go1.12 is no
	// longer supported.
)
