// Copyright ©2016 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import "math"

// Iparmq returns problem and machine dependent parameters useful for Dhseqr and
// related subroutines for eigenvalue problems.
//
// ispec specifies the parameter to return:
//  12: Crossover point between Dlahqr and Dlaqr0. Will be at least 11.
//  13: Deflation window size.
//  14: Nibble crossover point. Determines when to skip a multi-shift QR sweep.
//  15: Number of simultaneous shifts in a multishift QR iteration.
//  16: Select structured matrix multiply.
// For other values of ispec Iparmq will panic.
//
// name is the name of the calling function. name must be in uppercase but this
// is not checked.
//
// opts is not used and exists for future use.
//
// n is the order of the Hessenberg matrix H.
//
// ilo and ihi specify the block [ilo:ihi+1,ilo:ihi+1] that is being processed.
//
// lwork is the amount of workspace available.
//
// Except for ispec input parameters are not checked.
//
// Iparmq is an internal routine. It is exported for testing purposes.
func (Implementation) Iparmq(ispec int, name, opts string, n, ilo, ihi, lwork int) int {
	nh := ihi - ilo + 1
	ns := 2
	switch {
	case nh >= 30:
		ns = 4
	case nh >= 60:
		ns = 10
	case nh >= 150:
		ns = max(10, nh/int(math.Log(float64(nh))/math.Ln2))
	case nh >= 590:
		ns = 64
	case nh >= 3000:
		ns = 128
	case nh >= 6000:
		ns = 256
	}
	ns = max(2, ns-(ns%2))

	switch ispec {
	default:
		panic("lapack: bad ispec")

	case 12:
		// Matrices of order smaller than nmin get sent to Dlahqr, the
		// classic double shift algorithm. This must be at least 11.
		const nmin = 75
		return nmin

	case 13:
		const knwswp = 500
		if nh <= knwswp {
			return ns
		}
		return 3 * ns / 2

	case 14:
		// Skip a computationally expensive multi-shift QR sweep with
		// Dlaqr5 whenever aggressive early deflation finds at least
		// nibble*(window size)/100 deflations. The default, small,
		// value reflects the expectation that the cost of looking
		// through the deflation window with Dlaqr3 will be
		// substantially smaller.
		const nibble = 14
		return nibble

	case 15:
		return ns

	case 16:
		if len(name) != 6 {
			panic("lapack: bad name")
		}
		const (
			k22min = 14
			kacmin = 14
		)
		var acc22 int
		switch {
		case name[1:] == "GGHRD" || name[1:] == "GGHD3":
			acc22 = 1
			if nh >= k22min {
				acc22 = 2
			}
		case name[3:] == "EXC":
			if nh >= kacmin {
				acc22 = 1
			}
			if nh >= k22min {
				acc22 = 2
			}
		case name[1:] == "HSEQR" || name[1:5] == "LAQR":
			if ns >= kacmin {
				acc22 = 1
			}
			if ns >= k22min {
				acc22 = 2
			}
		}
		return acc22
	}
}
