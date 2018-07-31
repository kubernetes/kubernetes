// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

// Ilaenv returns algorithm tuning parameters for the algorithm given by the
// input string. ispec specifies the parameter to return:
//  1: The optimal block size for a blocked algorithm.
//  2: The minimum block size for a blocked algorithm.
//  3: The block size of unprocessed data at which a blocked algorithm should
//     crossover to an unblocked version.
//  4: The number of shifts.
//  5: The minimum column dimension for blocking to be used.
//  6: The crossover point for SVD (to use QR factorization or not).
//  7: The number of processors.
//  8: The crossover point for multi-shift in QR and QZ methods for non-symmetric eigenvalue problems.
//  9: Maximum size of the subproblems in divide-and-conquer algorithms.
//  10: ieee NaN arithmetic can be trusted not to trap.
//  11: infinity arithmetic can be trusted not to trap.
//  12...16: parameters for Dhseqr and related functions. See Iparmq for more
//           information.
//
// Ilaenv is an internal routine. It is exported for testing purposes.
func (impl Implementation) Ilaenv(ispec int, s string, opts string, n1, n2, n3, n4 int) int {
	// TODO(btracey): Replace this with a constant lookup? A list of constants?
	sname := s[0] == 'S' || s[0] == 'D'
	cname := s[0] == 'C' || s[0] == 'Z'
	if !sname && !cname {
		panic("lapack: bad name")
	}
	c2 := s[1:3]
	c3 := s[3:6]
	c4 := c3[1:3]

	switch ispec {
	default:
		panic("lapack: bad ispec")
	case 1:
		switch c2 {
		default:
			panic("lapack: bad function name")
		case "GE":
			switch c3 {
			default:
				panic("lapack: bad function name")
			case "TRF":
				if sname {
					return 64
				}
				return 64
			case "QRF", "RQF", "LQF", "QLF":
				if sname {
					return 32
				}
				return 32
			case "HRD":
				if sname {
					return 32
				}
				return 32
			case "BRD":
				if sname {
					return 32
				}
				return 32
			case "TRI":
				if sname {
					return 64
				}
				return 64
			}
		case "PO":
			switch c3 {
			default:
				panic("lapack: bad function name")
			case "TRF":
				if sname {
					return 64
				}
				return 64
			}
		case "SY":
			switch c3 {
			default:
				panic("lapack: bad function name")
			case "TRF":
				if sname {
					return 64
				}
				return 64
			case "TRD":
				return 32
			case "GST":
				return 64
			}
		case "HE":
			switch c3 {
			default:
				panic("lapack: bad function name")
			case "TRF":
				return 64
			case "TRD":
				return 32
			case "GST":
				return 64
			}
		case "OR":
			switch c3[0] {
			default:
				panic("lapack: bad function name")
			case 'G':
				switch c3[1:] {
				default:
					panic("lapack: bad function name")
				case "QR", "RQ", "LQ", "QL", "HR", "TR", "BR":
					return 32
				}
			case 'M':
				switch c3[1:] {
				default:
					panic("lapack: bad function name")
				case "QR", "RQ", "LQ", "QL", "HR", "TR", "BR":
					return 32
				}
			}
		case "UN":
			switch c3[0] {
			default:
				panic("lapack: bad function name")
			case 'G':
				switch c3[1:] {
				default:
					panic("lapack: bad function name")
				case "QR", "RQ", "LQ", "QL", "HR", "TR", "BR":
					return 32
				}
			case 'M':
				switch c3[1:] {
				default:
					panic("lapack: bad function name")
				case "QR", "RQ", "LQ", "QL", "HR", "TR", "BR":
					return 32
				}
			}
		case "GB":
			switch c3 {
			default:
				panic("lapack: bad function name")
			case "TRF":
				if sname {
					if n4 <= 64 {
						return 1
					}
					return 32
				}
				if n4 <= 64 {
					return 1
				}
				return 32
			}
		case "PB":
			switch c3 {
			default:
				panic("lapack: bad function name")
			case "TRF":
				if sname {
					if n4 <= 64 {
						return 1
					}
					return 32
				}
				if n4 <= 64 {
					return 1
				}
				return 32
			}
		case "TR":
			switch c3 {
			default:
				panic("lapack: bad function name")
			case "TRI":
				if sname {
					return 64
				}
				return 64
			case "EVC":
				if sname {
					return 64
				}
				return 64
			}
		case "LA":
			switch c3 {
			default:
				panic("lapack: bad function name")
			case "UUM":
				if sname {
					return 64
				}
				return 64
			}
		case "ST":
			if sname && c3 == "EBZ" {
				return 1
			}
			panic("lapack: bad function name")
		}
	case 2:
		switch c2 {
		default:
			panic("lapack: bad function name")
		case "GE":
			switch c3 {
			default:
				panic("lapack: bad function name")
			case "QRF", "RQF", "LQF", "QLF":
				if sname {
					return 2
				}
				return 2
			case "HRD":
				if sname {
					return 2
				}
				return 2
			case "BRD":
				if sname {
					return 2
				}
				return 2
			case "TRI":
				if sname {
					return 2
				}
				return 2
			}
		case "SY":
			switch c3 {
			default:
				panic("lapack: bad function name")
			case "TRF":
				if sname {
					return 8
				}
				return 8
			case "TRD":
				if sname {
					return 2
				}
				panic("lapack: bad function name")
			}
		case "HE":
			if c3 == "TRD" {
				return 2
			}
			panic("lapack: bad function name")
		case "OR":
			if !sname {
				panic("lapack: bad function name")
			}
			switch c3[0] {
			default:
				panic("lapack: bad function name")
			case 'G':
				switch c4 {
				default:
					panic("lapack: bad function name")
				case "QR", "RQ", "LQ", "QL", "HR", "TR", "BR":
					return 2
				}
			case 'M':
				switch c4 {
				default:
					panic("lapack: bad function name")
				case "QR", "RQ", "LQ", "QL", "HR", "TR", "BR":
					return 2
				}
			}
		case "UN":
			switch c3[0] {
			default:
				panic("lapack: bad function name")
			case 'G':
				switch c4 {
				default:
					panic("lapack: bad function name")
				case "QR", "RQ", "LQ", "QL", "HR", "TR", "BR":
					return 2
				}
			case 'M':
				switch c4 {
				default:
					panic("lapack: bad function name")
				case "QR", "RQ", "LQ", "QL", "HR", "TR", "BR":
					return 2
				}
			}
		}
	case 3:
		switch c2 {
		default:
			panic("lapack: bad function name")
		case "GE":
			switch c3 {
			default:
				panic("lapack: bad function name")
			case "QRF", "RQF", "LQF", "QLF":
				if sname {
					return 128
				}
				return 128
			case "HRD":
				if sname {
					return 128
				}
				return 128
			case "BRD":
				if sname {
					return 128
				}
				return 128
			}
		case "SY":
			if sname && c3 == "TRD" {
				return 32
			}
			panic("lapack: bad function name")
		case "HE":
			if c3 == "TRD" {
				return 32
			}
			panic("lapack: bad function name")
		case "OR":
			switch c3[0] {
			default:
				panic("lapack: bad function name")
			case 'G':
				switch c4 {
				default:
					panic("lapack: bad function name")
				case "QR", "RQ", "LQ", "QL", "HR", "TR", "BR":
					return 128
				}
			}
		case "UN":
			switch c3[0] {
			default:
				panic("lapack: bad function name")
			case 'G':
				switch c4 {
				default:
					panic("lapack: bad function name")
				case "QR", "RQ", "LQ", "QL", "HR", "TR", "BR":
					return 128
				}
			}
		}
	case 4:
		// Used by xHSEQR
		return 6
	case 5:
		// Not used
		return 2
	case 6:
		// Used by xGELSS and xGESVD
		return int(float64(min(n1, n2)) * 1.6)
	case 7:
		// Not used
		return 1
	case 8:
		// Used by xHSEQR
		return 50
	case 9:
		// used by xGELSD and xGESDD
		return 25
	case 10:
		// Go guarantees ieee
		return 1
	case 11:
		// Go guarantees ieee
		return 1
	case 12, 13, 14, 15, 16:
		// Dhseqr and related functions for eigenvalue problems.
		return impl.Iparmq(ispec, s, opts, n1, n2, n3, n4)
	}
}
